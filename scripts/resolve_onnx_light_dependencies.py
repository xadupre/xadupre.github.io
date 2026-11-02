"""Prepare requirement-selected ONNX Light wheels for pip and isolated builds.

Run with the same Python/platform, package indexes and project extras as the
subsequent install (use pip's PIP_INDEX_URL/PIP_EXTRA_INDEX_URL environment).
The wheelhouse and constraints are exported through GITHUB_ENV, not just passed
to the outer pip process: build-system.requires needs the same release access.
"""

from __future__ import annotations

import argparse
from email.parser import Parser
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
from urllib.error import URLError
from urllib.parse import quote, unquote, urlparse, urlsplit, urlunsplit
from urllib.request import Request, url2pathname, urlopen

from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename

PACKAGES = (
    "onnx-light",
    "onnx-light-cpu",
    "onnx-light-ort",
    "onnx-light-kernel-images",
    "onnx-light-viewer",
)


def applicable(requirement: Requirement, extras: set[str]) -> bool:
    return requirement.marker is None or any(
        requirement.marker.evaluate({"extra": extra}) for extra in {"", *extras}
    )


def project_requirements(target: str) -> tuple[Path, set[str], list[tuple[str, str]]]:
    """Read runtime, selected extras (including self references), and build inputs."""
    path_text, separator, extra_text = target.partition("[")
    extras = (
        {canonicalize_name(extra) for extra in extra_text.rstrip("]").split(",")}
        if separator else set()
    )
    path = Path(path_text).resolve()
    with (path / "pyproject.toml").open("rb") as stream:
        data = tomllib.load(stream)
    project = data["project"]
    optional = {
        canonicalize_name(extra): values
        for extra, values in project.get("optional-dependencies", {}).items()
    }
    pending = list(extras)
    while pending:
        extra = pending.pop()
        if extra not in optional:
            continue
        for text in optional[extra]:
            req = Requirement(text)
            if (
                canonicalize_name(req.name) == canonicalize_name(project["name"])
                and applicable(req, extras)
            ):
                for nested in {canonicalize_name(value) for value in req.extras} - extras:
                    extras.add(nested)
                    pending.append(nested)
    groups = {
        "project.dependencies": project.get("dependencies", []),
        "build-system.requires": data.get("build-system", {}).get("requires", []),
        **{
            f"project.optional-dependencies.{extra}": optional[extra]
            for extra in sorted(extras)
            if extra in optional
        },
    }
    requirements = []
    for group, values in groups.items():
        for text in values:
            req = Requirement(text)
            name = canonicalize_name(req.name)
            if (
                name in PACKAGES
                and (
                    name != canonicalize_name(project["name"])
                    or group == "build-system.requires"
                )
                and applicable(req, extras)
            ):
                source = f"{path / 'pyproject.toml'} [{group}]"
                print(f"{source}: {text}", flush=True)
                requirements.append((source, text))
    return path, extras, requirements


def github_release_page(url: str) -> list[dict]:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = "Bearer " + token
    try:
        with urlopen(Request(url, headers=headers), timeout=60) as response:
            return json.load(response)
    except URLError as exc:
        raise RuntimeError(
            f"Cannot discover release wheels at {url}: {exc}. "
            "Check GitHub availability and GH_TOKEN access; no latest-release fallback."
        ) from exc


def release_wheels(package: str, read_page=github_release_page) -> list[str]:
    """Discover every compatible full release wheel, not just the latest tag."""
    tags = set(sys_tags())
    links = []
    page = 1
    while True:
        url = (
            f"https://api.github.com/repos/xadupre/{package}/releases"
            f"?per_page=100&page={page}"
        )
        releases = read_page(url)
        for release in releases:
            if release["draft"]:
                continue
            for asset in release["assets"]:
                name = asset["name"]
                if not name.endswith(".whl") or "reduced" in name:
                    continue
                wheel_name, _, _, wheel_tags = parse_wheel_filename(name)
                if canonicalize_name(wheel_name) == package and tags & wheel_tags:
                    link = asset["browser_download_url"]
                    if urlparse(link).scheme != "https":
                        raise ValueError(f"Invalid release asset URL: {link}")
                    links.append(link)
        if len(releases) < 100:
            return links
        page += 1


def constraint(text: str) -> str:
    """Constraints cannot contain extras; markers have already been evaluated."""
    req = Requirement(text)
    if req.url:
        return f"{req.name} @ {req.url}"
    return f"{req.name}{req.specifier}"


def resolved_download_url(download_info: dict) -> str:
    """Return a pip-downloadable URL, pinning VCS sources at the resolved commit."""
    vcs_info = download_info.get("vcs_info")
    parts = urlsplit(download_info["url"])
    if vcs_info:
        url = urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))
        url = f"{vcs_info['vcs']}+{url}@{vcs_info['commit_id']}"
    else:
        url = download_info["url"]
    fragment = parts.fragment
    subdirectory = download_info.get("subdirectory")
    if subdirectory:
        encoded = quote(subdirectory, safe="/")
        fragment = f"{fragment}&subdirectory={encoded}" if fragment else f"subdirectory={encoded}"
    return f"{url}#{fragment}" if fragment else url


def isolated_build_requirements(temporary: Path) -> list[str]:
    """Find requirements installed for isolated build-backend hooks."""
    requirements = []
    for metadata_file in temporary.glob("pip-build-env-*/**/*.dist-info/METADATA"):
        metadata = Parser().parsestr(metadata_file.read_text(encoding="utf-8"))
        name = canonicalize_name(metadata.get("Name", ""))
        if name in PACKAGES:
            direct_url_file = metadata_file.parent / "direct_url.json"
            if direct_url_file.is_file():
                download_info = json.loads(direct_url_file.read_text(encoding="utf-8"))
                url = resolved_download_url(download_info)
                requirements.append(f"{name} @ {url}")
            else:
                requirements.append(name)
    return sorted(set(requirements))


def pip(*args: str, env: dict[str, str]) -> None:
    subprocess.run([sys.executable, "-m", "pip", *args], env=env, check=True)


def prepare(targets: list[str], wheelhouse: Path, links: list[str] | None = None) -> None:
    """Let pip resolve complete project graphs, then lock ecosystem selections."""
    wheelhouse = wheelhouse.resolve()
    wheelhouse.mkdir(parents=True, exist_ok=True)
    projects = [project_requirements(target) for target in targets]
    declarations = [item for _, _, reqs in projects for item in reqs]
    # Include all ecosystem repositories: a non-ecosystem dependency (e.g.
    # modelbuilder) can itself depend on one of these release-only packages.
    if links is None:
        links = [link for package in PACKAGES for link in release_wheels(package)]
    index = wheelhouse / "releases.html"
    index.write_text(
        "\n".join(f'<a href="{html.escape(link, quote=True)}">wheel</a>' for link in links),
        encoding="utf-8",
    )
    constraints = wheelhouse / "constraints.txt"
    constraints.write_text(
        "\n".join(dict.fromkeys(constraint(text) for _, text in declarations)) + "\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    find_links = " ".join(
        value for value in (env.get("PIP_FIND_LINKS", ""), str(index), str(wheelhouse))
        if value
    )
    # A previous invocation must not silently override this project's inputs.
    merged = set()
    for key in ("PIP_CONSTRAINT", "PIP_BUILD_CONSTRAINT"):
        previous = env.get(key)
        if previous and Path(previous).resolve() not in {constraints, *merged}:
            with constraints.open("a", encoding="utf-8") as stream:
                stream.write(Path(previous).read_text(encoding="utf-8"))
                stream.write("\n")
            merged.add(Path(previous).resolve())
        env[key] = str(constraints)
    env["PIP_FIND_LINKS"] = find_links
    report = wheelhouse / "resolution.json"
    roots = [
        str(path) + (f"[{','.join(sorted(extras))}]" if extras else "")
        for path, extras, _ in projects
    ]
    # Pass build requirements alongside runtime requirements so incompatible
    # runtime/build pins fail jointly rather than selecting two native ABIs.
    try:
        with tempfile.TemporaryDirectory(prefix="onnx-light-metadata-") as temporary:
            env["TMPDIR"] = temporary
            for _ in range(10):
                requirements = []
                for _, text in declarations:
                    req = Requirement(text)
                    req.marker = None
                    requirements.append(str(req))
                pip(
                    "install", "--dry-run", "--ignore-installed", "--no-clean",
                    "--report", str(report), *roots, *requirements, env=env,
                )
                # pip's report omits isolated build inputs. Inspect both unpacked
                # source declarations and packages installed for backend hooks, then
                # resolve those jointly with runtime requirements as well.
                sources = [
                    *Path(temporary).glob("pip-install-*/*/pyproject.toml"),
                    *Path(temporary).glob("pip-req-build-*/pyproject.toml"),
                ]
                current = json.loads(report.read_text(encoding="utf-8"))["install"]
                known = {constraint(text) for _, text in declarations}
                additional = []
                for text in isolated_build_requirements(Path(temporary)):
                    if constraint(text) not in known:
                        additional.append((f"{temporary} [isolated build environment]", text))
                        known.add(constraint(text))
                for item in current:
                    info = item["download_info"]
                    if "dir_info" in info and urlparse(info["url"]).scheme == "file":
                        path = Path(url2pathname(unquote(urlparse(info["url"]).path)))
                        source = path / "pyproject.toml"
                        if source.is_file():
                            sources.append(source)
                for source in sources:
                    with source.open("rb") as stream:
                        data = tomllib.load(stream)
                    for text in data.get("build-system", {}).get("requires", []):
                        req = Requirement(text)
                        if (
                            canonicalize_name(req.name) in PACKAGES
                            and applicable(req, set())
                        ):
                            label = f"{source} [build-system.requires]"
                            print(f"{label}: {text}", flush=True)
                            if constraint(text) not in known:
                                additional.append((label, text))
                                known.add(constraint(text))
                if not additional:
                    break
                declarations.extend(additional)
                with constraints.open("a", encoding="utf-8") as stream:
                    stream.write(
                        "\n".join(constraint(text) for _, text in additional) + "\n"
                    )
            else:
                raise RuntimeError("ONNX Light source build requirements did not converge")
    except subprocess.CalledProcessError as exc:
        sources = "\n".join(f"  {source}: {text}" for source, text in declarations)
        raise RuntimeError(
            "Cannot resolve ONNX Light runtime/build requirements for "
            f"{', '.join(targets)} on Python {sys.version.split()[0]} / {sys.platform}."
            f"\nDeclared ecosystem requirements:\n{sources}\nSee pip's diagnostic above."
        ) from exc
    env.pop("TMPDIR", None)
    if "TMPDIR" in os.environ:
        env["TMPDIR"] = os.environ["TMPDIR"]
    resolved = json.loads(report.read_text(encoding="utf-8"))["install"]
    selected = []
    for item in resolved:
        metadata = item["metadata"]
        name = canonicalize_name(metadata["name"])
        for text in metadata.get("requires_dist", []):
            req = Requirement(text)
            if canonicalize_name(req.name) in PACKAGES:
                print(f"{metadata['name']} {metadata['version']} Requires-Dist: {text}")
        if name not in PACKAGES:
            continue
        version = metadata["version"]
        print(f"Selected {name}=={version} from {item['download_info']['url']}", flush=True)
        selected.append(f"{name}=={version}")
        if "dir_info" not in item["download_info"]:
            url = resolved_download_url(item["download_info"])
            hashes = item["download_info"].get("archive_info", {}).get("hashes", {})
            if "sha256" in hashes:
                url += ("&" if "#" in url else "#") + f"sha256={hashes['sha256']}"
            download_env = {
                key: value for key, value in env.items()
                if key not in {"PIP_CONSTRAINT", "PIP_BUILD_CONSTRAINT"}
            }
            pip("download", "--no-deps", "--dest", str(wheelhouse), url, env=download_env)
    with constraints.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(selected) + "\n")
    # Only the selected, downloaded wheels are offered to subsequent installs.
    # Preserve pre-existing links for unrelated package sources.
    final_links = " ".join(
        value for value in (os.environ.get("PIP_FIND_LINKS", ""), str(wheelhouse)) if value
    )
    exported = {
        "PIP_FIND_LINKS": final_links,
        "PIP_CONSTRAINT": str(constraints),
        "PIP_BUILD_CONSTRAINT": str(constraints),
    }
    for key, value in exported.items():
        if "\n" in value or "\r" in value:
            raise ValueError(f"Invalid newline in {key}")
        print(f"{key}={value}", flush=True)
    if os.environ.get("GITHUB_ENV"):
        with Path(os.environ["GITHUB_ENV"]).open("a", encoding="utf-8") as stream:
            for key, value in exported.items():
                stream.write(f"{key}={value}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", action="append", required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.project, args.wheelhouse)


if __name__ == "__main__":
    main()
