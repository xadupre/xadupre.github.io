"""Prepare requirement-selected ONNX Light wheels for pip and isolated builds.

Run with the same Python/platform and project extras as the subsequent install.
The wheelhouse and constraints are exported through GITHUB_ENV, not just passed
to the outer pip process: build-system.requires needs the same release access.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib
from urllib.parse import urlparse
from urllib.request import Request, urlopen

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
    extras = set(extra_text.rstrip("]").split(",")) if separator else set()
    path = Path(path_text).resolve()
    with (path / "pyproject.toml").open("rb") as stream:
        data = tomllib.load(stream)
    project = data["project"]
    optional = project.get("optional-dependencies", {})
    pending = list(extras)
    while pending:
        extra = pending.pop()
        if extra not in optional:
            raise ValueError(f"{path}: unknown optional dependency group {extra!r}")
        for text in optional[extra]:
            req = Requirement(text)
            if (
                canonicalize_name(req.name) == canonicalize_name(project["name"])
                and applicable(req, extras)
            ):
                for nested in req.extras - extras:
                    extras.add(nested)
                    pending.append(nested)
    groups = {
        "project.dependencies": project.get("dependencies", []),
        "build-system.requires": data.get("build-system", {}).get("requires", []),
        **{
            f"project.optional-dependencies.{extra}": optional[extra]
            for extra in sorted(extras)
        },
    }
    requirements = []
    for group, values in groups.items():
        for text in values:
            req = Requirement(text)
            if canonicalize_name(req.name) in PACKAGES and applicable(req, extras):
                source = f"{path / 'pyproject.toml'} [{group}]"
                print(f"{source}: {text}", flush=True)
                requirements.append((source, text))
    return path, extras, requirements


def release_wheels(package: str) -> list[str]:
    """Discover every compatible full release wheel, not just the latest tag."""
    tags = set(sys_tags())
    links = []
    page = 1
    while True:
        url = (
            f"https://api.github.com/repos/xadupre/{package}/releases"
            f"?per_page=100&page={page}"
        )
        headers = {"Accept": "application/vnd.github+json"}
        token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = "Bearer " + token
        with urlopen(Request(url, headers=headers), timeout=60) as response:
            releases = json.load(response)
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
    for key in ("PIP_CONSTRAINT", "PIP_BUILD_CONSTRAINT"):
        previous = env.get(key)
        if previous and Path(previous).resolve() != constraints:
            with constraints.open("a", encoding="utf-8") as stream:
                stream.write(Path(previous).read_text(encoding="utf-8"))
        env[key] = str(constraints)
    env["PIP_FIND_LINKS"] = find_links
    report = wheelhouse / "resolution.json"
    roots = [
        str(path) + (f"[{','.join(sorted(extras))}]" if extras else "")
        for path, extras, _ in projects
    ]
    # Pass build requirements alongside runtime requirements so incompatible
    # runtime/build pins fail jointly rather than selecting two native ABIs.
    requirements = [text for _, text in declarations]
    try:
        pip(
            "install", "--dry-run", "--ignore-installed", "--report", str(report),
            *roots, *requirements, env=env,
        )
    except subprocess.CalledProcessError as exc:
        sources = "\n".join(f"  {source}: {text}" for source, text in declarations)
        raise RuntimeError(
            "Cannot resolve ONNX Light runtime/build requirements for "
            f"{', '.join(targets)} on Python {sys.version.split()[0]} / {sys.platform}."
            f"\nDeclared ecosystem requirements:\n{sources}\nSee pip's diagnostic above."
        ) from exc
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
            url = item["download_info"]["url"]
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
