"""Prepare requirement-selected ONNX Light wheels for pip and isolated builds.

Run with the same Python/platform, package indexes and project extras as the
subsequent install (use pip's PIP_INDEX_URL/PIP_EXTRA_INDEX_URL environment).
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
import tempfile
import tomllib
from urllib.error import URLError
from urllib.parse import parse_qs, quote, urlparse, urlsplit, urlunsplit
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


def source_key(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, "")
    )


def source_keys(source: str | dict) -> set[str]:
    """Return matching identities for a source URL or pip download report."""
    vcs_info = {}
    subdirectory = None
    if isinstance(source, dict):
        url = source["url"]
        vcs_info = source.get("vcs_info") or {}
        subdirectory = source.get("subdirectory")
    else:
        url = source

    parts = urlsplit(url)
    if "+" not in parts.scheme and not vcs_info:
        return {source_key(url)}

    vcs, separator, transport = parts.scheme.partition("+")
    if not separator:
        vcs = vcs_info["vcs"]
        transport = parts.scheme
    path = parts.path
    requested_revision = vcs_info.get("requested_revision")
    if separator:
        repository_path, revision_separator, url_revision = path.rpartition("@")
        if revision_separator:
            path = repository_path
            requested_revision = requested_revision or url_revision
    resolved_revision = vcs_info.get("commit_id")
    subdirectory = subdirectory or parse_qs(parts.fragment).get("subdirectory", [None])[0]
    repository = urlunsplit(
        (transport.lower(), parts.netloc.lower(), path, parts.query, "")
    )
    revisions = {revision for revision in (requested_revision, resolved_revision) if revision}
    if not revisions:
        revisions.add("")
    suffix = f"#subdirectory={quote(subdirectory, safe='/')}" if subdirectory else ""
    return {
        f"{vcs.lower()}+{repository}@{revision}{suffix}" if revision else
        f"{vcs.lower()}+{repository}{suffix}"
        for revision in revisions
    }


def selected_build_records(captured: list[dict], selected_sources: set[str]) -> list[dict]:
    return [
        record for record in captured
        if source_keys(record["source"]) & selected_sources
    ]


def selected_build_requirements(
    captured: list[dict], selected_sources: set[str], known: set[str],
    resolved: list[dict] | None = None,
) -> list[tuple[str, str]]:
    requirements = []
    packages = {
        canonicalize_name(item["metadata"]["name"]): item
        for item in resolved or []
    }
    visited = set()

    def include(text: str, extras: set[str], label: str) -> None:
        req = Requirement(text)
        if (
            canonicalize_name(req.name) in PACKAGES
            and applicable(req, extras)
            and constraint(text) not in known
        ):
            print(f"{label}: {text}", flush=True)
            requirements.append((label, text))
            known.add(constraint(text))

    def visit(text: str, extras: set[str], label: str) -> None:
        req = Requirement(text)
        if not applicable(req, extras):
            return
        name = canonicalize_name(req.name)
        item = packages.get(name)
        if item is None:
            return
        include(text, extras, label)
        key = (name, frozenset(req.extras))
        if key in visited:
            return
        visited.add(key)
        for dependency in item["metadata"].get("requires_dist", []):
            visit(dependency, set(req.extras), label)

    for record in selected_build_records(captured, selected_sources):
        for text in record["requirements"]:
            label = f"{record['source']} [build requirements]"
            include(text, set(), label)
            visit(text, set(), label)
    return requirements


def run_pip_with_build_requirement_capture(report: Path, args: list[str]) -> None:
    """Run pip while recording exact PEP 517 requirements by source URL."""
    from pip._internal.build_env import BuildEnvironment
    from pip._internal.cli.main import main as pip_main
    from pip._internal.distributions.sdist import SourceDistribution

    captured = []
    active = []
    original_prepare = SourceDistribution.prepare_distribution_metadata
    original_install = BuildEnvironment.install_requirements

    def prepare(distribution, *prepare_args, **prepare_kwargs):
        active.append(distribution)
        try:
            return original_prepare(distribution, *prepare_args, **prepare_kwargs)
        finally:
            active.pop()

    def install(environment, finder, requirements, prefix_as_string, *, kind):
        requirements = list(requirements)
        if active:
            captured.append({
                "source": active[-1].req.link.url,
                "requirements": requirements,
            })
        return original_install(
            environment, finder, requirements, prefix_as_string, kind=kind
        )

    SourceDistribution.prepare_distribution_metadata = prepare
    BuildEnvironment.install_requirements = install
    try:
        result = pip_main(args)
    finally:
        SourceDistribution.prepare_distribution_metadata = original_prepare
        BuildEnvironment.install_requirements = original_install
        report.write_text(json.dumps(captured), encoding="utf-8")
    if result:
        raise SystemExit(result)


def pip(
    *args: str, env: dict[str, str], build_requirements_report: Path | None = None
) -> None:
    if build_requirements_report is not None:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_run-pip",
            str(build_requirements_report),
            *args,
        ]
    else:
        command = [sys.executable, "-m", "pip", *args]
    subprocess.run(command, env=env, check=True)


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
    inherited_constraints = []
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
            inherited = Path(previous).read_text(encoding="utf-8")
            inherited_constraints.append(
                inherited if inherited.endswith("\n") else inherited + "\n"
            )
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
    root_declarations = declarations
    build_inputs = []
    try:
        with tempfile.TemporaryDirectory(prefix="onnx-light-metadata-") as temporary:
            for iteration in range(10):
                iteration_dir = Path(temporary) / str(iteration)
                iteration_dir.mkdir()
                env["TMPDIR"] = str(iteration_dir)
                constraint_lines = list(dict.fromkeys(
                    constraint(text) for _, text in declarations
                ))
                constraints.write_text(
                    "\n".join(constraint_lines) + "\n" + "\n".join(inherited_constraints),
                    encoding="utf-8",
                )
                requirements = []
                for _, text in declarations:
                    req = Requirement(text)
                    req.marker = None
                    requirements.append(str(req))
                build_requirements_report = iteration_dir / "build-requirements.json"
                pip(
                    "install", "--dry-run", "--ignore-installed", "--no-clean",
                    "--report", str(report), *roots, *requirements, *build_inputs, env=env,
                    build_requirements_report=build_requirements_report,
                )
                current = json.loads(report.read_text(encoding="utf-8"))["install"]
                selected_sources = set()
                for item in current:
                    selected_sources.update(source_keys(item["download_info"]))
                selected_sources.update(
                    key for path, _, _ in projects for key in source_keys(path.as_uri())
                )
                known = {constraint(text) for _, text in root_declarations}
                captured = json.loads(build_requirements_report.read_text(encoding="utf-8"))
                build_records = selected_build_records(captured, selected_sources)
                next_build_inputs = list(dict.fromkeys(
                    text for record in build_records for text in record["requirements"]
                ))
                additional = selected_build_requirements(
                    build_records, selected_sources, known, current
                )
                next_declarations = [*root_declarations, *additional]
                if {
                    constraint(text) for _, text in next_declarations
                } == {
                    constraint(text) for _, text in declarations
                } and next_build_inputs == build_inputs:
                    break
                declarations = next_declarations
                build_inputs = next_build_inputs
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
    if len(sys.argv) > 2 and sys.argv[1] == "--_run-pip":
        run_pip_with_build_requirement_capture(Path(sys.argv[2]), sys.argv[3:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", action="append", required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.project, args.wheelhouse)


if __name__ == "__main__":
    main()
