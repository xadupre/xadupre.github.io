"""Generates the workflow manifest displayed on the home page."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
DEFAULT_OUTPUT = ROOT / "assets" / "workflow-manifest.json"


def build_manifest(workflows: Path = WORKFLOWS) -> list[dict[str, object]]:
    """Builds the sorted workflow manifest."""
    manifest = []
    for path in sorted(workflows.glob("*.yml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        triggers = data.get("on", data.get(True, {})) or {}
        schedules = triggers.get("schedule", []) if isinstance(triggers, dict) else []
        crons = [
            item["cron"]
            for item in schedules
            if isinstance(item, dict) and isinstance(item.get("cron"), str)
        ]
        manifest.append(
            {
                "name": data.get("name", path.stem),
                "path": f".github/workflows/{path.name}",
                "crons": crons,
            }
        )
    return sorted(
        manifest, key=lambda item: (str(item["name"]).lower(), str(item["path"]))
    )


def main() -> None:
    """Writes the workflow manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(build_manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
