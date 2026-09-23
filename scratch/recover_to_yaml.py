#!/usr/bin/env python3
"""Convert recovered sphere JSON (from recover_spheres_console.js) into a
Bubblify-format spherization YAML that can be reloaded with --spherization_yml.

Usage:
    # Save the JSON printed in the browser console to recovered.json, then:
    uv run python scratch/recover_to_yaml.py recovered.json recovered_spheres.yml
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import yaml


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: recover_to_yaml.py <recovered.json> <output.yml>")
        return 2

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])

    data = json.loads(src.read_text())
    spheres = data["spheres"] if isinstance(data, dict) else data

    collision_spheres: dict[str, list] = {}
    for s in spheres:
        radius = s.get("radius")
        if radius is None or radius <= 0:
            continue
        center = [float(c) for c in s["center"]]
        collision_spheres.setdefault(s["link"], []).append(
            {"center": center, "radius": float(radius)}
        )

    out = {
        "collision_spheres": collision_spheres,
        "metadata": {
            "total_spheres": sum(len(v) for v in collision_spheres.values()),
            "links": list(collision_spheres.keys()),
            "export_timestamp": float(time.time()),
            "recovered": True,
        },
    }
    dst.write_text(yaml.dump(out, default_flow_style=False, sort_keys=False))
    print(f"Wrote {out['metadata']['total_spheres']} spheres to {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
