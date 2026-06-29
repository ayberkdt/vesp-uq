"""G7 -- provenance-completeness checker: every manifested artifact must match the file on disk.

For each study directory, recomputes the SHA-256 of every file listed in its run manifest and flags
missing or changed (tampered) artifacts. Files on disk that no manifest lists are reported but do not
fail the check. Exits non-zero if any directory has a missing or changed artifact.

    python scripts/run_provenance_check.py --dir outputs/benchmark_suite outputs/journal
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vesp.uq.io.run_artifacts import verify_manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="VESP-UQ provenance-completeness checker (G7).")
    parser.add_argument("--dir", nargs="+", required=True, help="study dir(s) holding a run manifest")
    parser.add_argument("--manifest-name", default=None,
                        help="force a manifest filename (default: manifest.json then run_manifest.json)")
    args = parser.parse_args(argv)

    all_ok = True
    for d in args.dir:
        report = verify_manifest(Path(d), manifest_name=args.manifest_name)
        if report["manifest"] is None:
            print(f"  {d}: NO MANIFEST")
            all_ok = False
            continue
        print(f"  {d}: {len(report['verified'])} verified, {len(report['changed'])} changed, "
              f"{len(report['missing'])} missing, {len(report['unlisted'])} unlisted")
        for name in report["changed"]:
            print(f"    CHANGED: {name}")
        for name in report["missing"]:
            print(f"    MISSING: {name}")
        all_ok = all_ok and report["ok"]

    print(f"[provenance] {'OK -- all artifacts match their manifests' if all_ok else 'FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
