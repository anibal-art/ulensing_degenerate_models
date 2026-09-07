#!/usr/bin/env python3

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--start", required=True, type=int)
    ap.add_argument("--stop", required=True, type=int)
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--workers", required=True, type=int)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    config_path = Path(args.config).resolve()

    cfg = json.loads(config_path.read_text())
    execution = cfg.get("execution", {})

    enabled = bool(execution.get("archive_chunk_artifacts", False))
    archive_format = str(execution.get("archive_format", "tar")).lower()
    delete_sources = bool(
        execution.get("archive_delete_sources_after_verify", False)
    )

    if not enabled:
        print("[archive] disabled by config")
        return

    if archive_format != "tar":
        raise RuntimeError(
            f"Unsupported archive_format={archive_format!r}; only tar is supported."
        )

    summary = run_dir / "logs" / "run_summary.parquet"
    if not summary.is_file():
        raise RuntimeError(f"Missing final run summary: {summary}")

    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    stem = f"artifacts_rows_{args.start}_{args.stop}"
    archive = artifact_dir / f"{stem}.tar"
    tmp_archive = artifact_dir / f"{stem}.tar.tmp"
    roots_file = artifact_dir / f"{stem}.roots.txt"
    manifest_file = artifact_dir / f"{stem}.manifest.json"

    roots = []

    for name in ("fits", "models", "results"):
        path = run_dir / name
        if path.is_dir():
            roots.append(name)

    logs_dir = run_dir / "logs"
    if logs_dir.is_dir():
        for path in sorted(logs_dir.iterdir()):
            if path.is_dir():
                roots.append(str(path.relative_to(run_dir)))

    roots_file.write_text(
        "".join(f"{root}\n" for root in roots)
    )

    if tmp_archive.exists():
        tmp_archive.unlink()

    print(f"[archive] roots={len(roots)}")
    print(f"[archive] creating {archive}")

    cmd = [
        "tar",
        "-cf",
        str(tmp_archive),
        "-C",
        str(run_dir),
        "-T",
        str(roots_file),
    ]

    subprocess.run(cmd, check=True)

    # Full archive integrity/readability check before deleting anything.
    subprocess.run(
        ["tar", "-tf", str(tmp_archive)],
        stdout=subprocess.DEVNULL,
        check=True,
    )

    tmp_archive.replace(archive)

    digest = sha256_file(archive)
    sha_file = archive.with_suffix(archive.suffix + ".sha256")
    sha_file.write_text(f"{digest}  {archive.name}\n")

    archive_size = archive.stat().st_size

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_tag": args.run_tag,
        "run_dir": str(run_dir),
        "catalog_row_start": args.start,
        "catalog_row_stop": args.stop,
        "workers": args.workers,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "archive": str(archive),
        "archive_format": "tar",
        "archive_sha256": digest,
        "archive_size_bytes": archive_size,
        "source_roots": roots,
        "delete_sources_after_verify": delete_sources,
    }

    manifest_file.write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print(f"[archive] verified sha256={digest}")
    print(f"[archive] size={archive_size} bytes")

    if delete_sources:
        print("[archive] deleting archived source trees")

        for root in roots:
            path = run_dir / root

            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()

    # Verify the archive once more after source deletion.
    subprocess.run(
        ["tar", "-tf", str(archive)],
        stdout=subprocess.DEVNULL,
        check=True,
    )

    print("[archive] completed successfully")


if __name__ == "__main__":
    main()
