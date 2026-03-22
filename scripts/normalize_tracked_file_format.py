#!/usr/bin/env python3
"""Restore tracked text files to their HEAD newline and EOF-newline style."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )


def git_show_bytes(path: str) -> bytes | None:
    result = subprocess.run(
        ["git", "show", f"HEAD:{path}"],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def modified_tracked_files() -> list[str]:
    result = run_git("diff", "--name-only", "--diff-filter=M", "HEAD", "--")
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git diff failed")
    return [line for line in result.stdout.splitlines() if line]


def is_binary(data: bytes) -> bool:
    return b"\x00" in data


def detect_newline(data: bytes) -> bytes:
    if b"\r\n" in data:
        return b"\r\n"
    if b"\n" in data:
        return b"\n"
    if b"\r" in data:
        return b"\r"
    return b"\n"


def normalize_newlines(data: bytes, newline: bytes) -> bytes:
    normalized = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    if newline != b"\n":
        normalized = normalized.replace(b"\n", newline)
    return normalized


def apply_eof_policy(data: bytes, newline: bytes, has_trailing_newline: bool) -> bytes:
    stripped = data
    while stripped.endswith((b"\r\n", b"\n", b"\r")):
        if stripped.endswith(b"\r\n"):
            stripped = stripped[:-2]
        else:
            stripped = stripped[:-1]

    if has_trailing_newline and stripped:
        return stripped + newline
    if has_trailing_newline and not stripped:
        return newline
    return stripped


def normalize_file(path_str: str) -> str:
    head_bytes = git_show_bytes(path_str)
    if head_bytes is None:
        return f"skip {path_str}: not present in HEAD"

    path = Path(path_str)
    if not path.exists():
        return f"skip {path_str}: file does not exist"

    work_bytes = path.read_bytes()
    if is_binary(head_bytes) or is_binary(work_bytes):
        return f"skip {path_str}: binary file"

    newline = detect_newline(head_bytes)
    has_trailing_newline = head_bytes.endswith((b"\r\n", b"\n", b"\r"))
    normalized = normalize_newlines(work_bytes, newline)
    normalized = apply_eof_policy(normalized, newline, has_trailing_newline)

    if normalized == work_bytes:
        return f"ok   {path_str}: already normalized"

    path.write_bytes(normalized)
    return f"fix  {path_str}: restored tracked file format"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize tracked text files to match the newline and EOF-newline "
            "style stored in HEAD."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Specific file paths to normalize. Defaults to all modified tracked files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        paths = args.paths or modified_tracked_files()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if not paths:
        print("No modified tracked files to normalize.")
        return 0

    for path in paths:
        print(normalize_file(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
