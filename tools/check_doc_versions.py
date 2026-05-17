#!/usr/bin/env python3
"""Check documented `chess-corners` dependency versions.

The workspace version is the source of truth. Public install snippets
should use the semver-compatible `major.minor` form:

    chess-corners = "0.11"

That avoids patch-release churn while still forcing docs to update for
minor releases. Historical changelogs and proposal documents are not
checked.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEP_RE = re.compile(r'(chess-corners\s*=\s*")([0-9]+\.[0-9]+(?:\.[0-9]+)?)(")')


def workspace_version() -> str:
    in_workspace_package = False
    for raw in (ROOT / "Cargo.toml").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line == "[workspace.package]":
            in_workspace_package = True
            continue
        if in_workspace_package and line.startswith("["):
            break
        if in_workspace_package:
            match = re.match(r'version\s*=\s*"([^"]+)"', line)
            if match:
                return match.group(1)
    raise RuntimeError("could not find [workspace.package].version in Cargo.toml")


def doc_version(version: str) -> str:
    parts = version.split(".")
    if len(parts) < 2:
        raise RuntimeError(f"workspace version must contain major.minor: {version!r}")
    return ".".join(parts[:2])


def active_doc_paths() -> list[Path]:
    paths: set[Path] = set()
    paths.add(ROOT / "README.md")
    paths.update((ROOT / "book" / "src").glob("**/*.md"))
    paths.update((ROOT / "crates").glob("*/README.md"))
    paths.update((ROOT / "docs").glob("*.md"))

    return sorted(
        p
        for p in paths
        if p.exists()
        and not p.name.startswith("proposal-")
        and "changelog" not in p.parts
    )


def check_file(path: Path, expected: str, fix: bool) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8")
    problems: list[tuple[int, str]] = []

    def replace(match: re.Match[str]) -> str:
        found = match.group(2)
        if found != expected:
            line = text.count("\n", 0, match.start()) + 1
            problems.append((line, found))
        return f"{match.group(1)}{expected}{match.group(3)}"

    updated = DEP_RE.sub(replace, text)
    if fix and updated != text:
        path.write_text(updated, encoding="utf-8")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", action="store_true", help="rewrite stale snippets in active docs")
    args = parser.parse_args()

    expected = doc_version(workspace_version())
    failures: list[tuple[Path, int, str]] = []

    for path in active_doc_paths():
        for line, found in check_file(path, expected, args.fix):
            failures.append((path, line, found))

    if failures:
        action = "Updated" if args.fix else "Found"
        print(f"{action} documented chess-corners version drift; expected {expected}:")
        for path, line, found in failures:
            rel = path.relative_to(ROOT)
            print(f"  {rel}:{line}: chess-corners = {found!r}")
        if not args.fix:
            print("\nRun: python3 tools/check_doc_versions.py --fix")
            return 1

    print(f"documented chess-corners dependency snippets match {expected}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
