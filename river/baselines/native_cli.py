"""Readiness commands for reproducible native baseline adapters."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from river.registry import available_baselines

from .native import (
    NativeBaselineManifest,
    _credential_status,
    _looks_like_commit,
    _normalize_command,
    load_native_manifests,
)


ALIASES = {"graphrag": "graphrag_local"}


def load_manifest(baseline_id: str) -> NativeBaselineManifest:
    manifests = load_native_manifests()
    manifest_id = ALIASES.get(baseline_id, baseline_id)
    try:
        return manifests[manifest_id]
    except KeyError as exc:
        raise SystemExit(f"No native baseline manifest for {baseline_id!r}") from exc


def credential_report(manifest: NativeBaselineManifest) -> dict[str, Any]:
    status = _credential_status(manifest.required_credentials)
    missing = [name for name, present in status.items() if not present]
    return {
        "required": manifest.required_credentials,
        "present": [name for name, present in status.items() if present],
        "missing": missing,
        "ok": not missing,
    }


def command_report(manifest: NativeBaselineManifest, runner_command: str | list[str] | None = None) -> dict[str, Any]:
    if manifest.baseline_id == "minicheck_no_revision":
        package_available = importlib.util.find_spec("minicheck") is not None
        return {
            "configured": True,
            "ok": package_available,
            "command": None,
            "message": (
                "Official MiniCheck package is importable."
                if package_available
                else "Install the optional dependency with: uv sync --extra minicheck"
            ),
        }

    command = _normalize_command(runner_command) or manifest.default_command
    if not command:
        return {
            "configured": False,
            "ok": False,
            "command": None,
            "message": "No JSON-stdin runner_command configured for the pinned official implementation.",
        }
    executable = command[0]
    found = shutil.which(executable) is not None or Path(executable).exists()
    return {
        "configured": True,
        "ok": found,
        "command": command,
        "message": (
            f"Found runner executable {executable!r}."
            if found
            else f"Runner executable {executable!r} was not found on PATH or as a file."
        ),
    }


def checkout_report(manifest: NativeBaselineManifest, checkout_path: str | None = None) -> dict[str, Any]:
    if not checkout_path:
        return {
            "checked": False,
            "ok": None,
            "path": None,
            "message": "No checkout path supplied; pass --checkout-path or configure official_checkout_path to verify the pinned commit.",
        }
    path = Path(checkout_path)
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return {
            "checked": True,
            "ok": False,
            "path": str(path),
            "message": f"{path} is not a readable Git checkout: {result.stderr.strip()}",
        }
    current = result.stdout.strip()
    expected = manifest.commit_or_tag
    if _looks_like_commit(expected) and current != expected:
        return {
            "checked": True,
            "ok": False,
            "path": str(path),
            "current_commit": current,
            "expected_commit": expected,
            "message": f"Checkout is at {current}, expected {expected}.",
        }
    return {
        "checked": True,
        "ok": True,
        "path": str(path),
        "current_commit": current,
        "expected_commit": expected,
        "message": "Checkout matches the manifest pin.",
    }


def diagnose_baseline(
    baseline_id: str,
    *,
    runner_command: str | list[str] | None = None,
    checkout_path: str | None = None,
) -> dict[str, Any]:
    manifest = load_manifest(baseline_id)
    registered = baseline_id in available_baselines() or ALIASES.get(baseline_id) in available_baselines()
    credentials = credential_report(manifest)
    command = command_report(manifest, runner_command)
    checkout = checkout_report(manifest, checkout_path)
    license_warning = manifest.license.lower().startswith("not declared")
    ready = registered and credentials["ok"] and command["ok"] and (checkout["ok"] is not False)
    return {
        "baseline_id": baseline_id,
        "manifest_id": manifest.baseline_id,
        "display_name": manifest.display_name,
        "registered": registered,
        "ready": ready,
        "license": manifest.license,
        "license_warning": license_warning,
        "supported_execution_modes": manifest.supported_execution_modes,
        "credentials": credentials,
        "runner_command": command,
        "checkout": checkout,
        "installation_command": manifest.installation_command,
        "isolation": manifest.isolation,
        "readiness_notes": manifest.readiness_notes,
    }


def list_availability() -> list[dict[str, Any]]:
    return [
        diagnose_baseline(manifest_id)
        for manifest_id in sorted(load_native_manifests())
    ]


def _print_table(rows: list[dict[str, Any]]) -> None:
    print("baseline_id\tregistered\tready\tmodes\tcredentials\trunner\tlicense")
    for row in rows:
        credentials = "ok" if row["credentials"]["ok"] else "missing:" + ",".join(row["credentials"]["missing"])
        runner = "ok" if row["runner_command"]["ok"] else row["runner_command"]["message"]
        license_status = "warning" if row["license_warning"] else "ok"
        print(
            "\t".join(
                [
                    row["baseline_id"],
                    str(row["registered"]).lower(),
                    str(row["ready"]).lower(),
                    ",".join(row["supported_execution_modes"]),
                    credentials,
                    runner,
                    license_status,
                ]
            )
        )


def _emit(payload: Any, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    elif isinstance(payload, list):
        _print_table(payload)
    else:
        for key, value in payload.items():
            print(f"{key}: {json.dumps(value, sort_keys=True, default=str)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect native baseline readiness without running experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-availability", help="List native baseline availability.")
    list_parser.add_argument("--json", action="store_true")

    diagnose_parser = subparsers.add_parser(
        "diagnose-missing-dependency",
        aliases=["diagnose"],
        help="Diagnose missing native baseline setup.",
    )
    diagnose_parser.add_argument("--baseline", required=True)
    diagnose_parser.add_argument("--runner-command")
    diagnose_parser.add_argument("--checkout-path")
    diagnose_parser.add_argument("--json", action="store_true")

    credentials_parser = subparsers.add_parser(
        "validate-credentials",
        help="Validate required environment variables without making paid calls.",
    )
    credentials_parser.add_argument("--baseline", required=True)
    credentials_parser.add_argument("--json", action="store_true")

    manifest_parser = subparsers.add_parser("print-manifest", help="Print a frozen native baseline manifest.")
    manifest_parser.add_argument("--baseline", required=True)
    manifest_parser.add_argument("--json", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "list-availability":
        rows = list_availability()
        _emit(rows, as_json=args.json)
        return 0
    if args.command in {"diagnose-missing-dependency", "diagnose"}:
        report = diagnose_baseline(
            args.baseline,
            runner_command=args.runner_command,
            checkout_path=args.checkout_path,
        )
        _emit(report, as_json=args.json)
        return 0 if report["ready"] else 1
    if args.command == "validate-credentials":
        manifest = load_manifest(args.baseline)
        report = credential_report(manifest)
        _emit(report, as_json=args.json)
        return 0 if report["ok"] else 1
    if args.command == "print-manifest":
        manifest = load_manifest(args.baseline)
        payload = manifest.as_dict()
        _emit(payload, as_json=True if args.json else True)
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
