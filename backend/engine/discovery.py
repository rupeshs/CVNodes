"""Shared auto-discovery used by both backend/nodes and backend/custom_nodes.

A node can be a single .py file, or its own subfolder (a package) that
optionally carries a requirements.txt - e.g. backend/custom_nodes/removebg/.
If a subfolder has one, its dependencies are installed automatically before
that node's module is imported, so adding a node with extra dependencies is
just "drop the folder in, restart the server" - no separate pip step.
"""

import hashlib
import importlib
import pkgutil
import subprocess
import sys
from pathlib import Path


def _ensure_requirements_installed(folder: Path) -> None:
    req_file = folder / "requirements.txt"
    if not req_file.exists():
        return

    digest = hashlib.sha1(req_file.read_bytes()).hexdigest()
    marker = folder / ".requirements_installed"
    if marker.exists() and marker.read_text().strip() == digest:
        return  # unchanged since the last successful install - skip pip entirely

    print(f"[cvnodes] Installing dependencies for {folder.name} ({req_file.name})...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r", str(req_file)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[cvnodes] Failed to install dependencies for {folder.name}:\n{result.stderr}")
        return  # the node's import will fail below and get skipped as usual

    marker.write_text(digest)
    print(f"[cvnodes] Dependencies installed for {folder.name}")


def import_all(package_name, package_path):
    """Imports every module in a package, so their @register_node decorators
    run. A module that fails to import (e.g. a custom node whose dependency
    install failed) is skipped with a warning instead of crashing the whole
    server."""

    root = Path(package_path[0])
    for _finder, module_name, is_pkg in pkgutil.iter_modules(package_path):
        if is_pkg:
            _ensure_requirements_installed(root / module_name)

        full_name = f"{package_name}.{module_name}"
        try:
            importlib.import_module(full_name)
        except Exception as exc:  # noqa: BLE001 - a bad node module must not take down the app
            print(f"[cvnodes] Skipped node module '{full_name}': {exc}")
