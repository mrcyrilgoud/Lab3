#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


RESULT_CELL_TAG = "lab3_autopilot_result"


class NotebookExecutionFailure(RuntimeError):
    def __init__(self, message: str, *, output_path: Path, payload: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.output_path = output_path
        self.payload = payload or {}


@contextmanager
def patched_environ(overrides: dict[str, str]) -> Iterator[None]:
    original: dict[str, str | None] = {}
    try:
        for key, value in overrides.items():
            original[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key, previous in original.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def stringify_env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_notebook_env(overrides: dict[str, Any]) -> dict[str, str]:
    return {
        key: stringify_env_value(value)
        for key, value in overrides.items()
        if value is not None
    }


def output_text(output: dict[str, Any]) -> str:
    if output.get("output_type") == "stream":
        text = output.get("text", "")
        if isinstance(text, list):
            return "".join(text)
        return text

    data = output.get("data", {})
    if "application/json" in data:
        payload = data["application/json"]
        return json.dumps(payload)
    if "text/plain" in data:
        text = data["text/plain"]
        if isinstance(text, list):
            return "".join(text)
        return text
    return ""


def parse_json_text(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def extract_result_payload(notebook: nbformat.NotebookNode) -> dict[str, Any]:
    tagged_cells: list[nbformat.NotebookNode] = []
    for cell in notebook.cells:
        if cell.get("cell_type") != "code":
            continue
        tags = cell.get("metadata", {}).get("tags", [])
        if RESULT_CELL_TAG in tags:
            tagged_cells.append(cell)

    candidate_cells = tagged_cells or [cell for cell in notebook.cells if cell.get("cell_type") == "code"]
    for cell in reversed(candidate_cells):
        for output in reversed(cell.get("outputs", [])):
            payload = parse_json_text(output_text(output))
            if not payload:
                continue
            required = {"candidate_id", "run_root", "summary_path", "report_path", "status"}
            if required.issubset(payload):
                return payload
    raise ValueError("Executed notebook did not emit the expected final JSON payload.")


def execute_notebook(
    *,
    notebook_path: Path,
    output_path: Path,
    working_dir: Path,
    env_overrides: dict[str, str],
) -> dict[str, Any]:
    notebook = nbformat.read(notebook_path, as_version=4)
    ensure_parent(output_path)
    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name="python3",
        resources={"metadata": {"path": str(working_dir)}},
    )

    try:
        with patched_environ(env_overrides):
            client.execute()
    except CellExecutionError as exc:
        nbformat.write(notebook, output_path)
        raise NotebookExecutionFailure(
            f"Notebook execution failed: {exc}",
            output_path=output_path,
        ) from exc
    except Exception as exc:
        nbformat.write(notebook, output_path)
        raise NotebookExecutionFailure(
            f"Notebook execution failed before completion: {exc}",
            output_path=output_path,
        ) from exc

    nbformat.write(notebook, output_path)
    try:
        payload = extract_result_payload(notebook)
    except ValueError as exc:
        raise NotebookExecutionFailure(str(exc), output_path=output_path) from exc
    return payload
