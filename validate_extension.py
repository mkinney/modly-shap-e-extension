#!/usr/bin/env python3
"""Static validator for the Modly Shap-E extension.

This script performs local contract checks only. It does not install packages,
run setup, download weights, or execute generation.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

EXT_DIR = Path(__file__).resolve().parent
REQUIRED_FILES = ("manifest.json", "setup.py", "generator.py", "requirements.txt")
TEST_FILES = ("test_setup.py",)


class ValidationError(Exception):
    pass


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def parse_python(path: Path) -> None:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValidationError(
            f"unable to read Python file {path}: {exc}. Ensure the file exists and is readable."
        ) from exc

    try:
        ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise ValidationError(f"{path.name} has invalid Python syntax: {exc}") from exc


def load_manifest() -> dict:
    path = EXT_DIR / "manifest.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValidationError(f"manifest.json is not valid JSON: {exc}") from exc


def validate_manifest(manifest: dict) -> None:
    expect(manifest.get("id") == "shap-e", "manifest.id must be shap-e")
    expect(manifest.get("type") == "model", "manifest.type must be model")
    expect(manifest.get("generator_class") == "ShapEGenerator", "generator_class must be ShapEGenerator")
    expect(manifest.get("hf_repo") == "openai/shap-e", "top-level hf_repo must be openai/shap-e")
    expect(manifest.get("download_check") == "model_index.json", "download_check must be model_index.json")
    expect(manifest.get("weight_owner_id") == "generate", "weight_owner_id must be generate")

    setup = manifest.get("setup") or {}
    expect(setup.get("downloads_weights") is False, "setup.downloads_weights must be false")
    expect(setup.get("default_downloads") is False, "setup.default_downloads must be false")

    nodes = manifest.get("nodes")
    expect(isinstance(nodes, list) and len(nodes) == 1, "manifest must declare exactly one node")
    node = nodes[0]
    expect(node.get("id") == "generate", "node.id must be generate")
    expect(node.get("input") == "text", "node.input must be text")
    expect(node.get("output") == "mesh", "node.output must be mesh")
    expect(node.get("hf_repo") == "openai/shap-e", "node.hf_repo must be openai/shap-e")
    expect(node.get("download_check") == "model_index.json", "node.download_check must be model_index.json")
    expect(node.get("weight_owner_id") == "generate", "node.weight_owner_id must be generate")

    params = node.get("params_schema")
    expect(isinstance(params, list) and params, "node.params_schema must be a non-empty list")
    ids = {param.get("id") for param in params if isinstance(param, dict)}
    expected = {
        "prompt",
        "num_inference_steps",
        "guidance_scale",
        "frame_size",
        "seed",
        "device",
        "torch_dtype",
        "upright_rotation",
        "output_name",
    }
    missing = sorted(expected - ids)
    expect(not missing, "params_schema missing ids: " + ", ".join(missing))


def validate_requirements() -> None:
    text = (EXT_DIR / "requirements.txt").read_text(encoding="utf-8").lower()
    expect("diffusers" in text, "requirements.txt must include diffusers")
    expect("transformers" in text, "requirements.txt must include transformers")
    expect("trimesh" in text, "requirements.txt must include trimesh")
    for line in text.splitlines():
        stripped = line.strip()
        expect(not stripped.startswith("torch"), "requirements.txt must not include torch; setup.py owns torch lane selection")
        expect(not stripped.startswith("torchvision"), "requirements.txt must not include torchvision; setup.py owns torch lane selection")


def validate_runtime_contract_sources() -> None:
    generator = (EXT_DIR / "generator.py").read_text(encoding="utf-8")
    setup = (EXT_DIR / "setup.py").read_text(encoding="utf-8")

    forbidden_download_calls = ("snapshot_download", "hf_hub_download", "._auto_download(", "_auto_download()")
    for token in forbidden_download_calls:
        expect(token not in generator, f"generator.py must not contain network/download call token: {token}")
    expect("local_files_only" in generator, "generator.py must load Diffusers with local_files_only")
    expect("ShapEPipeline.from_pretrained" in generator, "generator.py must load ShapEPipeline.from_pretrained")
    expect('output_type="mesh"' in generator, "generator.py must request output_type=\"mesh\"")
    expect("export_to_ply" in generator, "generator.py must export a PLY sidecar")
    expect(".glb" in generator and ".ply" in generator and "metadata" in generator, "generator.py must write GLB, PLY, and metadata outputs")
    expect("file=sys.stderr" in generator, "generator.py runtime logs must go to stderr")

    expect("download_models" in setup, "setup.py must parse/reject download_models")
    expect("setup-status.json" in setup, "setup.py must write setup-status.json")
    expect("setup.log" in setup, "setup.py must write setup.log")
    expect("downloads_started" in setup and "False" in setup, "setup.py must report that setup did not download weights")
    expect("snapshot_download" not in setup and "hf_hub_download" not in setup, "setup.py must not call Hugging Face download helpers")


def main() -> int:
    try:
        for filename in REQUIRED_FILES:
            expect((EXT_DIR / filename).is_file(), f"missing required file: {filename}")
        manifest = load_manifest()
        validate_manifest(manifest)
        validate_requirements()
        validate_runtime_contract_sources()
        parse_python(EXT_DIR / "setup.py")
        parse_python(EXT_DIR / "generator.py")
        for filename in TEST_FILES:
            parse_python(EXT_DIR / filename)
    except ValidationError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1

    print("OK: Shap-E extension static contract checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
