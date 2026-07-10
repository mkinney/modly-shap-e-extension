"""Modly Shap-E generator.

The Modly runner loads this module in an extension subprocess and communicates
over JSON on stdout. Keep runtime logs on stderr only.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

try:
    from services.generators.base import BaseGenerator, GenerationCancelled
except Exception:  # pragma: no cover - static validation fallback outside Modly.
    class GenerationCancelled(Exception):
        """Fallback cancellation exception for static imports outside Modly."""

    class BaseGenerator:
        MODEL_ID = ""
        DISPLAY_NAME = ""
        VRAM_GB = 0

        def __init__(self, model_dir: Path, outputs_dir: Path) -> None:
            self.model_dir = Path(model_dir)
            self.outputs_dir = Path(outputs_dir)
            self._model = None
            self.hf_repo = ""
            self.hf_skip_prefixes: list[str] = []
            self.download_check = ""
            self._params_schema: list[dict[str, Any]] = []

        def _report(self, progress_cb: Optional[Callable[[int, str], None]], pct: int, step: str) -> None:
            if progress_cb:
                progress_cb(pct, step)


EXTENSION_ID = "shap-e"
NODE_ID = "generate"
MODEL_ID = f"{EXTENSION_ID}/{NODE_ID}"
DISPLAY_NAME = "Shap-E Text to 3D"
HF_REPO = "openai/shap-e"
DOWNLOAD_CHECK = "model_index.json"
EXTENSION_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = EXTENSION_DIR / "models" / EXTENSION_ID / NODE_ID
SETUP_STATUS_PATH = EXTENSION_DIR / ".modly" / "setup" / "setup-status.json"

CORE_COMPONENT_HINTS = ("prior", "renderer", "shap_e_renderer", "scheduler", "text_encoder", "tokenizer")

PARAMS_SCHEMA: list[dict[str, Any]] = [
    {"id": "prompt", "label": "Prompt", "type": "string", "default": "", "required": True},
    {"id": "num_inference_steps", "label": "Inference Steps", "type": "int", "default": 64, "min": 1, "max": 100},
    {"id": "guidance_scale", "label": "Guidance Scale", "type": "float", "default": 15.0, "min": 0.0, "max": 30.0, "step": 0.5},
    {
        "id": "frame_size",
        "label": "Frame Size",
        "type": "select",
        "default": 256,
        "options": [{"value": 64, "label": "64"}, {"value": 128, "label": "128"}, {"value": 256, "label": "256"}],
    },
    {"id": "seed", "label": "Seed", "type": "int", "default": -1, "min": -1, "max": 2147483647},
    {
        "id": "device",
        "label": "Device",
        "type": "select",
        "default": "auto",
        "options": [{"value": "auto", "label": "Auto"}, {"value": "cuda", "label": "CUDA"}, {"value": "cpu", "label": "CPU"}],
    },
    {
        "id": "torch_dtype",
        "label": "Torch DType",
        "type": "select",
        "default": "auto",
        "options": [
            {"value": "auto", "label": "Auto"},
            {"value": "float16", "label": "float16"},
            {"value": "bfloat16", "label": "bfloat16"},
            {"value": "float32", "label": "float32"},
        ],
    },
    {
        "id": "upright_rotation",
        "label": "Upright Rotation",
        "type": "select",
        "default": "true",
        "options": [{"value": "true", "label": "Enabled"}, {"value": "false", "label": "Disabled"}],
    },
    {"id": "output_name", "label": "Output Name", "type": "string", "default": "", "required": False},
]


def _log(message: str) -> None:
    print(f"[ShapEGenerator] {message}", file=sys.stderr, flush=True)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value or not value.strip():
        return None
    return Path(os.path.expandvars(value.strip())).expanduser()


def _modly_home_model_dir() -> Path | None:
    if EXTENSION_DIR.parent.name.lower() != "extensions":
        return None
    return EXTENSION_DIR.parent.parent / "models" / EXTENSION_ID / NODE_ID


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path.absolute()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            result.append(resolved)
    return result


def _missing_dependencies() -> list[str]:
    import importlib.util

    modules = {
        "torch": "torch",
        "diffusers": "diffusers",
        "transformers": "transformers",
        "accelerate": "accelerate",
        "safetensors": "safetensors",
        "trimesh": "trimesh",
        "numpy": "numpy",
        "Pillow": "PIL",
    }
    return [package for package, module in modules.items() if importlib.util.find_spec(module) is None]


def _dependency_status() -> dict[str, bool]:
    import importlib.util

    modules = ("torch", "diffusers", "transformers", "accelerate", "safetensors", "trimesh", "numpy", "PIL")
    return {module: importlib.util.find_spec(module) is not None for module in modules}


def _readiness_result(
    *,
    ok: bool,
    machine_code: str,
    label_hint: str | None,
    reason: str,
    details: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "ok": ok,
        "machine_code": machine_code,
        "reason": reason,
        "details": details,
    }
    if label_hint:
        result["label_hint"] = label_hint
    return result


def _schema_defaults() -> dict[str, Any]:
    return {str(item["id"]): item.get("default") for item in PARAMS_SCHEMA}


def _param(params: Mapping[str, Any], defaults: Mapping[str, Any], name: str, fallback: Any) -> Any:
    value = params.get(name, defaults.get(name, fallback))
    return fallback if value is None else value


def _safe_int(value: Any, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def _safe_float(value: Any, default: float, *, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default
    if minimum is not None:
        result = max(minimum, result)
    if maximum is not None:
        result = min(maximum, result)
    return result


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _normalize_prompt(params: Mapping[str, Any]) -> str:
    for key in ("prompt", "text", "input_text"):
        value = params.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.strip().split())
    raise RuntimeError("Shap-E text-to-3D requires a non-empty prompt.")


def _sanitize_output_base(value: Any, default: str = "shap_e_mesh") -> str:
    text = str(value or "").strip()
    if not text:
        text = default
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return text[:80] or default


def _unique_output_paths(outputs_dir: Path, base_name: str) -> tuple[Path, Path, Path]:
    suffix = uuid.uuid4().hex[:12]
    stem = f"{base_name}_{suffix}"
    return outputs_dir / f"{stem}.glb", outputs_dir / f"{stem}.ply", outputs_dir / f"{stem}_metadata.json"


def _weights_probe(model_dir: Path) -> dict[str, Any]:
    sentinel = model_dir / DOWNLOAD_CHECK
    existing_components = [name for name in CORE_COMPONENT_HINTS if (model_dir / name).exists()]
    required_from_index: list[str] = []
    missing_from_index: list[str] = []
    model_index = _read_json(sentinel)
    if isinstance(model_index, dict):
        for key, value in model_index.items():
            if key.startswith("_"):
                continue
            if key in CORE_COMPONENT_HINTS and value is not None:
                required_from_index.append(key)
                if not (model_dir / key).exists():
                    missing_from_index.append(key)

    ok = sentinel.is_file() and not missing_from_index
    return {
        "ok": ok,
        "model_dir": str(model_dir),
        "sentinel": str(sentinel),
        "sentinel_present": sentinel.is_file(),
        "existing_components": existing_components,
        "required_from_index": required_from_index,
        "missing_from_index": missing_from_index,
    }


def _count_geometry(asset: Any) -> tuple[int, int]:
    vertices = 0
    faces = 0
    if hasattr(asset, "geometry"):
        for geom in asset.geometry.values():
            vertices += int(len(getattr(geom, "vertices", [])))
            faces += int(len(getattr(geom, "faces", [])))
    else:
        vertices = int(len(getattr(asset, "vertices", [])))
        faces = int(len(getattr(asset, "faces", [])))
    return vertices, faces


def _apply_upright_rotation(asset: Any) -> None:
    import trimesh

    transform = trimesh.transformations.rotation_matrix(-math.pi / 2.0, [1, 0, 0])
    asset.apply_transform(transform)


class ShapEGenerator(BaseGenerator):
    MODEL_ID = MODEL_ID
    DISPLAY_NAME = DISPLAY_NAME
    VRAM_GB = 8

    @classmethod
    def params_schema(cls) -> list[dict[str, Any]]:
        return PARAMS_SCHEMA

    @classmethod
    def capability_params_schema(cls, node_id: str) -> list[dict[str, Any]]:
        if node_id != NODE_ID:
            raise RuntimeError(f"Unsupported Shap-E node '{node_id}'. Supported node: {NODE_ID}.")
        return PARAMS_SCHEMA

    def __init__(self, model_dir: Path | str | None = None, outputs_dir: Path | str | None = None) -> None:
        provided_model_dir = Path(model_dir).expanduser() if model_dir is not None else None
        super().__init__(provided_model_dir or DEFAULT_MODEL_DIR, Path(outputs_dir or (EXTENSION_DIR / "outputs")))
        self.model_dir = provided_model_dir or DEFAULT_MODEL_DIR
        self._provided_model_dir = provided_model_dir
        self.outputs_dir = Path(outputs_dir or (EXTENSION_DIR / "outputs"))
        self.download_check = DOWNLOAD_CHECK
        self.hf_repo = HF_REPO
        self.hf_skip_prefixes = []
        self._model: Any | None = None
        self._loaded_model_dir: Path | None = None
        self._device_label: str | None = None
        self._dtype_label: str | None = None

    def _candidate_model_dirs(self) -> list[Path]:
        candidates: list[Path] = []
        env_model_dir = _env_path("MODEL_DIR")
        if env_model_dir is not None:
            candidates.append(env_model_dir)

        env_models_dir = _env_path("MODELS_DIR")
        if env_models_dir is not None:
            candidates.append(env_models_dir / EXTENSION_ID / NODE_ID)

        if self._provided_model_dir is not None:
            candidates.append(self._provided_model_dir)

        sibling_model_dir = _modly_home_model_dir()
        if sibling_model_dir is not None:
            candidates.append(sibling_model_dir)

        candidates.append(DEFAULT_MODEL_DIR)
        return _dedupe_paths(candidates)

    def _find_model_dir(self) -> Path | None:
        for candidate in self._candidate_model_dirs():
            if _weights_probe(candidate)["ok"]:
                return candidate
        return None

    def _expected_model_dir(self) -> Path:
        candidates = self._candidate_model_dirs()
        return candidates[0] if candidates else DEFAULT_MODEL_DIR

    def is_loaded(self) -> bool:
        return self._model is not None

    def is_downloaded(self) -> bool:
        return self._find_model_dir() is not None

    def readiness_status(self) -> dict[str, Any]:
        missing = _missing_dependencies()
        probes = [_weights_probe(candidate) for candidate in self._candidate_model_dirs()]
        model_dir = self._find_model_dir()
        details = {
            "model_id": MODEL_ID,
            "hf_repo": HF_REPO,
            "download_check": DOWNLOAD_CHECK,
            "candidate_model_dirs": [str(path) for path in self._candidate_model_dirs()],
            "expected_model_dir": str(self._expected_model_dir()),
            "expected_weights_path": str(self._expected_model_dir() / DOWNLOAD_CHECK),
            "resolved_model_dir": str(model_dir) if model_dir else None,
            "dependency_imports": _dependency_status(),
            "weights_probes": probes,
            "setup_status": _read_json(SETUP_STATUS_PATH),
        }
        if missing:
            return _readiness_result(
                ok=False,
                machine_code="missing_dependencies",
                label_hint="Run setup",
                reason="Missing Python imports: " + ", ".join(missing),
                details=details,
            )
        if model_dir is None:
            return _readiness_result(
                ok=False,
                machine_code="missing_weights",
                label_hint=None,
                reason="Shap-E weights are not present. Use Modly UI to download openai/shap-e into models/shap-e/generate.",
                details=details,
            )
        return _readiness_result(
            ok=True,
            machine_code="ready",
            label_hint="Ready",
            reason="Shap-E dependencies and model_index.json are available.",
            details=details,
        )

    def _select_device_and_dtype(self, params: Mapping[str, Any]) -> tuple[Any, str, Any, str]:
        import torch

        defaults = _schema_defaults()
        requested_device = str(_param(params, defaults, "device", "auto")).strip().lower()
        if requested_device not in {"auto", "cuda", "cpu"}:
            requested_device = "auto"
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for Shap-E, but torch.cuda.is_available() is false.")
        device_label = "cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device
        if device_label == "auto":
            device_label = "cpu"
        device = torch.device(device_label)

        requested_dtype = str(_param(params, defaults, "torch_dtype", "auto")).strip().lower()
        if requested_dtype not in {"auto", "float16", "bfloat16", "float32"}:
            requested_dtype = "auto"
        if requested_dtype == "auto":
            dtype_label = "float16" if device_label == "cuda" else "float32"
        elif requested_dtype in {"float16", "bfloat16"} and device_label == "cpu":
            _log(f"{requested_dtype} requested on CPU; using float32 for compatibility.")
            dtype_label = "float32"
        else:
            dtype_label = requested_dtype

        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype_label]
        return device, device_label, dtype, dtype_label

    def load(
        self,
        params: Mapping[str, Any] | None = None,
        progress_cb: Callable[..., Any] | None = None,
        cancel_event: Any | None = None,
    ) -> None:
        params = params or {}
        self._check_cancelled(cancel_event)
        missing = _missing_dependencies()
        if missing:
            raise RuntimeError("Shap-E runtime dependencies are missing: " + ", ".join(missing) + ". Run extension setup first.")

        model_dir = self._find_model_dir()
        if model_dir is None:
            searched = ", ".join(str(path) for path in self._candidate_model_dirs())
            raise FileNotFoundError(
                "Shap-E model_index.json was not found or the snapshot is incomplete. "
                "Use the Modly UI to download openai/shap-e into models/shap-e/generate. "
                f"Searched: {searched}"
            )

        import torch
        from diffusers import ShapEPipeline

        device, device_label, dtype, dtype_label = self._select_device_and_dtype(params)
        if (
            self._model is not None
            and self._loaded_model_dir == model_dir
            and self._device_label == device_label
            and self._dtype_label == dtype_label
        ):
            return

        self.unload()
        self._report(progress_cb, 10, "Loading Shap-E pipeline")
        self._check_cancelled(cancel_event)

        kwargs = {"torch_dtype": dtype, "local_files_only": True}
        try:
            variant_kwargs = dict(kwargs)
            if dtype_label in {"float16", "bfloat16"}:
                variant_kwargs["variant"] = "fp16"
            with contextlib.redirect_stdout(sys.stderr):
                pipe = ShapEPipeline.from_pretrained(str(model_dir), **variant_kwargs)
        except Exception as exc:
            if dtype_label in {"float16", "bfloat16"}:
                _log(f"Loading with variant='fp16' failed; retrying local snapshot without variant. Reason: {exc}")
                with contextlib.redirect_stdout(sys.stderr):
                    pipe = ShapEPipeline.from_pretrained(str(model_dir), **kwargs)
            else:
                raise

        self._report(progress_cb, 18, f"Moving Shap-E to {device_label}")
        self._check_cancelled(cancel_event)
        with contextlib.redirect_stdout(sys.stderr):
            pipe = pipe.to(device)

        self._model = pipe
        self._loaded_model_dir = model_dir
        self._device_label = device_label
        self._dtype_label = dtype_label
        _log(f"Loaded Shap-E from {model_dir} on {device_label} with {dtype_label}.")

    def unload(self) -> None:
        model = self._model
        self._model = None
        self._loaded_model_dir = None
        self._device_label = None
        self._dtype_label = None
        if model is not None:
            del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _check_cancelled(self, cancel_event: Optional[threading.Event]) -> None:
        if cancel_event and cancel_event.is_set():
            raise GenerationCancelled()

    def generate(
        self,
        image_bytes: bytes | None = None,
        params: Mapping[str, Any] | None = None,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        del image_bytes
        params = dict(params or {})
        defaults = _schema_defaults()
        prompt = _normalize_prompt(params)
        num_inference_steps = _safe_int(_param(params, defaults, "num_inference_steps", 64), 64, minimum=1, maximum=100)
        guidance_scale = _safe_float(_param(params, defaults, "guidance_scale", 15.0), 15.0, minimum=0.0, maximum=30.0)
        frame_size = _safe_int(_param(params, defaults, "frame_size", 256), 256, minimum=64, maximum=256)
        if frame_size not in {64, 128, 256}:
            frame_size = 256
        seed = _safe_int(_param(params, defaults, "seed", -1), -1, minimum=-1, maximum=2147483647)
        upright_rotation = _safe_bool(_param(params, defaults, "upright_rotation", "true"), True)
        output_base = _sanitize_output_base(_param(params, defaults, "output_name", "shap_e_mesh"))

        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        run_dir = self.outputs_dir / f"shap_e_{uuid.uuid4().hex[:12]}"
        run_dir.mkdir(parents=True, exist_ok=True)
        glb_path, ply_path, metadata_path = _unique_output_paths(run_dir, output_base)

        self._report(progress_cb, 3, "Validating Shap-E prompt")
        self._check_cancelled(cancel_event)
        self.load(params=params, progress_cb=progress_cb, cancel_event=cancel_event)
        if self._model is None:
            raise RuntimeError("Shap-E pipeline failed to load.")

        import torch
        import trimesh
        from diffusers.utils import export_to_ply

        generator = None
        actual_seed = seed
        if actual_seed < 0:
            actual_seed = random.randint(0, 2147483647)
        if self._device_label:
            generator = torch.Generator(device=self._device_label).manual_seed(actual_seed)

        _log(
            "Starting Shap-E generation "
            f"steps={num_inference_steps}, guidance_scale={guidance_scale}, frame_size={frame_size}, seed={actual_seed}."
        )
        self._report(progress_cb, 25, "Running Shap-E diffusion")
        self._check_cancelled(cancel_event)

        start_time = time.time()
        with torch.no_grad():
            with contextlib.redirect_stdout(sys.stderr):
                result = self._model(
                    prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    frame_size=frame_size,
                    generator=generator,
                    output_type="mesh",
                )

        self._check_cancelled(cancel_event)
        mesh = result.images[0] if getattr(result, "images", None) else None
        if mesh is None:
            raise RuntimeError("Shap-E returned no mesh output.")

        self._report(progress_cb, 82, "Writing Shap-E PLY sidecar")
        _log(f"Writing PLY sidecar to {ply_path}.")
        with contextlib.redirect_stdout(sys.stderr):
            export_to_ply(mesh, str(ply_path))
        if not ply_path.is_file() or ply_path.stat().st_size == 0:
            raise RuntimeError(f"Shap-E failed to write a non-empty PLY sidecar at {ply_path}.")

        self._report(progress_cb, 90, "Converting Shap-E mesh to GLB")
        loaded = trimesh.load(str(ply_path), force="scene")
        vertices, faces = _count_geometry(loaded)
        if vertices <= 0:
            raise RuntimeError("Shap-E produced an empty mesh. Try a different prompt or generation seed.")
        if upright_rotation:
            _apply_upright_rotation(loaded)

        _log(f"Writing GLB mesh to {glb_path}.")
        loaded.export(str(glb_path))
        if not glb_path.is_file() or glb_path.stat().st_size == 0:
            raise RuntimeError(f"Shap-E failed to write a non-empty GLB at {glb_path}.")

        metadata = {
            "schema": "modly.shap-e.run-metadata.v1",
            "model_id": MODEL_ID,
            "hf_repo": HF_REPO,
            "model_dir": str(self._loaded_model_dir) if self._loaded_model_dir else None,
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "params": {
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "frame_size": frame_size,
                "seed": actual_seed,
                "device": self._device_label,
                "torch_dtype": self._dtype_label,
                "upright_rotation": upright_rotation,
            },
            "outputs": {
                "glb": str(glb_path),
                "ply": str(ply_path),
            },
            "geometry": {
                "vertices": vertices,
                "faces": faces,
            },
            "duration_seconds": round(time.time() - start_time, 3),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        self._report(progress_cb, 100, "Shap-E generation complete")
        _log(f"Shap-E generation complete. Returning {glb_path}; PLY sidecar retained at {ply_path}.")
        return glb_path
