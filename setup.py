#!/usr/bin/env python3
"""Prepare the Modly Shap-E extension runtime.

Modly/Electron calls this script as:
    python setup.py '{"python_exe":"...","ext_dir":"...","gpu_sm":86,"cuda_version":128}'

This setup prepares Python dependencies and readiness evidence only. It never
downloads model weights; Modly UI owns Hugging Face model downloads.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import venv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXTENSION_ID = "shap-e"
NODE_ID = "generate"
HF_REPO = "openai/shap-e"
DOWNLOAD_CHECK = "model_index.json"
SCRIPT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_PATH = SCRIPT_DIR / "requirements.txt"
STATUS_RELATIVE_PATH = Path(".modly") / "setup" / "setup-status.json"
LOG_RELATIVE_PATH = Path(".modly") / "setup" / "logs" / "setup.log"
DEFAULT_MODEL_RELATIVE_PATH = Path("models") / EXTENSION_ID / NODE_ID
VENV_DIR_NAME = "venv"

DEFAULT_TORCH_VERSION = "2.5.1"
DEFAULT_TORCHVISION_VERSION = "0.20.1"
BLACKWELL_TORCH_VERSION = "2.7.0"
BLACKWELL_TORCHVISION_VERSION = "0.22.0"
PYTORCH_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"
PYTORCH_CUDA_INDEX_URLS = {
    "cu121": "https://download.pytorch.org/whl/cu121",
    "cu124": "https://download.pytorch.org/whl/cu124",
    "cu128-blackwell": "https://download.pytorch.org/whl/cu128",
}
PYTORCH_LANE_CUDA_VERSION = {
    "cu121": "12.1",
    "cu124": "12.4",
    "cu128-blackwell": "12.8",
}
PIP_FLAGS = ["--no-cache-dir", "--retries", "5", "--timeout", "60"]
DEPENDENCY_IMPORTS = {
    "torch": "torch",
    "torchvision": "torchvision",
    "diffusers": "diffusers",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "safetensors": "safetensors",
    "trimesh": "trimesh",
    "Pillow": "PIL",
    "numpy": "numpy",
    "huggingface_hub": "huggingface_hub",
}


@dataclass
class SetupConfig:
    python_exe: str
    ext_dir: Path
    gpu_sm: int | None = None
    cuda_version: int | None = None
    model_dir: Path | None = None
    validate_only: bool = False
    no_install: bool = False
    download_models: bool = False
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def venv_dir(self) -> Path:
        return self.ext_dir / VENV_DIR_NAME

    @property
    def venv_python(self) -> Path:
        if os.name == "nt":
            return self.venv_dir / "Scripts" / "python.exe"
        return self.venv_dir / "bin" / "python"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def log(message: str, *, stream: Any = sys.stdout) -> None:
    print(f"[setup:{EXTENSION_ID}] {message}", file=stream, flush=True)


def append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def run_command(cmd: list[str], log_path: Path, *, env: dict[str, str] | None = None) -> None:
    append_log(log_path, "Running: " + " ".join(cmd))
    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        check=False,
    )
    append_log(log_path, proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit {proc.returncode}: {' '.join(cmd)}. See {log_path}")


def modly_home_model_dir(ext_dir: Path) -> Path | None:
    if ext_dir.parent.name.lower() != "extensions":
        return None
    return ext_dir.parent.parent / DEFAULT_MODEL_RELATIVE_PATH


def resolve_expected_model_dir(config: SetupConfig) -> dict[str, Any]:
    if config.model_dir is not None:
        return {
            "path": config.model_dir,
            "source": "explicit_model_dir",
            "note": "Using the explicit model_dir provided to setup.",
        }

    models_dir = os.environ.get("MODELS_DIR")
    if models_dir and models_dir.strip():
        return {
            "path": Path(models_dir).expanduser().resolve() / EXTENSION_ID / NODE_ID,
            "source": "MODELS_DIR",
            "note": "Using MODELS_DIR/shap-e/generate from the environment.",
        }

    sibling_model_dir = modly_home_model_dir(config.ext_dir)
    if sibling_model_dir is not None:
        return {
            "path": sibling_model_dir,
            "source": "modly_home_sibling",
            "note": "Using the sibling Modly models directory derived from <modly_home>/extensions/<extension>.",
        }

    return {
        "path": config.ext_dir / DEFAULT_MODEL_RELATIVE_PATH,
        "source": "extension_local_fallback",
        "note": "Using the extension-local models directory as a fallback only.",
    }


def parse_setup_config(argv: list[str]) -> SetupConfig:
    if argv and argv[0].strip().startswith("{"):
        payload = json.loads(argv[0])
        return SetupConfig(
            python_exe=str(payload.get("python_exe") or sys.executable),
            ext_dir=Path(payload.get("ext_dir") or SCRIPT_DIR).expanduser().resolve(),
            gpu_sm=parse_int(payload.get("gpu_sm")),
            cuda_version=parse_int(payload.get("cuda_version")),
            model_dir=Path(payload["model_dir"]).expanduser().resolve() if payload.get("model_dir") else None,
            validate_only=parse_bool(payload.get("validate_only")),
            no_install=parse_bool(payload.get("no_install")),
            download_models=parse_bool(payload.get("download_models")),
            payload=payload,
        )

    if argv and not argv[0].startswith("-") and len(argv) >= 2:
        return SetupConfig(
            python_exe=argv[0],
            ext_dir=Path(argv[1]).expanduser().resolve(),
            gpu_sm=parse_int(argv[2]) if len(argv) >= 3 else None,
            cuda_version=parse_int(argv[3]) if len(argv) >= 4 else None,
        )

    parser = argparse.ArgumentParser(description="Prepare the Modly Shap-E extension runtime.")
    parser.add_argument("--python-exe", default=sys.executable, help="Python executable used by Modly runtime.")
    parser.add_argument("--ext-dir", default=str(SCRIPT_DIR), help="Installed extension directory.")
    parser.add_argument("--gpu-sm", type=int, default=None, help="Optional CUDA SM reported by Modly.")
    parser.add_argument("--cuda-version", type=int, default=None, help="Optional CUDA version reported by Modly.")
    parser.add_argument("--model-dir", default=None, help="Override model directory containing model_index.json.")
    parser.add_argument("--validate-only", action="store_true", help="Only write readiness evidence; do not install packages.")
    parser.add_argument("--no-install", action="store_true", help="Skip pip install even when imports are missing.")
    parser.add_argument("--download-models", action="store_true", help="Unsupported: Modly UI handles model downloads.")
    parser.add_argument("positional_payload_json", nargs="?", help="Optional Modly setup payload JSON when flags precede the payload.")
    args = parser.parse_args(argv)

    payload: dict[str, Any] = {}
    if args.positional_payload_json:
        if not args.positional_payload_json.strip().startswith("{"):
            parser.error("unexpected positional argument; pass a Modly payload JSON object or use legacy positional form")
        payload = json.loads(args.positional_payload_json)

    return SetupConfig(
        python_exe=str(payload.get("python_exe") or args.python_exe),
        ext_dir=Path(payload.get("ext_dir") or args.ext_dir).expanduser().resolve(),
        gpu_sm=args.gpu_sm if args.gpu_sm is not None else parse_int(payload.get("gpu_sm")),
        cuda_version=args.cuda_version if args.cuda_version is not None else parse_int(payload.get("cuda_version")),
        model_dir=Path(args.model_dir or payload["model_dir"]).expanduser().resolve() if (args.model_dir or payload.get("model_dir")) else None,
        validate_only=args.validate_only or parse_bool(payload.get("validate_only")),
        no_install=args.no_install or parse_bool(payload.get("no_install")),
        download_models=args.download_models or parse_bool(payload.get("download_models")),
        payload=payload,
    )


def select_torch_install_plan(config: SetupConfig) -> dict[str, Any]:
    cuda_signals: list[str] = []
    if config.gpu_sm is not None:
        cuda_signals.append(f"gpu_sm={config.gpu_sm}")
    if config.cuda_version is not None:
        cuda_signals.append(f"cuda_version={config.cuda_version}")

    cuda_expected = bool(cuda_signals)
    lane = "cpu"
    index_url = PYTORCH_CPU_INDEX_URL
    torch_version = DEFAULT_TORCH_VERSION
    torchvision_version = DEFAULT_TORCHVISION_VERSION
    note = "No CUDA signal was provided; selecting the explicit PyTorch CPU wheel index."

    if cuda_expected:
        blackwell_required = (
            (config.gpu_sm is not None and config.gpu_sm >= 120)
            or (config.cuda_version is not None and config.cuda_version >= 128)
        )
        if blackwell_required:
            lane = "cu128-blackwell"
            index_url = PYTORCH_CUDA_INDEX_URLS[lane]
            torch_version = BLACKWELL_TORCH_VERSION
            torchvision_version = BLACKWELL_TORCHVISION_VERSION
            note = "GB10/sm_12x or CUDA 12.8+ detected; selecting PyTorch cu128 Blackwell lane."
        elif config.cuda_version is None:
            lane = "cu121"
            index_url = PYTORCH_CUDA_INDEX_URLS[lane]
            note = "CUDA is expected but cuda_version was not provided; selecting cu121 fallback."
        elif config.cuda_version >= 124:
            lane = "cu124"
            index_url = PYTORCH_CUDA_INDEX_URLS[lane]
            note = "CUDA 12.4+ detected; selecting PyTorch cu124 wheel index."
        elif config.cuda_version >= 121:
            lane = "cu121"
            index_url = PYTORCH_CUDA_INDEX_URLS[lane]
            note = "CUDA 12.1+ detected; selecting PyTorch cu121 wheel index."
        else:
            note = f"CUDA version {config.cuda_version} is below supported cu121/cu124 lanes; selecting CPU wheels."

    return {
        "cuda_expected": cuda_expected,
        "cuda_signals": cuda_signals,
        "lane": lane,
        "index_url": index_url,
        "torch_version": torch_version,
        "torchvision_version": torchvision_version,
        "packages": [f"torch=={torch_version}", f"torchvision=={torchvision_version}"],
        "note": note,
    }


def ensure_venv(config: SetupConfig, log_path: Path) -> dict[str, Any]:
    if config.venv_python.exists():
        return {"created": False, "venv_python_exe": str(config.venv_python)}

    log(f"Creating extension venv at {config.venv_dir}")
    append_log(log_path, f"Creating venv at {config.venv_dir} with {config.python_exe}")
    builder = venv.EnvBuilder(with_pip=True, clear=False, symlinks=os.name != "nt")
    builder.create(str(config.venv_dir))
    if not config.venv_python.exists():
        raise RuntimeError(f"venv python was not created at {config.venv_python}")
    return {"created": True, "venv_python_exe": str(config.venv_python)}


def probe_imports(python_exe: str | Path) -> dict[str, Any]:
    code = (
        "import importlib.util, json, sys\n"
        f"mods = {json.dumps(DEPENDENCY_IMPORTS)}\n"
        "result = {'python': sys.executable, 'imports': {}}\n"
        "for pkg, mod in mods.items():\n"
        "    result['imports'][pkg] = importlib.util.find_spec(mod) is not None\n"
        "try:\n"
        "    import torch\n"
        "    result['torch'] = {'version': torch.__version__, 'cuda_version': getattr(torch.version, 'cuda', None), 'cuda_available': torch.cuda.is_available()}\n"
        "    try:\n"
        "        import torchvision\n"
        "        result['torchvision'] = {'version': torchvision.__version__}\n"
        "    except Exception as tv_exc:\n"
        "        result['torchvision_error'] = f'{type(tv_exc).__name__}: {tv_exc}'\n"
        "except Exception as exc:\n"
        "    result['torch_error'] = f'{type(exc).__name__}: {exc}'\n"
        "print(json.dumps(result, sort_keys=True))\n"
    )
    proc = subprocess.run([str(python_exe), "-c", code], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        return {
            "python": str(python_exe),
            "imports": {},
            "probe_error": proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}",
        }
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {
            "python": str(python_exe),
            "imports": {},
            "probe_error": f"Could not parse probe output: {type(exc).__name__}: {exc}",
            "raw_stdout": proc.stdout,
            "raw_stderr": proc.stderr,
        }


def missing_imports(probe: dict[str, Any]) -> list[str]:
    imports = probe.get("imports") or {}
    return [pkg for pkg in DEPENDENCY_IMPORTS if not imports.get(pkg)]


def torch_reinstall_needed(probe: dict[str, Any], plan: dict[str, Any]) -> bool:
    imports = probe.get("imports") or {}
    if not imports.get("torch") or not imports.get("torchvision"):
        return True
    torch_info = probe.get("torch") or {}
    torchvision_info = probe.get("torchvision") or {}
    torch_version = str(torch_info.get("version") or "").split("+", 1)[0]
    torchvision_version = str(torchvision_info.get("version") or "").split("+", 1)[0]
    if torch_version != str(plan.get("torch_version")):
        return True
    if torchvision_version != str(plan.get("torchvision_version")):
        return True
    if plan["lane"].startswith("cu") and not torch_info.get("cuda_version"):
        return True
    expected_cuda = PYTORCH_LANE_CUDA_VERSION.get(plan["lane"])
    if expected_cuda and str(torch_info.get("cuda_version")) != expected_cuda:
        return True
    return False


def pip_install_torch(python_exe: str | Path, log_path: Path, plan: dict[str, Any], *, force_reinstall: bool) -> None:
    cmd = [str(python_exe), "-m", "pip", "install", "--index-url", str(plan["index_url"]), *PIP_FLAGS]
    if force_reinstall:
        cmd.append("--force-reinstall")
    cmd.extend(str(package) for package in plan["packages"])
    log(f"Installing PyTorch lane {plan['lane']} from {plan['index_url']}")
    run_command(cmd, log_path)


def pip_install_requirements(python_exe: str | Path, log_path: Path) -> None:
    if not REQUIREMENTS_PATH.exists():
        raise RuntimeError(f"requirements.txt not found at {REQUIREMENTS_PATH}")
    cmd = [str(python_exe), "-m", "pip", "install", *PIP_FLAGS, "-r", str(REQUIREMENTS_PATH)]
    log("Installing Shap-E runtime Python requirements")
    run_command(cmd, log_path)


def write_status(config: SetupConfig, status: str, details: dict[str, Any]) -> Path:
    status_path = config.ext_dir / STATUS_RELATIVE_PATH
    status_path.parent.mkdir(parents=True, exist_ok=True)
    model_dir_resolution = resolve_expected_model_dir(config)
    expected_weights_path = model_dir_resolution["path"] / DOWNLOAD_CHECK
    payload = {
        "schema": "modly.setup-status.v1",
        "extension_id": EXTENSION_ID,
        "node_id": NODE_ID,
        "status": status,
        "checked_at": utc_now(),
        "python_exe": str(config.python_exe),
        "venv_dir": str(config.venv_dir),
        "venv_python_exe": str(config.venv_python),
        "ext_dir": str(config.ext_dir),
        "gpu_sm": config.gpu_sm,
        "cuda_version": config.cuda_version,
        "hf_repo": HF_REPO,
        "download_check": DOWNLOAD_CHECK,
        "expected_model_dir": str(model_dir_resolution["path"]),
        "expected_model_dir_source": model_dir_resolution["source"],
        "expected_model_dir_note": model_dir_resolution.get("note"),
        "expected_weights_path": str(expected_weights_path),
        "weights_present": expected_weights_path.is_file(),
        "weights_managed_by": "modly-ui",
        "downloads_started": False,
        "default_downloads": False,
        "setup_downloads_weights": False,
        "torch_install_lane": details.get("torch_install", {}).get("lane"),
        "torch_install_index_url": details.get("torch_install", {}).get("index_url"),
        "torch_packages": details.get("torch_install", {}).get("packages"),
        "dependency_imports": (details.get("probe_after") or details.get("probe_before") or {}).get("imports", {}),
        "missing_imports": details.get("missing_imports", []),
        "next_steps": details.get("next_steps", []),
        "details": details,
    }
    tmp_path = status_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, status_path)
    return status_path


def main(argv: list[str]) -> int:
    try:
        config = parse_setup_config(argv)
    except Exception as exc:
        log(f"Invalid setup arguments: {exc}", stream=sys.stderr)
        return 2

    if config.download_models:
        log("--download-models is intentionally unsupported. Use Modly's model-download UI for openai/shap-e.", stream=sys.stderr)
        return 2

    config.ext_dir.mkdir(parents=True, exist_ok=True)
    log_path = config.ext_dir / LOG_RELATIVE_PATH
    append_log(log_path, f"Started at {utc_now()}")
    append_log(log_path, f"Extension dir: {config.ext_dir}")
    model_dir_resolution = resolve_expected_model_dir(config)
    append_log(log_path, f"Expected weights ({model_dir_resolution['source']}): {model_dir_resolution['path'] / DOWNLOAD_CHECK}")

    details: dict[str, Any] = {
        "argv_payload": config.payload,
        "validate_only": config.validate_only,
        "no_install": config.no_install,
        "torch_install": select_torch_install_plan(config),
        "model_dir_resolution": {
            "path": str(model_dir_resolution["path"]),
            "source": model_dir_resolution["source"],
            "note": model_dir_resolution.get("note"),
        },
        "installs_started": False,
        "downloads_started": False,
        "next_steps": [],
    }

    try:
        if not config.validate_only:
            details["venv"] = ensure_venv(config, log_path)
        else:
            details["venv"] = {"created": False, "validate_only": True, "venv_python_exe": str(config.venv_python)}

        python_for_probe: Path | str = config.venv_python if config.venv_python.exists() else config.python_exe
        details["probe_before"] = probe_imports(python_for_probe)

        if not config.validate_only and not config.no_install:
            details["installs_started"] = True
            run_command([str(config.venv_python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], log_path)
            if torch_reinstall_needed(details["probe_before"], details["torch_install"]):
                pip_install_torch(config.venv_python, log_path, details["torch_install"], force_reinstall=True)
            pip_install_requirements(config.venv_python, log_path)
            details["probe_after"] = probe_imports(config.venv_python)
            run_command([str(config.venv_python), "-m", "pip", "check"], log_path)
        else:
            details["probe_after"] = details["probe_before"]
            if config.validate_only:
                details["next_steps"].append("Run setup without validate_only to install missing dependencies.")
            if config.no_install:
                details["next_steps"].append("Run setup without no_install to install missing dependencies.")

        missing = missing_imports(details["probe_after"])
        details["missing_imports"] = missing
        weights_present = (model_dir_resolution["path"] / DOWNLOAD_CHECK).is_file()
        if missing:
            status = "failed"
            details["next_steps"].append("Run extension setup/repair to install missing imports: " + ", ".join(missing))
        elif not weights_present:
            status = "needs_weights"
            details["next_steps"].append("Download openai/shap-e from the Modly UI.")
        else:
            status = "ready"

        status_path = write_status(config, status, details)
        log(f"Setup status: {status}. Evidence written to {status_path}")
        return 0 if status in {"ready", "needs_weights"} else 1
    except Exception as exc:
        details["error"] = f"{type(exc).__name__}: {exc}"
        details["next_steps"].append(f"Inspect setup log: {log_path}")
        try:
            status_path = write_status(config, "failed", details)
            log(f"Failure status written to {status_path}")
        except Exception as status_exc:
            log(f"Could not write setup status: {status_exc}", stream=sys.stderr)
        log(f"Setup failed: {exc}", stream=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
