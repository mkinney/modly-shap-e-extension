# Shap-E Modly Extension

This repository provides a Modly model extension for text-to-3D mesh generation with Shap-E. The extension uses Hugging Face Diffusers' `ShapEPipeline` with the `openai/shap-e` model snapshot and exposes a single Modly node, `shap-e/generate`.

The repository contains the Modly integration, dependency setup, and runtime adapter. It does **not** vendor OpenAI's Shap-E source code or model weights. Modly downloads the model snapshot separately through its model-management UI. This project is independently maintained and is not authored, sponsored, or endorsed by OpenAI.

## Modly contract

- Extension ID: `shap-e`
- Generator class: `ShapEGenerator`
- Node: `generate` (`model.generate`)
- Capability: text-to-3D
- Input: a non-empty text prompt
- Primary return value: a GLB mesh
- Sidecars: a PLY mesh and JSON run metadata
- Model repository: `openai/shap-e`
- Download sentinel: `model_index.json`

`setup.py` creates an extension-local virtual environment, installs runtime dependencies, selects an appropriate PyTorch wheel lane from the supplied CUDA signals, and writes setup evidence. It never downloads model weights. At inference time, `generator.py` loads only a local model snapshot by passing `local_files_only=True` to Diffusers.

## Requirements and support

- A Modly installation that supports model extensions.
- A Python interpreter with `venv` and `pip` support.
- Network access during setup to install Python packages.
- A model snapshot downloaded by Modly from `openai/shap-e`.
- CPU or NVIDIA CUDA execution. CPU is supported but expected to be slow.

The setup script selects CPU, CUDA 12.1, CUDA 12.4, or CUDA 12.8/Blackwell PyTorch wheels according to `gpu_sm` and `cuda_version`. If no CUDA signal is supplied, it selects CPU wheels. The manifest declares 8 GB VRAM and recommends 12 GB; actual usage depends on the selected parameters and runtime environment.

## Setup

Install the extension under Modly's extension directory, conventionally:

```text
<modly-home>/extensions/shap-e/
```

Modly can invoke setup with its JSON payload contract:

```bash
python setup.py '{"python_exe":"/path/to/python","ext_dir":"/path/to/modly/extensions/shap-e","gpu_sm":86,"cuda_version":124}'
```

The equivalent flag-based form is:

```bash
python setup.py \
  --python-exe /path/to/python \
  --ext-dir /path/to/modly/extensions/shap-e \
  --gpu-sm 86 \
  --cuda-version 124
```

Use hardware values reported by Modly rather than copying the example values blindly. Setup creates `venv/` inside the installed extension and writes:

```text
.modly/setup/setup-status.json
.modly/setup/logs/setup.log
```

Useful setup controls include `--model-dir`, `--validate-only`, and `--no-install`. The `--download-models` option is intentionally rejected; model downloads belong to the Modly UI.

By default, an extension installed at `<modly-home>/extensions/shap-e` expects the model snapshot at:

```text
<modly-home>/models/shap-e/generate/model_index.json
```

The runtime also recognizes `MODEL_DIR` as a direct snapshot path and `MODELS_DIR` as a model-root override. When the extension is run outside the conventional Modly layout, its final fallback is `models/shap-e/generate/` under the extension directory.

## Usage

1. Run or repair the extension setup from Modly.
2. Use Modly's model-management UI to download `openai/shap-e` for the `generate` node.
3. Add or select the **Shap-E Text to 3D** node.
4. Enter a text prompt and adjust the generation parameters if needed.
5. Run the node. Modly receives the generated GLB path; the PLY and metadata files remain beside it.

For a quick smoke test, use a prompt such as `a small red toy robot`, set **Inference Steps** to `8`-`16`, and select **Frame Size** `64`. These lower values trade quality for a faster first run. The manifest defaults are intended for normal generation.

## Parameters

| Parameter | Type and default | Accepted values | Behavior |
| --- | --- | --- | --- |
| `prompt` | String, required | Non-empty text | Text condition for Shap-E generation. |
| `num_inference_steps` | Integer, `64` | `1`-`100` | Number of denoising steps. |
| `guidance_scale` | Float, `15.0` | `0.0`-`30.0`, step `0.5` | Classifier-free guidance strength. |
| `frame_size` | Select, `256` | `64`, `128`, `256` | Mesh renderer frame size; smaller values reduce time and memory use. |
| `seed` | Integer, `-1` | `-1`-`2147483647` | `-1` chooses a random seed; other values make the sampling seed explicit. |
| `device` | Select, `auto` | `auto`, `cuda`, `cpu` | `auto` uses CUDA when available and CPU otherwise. Explicit CUDA fails if CUDA is unavailable. |
| `torch_dtype` | Select, `auto` | `auto`, `float16`, `bfloat16`, `float32` | `auto` uses `float16` on CUDA and `float32` on CPU. Half-precision requests on CPU fall back to `float32`. |
| `upright_rotation` | Select, `true` | `true`, `false` | Applies a -90-degree X-axis rotation before GLB export when enabled. |
| `output_name` | Optional string, empty | Filename-safe text | Supplies the output filename stem after runtime sanitization. An empty value uses `shap_e_mesh`. |

## Outputs

Each generation creates a unique directory under `outputs/` and writes three files:

- `<name>_<id>.glb` — the primary mesh returned to Modly.
- `<name>_<id>.ply` — the PLY sidecar exported from the Diffusers mesh result.
- `<name>_<id>_metadata.json` — run details, including the prompt and its SHA-256 hash, resolved parameters, model path, output paths, geometry counts, and duration.

The manifest declares the mesh output as non-PBR.

## Validation

Run the repository's static contract validator:

```bash
python validate_extension.py
```

It validates the manifest contract, Python syntax, dependency declarations, local-only Diffusers loading, and expected output behavior. It does not install packages, download weights, or execute generation.

Validate the manifest JSON independently with:

```bash
python -m json.tool manifest.json
```

For installed-runtime diagnostics, inspect `.modly/setup/setup-status.json` and `.modly/setup/logs/setup.log` under the extension directory.

## Upstream references and acknowledgements

Shap-E was created by OpenAI. This extension runs the model through Hugging Face Diffusers rather than vendoring the original Shap-E implementation.

- [Official Shap-E source](https://github.com/openai/shap-e)
- [Shap-E model assets](https://huggingface.co/openai/shap-e)
- [Shap-E: Generating Conditional 3D Implicit Functions](https://arxiv.org/abs/2305.02463)

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream attribution and licensing details.

## Licensing

The Modly extension code is licensed under the [MIT License](LICENSE), Copyright (c) 2026 DrHepa.

OpenAI's Shap-E source and the separately downloaded model assets remain third-party materials governed by their upstream terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
