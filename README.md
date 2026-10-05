# HUMER

HUMER is a training data scheduling tool for source-code vulnerability
detection models. It organizes when and how training samples are presented to
an existing detector without changing the detector architecture.

HUMER schedules training in three phases:

1. **Curriculum construction** assigns code-based difficulty scores and builds
   curriculum from easy to hard.
2. **Progressive learning and adaptive review** introduces one bucket at a
   time, measures the model's learning state, and allocates review slots to
   previously seen samples and persistent errors.
3. **Knowledge consolidation** trains on the complete training set and selects
   the final vulnerability detection model.

The repository provides model adapters, dataset preparation utilities,
reproducible experiment configurations, training diagnostics, and checkpoint
management for the complete scheduling workflow.

## Supported models

| Adapter | Base checkpoint |
|---|---|
| CodeBERT | `microsoft/codebert-base` |
| UniXcoder | `microsoft/unixcoder-base` |
| CodeT5 | `Salesforce/codet5-base` |
| LineVul | `microsoft/codebert-base` |
| VulGPT | `microsoft/codebert-base-mlm` |
| EPVD | `microsoft/codebert-base` |

Pretrained weights are not stored in this repository. They are downloaded
from Hugging Face when first used, unless a local checkpoint is supplied.

## System requirements

- Linux
- Python 3.10--3.13; Python 3.11 is recommended
- NVIDIA GPU and a CUDA-compatible PyTorch installation for full experiments
- Approximately 10 GB of free disk space for dependencies, caches, and model
  checkpoints; EPVD requires additional space for its official artifact

The reference environment uses PyTorch 2.5 and Transformers 4.46.3. GPU memory
requirements depend on the selected detector and dataset.

## 1. Get the source code

The source code is available from the anonymous repository:

<https://anonymous.4open.science/r/Humer/>

Download the repository and run the commands below from its root directory.

## 2. Create the environment

The shortest installation path is:

```bash
conda env create -f environment.yml
conda activate humer
```

For an explicit CUDA 12.1 installation:

```bash
conda create -n humer python=3.11 pip -y
conda activate humer
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

Install EPVD dependencies only when EPVD is needed:

```bash
pip install -e ".[epvd]"
```

Verify the package without loading a pretrained model:

```bash
python -m humer --help
PYTHONPATH=src python -m unittest discover -s tests -v
```

## 3. Download pretrained models

### Automatic download

By default, Transformers downloads the selected checkpoint during the first
run and stores it under `.cache/huggingface`.

### Download before training

Download all unique checkpoints:

```bash
python scripts/download_models.py --output-dir models
```

Download only the checkpoints needed by selected adapters:

```bash
python scripts/download_models.py \
  --output-dir models \
  --models unixcoder codet5
```

Use a downloaded checkpoint by overriding `--model-name`:

```bash
--model-name models/microsoft--unixcoder-base --local-files-only
```

The checkpoints remain subject to their respective Hugging Face model
licenses.

## 4. Prepare the dataset

HUMER expects the following directory structure:

```text
data/Devign/
|-- train.json
|-- val.json
`-- test.json
```

Each split is a JSON list. Every item must contain `code`, `label`, and a
split-unique integer `id`:

```json
[
  {
    "id": 1001,
    "code": "int example(void) { return 0; }",
    "label": 0
  }
]
```

Labels must be `0` for non-vulnerable code and `1` for vulnerable code. HUMER
does not create or modify train/validation/test splits.

Check the installation and data before starting a full experiment:

```bash
python scripts/check_install.py \
  --config configs/experiments/unixcoder_devign.yaml
```

Datasets are not redistributed by this repository. Obtain them from their
official sources:

- Devign: <https://github.com/epicosy/devign>
- PrimeVul: <https://github.com/DLVulDet/PrimeVul>

If the downloaded splits use JSONL or the common `func`/`target`/`idx` fields,
convert the already-defined splits without changing their membership:

```bash
python scripts/prepare_dataset.py \
  --train /path/to/train.jsonl \
  --validation /path/to/valid.jsonl \
  --test /path/to/test.jsonl \
  --output-dir data/PrimeVul
```

The converter never creates a random split. Users must provide the official or
otherwise explicitly selected train, validation, and test files and comply
with their licenses and access terms.

## 5. EPVD setup

EPVD additionally requires its official CFG implementation and compiled
Tree-sitter C parser. HUMER does not redistribute those files because the
Zenodo artifact does not declare a license.

1. Download `EPVD.zip` from the official artifact:
   <https://doi.org/10.5281/zenodo.7123322>
2. Extract the archive.
3. Install the two runtime files:

```bash
python scripts/prepare_epvd.py --source-dir /path/to/extracted/EPVD
```

If `my-languages.so` is incompatible with the local system, rebuild it by
following the instructions in the official EPVD README, then rerun the setup
script.

## 6. Run HUMER

Strict deterministic mode is enabled in the supplied configurations. Set the
cuBLAS workspace before starting Python:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Example: UniXcoder on Devign with seed 42:

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m humer \
  --config configs/experiments/unixcoder_devign.yaml \
  --output-dir outputs/unixcoder_devign_seed42 \
  --seed 42
```

Configuration files for all six adapters and both datasets are available in
`configs/experiments/`.

HUMER refuses to overwrite an existing output directory. Choose a new
directory for every run.

## Output files

Each run writes:

```text
config.json
training.log
difficulty_buckets.csv
stage_diagnostics.csv
error_tracking.csv
consolidation_history.csv
stage_1_best.pt ... stage_5_best.pt
consolidation_best.pt
final_model.pt
final_test_metrics.json
```

`config.json` stores the fully resolved configuration and fixed algorithm
choices. `consolidation_history.csv` contains validation metrics only. Test
metrics are written only to `final_test_metrics.json`.

## Configuration

Select a YAML file from `configs/experiments/` for the model and dataset. The
following command-line arguments can override run-specific values without
editing that file:

```text
--config
--output-dir
--data-dir
--model-name
--seed
--device
--local-files-only
```

Training and scheduling parameters are stored in YAML so that each run has an
explicit, reusable configuration. The resolved configuration is copied to
`config.json` in the output directory.

## Reproducibility notes

- Use a separate output directory for every run.
- Keep the same dataset files; difficulty caches are keyed by the SHA-256 hash
  of `train.json`.
- Keep `CUBLAS_WORKSPACE_CONFIG=:4096:8` when strict deterministic mode is
  enabled.
- The exact result can still depend on GPU architecture, CUDA, PyTorch, and
  low-level kernels. Record those versions with every published run.

## Third-party resources

Third-party model weights, datasets, and EPVD files are not included in this
repository and remain subject to their respective terms.
