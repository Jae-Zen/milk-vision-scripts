# Colab evaluation lab

[Quick start](../README.md) · [Observations](observations.md) · [Development](development.md)

Use the existing `evaluate_milk_colab.ipynb` in Google Colab. Select a GPU runtime.
The notebook is the testing interface; algorithm code comes from a published GitHub
branch, tag or commit. The local notebook file mirrors the online source and has
no saved outputs. No ZIP upload is needed for this workflow.

## Publish code first

Publish the shared-observation implementation and subsequent algorithm revisions
to `https://github.com/Jae-Zen/milk-vision-scripts.git` yourself. The notebook does
not commit or push code. An older revision without the shared evaluator or required
CLI options is rejected with an actionable message, never silently run as a
two-algorithm test.

Cell 2 contains `GIT_REF` (default `main`) and `MODEL_NAME` (default
`milk-vision-v4s.pt`). Cell 3 fetches that ref and resolves an exact commit SHA.
It creates/reuses a detached, commit-specific worktree rather than deleting the
runtime checkout. If a worktree contains edits, setup stops and preserves them.
Changing the ref or repository requires refreshing source before another run.

## First run in a new runtime

Keep the complete annotated dataset and checkpoint in Drive:

```text
MyDrive/Yolo_Cloud/
  Dataset/videos/annotations.json
  Dataset/videos/<food>/<net folder>/<video>
  Models/milk-vision-v4s.pt
```

1. Run cell 1 to mount Drive and cell 2 to configure the source ref/paths.
2. Run cell 3 to load the published source and install its requirements.
3. Run cell 4 to synchronize the complete dataset and selected model to the
   disposable runtime disk. Only these runtime copies are replaced; Drive is untouched.
4. Run cell 5 for GPU/input readiness and reusable helper definitions. This performs
   no inference or TypeSafe requests.
5. Choose algorithms and settings in cell 6, then run that cell explicitly.
6. Run cell 7 to inspect results and experiment history.

The annotation loader validates complete dataset coverage, including food classes
not selected for milk evaluation. The default selection is 133 milk clips plus 36
other-object negatives; the 55 cheese-labelled clips are excluded. Scenario and
limit controls can select a smaller test.

## Repeat after algorithm changes

The normal loop is **publish changes → source cell 3 → preflight cell 5 → run
cell 6 → review cell 7**. Do not rerun cell 4 just to change algorithm code.
Rerun input synchronization when dataset contents or the checkpoint change, and
rerun preflight after reconnecting or changing the GPU.

Cell 6 exposes three independent checkboxes: Placement, Motion and Jev. Enable
any nonempty combination. Controls live in the run cell itself, so it reconstructs
selection every time that cell runs. Selecting none fails before starting work.

- **Fresh predictions** (default) reuses detector/tracker observations and reruns
  only the selected algorithms, including fresh Jev calls.
- **Resume** reuses compatible completed predictions. Use it after interruption.
- An optional experiment label names the run; timestamps, commit and a unique ID
  prevent reports being overwritten.
- `INCLUDE_NEGATIVES` defaults to true, blank scenario means all scenarios, and
  limit zero means every selected video. Jev defaults to `jev-1.14.0`.

GPU accelerates YOLO collection; Jev still runs through TypeSafe's API. Only Jev
runs access `TYPESAFE_API_KEY`, using an existing runtime environment value or
Colab Secrets with notebook access enabled. Never paste credentials into a code
cell or upload `.env`. Keys are not command arguments or saved metadata.

## Results, caches and comparisons

Videos and weights are read from the runtime disk. Completed observations persist
in GPU-model-specific directories under `Yolo_Cloud/cache`; prediction versions
use the sibling `predictions` directory. Checkpoint, extraction settings and
software/code fingerprints control compatibility. Source changes that do not alter
collection can reuse observations. Runtime resets preserve Drive caches but require
setup and input copies again. There is no silent CPU fallback.

Each experiment has a directory under `Yolo_Cloud/results/evaluation` containing
`metadata.json`, `report.json`, `run.log`, and an exported `summary.csv` after
review. Metadata records the requested ref, exact commit, selected videos/methods,
settings, detector/annotation hashes, Python/Torch/CUDA information and GPU name.
Evaluation reports retain observation hashes, actual Jev version, confidence,
probabilities, usage, requests and responses. Reports update after each result.

The history table lists code revisions, labels, methods, progress and scores.
Scope/settings fingerprints highlight changed selections, labels, models and
settings. Completed evidence fingerprints also distinguish different observations;
matching scope alone does not prove identical video evidence. Partial/failed runs
are labelled explicitly, rather than presented as complete benchmark scores.

The current-run tables separate processing failures, label mismatches and empty
versus nonempty detection inputs. An empty input selecting `none` can correctly
describe observed motion while missing an inventory event. Existing inventory
event annotations are not independently reviewed visible-motion labels. Episode
order is checked without invented Jev frame boundaries.

## Verification boundaries

Local tests load only notebook helper function code, with mocked Git/subprocess
operations. They cover all seven algorithm combinations, no selection, modes,
filters, obsolete revisions, edited-worktree protection and run-history statuses.
They do not mount Drive, install runtime dependencies, infer videos or call Jev.

Notebook source must also pass the configured Colab MCP runtime static checker.
That check analyzes source without running the evaluation cells. Google access,
GPU availability, dataset presence and actual inference are checked when you run
the notebook. Do not claim a completed GPU benchmark from static checks alone.
