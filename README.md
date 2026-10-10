# VLM PPE & Hazard Detection

A two-stage Vision-Language Model pipeline (Qwen2-VL-2B-Instruct) for PPE compliance and hazard detection, with a Gradio UI. Built during an industry internship.

## How it works

1. **Stage 1:** the model generates a free-text description of the image.
2. **Stage 2:** the model answers closed Yes/No sub-questions about PPE and hazards.
3. **Verdict:** the final Safe/Unsafe result is computed in code (`extract_ppe_hazard_verdict()`), not taken from the model's own label, because small models sometimes contradict their own sub-answers.

The UI (`queen_ui.py`) supports batch upload and shows results in a table.

## Reproducibility

Greedy decoding (`do_sample=False`), a fixed seed, and deterministic CUDA settings give consistent output from run to run. `stop_strings` prevents duplicated output blocks.

## Setup

Requires an NVIDIA GPU (the model is loaded with `device_map="cuda"`) and Docker with GPU support (Docker Desktop with WSL2 on Windows).

1. Download `Qwen/Qwen2-VL-2B-Instruct` from Hugging Face. The weights are used unmodified, with no fine-tuning.
2. Place the weights in `./models/qwen2-vl-2b`, or set the `MODEL_DIR` environment variable to their location.
3. Download these wheels (Python 3.14, CUDA 12.6) and place them next to the `Dockerfile`:
   - `torch-2.13.0+cu126-cp314-cp314-manylinux_2_28_x86_64.whl` and
     `torchvision-0.28.0+cu126-cp314-cp314-manylinux_2_28_x86_64.whl`
     from https://download.pytorch.org/whl/cu126/
   - `nvidia_cublas_cu12-12.6.4.1-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl`,
     `nvidia_cudnn_cu12-9.10.2.21-py3-none-manylinux_2_27_x86_64.whl` and
     `nvidia_cusparselt_cu12-0.7.1-py3-none-manylinux2014_x86_64.whl`
     from PyPI (packages `nvidia-cublas-cu12`, `nvidia-cudnn-cu12`, `nvidia-cusparselt-cu12`)
4. Run `docker compose up --build`, then open http://localhost:7860.

## Evaluation

26 images collected from the internet (not redistributed). Ground truth was labelled manually by the author, who is not a certified HSE professional.

Labelling criteria:
- **PPE OK:** hardhat and industrial clothing are both present.
- **Active hazard:** a worker is touching or reaching toward operating equipment.
- **Verdict:** Unsafe if PPE is not OK or an active hazard is present; otherwise Safe. The same rule is used in the code.

Accuracy at each stage (n = 26):

| Stage | PPE | Hazard |
|---|---|---|
| Stage 1 description (claim vs. ground truth) | 76.9% | 61.5% |
| Stage 2 sub-answer vs. ground truth | 80.8% | 76.9% |

Final verdict accuracy: **84.6% (22/26)**. Recall on Unsafe: **83.3% (10/12)**.

| | Model: Safe | Model: Unsafe |
|---|---|---|
| Ground truth: Safe | 12 | 2 |
| Ground truth: Unsafe | 2 | 10 |

The two false negatives are Unsafe scenes the model called Safe. In both, the model missed an active hazard.

## Limitations

- Small evaluation set (26 images): the 95% interval on 84.6% accuracy is roughly 66% to 94%, so results are indicative only.
- Single annotator, not an HSE expert, with project-specific definitions of "PPE OK" and "active hazard".
- Hazard perception is the weakest stage (61.5% in the description stage).
- The evaluation images are not included, so the results cannot be reproduced exactly; the pipeline can be run on your own images.
- Tested on one machine (RTX 3050, 4GB).
- The pinned wheels are specific to Python 3.14 and CUDA 12.6.

## Author

Arqia M, Universitas Gadjah Mada
