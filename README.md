# DRCSR

Official implementation of **"Dual-Phase Reliability Calibration for Robust Multi-modal Sequential Recommendation"**.

## Overview

Multimodal sequential recommendation benefits from fusing visual and textual semantics but suffers from two intertwined challenges:

- **C1 – Cascading Reliability Degradation:** A noisy interaction with internally consistent multimodal content is amplified by fusion and then propagates through sequential state updates, forming a compound degradation chain.
- **C2 – Precision–Coverage Dilemma:** Suppressing noise (precision) risks eliminating genuine niche interests, while preserving coverage reintroduces contamination.

DRCSR resolves both challenges through a dual-phase design:

- **Phase I** (state-writing step): Factor-level reliability calibration + gated factor memory → breaks the degradation chain, achieving training robustness.
- **Phase II** (scoring step): Candidate-aware motive composition → activates genuine niche interests without reintroducing suppressed noise, achieving inference robustness.

## Requirements

- Python 3.10
- PyTorch 2.1+ (with CUDA)
- NVIDIA GPU (experiments were run on a single RTX 4090D)

```bash
conda create -n DRCSR python=3.10
conda activate DRCSR
pip install -r requirements.txt
```

> **Note:** This repository includes a local copy of [RecBole](https://recbole.io/) in the `recbole/` directory. Do **not** install RecBole via pip; the included version is used directly.

## Repository Structure

```
DRCSR/
├── DRCSR.py                 # Model implementation
├── robustness_eval.py       # Robustness evaluation (NNDCG / RNDCG)
├── run_drcsr_baby.py        # Run script for Amazon-Baby
├── run_drcsr_games.py       # Run script for Amazon-Games
├── run_drcsr_office.py      # Run script for Amazon-Office
├── requirements.txt
├── config/
│   └── data.yaml            # RecBole data & evaluation settings
├── recbole/                 # Local RecBole framework
└── dataset/
    ├── Baby/
    │   ├── Baby.inter       # User-item interactions
    │   ├── txt_emb.pt       # Pretrained text embeddings
    │   ├── img_emb.pt       # Pretrained image embeddings
    │   └── ...
    ├── Games/
    └── Office/
```

## Dataset Preparation

We evaluate on three [Amazon review datasets](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/): **Video Games**, **Office Products**, and **Baby**.

Download the processed datasets from [this link](TODO) and unzip into the `dataset/` folder. Each dataset directory should contain the `.inter` interaction file and the pretrained embedding files (`txt_emb.pt`, `img_emb.pt`). Data splitting (leave-one-out) is handled automatically by RecBole.

## Running

```bash
# Amazon-Games
python run_drcsr_games.py

# Amazon-Office
python run_drcsr_office.py
```

Each script trains the model, evaluates on the test set, and runs the robustness evaluation (NNDCG@10 and RNDCG@10 under 5%/15%/25% perturbation).



## Citation

```bibtex
TODO
```

## Acknowledgement

Our implementation is built on [RecBole](https://github.com/RUCAIBox/RecBole) and the codebase of [HM4SR](https://github.com/SStarCCat/HM4SR). We thank the authors for making their code publicly available.
