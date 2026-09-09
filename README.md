# DRCSR

Official implementation of **"Dual-Phase Reliability Calibration for Robust Multi-modal Sequential Recommendation"**.


## Overview

Multimodal sequential recommendation enriches user modeling with visual and textual content, but rich content does not necessarily provide reliable preference evidence. Unreliable interactions—such as accidental clicks or gift purchases—may carry strong multimodal semantics that mislead user states and propagate through sequence updates.

Existing robust methods estimate reliability at the **whole-interaction level** before candidate matching, making preference-irrelevant and ranking-useful semantic aspects difficult to distinguish. This leads to a **Precision–Coverage Dilemma**:

- **Preference-Learning Robustness** emphasizes *precision*: avoiding misleading semantics as stable preference patterns.
- **Intent-Ranking Robustness** emphasizes *coverage*: preserving the user's genuine intents—including niche interests—in the final ranking.

**DRCSR** resolves this dilemma through a dual-phase design operating at the **multimodal semantic factor** level:

- **Phase I** (state-writing): Factor-level reliability calibration + gated factor memory attenuate unreliable factors before they contaminate the user state, while residual retention preserves recoverable evidence.
- **Phase II** (scoring): Candidate-aware factor recalibration adjusts retained factor contributions according to candidate matching evidence, reactivating genuine intents without reintroducing suppressed noise.

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
    └── Sports/
```

## Dataset Preparation

We evaluate on three [Amazon review datasets](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/): **Video Games**, **Office Products**, and **Baby**, and **Sports and Outdoors**.

Download the processed datasets from [this link](TODO) and unzip into the `dataset/` folder. Each dataset directory should contain the `.inter` interaction file and the pretrained embedding files (`txt_emb.pt`, `img_emb.pt`). Data splitting (leave-one-out) is handled automatically by RecBole.

## Running

```bash
# Amazon-Games
python run_drcsr_games.py
```

Each script trains the model, evaluates on the test set, and runs the robustness evaluation (NNDCG@10 and RNDCG@10 under 5%/15%/25% perturbation).



## Citation

```bibtex
TODO
```

## Acknowledgement

Our implementation is built on [RecBole](https://github.com/RUCAIBox/RecBole) and the codebase of [HM4SR](https://github.com/SStarCCat/HM4SR). We thank the authors for making their code publicly available.
