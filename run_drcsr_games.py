"""
Run script for DRCSR on the Amazon-Games dataset.

Reproduces the results reported in Table 2 and Table 3 of the paper.

Usage:
    python run_drcsr_games.py
"""
import importlib.util
import sys
import torch
from pathlib import Path
from logging import getLogger

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, init_seed, get_trainer, set_color
from robustness_eval import robustness_eval


def load_local_model(model_file, class_name):
    model_path = Path(model_file).resolve()
    spec = importlib.util.spec_from_file_location(class_name, model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[class_name] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def run(dataset="Games", saved=True):
    model_cls = load_local_model("DRCSR.py", "DRCSR")
    config_file_list = ["./config/data.yaml"]

    config_dict = {
        "model": "DRCSR",

        # --- Model architecture ---
        "hidden_size": 128,
        "factor_num": 4,
        "residual_beta": 0.42,
        "candidate_logit_scale": 0.18,
        "t0": 0.8,
        "gamma": 0.3,
        "dropout_prob": 0.2,

        # --- Training signal ---
        "num_train_negs": 384,
        "logq_correction": 1.0,

        # --- Loss weights ---
        "lambda_sep": 1e-4,
        "lambda_stab": 1e-4,
        "lambda_align": 5e-5,
        "lambda_comp": 1e-5,
        "comp_tau": 0.28,
        "label_smoothing": 0.02,
        "train_score_temp": 1.15,

        # --- Gated factor memory ---
        "use_mem_gate": True,

        # --- Train-only robust augmentation ---
        "train_aug_noise": 0.05,
        "train_aug_drop": 0.05,
        "embed_noise_std": 0.03,

        # --- Optimizer ---
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "train_batch_size": 128,
        "eval_batch_size": 32,

        # --- Multimodal embedding ---
        "freeze_mm": True,

        # --- Early stopping ---
        "stopping_step": 10,

        "metrics": ["Recall", "NDCG"],
        "topk": [5, 10, 20, 50],
        "neg_chunk_size": 0,
    }

    config = Config(model=model_cls, dataset=dataset,
                    config_file_list=config_file_list, config_dict=config_dict)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(">>> DRCSR on Games")

    dataset_obj = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset_obj)

    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = model_cls(config, train_data._dataset).to(config["device"])

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f">>> Trainable params: {trainable:,}")

    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved,
        show_progress=config["show_progress"])

    test_result = trainer.evaluate(
        test_data, load_best_model=saved,
        show_progress=config["show_progress"])

    logger.info(set_color("best valid", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    if saved:
        checkpoint = torch.load(trainer.saved_model_file,
                                map_location=config["device"],
                                weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])

    robustness_eval(model, test_data, config, logger,
                    drop_ratios=(0.05, 0.15, 0.25),
                    eval_k=10, top_k=20, n_repeats=3)


if __name__ == "__main__":
    run(dataset="Games", saved=True)
