"""
Robustness Evaluation Module

Default perturbation ratios: (0.05, 0.15, 0.25)

References:
    - RecDenoiser (WWW 2022 / TOIS 2024) adopts a noise injection protocol
      ranging from 0% to 25%.
      A perturbation ratio of 25% is commonly regarded as a high-noise setting,
      while 50% is generally beyond the meaningful perturbation range for
      sequential recommendation.
    - CL4SRec / CoSeRec typically use item-dropping ratios in the range of 0.1–0.3.

Metric 1: NNDCG@k (Noise-Robust NDCG)
    Randomly replace r% of items in each interaction sequence with random items
    and report the absolute NDCG@k.
    Higher is better. This metric directly reflects recommendation quality
    under noisy interactions.

Metric 2: RNDCG@k (Robust NDCG under Sparsity)
    Remove r% of items from each interaction sequence, compact the remaining
    sequence, and report the absolute NDCG@k.
    Higher is better. This metric directly reflects recommendation quality
    under sparse interaction histories.
"""

import random
import numpy as np
import torch


def _compute_ndcg_at_k(scores, pos_items, k=10):
    """Compute NDCG@k for single-positive-item evaluation."""
    if scores.size(0) == 0:
        return 0.0

    _, topk_idx = scores.topk(k, dim=1)
    hits = (topk_idx == pos_items.unsqueeze(1)).float()

    positions = torch.arange(
        1, k + 1,
        dtype=torch.float,
        device=hits.device
    ).unsqueeze(0)

    dcg = (hits / torch.log2(positions + 1)).sum(dim=1)

    return dcg.mean().item()


# =====================================================================
# Perturbation Type 1: Noise Injection
# =====================================================================
def _inject_noise_by_ratio(
    interaction,
    item_seq_field,
    item_seq_len_field,
    n_items,
    noise_ratio=0.1
):
    """
    Replace a given ratio of items in each sequence with randomly sampled items.

    Parameters
    ----------
    interaction : dict-like
        Interaction batch containing item sequences and sequence lengths.
    item_seq_field : str
        Field name of the item sequence.
    item_seq_len_field : str
        Field name of the sequence length.
    n_items : int
        Total number of items.
    noise_ratio : float, optional
        Fraction of valid sequence items to replace with random items.

    Returns
    -------
    dict
        A perturbed interaction dictionary.
    """
    item_seq = interaction[item_seq_field].clone()
    item_seq_len = interaction[item_seq_len_field]

    B, L = item_seq.size()

    for b in range(B):
        seq_len = item_seq_len[b].item()

        n_noise = max(1, round(seq_len * noise_ratio))
        n_noise = min(n_noise, seq_len - 1)

        if n_noise <= 0:
            continue

        valid_positions = list(range(0, seq_len))
        noise_positions = random.sample(valid_positions, n_noise)

        for pos in noise_positions:
            rand_item = random.randint(1, max(n_items - 1, 1))
            item_seq[b, pos] = rand_item

    new_interaction = {}

    for key in interaction.keys():
        if key == item_seq_field:
            new_interaction[key] = item_seq
        else:
            new_interaction[key] = interaction[key]

    return new_interaction


# =====================================================================
# Perturbation Type 2: Interaction Sparsification
# =====================================================================
def _drop_items_by_ratio(
    interaction,
    item_seq_field,
    item_seq_len_field,
    drop_ratio=0.1,
    timestamp_field="timestamp_list"
):
    """
    Remove a given ratio of items from each sequence and compact the remainder.

    If timestamp information is available, the timestamps corresponding to the
    removed items are also deleted and compacted accordingly.

    Parameters
    ----------
    interaction : dict-like
        Interaction batch containing item sequences and sequence lengths.
    item_seq_field : str
        Field name of the item sequence.
    item_seq_len_field : str
        Field name of the sequence length.
    drop_ratio : float, optional
        Fraction of valid sequence items to remove.
    timestamp_field : str, optional
        Field name of the timestamp sequence.

    Returns
    -------
    dict
        A sparsified interaction dictionary.
    """
    item_seq = interaction[item_seq_field].clone()
    item_seq_len = interaction[item_seq_len_field].clone()

    has_ts = timestamp_field in interaction

    if has_ts:
        timestamp = interaction[timestamp_field].clone()

    B, L = item_seq.size()

    for b in range(B):
        seq_len = item_seq_len[b].item()

        n_drop = max(1, round(seq_len * drop_ratio))
        n_drop = min(n_drop, seq_len - 1)

        if n_drop <= 0:
            continue

        valid_positions = list(range(0, seq_len))
        drop_set = set(random.sample(valid_positions, n_drop))

        keep_items = []
        keep_ts = []

        for pos in valid_positions:
            if pos not in drop_set:
                keep_items.append(item_seq[b, pos].item())

                if has_ts:
                    keep_ts.append(timestamp[b, pos].item())

        new_len = len(keep_items)

        # Rebuild the compacted item sequence.
        item_seq[b, :] = 0

        for i, val in enumerate(keep_items):
            item_seq[b, i] = val

        # Rebuild the corresponding timestamp sequence if available.
        if has_ts:
            timestamp[b, :] = 0

            for i, val in enumerate(keep_ts):
                timestamp[b, i] = val

        item_seq_len[b] = new_len

    new_interaction = {}

    for key in interaction.keys():
        if key == item_seq_field:
            new_interaction[key] = item_seq

        elif key == item_seq_len_field:
            new_interaction[key] = item_seq_len

        elif key == timestamp_field and has_ts:
            new_interaction[key] = timestamp

        else:
            new_interaction[key] = interaction[key]

    return new_interaction


# =====================================================================
# Utility Wrapper
# =====================================================================
class _InteractionWrapper:
    """
    Lightweight wrapper that provides a RecBole-like interaction interface.
    """

    def __init__(self, data_dict):
        self._data = data_dict
        self.interaction = data_dict

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def to(self, device):
        """Move all tensor fields to the specified device."""
        new_data = {}

        for k, v in self._data.items():
            if isinstance(v, torch.Tensor):
                new_data[k] = v.to(device)
            else:
                new_data[k] = v

        return _InteractionWrapper(new_data)


# =====================================================================
# Main Robustness Evaluation Function
# =====================================================================
def robustness_eval(
    model,
    test_data,
    config,
    logger,
    drop_ratios=(0.05, 0.15, 0.25),
    eval_k=10,
    top_k=20,
    n_repeats=3
):
    """
    Evaluate recommendation robustness along two perturbation dimensions.

    Metrics
    -------
    NNDCG@k:
        Absolute NDCG@k after replacing r% of historical items with random
        items. Higher values indicate stronger robustness to noisy interactions.

    RNDCG@k:
        Absolute NDCG@k after removing r% of historical items and compacting
        the sequence. Higher values indicate stronger robustness to sparse
        interaction histories.

    Parameters
    ----------
    model
        Sequential recommendation model.
    test_data
        Test data loader.
    config
        Configuration object containing the evaluation device.
    logger
        Logger used to print evaluation results.
    drop_ratios : tuple of float, optional
        Perturbation ratios used for both noise injection and item dropping.
        Default: (0.05, 0.15, 0.25).
    eval_k : int, optional
        Cutoff value for NDCG evaluation.
    top_k : int, optional
        Reserved top-k parameter.
    n_repeats : int, optional
        Number of repeated perturbation trials for each ratio.

    Returns
    -------
    dict or None
        {
            "baseline_ndcg": float,
            "noise": {
                ratio: (mean, std),
                ...
            },
            "noise_avg": float,
            "sparsity": {
                ratio: (mean, std),
                ...
            },
            "sparsity_avg": float,
            "ratios": tuple,
            "eval_k": int,
            "n_samples": int,
            "avg_seq_len": float,
        }

        Returns None if the test set contains fewer than 10 samples.
    """
    model.eval()
    device = config["device"]

    item_seq_field = model.ITEM_SEQ
    item_seq_len_field = model.ITEM_SEQ_LEN

    item_id_field = (
        model.ITEM_ID
        if hasattr(model, "ITEM_ID")
        else "item_id"
    )

    n_items = model.n_items

    logger.info("=" * 60)
    logger.info(
        "Robustness Evaluation "
        "(Two Absolute Metrics, Perturbation Ratios: 5% / 15% / 25%)"
    )
    logger.info(
        f"  NNDCG@{eval_k} = Noise-Robust NDCG "
        f"(absolute NDCG after injecting r% noise; higher is better)"
    )
    logger.info(
        f"  RNDCG@{eval_k} = Sparsity-Robust NDCG "
        f"(absolute NDCG after removing r% interactions; higher is better)"
    )
    logger.info(
        f"  ratios={[f'{r:.0%}' for r in drop_ratios]}, "
        f"n_repeats={n_repeats}"
    )
    logger.info("=" * 60)

    # =================================================================
    # Phase 1: Collect Original Evaluation Data
    # =================================================================
    all_original_scores = []
    all_pos_items = []
    all_interactions = []
    all_seq_lens = []

    with torch.no_grad():
        for batch_data_raw in test_data:

            if isinstance(batch_data_raw, (tuple, list)):
                interaction = batch_data_raw[0]
            else:
                interaction = batch_data_raw

            interaction = interaction.to(device)

            scores = model.full_sort_predict(interaction)
            pos_items = interaction[item_id_field]
            seq_lens = interaction[item_seq_len_field]

            all_original_scores.append(scores.cpu())
            all_pos_items.append(pos_items.cpu())
            all_seq_lens.append(seq_lens.cpu())

            # Store a CPU copy of the interaction batch for perturbation.
            batch_data = {}

            if hasattr(interaction, "interaction"):
                iter_keys = interaction.interaction.keys()

            elif hasattr(interaction, "columns"):
                iter_keys = interaction.columns

            else:
                iter_keys = interaction.keys()

            for key in iter_keys:
                val = interaction[key]

                if isinstance(val, torch.Tensor):
                    batch_data[key] = val.cpu()
                else:
                    batch_data[key] = val

            all_interactions.append(batch_data)

    all_original_scores = torch.cat(all_original_scores, dim=0)
    all_pos_items = torch.cat(all_pos_items, dim=0)
    all_seq_lens = torch.cat(all_seq_lens, dim=0)

    N = all_original_scores.size(0)

    if N < 10:
        logger.info(
            "[Robustness] Too few test samples; skipping robustness evaluation."
        )
        return None

    original_ndcg = _compute_ndcg_at_k(
        all_original_scores,
        all_pos_items,
        k=eval_k
    )

    avg_len = all_seq_lens.float().mean().item()

    logger.info(
        f"[Baseline] NDCG@{eval_k} = {original_ndcg:.5f}  "
        f"(samples={N}, avg_sequence_length={avg_len:.1f})"
    )

    # =================================================================
    # Phase 2: NNDCG — Robustness to Noise Injection
    # =================================================================
    logger.info("-" * 60)
    logger.info(
        ">>> NNDCG: Noise-Robust NDCG "
        "(absolute NDCG after injecting r% random items)"
    )
    logger.info("-" * 60)

    noise_results = {}

    for ratio in drop_ratios:
        nndcg_list = []

        for repeat in range(n_repeats):
            noisy_scores_list = []

            with torch.no_grad():
                for batch_data in all_interactions:

                    noisy = _inject_noise_by_ratio(
                        batch_data,
                        item_seq_field,
                        item_seq_len_field,
                        n_items=n_items,
                        noise_ratio=ratio
                    )

                    wrapped = _InteractionWrapper(noisy).to(device)

                    noisy_scores = model.full_sort_predict(wrapped)
                    noisy_scores_list.append(noisy_scores.cpu())

            all_noisy_scores = torch.cat(
                noisy_scores_list,
                dim=0
            )

            noisy_ndcg = _compute_ndcg_at_k(
                all_noisy_scores,
                all_pos_items,
                k=eval_k
            )

            nndcg_list.append(noisy_ndcg)

        avg_nndcg = float(np.mean(nndcg_list))
        std_nndcg = float(np.std(nndcg_list))

        retention = avg_nndcg / (original_ndcg + 1e-8)

        noise_results[ratio] = (
            avg_nndcg,
            std_nndcg
        )

        logger.info(
            f"[Noise={ratio:.0%}] "
            f"NNDCG@{eval_k} = {avg_nndcg:.5f} "
            f"(±{std_nndcg:.5f})  |  "
            f"retention = {retention:.1%}"
        )

    noise_avg = float(
        np.mean([
            v[0]
            for v in noise_results.values()
        ])
    )

    logger.info(
        f"[Noise Avg ] NNDCG@{eval_k} = {noise_avg:.5f}  "
        f"(mean over {[f'{r:.0%}' for r in drop_ratios]})"
    )

    # =================================================================
    # Phase 3: RNDCG — Robustness to Interaction Sparsity
    # =================================================================
    logger.info("-" * 60)
    logger.info(
        ">>> RNDCG: Sparsity-Robust NDCG "
        "(absolute NDCG after removing r% interactions)"
    )
    logger.info("-" * 60)

    sparsity_results = {}

    for ratio in drop_ratios:
        rndcg_list = []

        for repeat in range(n_repeats):
            drop_scores_list = []

            with torch.no_grad():
                for batch_data in all_interactions:

                    dropped = _drop_items_by_ratio(
                        batch_data,
                        item_seq_field,
                        item_seq_len_field,
                        drop_ratio=ratio,
                        timestamp_field="timestamp_list"
                    )

                    wrapped = _InteractionWrapper(
                        dropped
                    ).to(device)

                    drop_scores = model.full_sort_predict(
                        wrapped
                    )

                    drop_scores_list.append(
                        drop_scores.cpu()
                    )

            all_drop_scores = torch.cat(
                drop_scores_list,
                dim=0
            )

            drop_ndcg = _compute_ndcg_at_k(
                all_drop_scores,
                all_pos_items,
                k=eval_k
            )

            rndcg_list.append(drop_ndcg)

        avg_rndcg = float(np.mean(rndcg_list))
        std_rndcg = float(np.std(rndcg_list))

        retention = avg_rndcg / (
            original_ndcg + 1e-8
        )

        sparsity_results[ratio] = (
            avg_rndcg,
            std_rndcg
        )

        logger.info(
            f"[Drop={ratio:.0%}] "
            f"RNDCG@{eval_k} = {avg_rndcg:.5f} "
            f"(±{std_rndcg:.5f})  |  "
            f"retention = {retention:.1%}"
        )

    sparsity_avg = float(
        np.mean([
            v[0]
            for v in sparsity_results.values()
        ])
    )

    logger.info(
        f"[Sparsity Avg] RNDCG@{eval_k} = {sparsity_avg:.5f}  "
        f"(mean over {[f'{r:.0%}' for r in drop_ratios]})"
    )

    # =================================================================
    # Final Summary
    # =================================================================
    logger.info("=" * 60)
    logger.info("Summary (ready to copy into a paper table):")

    ratio_hdr = "  ".join([
        f"{r:>6.0%}"
        for r in drop_ratios
    ])

    logger.info(
        f"  Baseline NDCG@{eval_k} = "
        f"{original_ndcg:.5f}"
    )

    logger.info(
        f"  ratios           : "
        f"{ratio_hdr}     Avg"
    )

    noise_row = "  ".join([
        f"{noise_results[r][0]:.5f}"
        for r in drop_ratios
    ])

    sparsity_row = "  ".join([
        f"{sparsity_results[r][0]:.5f}"
        for r in drop_ratios
    ])

    logger.info(
        f"  NNDCG@{eval_k}         : "
        f"{noise_row}   {noise_avg:.5f}"
    )

    logger.info(
        f"  RNDCG@{eval_k}         : "
        f"{sparsity_row}   {sparsity_avg:.5f}"
    )

    logger.info("=" * 60)

    return {
        "baseline_ndcg": original_ndcg,
        "noise": {
            r: noise_results[r]
            for r in drop_ratios
        },
        "noise_avg": noise_avg,
        "sparsity": {
            r: sparsity_results[r]
            for r in drop_ratios
        },
        "sparsity_avg": sparsity_avg,
        "ratios": tuple(drop_ratios),
        "eval_k": eval_k,
        "n_samples": N,
        "avg_seq_len": avg_len,
    }

