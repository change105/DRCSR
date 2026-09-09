"""
鲁棒性评估模块  
   默认扰动比例  (0.05, 0.15, 0.25)
     - 引文支撑: RecDenoiser (WWW 2022 / TOIS 2024) 噪声注入协议为 0-25%
       25% 是标准高扰动档, 50% 在序列推荐里已超出有意义扰动范围
     - CL4SRec / CoSeRec item dropping 典型比例也在 0.1-0.3
指标 1: NNDCG@k (Noise-robust NDCG)
  将 r% 的 item 替换为随机 item, 报告绝对 NDCG@k
  越高越好, 直接反映噪声交互下的推荐质量
指标 2: RNDCG@k (Robust NDCG under Sparsity)
  删除 r% 的 item 并紧凑重建, 报告绝对 NDCG@k
  越高越好, 直接反映数据稀疏下的推荐质量
"""
import random
import numpy as np
import torch

def _compute_ndcg_at_k(scores, pos_items, k=10):
    """单正样本场景下的 NDCG@k"""
    if scores.size(0) == 0:
        return 0.0
    _, topk_idx = scores.topk(k, dim=1)
    hits = (topk_idx == pos_items.unsqueeze(1)).float()
    positions = torch.arange(1, k + 1, dtype=torch.float, device=hits.device).unsqueeze(0)
    dcg = (hits / torch.log2(positions + 1)).sum(dim=1)
    return dcg.mean().item()

# =====================================================================
#  扰动方式 1: 噪声注入
# =====================================================================
def _inject_noise_by_ratio(interaction, item_seq_field, item_seq_len_field,
                           n_items, noise_ratio=0.1):
    """将序列中 noise_ratio 比例的 item 替换为随机 item"""
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
#  扰动方式 2: 数据稀疏 (紧凑重建)
# =====================================================================
def _drop_items_by_ratio(interaction, item_seq_field, item_seq_len_field,
                         drop_ratio=0.1, timestamp_field='timestamp_list'):
    """删除序列中 drop_ratio 比例的 item 并紧凑重建"""
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
        item_seq[b, :] = 0
        for i, val in enumerate(keep_items):
            item_seq[b, i] = val
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
#  工具类
# =====================================================================
class _InteractionWrapper:
    """让 dict 表现得像 RecBole 的 Interaction 对象"""

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
        new_data = {}
        for k, v in self._data.items():
            if isinstance(v, torch.Tensor):
                new_data[k] = v.to(device)
            else:
                new_data[k] = v
        return _InteractionWrapper(new_data)

# =====================================================================
#  主评估函数
# =====================================================================
def robustness_eval(model, test_data, config, logger,
                    drop_ratios=(0.05, 0.15, 0.25),
                    eval_k=10, top_k=20, n_repeats=3):
    """
    双维度鲁棒性评估 (绝对指标版):
      NNDCG@k: 注入 r% 噪声后的绝对 NDCG@k (越高越好)
      RNDCG@k: 删除 r% item 后的绝对 NDCG@k (越高越好)

    Returns
    -------
    dict:
      {
        "baseline_ndcg": float,
        "noise":    {ratio: (mean, std), ...},
        "noise_avg":    float,        # 三档均值
        "sparsity": {ratio: (mean, std), ...},
        "sparsity_avg": float,
        "ratios": tuple,
        "eval_k": int,
        "n_samples": int,
        "avg_seq_len": float,
      }
    """
    model.eval()
    device = config["device"]

    item_seq_field = model.ITEM_SEQ
    item_seq_len_field = model.ITEM_SEQ_LEN
    item_id_field = model.ITEM_ID if hasattr(model, 'ITEM_ID') else 'item_id'
    n_items = model.n_items

    logger.info("=" * 60)
    logger.info("鲁棒性评估 (双绝对指标, 扰动 5%/15%/25%)")
    logger.info(f"  NNDCG@{eval_k} = 噪声鲁棒NDCG (注入r%噪声后, 越高越好)")
    logger.info(f"  RNDCG@{eval_k} = 稀疏鲁棒NDCG (删除r%交互后, 越高越好)")
    logger.info(f"  ratios={[f'{r:.0%}' for r in drop_ratios]}, n_repeats={n_repeats}")
    logger.info("=" * 60)

    # ========== Phase 1: 收集原始数据 ==========
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

            batch_data = {}
            if hasattr(interaction, 'interaction'):
                iter_keys = interaction.interaction.keys()
            elif hasattr(interaction, 'columns'):
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
        logger.info("[Robustness] 测试样本太少, 跳过")
        return None

    original_ndcg = _compute_ndcg_at_k(all_original_scores, all_pos_items, k=eval_k)
    avg_len = all_seq_lens.float().mean().item()
    logger.info(f"[Baseline] NDCG@{eval_k} = {original_ndcg:.5f}  "
                f"(用户数={N}, 平均序列长={avg_len:.1f})")

    # ========== Phase 2: NNDCG (噪声鲁棒 NDCG) ==========
    logger.info("-" * 60)
    logger.info(">>> NNDCG: 噪声鲁棒NDCG (注入r%随机item后的绝对NDCG)")
    logger.info("-" * 60)

    noise_results = {}
    for ratio in drop_ratios:
        nndcg_list = []

        for repeat in range(n_repeats):
            noisy_scores_list = []

            with torch.no_grad():
                for batch_data in all_interactions:
                    noisy = _inject_noise_by_ratio(
                        batch_data, item_seq_field, item_seq_len_field,
                        n_items=n_items, noise_ratio=ratio
                    )
                    wrapped = _InteractionWrapper(noisy).to(device)
                    noisy_scores = model.full_sort_predict(wrapped)
                    noisy_scores_list.append(noisy_scores.cpu())

            all_noisy_scores = torch.cat(noisy_scores_list, dim=0)
            noisy_ndcg = _compute_ndcg_at_k(all_noisy_scores, all_pos_items, k=eval_k)
            nndcg_list.append(noisy_ndcg)

        avg_nndcg = float(np.mean(nndcg_list))
        std_nndcg = float(np.std(nndcg_list))
        retention = avg_nndcg / (original_ndcg + 1e-8)
        noise_results[ratio] = (avg_nndcg, std_nndcg)

        logger.info(
            f"[Noise={ratio:.0%}] "
            f"NNDCG@{eval_k} = {avg_nndcg:.5f} (±{std_nndcg:.5f})  |  "
            f"retention = {retention:.1%}"
        )

    noise_avg = float(np.mean([v[0] for v in noise_results.values()]))
    logger.info(
        f"[Noise Avg ] NNDCG@{eval_k} = {noise_avg:.5f}  "
        f"(mean over {[f'{r:.0%}' for r in drop_ratios]})"
    )

    # ========== Phase 3: RNDCG (稀疏鲁棒 NDCG) ==========
    logger.info("-" * 60)
    logger.info(">>> RNDCG: 稀疏鲁棒NDCG (删除r%交互后的绝对NDCG)")
    logger.info("-" * 60)

    sparsity_results = {}
    for ratio in drop_ratios:
        rndcg_list = []

        for repeat in range(n_repeats):
            drop_scores_list = []

            with torch.no_grad():
                for batch_data in all_interactions:
                    dropped = _drop_items_by_ratio(
                        batch_data, item_seq_field, item_seq_len_field,
                        drop_ratio=ratio, timestamp_field='timestamp_list'
                    )
                    wrapped = _InteractionWrapper(dropped).to(device)
                    drop_scores = model.full_sort_predict(wrapped)
                    drop_scores_list.append(drop_scores.cpu())

            all_drop_scores = torch.cat(drop_scores_list, dim=0)
            drop_ndcg = _compute_ndcg_at_k(all_drop_scores, all_pos_items, k=eval_k)
            rndcg_list.append(drop_ndcg)

        avg_rndcg = float(np.mean(rndcg_list))
        std_rndcg = float(np.std(rndcg_list))
        retention = avg_rndcg / (original_ndcg + 1e-8)
        sparsity_results[ratio] = (avg_rndcg, std_rndcg)

        logger.info(
            f"[Drop={ratio:.0%}] "
            f"RNDCG@{eval_k} = {avg_rndcg:.5f} (±{std_rndcg:.5f})  |  "
            f"retention = {retention:.1%}"
        )

    sparsity_avg = float(np.mean([v[0] for v in sparsity_results.values()]))
    logger.info(
        f"[Spars Avg ] RNDCG@{eval_k} = {sparsity_avg:.5f}  "
        f"(mean over {[f'{r:.0%}' for r in drop_ratios]})"
    )

    # ========== 最终 Summary ==========
    logger.info("=" * 60)
    logger.info("Summary (复制到 paper table):")
    ratio_hdr = "  ".join([f"{r:>6.0%}" for r in drop_ratios])
    logger.info(f"  Baseline NDCG@{eval_k} = {original_ndcg:.5f}")
    logger.info(f"  ratios           : {ratio_hdr}     Avg")
    noise_row = "  ".join([f"{noise_results[r][0]:.5f}" for r in drop_ratios])
    sparsity_row = "  ".join([f"{sparsity_results[r][0]:.5f}" for r in drop_ratios])
    logger.info(f"  NNDCG@{eval_k}         : {noise_row}   {noise_avg:.5f}")
    logger.info(f"  RNDCG@{eval_k}         : {sparsity_row}   {sparsity_avg:.5f}")
    logger.info("=" * 60)

    return {
        "baseline_ndcg": original_ndcg,
        "noise":    {r: noise_results[r] for r in drop_ratios},
        "noise_avg":    noise_avg,
        "sparsity": {r: sparsity_results[r] for r in drop_ratios},
        "sparsity_avg": sparsity_avg,
        "ratios": tuple(drop_ratios),
        "eval_k": eval_k,
        "n_samples": N,
        "avg_seq_len": avg_len,
    }

