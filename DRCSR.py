"""
DRCSR: Dual-phase Reliability Calibration for Robust Multimodal
Sequential Recommendation.

Modules:
    (a) Train-only Robust Augmentation
    (b) Multimodal Factor Encoding
    (c) Reliability Calibration  (Phase I)
    (d) Calibrated State-Memory Encoding  (Phase I)
    (e) Candidate-Aware Motive Composition  (Phase II)
"""
import os
import math
import torch
from torch import nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender


class DRCSR(SequentialRecommender):
    """
    DRCSR  –  Dual-phase Reliability Calibration for Robust
    Multimodal Sequential Recommendation.

    Phase I (state-writing step):
        Factor-level reliability calibration + gated factor memory
        → breaks the cascading reliability degradation chain (C1).
    Phase II (scoring step):
        Candidate-aware motive composition
        → resolves the precision–coverage dilemma (C2).
    """

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.config = config
        self.dataset_name = config["dataset"].split("/")[-1]

        # --- model hyper-parameters ---
        self.hidden_size = self._cfg(["hidden_size", "embedding_size"], 128)
        self.factor_num = self._cfg(["factor_num", "num_factors"], 4)
        self.dropout_prob = self._sanitize_dropout(
            self._cfg(["dropout_prob", "hidden_dropout_prob", "dropout", "attn_dropout_prob"], 0.1)
        )

        self.lambda_sep = self._cfg(["lambda_sep"], 1e-4)
        self.lambda_stab = self._cfg(["lambda_stab"], 1e-4)
        self.lambda_align = self._cfg(["lambda_align"], 5e-5)
        self.t0 = self._cfg(["t0"], 0.8)
        self.gamma = self._cfg(["gamma"], 0.3)
        self.num_train_negs = int(self._cfg(["num_train_negs"], 64))
        self.residual_beta = float(self._cfg(["residual_beta"], 0.35))
        self.candidate_logit_scale = float(self._cfg(["candidate_logit_scale"], 0.45))

        self.logq_correction = float(self._cfg(["logq_correction"], 1.0))
        self.lambda_comp = float(self._cfg(["lambda_comp"], 1e-5))
        self.comp_tau = float(self._cfg(["comp_tau"], 0.4))
        self.freeze_mm = bool(self._cfg(["freeze_mm"], True))
        self.label_smoothing = float(self._cfg(["label_smoothing"], 0.0))
        self.train_score_temp = float(self._cfg(["train_score_temp"], 1.0))

        self.max_seq_length = self._cfg(["MAX_ITEM_LIST_LENGTH", "max_seq_length"], 50)

        # --- gated factor memory ---
        self.use_mem_gate = bool(self._cfg(["use_mem_gate"], True))

        # --- train-only robust augmentation ---
        self.train_aug_noise = float(self._cfg(["train_aug_noise"], 0.0))
        self.train_aug_drop = float(self._cfg(["train_aug_drop"], 0.0))
        self.embed_noise_std = float(self._cfg(["embed_noise_std"], 0.0))

        # --- load pretrained multimodal embeddings ---
        txt_weight = self._load_mm_table([
            f"./dataset/{self.dataset_name}/txt_emb.pt",
            f"./dataset/{self.dataset_name}/text_emb.pt",
        ])
        img_weight = self._load_mm_table([
            f"./dataset/{self.dataset_name}/img_emb.pt",
            f"./dataset/{self.dataset_name}/image_emb.pt",
        ])
        txt_dim = txt_weight.size(1)
        img_dim = img_weight.size(1)

        # --- trainable layers (defined before self.apply) ---
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.txt_proj = nn.Linear(txt_dim, self.hidden_size)
        self.img_proj = nn.Linear(img_dim, self.hidden_size)

        self.base_encoder = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.shared_factor_proj = nn.Linear(self.hidden_size, self.factor_num * self.hidden_size)
        self.txt_factor_proj = nn.Linear(self.hidden_size, self.factor_num * self.hidden_size)
        self.img_factor_proj = nn.Linear(self.hidden_size, self.factor_num * self.hidden_size)

        self.global_prior = nn.Parameter(torch.zeros(self.factor_num))

        self.local_adjust = nn.Sequential(
            nn.Linear(self.hidden_size * 2 + 3, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
        )
        self.query_adjust = nn.Sequential(
            nn.Linear(self.hidden_size * 2 + 1, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
        )
        self.comp_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 2 + 1, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
        )
        self.amb_mlp = nn.Sequential(
            nn.Linear(self.hidden_size + self.factor_num, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid(),
        )

        self.init_state = nn.Parameter(torch.zeros(1, self.hidden_size))
        self.init_state_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.content_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.gru_cell = nn.GRUCell(self.hidden_size, self.hidden_size)

        if self.use_mem_gate:
            self.mem_gate = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.score_scale = math.sqrt(self.hidden_size)
        self.eps = 1e-8

        # --- weight initialisation ---
        self.apply(self._init_weights)

        # --- multimodal embeddings (loaded after self.apply) ---
        self.txt_embedding = nn.Embedding.from_pretrained(
            txt_weight, freeze=self.freeze_mm, padding_idx=0)
        self.img_embedding = nn.Embedding.from_pretrained(
            img_weight, freeze=self.freeze_mm, padding_idx=0)

        effective_n = max(self.n_items - 1, 2)
        effective_k = max(self.num_train_negs, 1)
        self._logq_term = math.log(effective_n / effective_k) * self.logq_correction

    # ================================================================
    # Utilities
    # ================================================================
    def _cfg(self, keys, default):
        for k in keys:
            try:
                v = self.config[k]
                if v is not None:
                    return v
            except Exception:
                pass
        return default

    def _sanitize_dropout(self, p):
        if p is None:
            return 0.1
        p = float(p)
        return max(0.0, min(1.0, p))

    def _load_mm_table(self, candidate_paths):
        path = None
        for p in candidate_paths:
            if os.path.exists(p):
                path = p
                break
        if path is None:
            raise FileNotFoundError(
                f"Cannot find multimodal embedding file. Tried: {candidate_paths}")
        weight = torch.load(path, map_location="cpu")
        if not isinstance(weight, torch.Tensor):
            weight = torch.tensor(weight)
        weight = weight.float()
        if weight.size(0) == self.n_items - 1:
            pad = torch.zeros(1, weight.size(1), dtype=weight.dtype)
            weight = torch.cat([pad, weight], dim=0)
        elif weight.size(0) == self.n_items:
            pass
        elif weight.size(0) > self.n_items:
            weight = weight[: self.n_items]
        else:
            pad = torch.zeros(self.n_items - weight.size(0), weight.size(1),
                              dtype=weight.dtype)
            weight = torch.cat([weight, pad], dim=0)
        return weight

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _factorize(self, x, proj):
        out = proj(x).view(*x.shape[:-1], self.factor_num, self.hidden_size)
        return torch.tanh(out)

    def _safe_timestamp_key(self, interaction):
        if "timestamp_list" in interaction:
            return "timestamp_list"
        if "time_list" in interaction:
            return "time_list"
        keys = list(interaction.interaction.keys())
        for k in keys:
            lk = k.lower()
            if k.endswith("_list") and ("time" in lk or "timestamp" in lk):
                return k
        raise KeyError(f"No timestamp list field found. Available keys: {keys}")

    # ================================================================
    # (b) Multimodal Factor Encoding  +  (a) Embedding Perturbation
    # ================================================================
    def _encode_item_tensor(self, item_ids, position_ids=None):
        item_e = self.item_embedding(item_ids)
        txt_e = self.txt_proj(self.txt_embedding(item_ids))
        img_e = self.img_proj(self.img_embedding(item_ids))

        if self.training and self.embed_noise_std > 0:
            item_e = item_e + torch.randn_like(item_e) * self.embed_noise_std
            txt_e = txt_e + torch.randn_like(txt_e) * self.embed_noise_std
            img_e = img_e + torch.randn_like(img_e) * self.embed_noise_std

        if position_ids is not None:
            pos_e = self.position_embedding(position_ids)
            item_e = item_e + pos_e
            txt_e = txt_e + pos_e
            img_e = img_e + pos_e

        h = self.base_encoder(torch.cat([item_e, txt_e, img_e], dim=-1))
        h = self.layer_norm(h)
        z = self._factorize(h, self.shared_factor_proj)
        z_txt = self._factorize(txt_e, self.txt_factor_proj)
        z_img = self._factorize(img_e, self.img_factor_proj)
        return item_e, h, z, z_txt, z_img

    # ================================================================
    # Temporal drift
    # ================================================================
    def _gap_surprise(self, timestamp):
        ts = timestamp.float()
        delta = ts[:, 1:] - ts[:, :-1]
        delta = torch.clamp(delta, min=0.0)
        delta = torch.log1p(delta)
        first = torch.zeros(ts.size(0), 1, device=ts.device, dtype=ts.dtype)
        delta = torch.cat([first, delta], dim=1)
        cumsum = torch.cumsum(delta, dim=1)
        idx = torch.arange(1, delta.size(1) + 1,
                           device=delta.device, dtype=delta.dtype).view(1, -1)
        mean_prev = cumsum / idx
        surprise = torch.abs(delta - mean_prev)
        surprise = surprise / (surprise.mean(dim=1, keepdim=True) + self.eps)
        surprise = torch.clamp(surprise, 0.0, 5.0)
        return surprise.unsqueeze(-1)

    # ================================================================
    # Motive regularisation: separation loss
    # ================================================================
    def _sep_loss(self, factors, mask):
        z = F.normalize(factors, dim=-1, eps=self.eps)
        gram = torch.matmul(z, z.transpose(-1, -2))
        eye = torch.eye(self.factor_num, device=z.device).view(
            1, 1, self.factor_num, self.factor_num)
        off_diag = (gram ** 2) * (1.0 - eye)
        denom = mask.sum() * self.factor_num * max(self.factor_num - 1, 1)
        return off_diag.sum() / (denom + self.eps)

    # ================================================================
    # (c) Reliability Calibration
    # ================================================================
    def _local_reliability(self, prev_state, factors, z_txt, z_img,
                           drift, hist_factor_memory):
        state_expand = prev_state.unsqueeze(1).expand(-1, self.factor_num, -1)
        consistency = F.cosine_similarity(
            z_txt, z_img, dim=-1, eps=self.eps).unsqueeze(-1)
        persistence = F.cosine_similarity(
            factors, hist_factor_memory, dim=-1, eps=self.eps).unsqueeze(-1)
        drift_expand = drift.unsqueeze(1).expand(-1, self.factor_num, -1)
        feat = torch.cat([state_expand, factors,
                          drift_expand, consistency, persistence], dim=-1)
        delta = torch.sigmoid(self.local_adjust(feat)).squeeze(-1)
        prior = torch.sigmoid(self.global_prior).unsqueeze(0)
        return torch.clamp(prior * delta, 0.05, 0.95)

    def _residualize_weight(self, w):
        return self.residual_beta + (1.0 - self.residual_beta) * w

    # ================================================================
    # (e) Candidate-Aware Motive Composition
    # ================================================================
    def _query_weights(self, state, factor_memory, last_drift):
        state_expand = state.unsqueeze(1).expand(-1, self.factor_num, -1)
        drift_expand = last_drift.unsqueeze(1).expand(-1, self.factor_num, -1)
        feat = torch.cat([state_expand, factor_memory, drift_expand], dim=-1)
        delta = torch.sigmoid(self.query_adjust(feat)).squeeze(-1)
        prior = torch.sigmoid(self.global_prior).unsqueeze(0)
        w_query = torch.clamp(prior * delta, 0.05, 0.95)

        amb_in = torch.cat([state, w_query], dim=-1)
        ambiguity = self.amb_mlp(amb_in).squeeze(-1)
        temperature = torch.clamp(self.t0 + self.gamma * ambiguity,
                                  min=0.5, max=2.0)

        comp_in = torch.cat([state_expand, factor_memory,
                             w_query.unsqueeze(-1)], dim=-1)
        comp_score = torch.clamp(self.comp_mlp(comp_in).squeeze(-1), -10.0, 10.0)
        alpha = F.softmax(comp_score / temperature.unsqueeze(-1), dim=-1)
        alpha = torch.clamp(alpha, min=1e-6)
        alpha = alpha / alpha.sum(dim=-1, keepdim=True)
        return w_query, alpha

    def _refine_alpha_with_candidate(self, base_alpha, item_factors,
                                     factor_memory):
        if item_factors.dim() == 3:
            sim = F.cosine_similarity(item_factors, factor_memory,
                                      dim=-1, eps=self.eps)
            logits = (torch.log(torch.clamp(base_alpha, min=1e-6))
                      + self.candidate_logit_scale * sim)
            return F.softmax(logits, dim=-1)
        if item_factors.dim() == 4:
            mem = factor_memory.unsqueeze(1).expand(
                -1, item_factors.size(1), -1, -1)
            base = base_alpha.unsqueeze(1).expand(
                -1, item_factors.size(1), -1)
            sim = F.cosine_similarity(item_factors, mem, dim=-1, eps=self.eps)
            logits = (torch.log(torch.clamp(base, min=1e-6))
                      + self.candidate_logit_scale * sim)
            return F.softmax(logits, dim=-1)
        raise ValueError(f"Unexpected item_factors dim: {item_factors.dim()}")

    # ================================================================
    # Item representation / scoring
    # ================================================================
    def _compose_item_rep(self, item_ids, factor_weights, factor_memory=None):
        item_e, _, z, _, _ = self._encode_item_tensor(item_ids, position_ids=None)
        if factor_memory is not None:
            factor_weights = self._refine_alpha_with_candidate(
                factor_weights, z, factor_memory)
        m = torch.einsum("...k,...kd->...d", factor_weights, z)
        rep = self.layer_norm(item_e + self.content_proj(m))
        rep = torch.tanh(rep)
        return rep

    def _all_item_factors(self):
        device = self.item_embedding.weight.device
        all_ids = torch.arange(self.n_items, device=device)
        item_e, _, z, _, _ = self._encode_item_tensor(all_ids, position_ids=None)
        return item_e, z

    def _score_full(self, state, alpha, factor_memory):
        item_e, all_factors = self._all_item_factors()
        mem = factor_memory.unsqueeze(1).expand(-1, all_factors.size(0), -1, -1)
        all_factors_b = all_factors.unsqueeze(0).expand(
            state.size(0), -1, -1, -1)
        base_alpha = alpha.unsqueeze(1).expand(-1, all_factors.size(0), -1)
        sim = F.cosine_similarity(all_factors_b, mem, dim=-1, eps=self.eps)
        logits = (torch.log(torch.clamp(base_alpha, min=1e-6))
                  + self.candidate_logit_scale * sim)
        alpha_item = F.softmax(logits, dim=-1)

        content = torch.einsum("bnk,nkd->bnd", alpha_item, all_factors)
        rep = self.layer_norm(item_e.unsqueeze(0) + self.content_proj(content))
        rep = torch.tanh(rep)

        state = F.normalize(state, dim=-1, eps=self.eps)
        rep = F.normalize(rep, dim=-1, eps=self.eps)

        score = torch.einsum("bd,bnd->bn", state, rep) * self.score_scale
        score[:, 0] = -1e9
        return score

    def _sample_neg_items(self, pos_items, device):
        B = pos_items.size(0)
        K = self.num_train_negs
        neg_items = torch.randint(low=1, high=max(self.n_items, 2),
                                  size=(B, K), device=device,
                                  dtype=pos_items.dtype)
        pos_expand = pos_items.unsqueeze(1).expand(B, K)
        neg_items = torch.where(neg_items.eq(pos_expand),
                                (neg_items % (self.n_items - 1)) + 1,
                                neg_items)
        return neg_items.clamp(min=1, max=self.n_items - 1)

    # ================================================================
    # (a) Train-only Robust Augmentation: Sequence Perturbation
    # ================================================================
    def _train_augment_seq(self, item_seq, item_seq_len):
        """Perturb the input sequence during training: inject noise items
        and randomly drop items. The last item is always protected."""
        aug = item_seq.clone()
        valid = (item_seq > 0)
        B = item_seq.size(0)
        device = item_seq.device
        last_idx = (item_seq_len - 1).clamp(min=0)
        protect = torch.zeros_like(valid)
        protect[torch.arange(B, device=device), last_idx] = True

        if self.train_aug_noise > 0:
            noise_mask = ((torch.rand_like(item_seq.float()) < self.train_aug_noise)
                          & valid & ~protect)
            random_items = torch.randint(1, self.n_items, item_seq.shape,
                                         device=device, dtype=item_seq.dtype)
            aug = torch.where(noise_mask, random_items, aug)

        if self.train_aug_drop > 0:
            drop_mask = ((torch.rand_like(item_seq.float()) < self.train_aug_drop)
                         & valid & ~protect)
            aug[drop_mask] = 0

        return aug

    class _AugInteraction:
        """Lightweight wrapper that substitutes item_seq while forwarding
        all other fields from the original interaction."""
        def __init__(self, base, seq_key, aug_seq):
            self._base = base
            self._seq_key = seq_key
            self._aug_seq = aug_seq
        def __getitem__(self, key):
            if key == self._seq_key:
                return self._aug_seq
            return self._base[key]
        def __contains__(self, key):
            return key in self._base
        @property
        def interaction(self):
            return self._base.interaction

    # ================================================================
    # (d) Calibrated State-Memory Encoding
    # ================================================================
    def encode_user_state(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        time_key = self._safe_timestamp_key(interaction)
        timestamp = interaction[time_key]

        B, L = item_seq.size()
        device = item_seq.device
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)

        item_e, _, z, z_txt, z_img = self._encode_item_tensor(
            item_seq, position_ids=pos_ids)
        mask = (item_seq > 0).float()
        drift_seq = self._gap_surprise(timestamp)

        state = torch.tanh(self.init_state_proj(self.init_state)).expand(B, -1)
        factor_memory = torch.zeros(
            B, self.factor_num, self.hidden_size, device=device)
        factor_count = torch.zeros(B, 1, 1, device=device)

        stab_terms = []
        for t in range(L):
            valid = mask[:, t].unsqueeze(-1)
            w_t = self._local_reliability(
                state, z[:, t], z_txt[:, t], z_img[:, t],
                drift_seq[:, t], factor_memory)
            w_train = self._residualize_weight(w_t)
            m_train = torch.einsum("bk,bkd->bd", w_train, z[:, t])
            v_train = torch.tanh(self.layer_norm(
                item_e[:, t] + self.content_proj(m_train)))
            next_state = torch.tanh(self.gru_cell(v_train, state))
            state = valid * next_state + (1.0 - valid) * state

            valid_k = valid.unsqueeze(-1)
            z_write = z[:, t] * w_train.unsqueeze(-1)

            if self.use_mem_gate:
                gate_in = torch.cat([factor_memory, z_write], dim=-1)
                gate = torch.sigmoid(self.mem_gate(gate_in))
                new_mem = gate * factor_memory + (1.0 - gate) * z_write
                factor_memory = (valid_k * new_mem
                                 + (1.0 - valid_k) * factor_memory)
            else:
                factor_memory = (
                    (factor_memory * factor_count + z_write * valid_k)
                    / (factor_count + valid_k + self.eps))
                factor_count = factor_count + valid_k

            stab_terms.append(
                (w_t.mean(dim=-1, keepdim=True) * drift_seq[:, t]).squeeze(-1))

        sep_loss = self._sep_loss(z, mask)
        stab_loss = torch.clamp(
            torch.cat(stab_terms, dim=0).mean(), 0.0, 10.0)

        last_index = (item_seq_len - 1).clamp(min=0)
        last_drift = drift_seq[torch.arange(B, device=device), last_index]
        w_query, alpha = self._query_weights(state, factor_memory, last_drift)

        return {
            "state": state, "w_query": w_query, "alpha": alpha,
            "factor_memory": factor_memory,
            "sep_loss": sep_loss, "stab_loss": stab_loss,
        }

    # ================================================================
    # Training
    # ================================================================
    def calculate_loss(self, interaction):
        if self.training and (self.train_aug_noise > 0
                              or self.train_aug_drop > 0):
            item_seq = interaction[self.ITEM_SEQ]
            item_seq_len = interaction[self.ITEM_SEQ_LEN]
            aug_seq = self._train_augment_seq(item_seq, item_seq_len)
            interaction = self._AugInteraction(
                interaction, self.ITEM_SEQ, aug_seq)

        encoded = self.encode_user_state(interaction)
        state = F.normalize(encoded["state"], dim=-1, eps=self.eps)
        alpha = encoded["alpha"]
        factor_memory = encoded["factor_memory"]

        pos_items = interaction[self.POS_ITEM_ID]
        B = pos_items.size(0)
        device = pos_items.device

        pos_rep = self._compose_item_rep(pos_items, alpha, factor_memory)
        pos_rep = F.normalize(pos_rep, dim=-1, eps=self.eps)
        pos_score = (torch.sum(state * pos_rep, dim=-1, keepdim=True)
                     * self.score_scale)

        neg_items = self._sample_neg_items(pos_items, device)
        K = neg_items.size(1)
        flat_neg = neg_items.reshape(-1)
        alpha_exp = (alpha.unsqueeze(1)
                     .expand(B, K, self.factor_num)
                     .reshape(-1, self.factor_num))
        fm_exp = (factor_memory.unsqueeze(1)
                  .expand(B, K, self.factor_num, self.hidden_size)
                  .reshape(B * K, self.factor_num, self.hidden_size))

        neg_rep = self._compose_item_rep(flat_neg, alpha_exp, fm_exp)
        neg_rep = (F.normalize(neg_rep, dim=-1, eps=self.eps)
                   .view(B, K, self.hidden_size))
        neg_score = (torch.einsum("bd,bmd->bm", state, neg_rep)
                     * self.score_scale)
        neg_score = neg_score + self._logq_term

        logits = torch.cat([pos_score, neg_score], dim=1)
        logits = logits / self.train_score_temp
        labels = torch.zeros(B, dtype=torch.long, device=device)
        rec_loss = F.cross_entropy(logits, labels,
                                   label_smoothing=self.label_smoothing)

        align_target = encoded["w_query"] / (
            encoded["w_query"].sum(dim=-1, keepdim=True) + self.eps)
        align_loss = F.mse_loss(alpha, align_target)
        alpha_concentration = torch.sum(alpha ** 2, dim=-1)
        comp_loss = torch.mean((alpha_concentration - self.comp_tau) ** 2)

        total_loss = (
            rec_loss
            + self.lambda_sep * encoded["sep_loss"]
            + self.lambda_stab * encoded["stab_loss"]
            + self.lambda_align * align_loss
            + self.lambda_comp * comp_loss
        )
        return torch.nan_to_num(total_loss, nan=1e3, posinf=1e3, neginf=1e3)

    # ================================================================
    # Inference
    # ================================================================
    def predict(self, interaction):
        encoded = self.encode_user_state(interaction)
        state = F.normalize(encoded["state"], dim=-1, eps=self.eps)
        alpha = encoded["alpha"]
        test_item = interaction[self.ITEM_ID]
        test_rep = F.normalize(
            self._compose_item_rep(test_item, alpha, encoded["factor_memory"]),
            dim=-1, eps=self.eps)
        score = torch.sum(state * test_rep, dim=-1) * self.score_scale
        return torch.nan_to_num(score, nan=0.0, posinf=1e4, neginf=-1e4)

    def full_sort_predict(self, interaction):
        encoded = self.encode_user_state(interaction)
        score = self._score_full(
            encoded["state"], encoded["alpha"], encoded["factor_memory"])
        return torch.nan_to_num(score, nan=0.0, posinf=1e4, neginf=-1e4)
