"""Test-time interventions on FrameSamp perceptual memory.

The released checkpoint is fixed, but everything that decides *what the memory
contains* happens in numpy on the server side, after the vision encoder and
before the JIT boundary. That makes a handful of the questions we care about
answerable without retraining: how much the memory is worth at all, whether its
temporal ordering is used, whether 32 frames is simply too few, and whether the
same budget spent differently would do better.

Engaged by ROBOMME_MEM_MODE; absent or "baseline" leaves the stock path alone.

    off        memory zeroed and masked out -> the modulation contributes only
               its bias, so the gap to baseline is what memory is worth under
               these exact weights (unlike the paper's pi0.5 row, which is a
               separately trained model)
    timeshuf   the same 32 frames, both temporal channels destroyed: frame rows
               are permuted (breaking the slot-index RoPE, which is monotone in
               time only because indices arrive sorted) and pos_emb is permuted
               independently (breaking the absolute timestamp). If this does not
               hurt, the memory is being read as an unordered bag of frames.
    budget128  32 -> 128 frames. Only the JIT input shape depends on `budget`;
               no weight does, and pos_emb comes from the data rather than a
               learned table, so a longer memory sequence is well defined.
    primacy    same 32-frame budget, 12 of them log-spaced into the first
               eighth of the history. Occlusion tasks put their evidence in the
               opening frames, which uniform sampling reaches about twice.
"""

from __future__ import annotations

import os

import numpy as np

MODE = os.environ.get("ROBOMME_MEM_MODE", "baseline").strip().lower()
_RNG = np.random.default_rng(int(os.environ.get("ROBOMME_MEM_SEED", "0")))
_APPLIED = False

# budget // (token_per_image * num_views) is the frame count, so scaling the
# budget scales the frames at unchanged per-frame resolution. 8/32/128 against
# `off` gives a dose-response curve: if a quarter of the frames costs nothing and
# four times as many buys nothing, capacity is not what limits this memory.
BUDGET_SCALE = {"budget128": 4.0, "budget8": 0.25}

# The clean version of "does more temporal resolution help".
#
# budget8/budget128 change the number of memory tokens, and the action expert's
# query RoPE starts at that length (`q_positions = arange(mem_len, ...)`), so
# they move the queries into positions the model never trained on -- they测 the
# RoPE, not the frame count. Cutting token_per_image from 16 to 4 instead buys
# 128 frames at 4 tokens each: 512 memory tokens, the same mem_len, the same
# RoPE, four times the temporal resolution. The probe found 2x2 features decode
# as well as 4x4 on both labelled tasks, so the spatial cost is near zero.
TOKENS_PER_IMAGE = {"frames128": 4}


def _primacy_indices(step_idx: int, max_size: int) -> list[int]:
    """Uniform coverage, but with a twelfth of the budget crowded into the start."""
    n_early = max(1, max_size // 8 * 3)
    early_span = max(1, step_idx // 8)
    early = np.unique(np.geomspace(1, early_span + 1, n_early).astype(np.int64) - 1)
    rest = np.linspace(0, step_idx, max_size - len(early), dtype=np.int64)
    return sorted(set(early.tolist()) | set(rest.tolist()))


def apply() -> str:
    """Patch the memory buffer in place. Safe to call more than once."""
    global _APPLIED
    if _APPLIED or MODE in ("", "baseline"):
        return MODE

    from mme_vla_suite.shared import mem_buffer as mb
    from mme_vla_suite.shared.data_utils import even_sampling_indices

    if MODE in BUDGET_SCALE or MODE in TOKENS_PER_IMAGE:
        scale = BUDGET_SCALE.get(MODE, 1.0)
        tpi = TOKENS_PER_IMAGE.get(MODE)
        from mme_vla_suite.models.config import utils as cfg_utils

        _orig_get = cfg_utils.get_history_config

        def _scaled_get(history_config):
            cfg = _orig_get(history_config)
            if getattr(cfg, "representation_type", None) == "perceptual":
                cfg.budget = int(round(int(cfg.budget) * scale))
                if tpi is not None:
                    cfg.token_per_image = tpi
            return cfg

        cfg_utils.get_history_config = _scaled_get
        # history_pi0 binds the name at import time, so patching the defining
        # module alone would miss the call site that actually shapes the input.
        from mme_vla_suite.models.integration import history_pi0

        history_pi0.get_history_config = _scaled_get

    _orig_indices = mb.MemoryBuffer.get_frame_sampling_indices
    _orig_prepare = mb.MemoryBuffer._prepare_frame_sampling

    def _indices(self, step_idx, token_budget, token_per_image):
        max_size = token_budget // (token_per_image * self.num_views)
        if MODE == "primacy" and step_idx >= max_size:
            return _primacy_indices(step_idx, max_size)
        return even_sampling_indices(step_idx, max_size)

    def _prepare(self, history_feats, indices_to_load, token_budget, token_per_image):
        img, pos, state, mask = _orig_prepare(
            self, history_feats, indices_to_load, token_budget, token_per_image
        )
        if MODE == "off":
            # Zeroing matters as much as masking: an all-False mask alone leaves
            # softmax uniform over real frames, which attends to everything
            # rather than to nothing.
            img = np.zeros_like(img)
            pos = np.zeros_like(pos)
            state = np.zeros_like(state)
            mask = np.zeros_like(mask)
        elif MODE == "repeatcur":
            # A better "no memory" control than `off`. Every slot gets the newest
            # sampled frame -- which is the current observation, since the sampled
            # indices end at step_idx -- so the encoder, the cross-attention and
            # the modulating Dense all see inputs of the kind they were trained on,
            # rather than zeros they never saw. What is removed is the history:
            # the current frame already reaches the VLM through the observation, so
            # a memory made of 32 copies of it tells the policy nothing new.
            #
            # pos_emb is deliberately left correct. The memory then asserts "at
            # each of these past times the scene looked as it does now" -- false,
            # but positionally well formed, so only the visual history is ablated.
            per = self.num_views * token_per_image
            n = int(mask.sum()) // per
            if n > 1:
                img = img.copy()
                img[: n * per] = np.tile(img[(n - 1) * per : n * per], (n, 1))
        elif MODE == "timeshuf":
            # mask comes back already expanded to patch granularity, so the frame
            # count is its sum divided by the tokens each frame contributes.
            per = self.num_views * token_per_image
            n = int(mask.sum()) // per
            if n > 1:
                def shuffled(a):
                    blocks = a[: n * per].reshape(n, per, a.shape[-1])
                    out = a.copy()
                    out[: n * per] = blocks[_RNG.permutation(n)].reshape(n * per, a.shape[-1])
                    return out
                # img and pos are permuted independently: one breaks the slot
                # order the RoPE reads, the other the absolute timestamp.
                img, pos = shuffled(img), shuffled(pos)
        return img, pos, state, mask

    mb.MemoryBuffer.get_frame_sampling_indices = _indices
    mb.MemoryBuffer._prepare_frame_sampling = _prepare
    _APPLIED = True
    print(f"[mem_intervention] mode={MODE}", flush=True)
    return MODE
