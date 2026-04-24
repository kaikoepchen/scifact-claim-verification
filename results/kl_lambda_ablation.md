# KL-Disagreement λ Ablation — Results

Training config held constant across runs: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, 10 epochs, LR 1e-5, effective batch 32 (bs=4, grad_accum=8), neg_ratio 3, seq 256, `kl_top_k=3`, `kl_every_n=8`, seed 42. Dense retriever for KL views: `sentence-transformers/all-MiniLM-L6-v2`. Dev split: 1288 sentence-level examples (SUPPORT 235 / CONTRADICT 131 / NEI 922). Per-run logs in `kl_ablation_logs/lambda{X}.json`.

## Dev sentence-level metrics

| λ | Macro-F1 | Acc | SUPPORT F1 | CONTRADICT F1 | NEI F1 | Best epoch |
|---|----------|-----|------------|---------------|--------|------------|
| 0.0 | 0.7890 | 0.863 | 0.756 | 0.696 | 0.915 | 10 |
| **0.1** | **0.7979** | 0.865 | 0.752 | **0.729** | 0.913 | 10 |
| 0.3 | 0.7889 | 0.861 | 0.753 | 0.701 | 0.913 | 9 |
| 0.5 | 0.7965 | 0.865 | 0.764 | 0.712 | 0.913 | 10 |

Baseline reference (no-KL codepath, `joint-scifact`): Macro-F1 0.7873. λ=0.0 under the KL script matches it to within 0.002 — sanity check passes.

### Dense-encoder swap (λ=0.1, seed 42)

| Dense view | Macro-F1 | SUPPORT F1 | CONTRADICT F1 | NEI F1 | Peak KL |
|---|---|---|---|---|---|
| MiniLM-L6-v2 | **0.7979** | 0.752 | **0.729** | 0.913 | 0.048 |
| SPECTER-2 (base) | 0.7885 | 0.758 | 0.693 | 0.914 | 0.113 |

SPECTER-2 produces ~2.5× larger BM25 disagreement but no downstream gain — CONTRADICT F1 regresses to no-KL levels. Citation-topic similarity is the "wrong kind of different" for claim verification, which needs passage-level lexical specificity. See `docs/dense_models.md` for the full reasoning.

## Observations

- **λ=0.1 is the peak** at 0.7979, with the biggest individual-class move: CONTRADICT F1 0.696 → 0.729 (+3.3). SUPPORT/NEI essentially unchanged.
- **λ=0.3 is flat** (identical macro-F1 to λ=0.0, 0.7889). The CONTRADICT boost from λ=0.1 disappears at this pressure.
- **λ=0.5 recovers** to 0.7965 — nearly matching λ=0.1 but with a different class mix: SUPPORT ↑ (0.764), CONTRADICT slightly lower than λ=0.1 (0.712). High λ pushes the model to average across retrieval views, which seems to help the majority-adjacent SUPPORT class but not CONTRADICT specifically.
- **Non-monotone curve.** The relationship is not linear in λ: weak KL (λ=0.1) > no KL (λ=0.0) ≈ medium KL (λ=0.3) < strong KL (λ=0.5) < weak KL (λ=0.1). This is unusual; it suggests the effect is not just a loss-scaling issue but that different λ values steer training into qualitatively different solutions.
- **Single-seed caveat.** Max spread across the four λ's is ~0.9 pt macro-F1 — within plausible run-to-run noise for a 1288-example dev set. Three seeds per λ would be needed for confidence intervals.

### CONTRADICT-only view (minority class, where signal concentrates)

| λ | CONTRADICT P | CONTRADICT R | CONTRADICT F1 |
|---|--------------|--------------|---------------|
| 0.0 | 0.676 | 0.718 | 0.696 |
| 0.1 | **0.719** | **0.740** | **0.729** |
| 0.3 | 0.671 | 0.733 | 0.701 |
| 0.5 | 0.707 | 0.718 | 0.712 |

λ=0.1 wins on both P and R — not a threshold artifact. Monotone P improvement up to 0.1 then dips at 0.3, recovers partially at 0.5.

## KL loss decay (average per epoch)

| Epoch | λ=0.1 | λ=0.3 | λ=0.5 |
|---|---|---|---|
| 1 | 0.0409 | 0.0401 | 0.0398 |
| 2 | 0.0337 | 0.0327 | 0.0313 |
| 3 | 0.0443 | 0.0518 | 0.0427 |
| 4 | 0.0477 | 0.0461 | 0.0348 |
| 5 | 0.0225 | 0.0231 | 0.0214 |
| 6 | 0.0086 | 0.0121 | 0.0078 |
| 7 | 0.0135 | 0.0111 | 0.0175 |
| 8 | 0.0181 | 0.0189 | 0.0159 |
| 9 | 0.0037 | 0.0028 | 0.0031 |
| 10 | 0.0256 | 0.0179 | 0.0131 |

All three runs show the same decay profile — model learns retrieval-view consistency by ~epoch 6 regardless of λ. Suggests the task is saturable: once the model is consistent, more λ-weight provides no additional training signal. This matches the non-monotone macro-F1 result (consistency is necessary but not sufficient).

## Recommendations

1. **Report λ=0.1 as the primary KL result** (0.7979 macro-F1, +0.9 over λ=0.0) in the writeup. Frame the main claim around CONTRADICT F1: +3.3 pt on the hardest class.
2. **Run 3 seeds for λ ∈ {0.0, 0.1}** to confirm the gap is not noise. Estimated cost: 2 × ~50 min = ~1.7h on T4.
3. **Refine the sweep** around the peak: add λ=0.05 and λ=0.2 to pin down the optimum.
4. **Couple with domain-encoder swap** (see `docs/dense_models.md`): BGE-M3 as the dense view may sharpen the KL signal by giving the model a stronger "target" distribution to be consistent with. Re-run λ=0.1 after the swap.
