# Domain-Specific Dense Retrievers: Analysis & Recommendation

Written 2026-04-20. Context: today's retriever setup is **BM25 (sparse) + all-MiniLM-L6-v2 (dense)**. MiniLM is a general-purpose 22M-param encoder; it's cheap but generic. For a corpus of scientific abstracts, a domain-tuned encoder should (a) retrieve better on its own and (b) change the character of the BM25-vs-dense disagreement signal that drives the whole research angle.

Note: `src/claimverify/retrieval/dense.py` already declares `DEFAULT_MODEL = "BAAI/bge-m3"`, but the KL-finetuning script (`scripts/12_finetune_kl_disagreement.py:269`) still hard-codes `all-MiniLM-L6-v2`. The λ ablation was started under MiniLM to stay comparable with the existing λ=0.3 result — swapping encoders is a separate experiment.

---

## Candidate models

### 1. SPECTER-2 (`allenai/specter2_base`)

- **Architecture**: SciBERT backbone (110M params), fine-tuned with triplet loss on citation pairs from 6M scientific papers. Produces a **document-level** embedding designed for "cite-worthy" similarity.
- **Domain**: all-science; trained across CS, biomed, physics, humanities.
- **Native unit**: paper (title + abstract). Not designed for sentence-level retrieval.
- **Adapters**: Proximity, Ad-Hoc Query, Classification, Regression — SPECTER-2 ships four task-specific adapters. For SciFact (claim → abstract), the **Ad-Hoc Query** adapter is the right fit (queries are short strings, documents are abstracts).
- **Integration cost**: Low. Sentence-Transformers supports it; need to download the adapter weights separately. Embedding dim 768.

### 2. BGE-M3 (`BAAI/bge-m3`)

- **Architecture**: XLM-RoBERTa backbone (~560M params), trained with a hybrid loss that produces **three outputs simultaneously**: dense embedding, sparse (lexical) weights, and multi-vector (ColBERT-style) representations.
- **Domain**: general-purpose, multilingual (100+ languages), trained on MS-MARCO-scale data + synthetic queries.
- **Native unit**: passage up to 8192 tokens — ideal for full abstracts without truncation.
- **Integration cost**: Zero additional work on the retrieval class — the current `DenseRetriever` default already points at BGE-M3. Just need to build the index. Embedding dim 1024.
- **Unique value**: The multi-vector + sparse heads mean BGE-M3 itself could replace our BM25+dense stack as a single-model hybrid. For our research angle, though, we specifically *want* two sources of disagreement, so we'd keep BM25 and use only BGE-M3's dense head.

### 3. PubMedBERT / BiomedNLP-PubMedBERT

- **Architecture**: BERT-base (110M params) pretrained from scratch on 14M PubMed abstracts. **It's a masked-LM, not a retriever** — out of the box it produces token embeddings, not good sentence embeddings.
- **For retrieval, use the SBERT variant**: `pritamdeka/S-PubMedBert-MS-MARCO` (or `MS-MARCO-SPECTER` variants). This is PubMedBERT fine-tuned on MS-MARCO with contrastive loss, so it actually encodes queries and passages into a comparable space.
- **Domain**: biomedical only. Tightly aligned with SciFact (SciFact claims are largely biomedical: drug effects, diseases, clinical findings).
- **Integration cost**: Low via Sentence-Transformers. Embedding dim 768, max seq 512.

---

## How each interacts with the disagreement signal

The current BM25-vs-MiniLM Jaccard@10 averages **0.224** (README). That's low — retrievers often surface different docs, and when they do agree accuracy rises from 69% → 81%. The signal works because the two views capture complementary notions of relevance: **BM25 = lexical overlap, MiniLM = generic semantic similarity**.

Swapping MiniLM for a domain encoder changes the second view:

| Swap | Likely Jaccard trend | Why |
|---|---|---|
| MiniLM → **SPECTER-2** | ↓ (more disagreement) | SPECTER optimizes for citation-worthiness — it ranks papers by "is this the kind of paper one would cite for this topic," which is orthogonal to lexical overlap. Expected: more distinctive dense ranking, lower overlap with BM25. |
| MiniLM → **BGE-M3** | ≈ or ↑ slightly | BGE-M3 is trained on MS-MARCO — partly lexical in what it learned to reward. Expect similar or modestly higher overlap with BM25 than MiniLM. |
| MiniLM → **S-PubMedBERT** | ↓↓ | Biomedical-only pretraining means the embedding space clusters around domain semantics (drug → compound → therapy), not surface words. Strongest differentiation from BM25 on biomedical claims. |

**Implication for KL regularization**: more disagreement isn't automatically better. The KL loss penalizes the *model's* disagreement given two evidence sets; if the two retrievers return genuinely contradictory evidence, forcing consistent predictions could *hurt* (teaching the model to average over bad retrieval). But if disagreement mostly reflects complementary-but-valid evidence (common for biomedical), then stronger disagreement = richer training signal.

This is exactly the thing we can measure, and doing so is probably worth more than the λ grid itself.

---

## Recommendation: three-stage rollout

### Stage 1 — Swap in BGE-M3 for retrieval-only eval (low risk, 1 hour)

Only touches inference. Rebuild the dense index with BGE-M3 (the code already defaults to it). Re-run `scripts/02_dense_retrieval.py` and `scripts/03_disagreement_analysis.py`. Measure:

- nDCG@10, Recall@10 vs. the current MiniLM numbers
- New Jaccard@10 with BM25
- Agreement-accuracy gap on dev

If BGE-M3 beats MiniLM on retrieval metrics, keep it as the new dense baseline. No re-training required.

### Stage 2 — Add SPECTER-2 as a third retriever (1–2 days)

Subclass or parameterize `DenseRetriever` to load SPECTER-2 with the ad-hoc-query adapter. Add a `TripleRetriever` wrapper or extend `pipeline.py` to support three-way RRF. Then:

- Measure pairwise Jaccard: BM25↔BGE, BM25↔SPECTER, BGE↔SPECTER
- If the three views are genuinely diverse (pairwise Jaccard < 0.4), there's real signal in three-way disagreement
- Extend the KL loss from pairwise to three-way (sum of three pairwise symmetric KLs, or a single KL to the mean distribution)

This is **FINDINGS.md Priority 3** ("Multi-Retriever KL with 3+ Retrievers") and the novelty argument rests partly on it.

### Stage 3 — Biomedical specialization via S-PubMedBERT (optional)

Only worth doing if (a) BGE-M3 + SPECTER-2 already show the three-way signal works, and (b) we want to argue domain adaptation is an additional win. S-PubMedBERT is narrower than SPECTER (biomed-only vs all-science) so it competes with SPECTER more than complements it.

Skip this unless we also evaluate on **HealthVer** (FINDINGS.md Priority 5), where the domain match matters.

---

## Usefulness — bottom line

| Model | Expected retrieval lift | Value for research angle | Integration effort | Verdict |
|---|---|---|---|---|
| BGE-M3 | Medium–large (generic passages → strong encoder) | Moderate — changes the baseline but doesn't add a new view | Zero (already the default) | **Do now** |
| SPECTER-2 | Medium on sci corpus | **High** — genuinely different view, enables three-way disagreement | Low–medium | **Do after BGE-M3 swap** |
| S-PubMedBERT | Medium on biomed | Low unless evaluating on HealthVer | Low | Skip for SciFact; revisit for HealthVer |

The biggest research payoff is **SPECTER-2 as a third, citation-semantic retriever**, because it adds a qualitatively different notion of relevance and unlocks three-way KL — which nobody in the related work does. BGE-M3 is cheap and should just happen. S-PubMedBERT only makes sense once we're evaluating outside SciFact.

---

## Empirical update (2026-04-20): SPECTER-2 as a drop-in replacement for MiniLM

Ran λ=0.1 with SPECTER-2 as the dense view (seed 42, identical config otherwise). Result: macro-F1 0.788 vs MiniLM's 0.798 at the same λ. CONTRADICT F1 **regressed** from 0.729 (MiniLM) to 0.693 (SPECTER-2) — back to no-KL-baseline levels.

KL-loss trajectory tells the story: SPECTER-2 produces ~2.5× stronger disagreement with BM25 at its peak (KL peaks at 0.113 vs MiniLM's stable ~0.04), confirming the "genuinely different view" hypothesis. But the disagreement is **the wrong kind of different** for claim verification — SPECTER-2 optimizes for citation-topic similarity, so it retrieves papers *about the topic* rather than passages with *specific counter-evidence*. For CONTRADICT, which needs lexical specificity ("X does NOT cause Y"), SPECTER-2's view systematically misleads the KL term.

**Implication**: pair-wise KL with a single dense retriever may not be improvable by picking a "better" dense encoder — the right move is three-way KL (BM25 + MiniLM + SPECTER-2) where each retriever contributes its own kind of relevance. SPECTER-2 alone degrades the signal; SPECTER-2 alongside MiniLM may enrich it.
