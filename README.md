# Does Bigger Mean Better Everywhere?

## Cross-Domain Sentiment Analysis with TF-IDF and DistilBERT

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

This repository accompanies the paper _Does Bigger Mean Better Everywhere? Cross-Domain Sentiment Analysis with TF-IDF and DistilBERT_.

It compares a classical TF-IDF + Logistic Regression pipeline against fine-tuned DistilBERT for binary sentiment classification. Both models are trained on Sentiment140 tweets and evaluated **zero-shot** on IMDB movie reviews.

**Main result:** DistilBERT has a clear in-domain advantage on Twitter (7.3 accuracy points, p < 0.001), but that gap disappears entirely under domain shift. Cross-domain, the two models reach statistical parity (χ² = 0.042, p = 0.838), and the cheaper classical model is the more defensible choice when retraining is not feasible.

---

## Abstract

A model with a 7-point in-domain accuracy advantage can lose it entirely under domain shift—even against a far simpler baseline. We train TF-IDF + Logistic Regression (TF-IDF+LR) and fine-tuned DistilBERT on Sentiment140 and evaluate both zero-shot on IMDB movie reviews. In-domain, DistilBERT leads by 7.3 accuracy points (85.0% vs. 77.7%, `p < 0.001`, McNemar's test). Cross-domain, DistilBERT degrades 2.3× faster (12.6 vs. 5.4 points) and the two models reach statistical parity (`χ² = 0.042`, `p = 0.838`). The degradation is precision-dominated (DistilBERT precision drop −15.6 vs. recall drop −4.3 points; 3.6× asymmetry), consistent with over-triggering on local positive-affect tokens that do not carry sentiment in long-form text. We recommend the precision-to-recall degradation ratio as a lightweight cross-domain diagnostic, and conclude that in-domain accuracy alone is an insufficient basis for model selection under distribution shift.

---

## Results

**Dataset sizes:** 1.6M Sentiment140 tweets · 160K used for DistilBERT fine-tuning · 1.28M for TF-IDF+LR · 25K IMDB examples for zero-shot evaluation.

>  **Training data asymmetry:** TF-IDF+LR was trained on 1.28M tweets; DistilBERT on 160K due to GPU constraints. This is a known confound — each pipeline is treated as representative of its practical deployment scenario. The cross-domain evaluation is unaffected since both models are evaluated zero-shot on the same 25K IMDB examples.

### Table 1 — Full metrics (bold = best within domain)

| Model       | Domain              | Accuracy  | Precision | Recall    | F1        |
| ----------- | ------------------- | --------- | --------- | --------- | --------- |
| TF-IDF + LR | Twitter (in-domain) | 0.777     | 0.765     | 0.798     | 0.781     |
| TF-IDF + LR | IMDB (cross-domain) | **0.723** | **0.701** | 0.777     | 0.737     |
| DistilBERT  | Twitter (in-domain) | **0.850** | **0.846** | **0.853** | **0.850** |
| DistilBERT  | IMDB (cross-domain) | **0.723** | 0.690     | **0.810** | **0.746** |

In-domain McNemar: χ² = 1113.5, p < 0.001. Cross-domain: χ² = 0.042, p = 0.838.

### Table 2 — Cross-domain degradation Twitter → IMDB

| Metric    | TF-IDF + LR | DistilBERT |
| --------- | ----------- | ---------- |
| Accuracy  | −0.054      | −0.126     |
| Precision | −0.064      | −0.156     |
| Recall    | −0.021      | −0.043     |
| F1        | −0.044      | −0.104     |

DistilBERT degrades 2.3× faster on accuracy and 2.4× on precision, yet both models reach statistical parity on IMDB.

---

## Key Findings

- DistilBERT leads TF-IDF+LR by **7.3 accuracy points** in-domain (p < 0.001).
- Cross-domain, TF-IDF+LR drops 5.4 points; DistilBERT drops **12.6 points (2.3× faster)**.
- The two models reach **statistical parity** on IMDB (χ² = 0.042, p = 0.838) — the 7.3-point advantage is entirely erased.
- DistilBERT's degradation is **precision-dominated** (−15.6 pt precision vs. −4.3 pt recall; 3.6× asymmetry), consistent with over-triggering on local positive-affect tokens in long-form reviews.
- **In-domain accuracy alone is insufficient** for model selection under distribution shift.

### Why precision drops more sharply

On Twitter, positive sentiment is local — a tweet with "love" or "amazing" is almost always positive. DistilBERT learns strong associations between positive-affect tokens and the positive class. On IMDB, the same tokens appear in negative reviews ("I wanted to love this film, but..."). Sentiment is determined by a narrative arc spanning hundreds of tokens — a structure a model trained on 12-token tweets is not equipped to capture.

TF-IDF+LR is less susceptible because term frequency is diluted by document length: "love" in a 300-word review contributes far less weight than in a 10-word tweet, producing more symmetric degradation.

---

## Experimental Setup

| | TF-IDF + LR | DistilBERT |
|---|---|---|
| Preprocessing | Non-alphabetic removal, lowercasing, stopword removal, Porter stemming | Raw tweet text, WordPiece tokenization |
| Training data | 1,280,000 tweets (80/20 split, seed 42) | 160,000 tweets (GPU constraint) |
| Model | Logistic Regression (C=1.0, L2) | distilbert-base-uncased, lr=2×10⁻⁵, 3 epochs, batch 32, FP16 |
| In-domain test | 320,000 examples | 40,000 examples |
| Cross-domain test | 25,000 IMDB examples (zero-shot) | 25,000 IMDB examples (zero-shot) |

McNemar's test uses the continuity-corrected form. In-domain McNemar is restricted to the shared 40K subset for fair pairing.

---

## Project Structure

```
sentiment_analysis/
├── notebooks/
│   └── sentiment_analysis.ipynb
├── requirements.txt
├── CITING.bib
├── Contributing.md
└── LICENSE
```

---

## How to Run

### Google Colab

1. Open the notebook in Google Colab.
2. Upload `notebooks/sentiment_analysis.ipynb`.
3. Switch the runtime to a **T4 GPU**.
4. Run the notebook top to bottom.

### Local

1. Create a Python 3.10 environment.
2. Install PyTorch for your CUDA version.
3. Install remaining dependencies: `pip install -r requirements.txt`
4. Open `notebooks/sentiment_analysis.ipynb` in Jupyter and run it.

### CPU only

DistilBERT training on CPU is very slow. If you only need the TF-IDF pipeline, run the preprocessing and classical modeling sections of the notebook only.

---

## Datasets

- **Sentiment140** — 1.6M tweets with binary labels derived from emoticons (Go et al., 2009).
- **IMDB Movie Reviews** — 50K human-annotated reviews; fixed, deterministic 25K balanced subset of the standard test split used for zero-shot evaluation (Maas et al., 2011).

---

## References

1. Ben-David, S. et al. (2010). A theory of learning from different domains. _Machine Learning 79(1)_.
2. Blitzer, J. et al. (2007). Biographies, Bollywood, boom-boxes and blenders: Domain adaptation for sentiment classification. _ACL 2007_.
3. Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. _NAACL 2019_.
4. Dror, R. et al. (2018). The hitchhiker's guide to testing statistical significance in NLP. _ACL 2018_.
5. Go, A., Bhayani, R., & Huang, L. (2009). Twitter Sentiment Classification using Distant Supervision. _CS224N Stanford_.
6. Gururangan, S. et al. (2020). Don't Stop Pretraining: Adapt Language Models to Domains and Tasks. _ACL 2020_.
7. Liu, Y. et al. (2019). RoBERTa: A robustly optimized BERT pretraining approach. _arXiv:1907.11692_.
8. Maas, A. et al. (2011). Learning Word Vectors for Sentiment Analysis. _ACL 2011_.
9. Manning, C.D., Raghavan, P., Schütze, H. (2008). Introduction to Information Retrieval. Cambridge University Press.
10. McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. _Psychometrika 12(2)_.
11. Nguyen, D.Q. et al. (2020). BERTweet: A pre-trained language model for English tweets. _EMNLP 2020_.
12. Pan, S.J. & Yang, Q. (2010). A survey on transfer learning. _IEEE TKDE 22(10)_.
13. Porter, M.F. (1980). An algorithm for suffix stripping. _Program 14(3)_.
14. Ramponi, A. & Plank, B. (2020). Neural unsupervised domain adaptation in NLP — A survey. _COLING 2020_.
15. Ribeiro, M.T. et al. (2016). "Why should I trust you?": Explaining the predictions of any classifier. _KDD 2016_.
16. Sanh, V. et al. (2019). DistilBERT, a distilled version of BERT. _EMC2 @ NeurIPS 2019_.
17. Sundararajan, M. et al. (2017). Axiomatic attribution for deep networks. _ICML 2017_.

---

## Cite This Work

```bibtex
@misc{anonymous2026crossdomain,
  title        = {Does Bigger Mean Better Everywhere? Cross-Domain Sentiment Analysis with TF-IDF and DistilBERT},
  year         = {2026},
  howpublished = {\url{https://anonymous.4open.science/r/sentiment_analysis-DBC8}},
  note         = {Cross-domain sentiment analysis with TF-IDF + Logistic Regression and fine-tuned DistilBERT.}
}
```

A machine-readable copy is available in `CITING.bib`.

---

## License

MIT License
