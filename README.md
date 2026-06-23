# Does Bigger Mean Better Everywhere?

## Cross-Domain Sentiment Analysis with TF-IDF and DistilBERT

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

This repository accompanies the paper _Does Bigger Mean Better Everywhere? Cross-Domain Sentiment Analysis with TF-IDF and DistilBERT_.

It compares a classical TF-IDF + Logistic Regression pipeline against fine-tuned DistilBERT for binary sentiment classification. Both models are trained on Sentiment140 tweets and evaluated zero-shot on IMDB movie reviews.

**Main result:** DistilBERT has a clear in-domain advantage on Twitter, but that gap disappears under domain shift. Cross-domain, TF-IDF + Logistic Regression is statistically indistinguishable from DistilBERT, and the cheaper classical model is the more defensible choice when retraining is not feasible.

---

## Abstract

A model with a 7-point in-domain accuracy advantage can lose it entirely under domain shift, even against a far simpler baseline. We train TF-IDF + Logistic Regression and fine-tuned DistilBERT on Sentiment140 and evaluate both zero-shot on IMDB movie reviews. In-domain, DistilBERT leads by 7.3 accuracy points (85.0% vs. 77.7%, `p < 0.001`, McNemar's test). Cross-domain, DistilBERT degrades 2.3x faster (12.6 vs. 5.4 points) and the two models reach statistical parity (`chi^2 = 0.042`, `p = 0.838`). The degradation is precision-dominated (DistilBERT precision drop −15.6 vs. recall drop −4.3 points; 3.6× asymmetry), consistent with over-triggering on local positive-affect tokens that do not carry sentiment in long-form text. We recommend the precision-to-recall degradation ratio as a lightweight cross-domain diagnostic, and conclude that in-domain accuracy alone is an insufficient basis for model selection under distribution shift.

---

## Results

Full dataset: 1.6M Sentiment140 tweets, 160K DistilBERT training examples, and 25K IMDB evaluation examples.

| Model       | Test domain         | Accuracy | Precision | Recall | F1    |
| ----------- | ------------------- | -------- | --------- | ------ | ----- |
| TF-IDF + LR | Twitter (in-domain) | 0.777    | 0.765     | 0.798  | 0.781 |
| TF-IDF + LR | IMDB (cross-domain) | 0.723    | 0.701     | 0.777  | 0.737 |
| DistilBERT  | Twitter (in-domain) | 0.850    | 0.846     | 0.853  | 0.850 |
| DistilBERT  | IMDB (cross-domain) | 0.723    | 0.690     | 0.810  | 0.746 |

**Cross-domain degradation from Twitter to IMDB**

| Metric    | TF-IDF + LR drop | DistilBERT drop |
| --------- | ---------------- | --------------- |
| Accuracy  | −0.054           | −0.126          |
| Precision | −0.064           | −0.156          |
| Recall    | −0.021           | −0.043          |
| F1        | −0.044           | −0.104          |

---

## Key Findings

- DistilBERT leads TF-IDF + LR by 7.3 accuracy points in-domain (p < 0.001).
- Cross-domain, TF-IDF + LR drops 5.4 points while DistilBERT drops 12.6 points (2.3× faster).
- The two models reach statistical parity on IMDB (χ² = 0.042, p = 0.838).
- DistilBERT's degradation is precision-dominated (−15.6 pt precision vs. −4.3 pt recall; 3.6× asymmetry), suggesting over-triggering on isolated positive-affect words in long-form reviews.
- In-domain accuracy alone is not enough to choose a model when distribution shift is expected.

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
3. Switch the runtime to a T4 GPU.
4. Run the notebook top to bottom.

### Local

1. Create a Python 3.10 environment.
2. Install PyTorch for your CUDA version.
3. Install the remaining dependencies from `requirements.txt`.
4. Open `notebooks/sentiment_analysis.ipynb` in Jupyter and run it.

### CPU only

DistilBERT training on CPU is slow. If you only need the TF-IDF pipeline, run the notebook sections for preprocessing and classical modeling.

---

## Datasets

- **Sentiment140**: 1.6M tweets with binary labels derived from emoticons.
- **IMDB Movie Reviews**: 50K reviews, with the standard 25K test split used for evaluation.

---

## References

1. Go, A., Bhayani, R., & Huang, L. (2009). _Twitter Sentiment Classification using Distant Supervision_.
2. Devlin, J. et al. (2019). _BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding_.
3. Sanh, V. et al. (2019). _DistilBERT, a distilled version of BERT_.
4. Maas, A. et al. (2011). _Learning Word Vectors for Sentiment Analysis_.
5. Gururangan, S. et al. (2020). _Don't Stop Pretraining: Adapt Language Models to Domains and Tasks_.
6. McNemar, Q. (1947). _Note on the Sampling Error of the Difference between Correlated Proportions or Percentages_.

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
