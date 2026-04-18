# Cross-Domain Sentiment Analysis: TF-IDF vs DistilBERT

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Open in Colab](https://img.shields.io/badge/Open%20in-Colab-orange?style=flat-square)](https://colab.research.google.com)

Benchmarks a classical TF-IDF + Logistic Regression pipeline against fine-tuned DistilBERT on binary sentiment classification. Both models are trained on Twitter (Sentiment140) and evaluated in-domain on Twitter and out-of-domain on IMDB movie reviews — without any retraining on the target domain.

**Main finding:** DistilBERT outperforms TF-IDF+LR by 7.1 points in-domain (McNemar χ² = 1123.2, p < 0.001), but degrades 2.3× faster under domain shift. On IMDB, the gap disappears entirely — the two models reach statistical parity (χ² = 1.526, p = 0.217). In-domain accuracy alone is an unreliable basis for model selection when distribution shift is expected.

---

## Results

Full dataset: 1.6M Sentiment140 tweets, 160K DistilBERT training examples, 25K IMDB evaluation examples. Verified on Google Colab T4 GPU (~45–60 min).

| Model | Test domain | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| TF-IDF + LR | Twitter (in-domain) | 0.777 | 0.765 | 0.798 | 0.781 |
| TF-IDF + LR | IMDB (cross-domain) | 0.723 | 0.701 | 0.777 | 0.737 |
| DistilBERT | Twitter (in-domain) | 0.848 | 0.841 | 0.860 | 0.848 |
| DistilBERT | IMDB (cross-domain) | 0.727 | 0.704 | 0.782 | 0.741 |

**Cross-domain degradation (Twitter → IMDB):**

| Metric | TF-IDF+LR drop | DistilBERT drop |
|---|---|---|
| Accuracy | 0.054 | 0.123 |
| Precision | 0.064 | 0.137 |
| Recall | 0.021 | 0.078 |
| F1 | 0.044 | 0.107 |

---

## Key findings

**In-domain:** DistilBERT achieves 84.8% vs 77.7% for TF-IDF+LR — a statistically significant 7.1-point gap (McNemar χ² = 1123.2, p < 0.001).

**Cross-domain:** TF-IDF+LR drops 5.4 accuracy points on IMDB. DistilBERT drops 12.3 points — ~2.3× faster degradation. Despite this, both models reach near-identical IMDB accuracy; the difference is not statistically significant (χ² = 1.526, p = 0.217).

**Precision/recall asymmetry:** DistilBERT's precision drops 13.7 points on IMDB while recall drops only 7.8 (a 1.8× ratio). The model keeps finding positive reviews but over-triggers on positive-sentiment words appearing in the setup of negative reviews ("I wanted to love this film..."). TF-IDF+LR degrades more symmetrically — global word-count features are less sensitive to local trigger words.

**Practical implication:** The cheaper, GPU-free TF-IDF+LR pipeline is statistically indistinguishable from DistilBERT under domain shift. For cross-domain deployment, model selection based on in-domain benchmark accuracy is misleading.

---

## Project structure

```
sentiment_analysis/
├── notebooks/
│   └── sentiment_analysis.ipynb   # Main notebook — runs end to end
├── src/
│   ├── preprocess.py              # preprocess_tweet(), build_tfidf()
│   └── evaluate.py                # compute_metrics(), mcnemar_test(), degradation_table()
├── outputs/
│   ├── figures/                   # fig1_model_comparison.png, fig2_confusion_matrices.png, fig3_precision_recall.png
│   └── results/                   # results_table.csv, bert_wins_examples.csv, lr_wins_examples.csv
├── requirements.txt
└── CITING.bib
```

---

## How to run

### Option 1 — Google Colab (recommended)

Fastest path. No local setup, no dependency management.

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. File → Upload notebook → select `notebooks/sentiment_analysis.ipynb`
3. Runtime → Change runtime type → T4 GPU → Save
4. Add this cell at the top and run it first:

```python
import os, subprocess

for path in [
    "/content/data/raw/sentiment140", "/content/data/raw/imdb",
    "/content/data/processed", "/content/outputs/figures",
    "/content/outputs/results", "/content/outputs/models"
]:
    os.makedirs(path, exist_ok=True)

subprocess.run([
    "sed", "-i",
    "s|../data/|/content/data/|g;s|../outputs/|/content/outputs/|g",
    "/content/sentiment_analysis.ipynb"
])
print("Paths fixed")
```

5. Runtime → Run all. Both datasets download automatically.

| Step | Dev mode (~13 min) | Paper mode (~45–60 min) |
|---|---|---|
| Sentiment140 download | ~1 min | ~3 min |
| TF-IDF preprocessing | ~1 min | ~4 min |
| TF-IDF training | ~30 sec | ~30 sec |
| DistilBERT fine-tuning | ~9 min | ~30–40 min |
| IMDB evaluation + figures | ~1.5 min | ~5.5 min |

### Option 2 — Local with GPU

Tested on Windows 11, NVIDIA RTX 3050 Laptop.

```bash
conda create -n sentiment_analysis python=3.10
conda activate sentiment_analysis

# Install PyTorch — match your CUDA version (check with nvidia-smi)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify GPU
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Install remaining dependencies in this order
pip install transformers==4.40.0
pip install accelerate==0.30.0
pip install -r requirements.txt

python -m ipykernel install --user --name sentiment_analysis --display-name "sentiment_analysis"
jupyter notebook notebooks/sentiment_analysis.ipynb
```

> **Windows note:** if you hit `ImportError: accelerate>=1.1.0 required`, run `pip install transformers[torch] --upgrade --force-reinstall` then fully close and reopen Jupyter (kernel restart alone won't pick up the reinstalled packages).

### Option 3 — Local CPU only

DistilBERT training on CPU takes 3–6 hours. For TF-IDF results only, run sections 1, 2, and 4 (skip section 3).

For a quick end-to-end CPU test (~20 min, non-representative results):
```python
# In cell 3.1
BERT_TRAIN_SIZE = 2_000
BERT_TEST_SIZE  = 500
```

---

## Dev mode vs paper mode

The notebook defaults to `FULL_DATASET = True` and `BERT_TRAIN_SIZE = 160_000` (paper-quality settings). To switch to dev mode:

- In cell 2.2: set `FULL_DATASET = False`
- In cell 3.1: set `BERT_TRAIN_SIZE = 50_000`
- Re-run from cell 2.2

| Setting | TF-IDF+LR accuracy | DistilBERT accuracy | Runtime (T4) |
|---|---|---|---|
| Dev (200K / 50K) | ~76.7% | ~83.4% | ~13 min |
| Paper (1.6M / 160K) | ~77.7% | ~84.8% | ~45–60 min |

---

## Reproducing specific results

| Result | Notebook section |
|---|---|
| TF-IDF+LR Twitter metrics | 2.5 |
| DistilBERT Twitter metrics | 3.6 |
| Preprocessing ablation (raw vs stemmed BERT) | 3.7 |
| TF-IDF+LR IMDB cross-domain | 4.2 |
| DistilBERT IMDB cross-domain | 4.3 |
| Results table (Table 1) | 5.1 |
| Domain degradation gap | 6.1 |
| McNemar's test | 6.2b |
| Qualitative error examples | 6.3 |

---

## Datasets

**Sentiment140** — 1.6M tweets, binary labels via emoticon distant supervision. Go, Bhayani & Huang (2009). Downloaded automatically from Stanford NLP (~80MB, gitignored).

**IMDB Movie Reviews** — 50K reviews, human-annotated, balanced classes. Maas et al. (2011). Loaded via HuggingFace `datasets`, cached to `data/raw/imdb/`.

---

## References

1. Go, A., Bhayani, R., & Huang, L. (2009). Twitter Sentiment Classification using Distant Supervision. Stanford Technical Report.
2. Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL.
3. Sanh, V. et al. (2019). DistilBERT, a distilled version of BERT. NeurIPS Workshop on EMC².
4. Maas, A. et al. (2011). Learning Word Vectors for Sentiment Analysis. ACL.
5. Gururangan, S. et al. (2020). Don't Stop Pretraining: Adapt Language Models to Domains and Tasks. ACL.
6. McNemar, Q. (1947). Note on the Sampling Error of the Difference between Correlated Proportions or Percentages. Psychometrika.

---

## Contributing

Contributions welcome. Please open an issue before starting large changes.

Good first contributions:
- Run the preprocessing ablation (`RUN_ABLATION = True` in section 3.7) and report numbers
- Extend cross-domain evaluation to SST-2, Yelp, or Amazon Reviews
- Add domain-adaptive pre-training as a third condition (Gururangan et al. 2020)
- Add token-level attribution (integrated gradients or LIME) to validate the precision-collapse hypothesis

---

## Cite this work

```bibtex
@misc{ojha2025crossdomain,
  author       = {Ojha, Aarogya},
  title        = {Cross-Domain Sentiment Analysis: {TF-IDF} vs {DistilBERT}},
  year         = {2025},
  howpublished = {\url{https://github.com/aarogyaojha/sentiment_analysis}},
  note         = {Compares TF-IDF + Logistic Regression against fine-tuned DistilBERT
                  trained on Sentiment140 and evaluated cross-domain on IMDB.}
}
```

A machine-readable copy is in `CITING.bib`.

---

## License

MIT © Aarogya Ojha
