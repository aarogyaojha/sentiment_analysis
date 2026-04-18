# Cross-Domain Sentiment Analysis: TF-IDF vs DistilBERT

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/[USERNAME]/sentiment_analysis/blob/main/LICENSE)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)
[![GitHub](https://img.shields.io/badge/GitHub-[USERNAME]-black?logo=github)](https://github.com/[USERNAME]/sentiment_analysis)

Comparing a classical TF-IDF + Logistic Regression pipeline against fine-tuned DistilBERT on binary sentiment classification — trained on Twitter (Sentiment140) and evaluated both in-domain and out-of-domain on IMDB movie reviews without any retraining.

The main finding: DistilBERT outperforms TF-IDF+LR by 7.1 points on Twitter, and the gap is statistically significant (McNemar χ² = 1123.2, p < 0.001). But DistilBERT degrades approximately 2.3× faster under domain shift. On IMDB, the two models reach statistical parity (χ² = 1.526, p = 0.217) — a 7-point in-domain advantage vanishes entirely.

---

## Results

> Full dataset run: 1.6M Sentiment140 tweets, 160K DistilBERT training examples, 25K IMDB evaluation examples. Verified on Google Colab T4 GPU — total runtime ~45–60 min.
>
> For dev-mode numbers (200K / 50K, ~13 min), see notebook section 2.2b.

| Model | Test domain | Accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| TF-IDF + LR | Twitter (in-domain) | 0.777 | 0.765 | 0.798 | 0.781 |
| TF-IDF + LR | IMDB (cross-domain) | 0.723 | 0.701 | 0.777 | 0.737 |
| DistilBERT | Twitter (in-domain) | 0.848 | 0.841 | 0.860 | 0.848 |
| DistilBERT | IMDB (cross-domain) | 0.727 | 0.704 | 0.782 | 0.741 |

> **Note on test set sizes:** TF-IDF+LR in-domain evaluation uses 320,000 Twitter examples; DistilBERT in-domain uses 40,000 examples drawn from the same split (limited by inference cost). Both models use all 25,000 IMDB test examples for cross-domain evaluation and McNemar's test.

**Cross-domain degradation (Twitter → IMDB):**

| Metric | TF-IDF+LR drop | DistilBERT drop |
| --- | --- | --- |
| Accuracy | +0.054 | +0.123 |
| Precision | +0.064 | +0.137 |
| Recall | +0.021 | +0.078 |
| F1 | +0.044 | +0.107 |

---

## Project structure

```
sentiment_analysis/
│
├── data/
│   ├── raw/
│   │   ├── sentiment140/       # Downloaded automatically (~80MB, gitignored)
│   │   └── imdb/               # Cached by HuggingFace datasets (gitignored)
│   └── processed/              # Intermediate files (generated, gitignored)
│
├── notebooks/
│   └── sentiment_analysis.ipynb   # Main notebook — runs end to end
│
├── outputs/
│   ├── figures/                # fig1_model_comparison.png
│   │                           # fig2_confusion_matrices.png
│   │                           # fig3_precision_recall_breakdown.png
│   ├── results/                # results_table.csv
│   │                           # bert_wins_examples.csv
│   │                           # lr_wins_examples.csv
│   └── models/                 # Saved model weights (gitignored — large files)
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py           # preprocess_tweet(), build_tfidf()
│   └── evaluate.py             # compute_metrics(), mcnemar_test(), degradation_table()
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## How to run

### Option 1 — Google Colab (recommended)

The fastest way. No local setup, no dependency issues.

**Step 1:** Go to [colab.research.google.com](https://colab.research.google.com)

**Step 2:** File → Upload notebook → select `notebooks/sentiment_analysis.ipynb`

**Step 3:** Runtime → Change runtime type → **T4 GPU** → Save

**Step 4:** Add this cell at the very top of the notebook and run it first:

```python
import os

os.makedirs("/content/data/raw/sentiment140", exist_ok=True)
os.makedirs("/content/data/raw/imdb", exist_ok=True)
os.makedirs("/content/data/processed", exist_ok=True)
os.makedirs("/content/outputs/figures", exist_ok=True)
os.makedirs("/content/outputs/results", exist_ok=True)
os.makedirs("/content/outputs/models", exist_ok=True)

import subprocess
subprocess.run([
    "sed", "-i",
    "s|../data/|/content/data/|g;s|../outputs/|/content/outputs/|g",
    "/content/sentiment_analysis.ipynb"
])
print("Paths fixed")
```

**Step 5:** Runtime → Run all

Both datasets download automatically. No `pip install` needed — Colab has all dependencies pre-installed.

| Step | Dev mode (200K / 50K) | Paper mode (1.6M / 160K) |
| --- | --- | --- |
| Sentiment140 download + load | ~1 min | ~3 min |
| TF-IDF preprocessing | ~1 min | ~4 min |
| TF-IDF training | ~30 sec | ~30 sec |
| DistilBERT fine-tuning | ~9 min | ~30–40 min |
| IMDB cross-domain evaluation | ~1 min | ~5 min |
| Figures + analysis | ~30 sec | ~30 sec |
| **Total** | **~13 min** | **~45–60 min** |

---

### Option 2 — Local with GPU (conda)

Tested on Windows 11, NVIDIA RTX 3050 Laptop GPU.

```bash
# 1. Create environment
conda create -n sentiment_analysis python=3.10
conda activate sentiment_analysis

# 2. Check your CUDA version
nvidia-smi   # note the CUDA version in the top right

# 3. Install PyTorch with CUDA — match your version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# For CUDA 11.8 use: --index-url https://download.pytorch.org/whl/cu118

# 4. Verify GPU is detected
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Should print: True  NVIDIA GeForce ...

# 5. Install remaining dependencies in this exact order
pip install transformers==4.40.0
pip install accelerate==0.30.0
pip install -r requirements.txt

# 6. Register kernel so Jupyter sees the environment
python -m ipykernel install --user --name sentiment_analysis --display-name "sentiment_analysis"

# 7. Launch
jupyter notebook notebooks/sentiment_analysis.ipynb
```

In Jupyter, check the kernel shows **sentiment_analysis** in the top right. If not: Kernel → Change kernel → sentiment_analysis. Then Cell → Run All.

> **Note on Windows:** if you hit `ImportError: Using the Trainer with PyTorch requires accelerate>=1.1.0` even after installing accelerate, run:
>
> ```bash
> pip install transformers[torch] --upgrade --force-reinstall
> ```
>
> Then **fully close Jupyter and reopen it** — restarting the kernel alone does not pick up the reinstalled packages.

---

### Option 3 — Local CPU only

DistilBERT training on CPU takes 3–6 hours. For TF-IDF results only:

```bash
conda create -n sentiment_analysis python=3.10
conda activate sentiment_analysis
pip install -r requirements.txt
python -m ipykernel install --user --name sentiment_analysis --display-name "sentiment_analysis"
jupyter notebook notebooks/sentiment_analysis.ipynb
```

Run Sections 1, 2, and 4 only (skip Section 3 — DistilBERT fine-tuning). You still get the full TF-IDF baseline and cross-domain IMDB results.

For a quick CPU end-to-end test, in cell **3.1** change:

```python
BERT_TRAIN_SIZE = 2_000
BERT_TEST_SIZE  = 500
```

Completes in ~20 min but results are not representative.

---

## Running the full dataset

The notebook defaults to `FULL_DATASET = True` and `BERT_TRAIN_SIZE = 160_000` — these are the paper-quality settings. The results table above reflects this run.

To switch to dev mode for fast iteration:

1. In cell **2.2**, set `FULL_DATASET = False`
2. In cell **3.1**, set `BERT_TRAIN_SIZE = 50_000`
3. Re-run from cell 2.2 onwards

| Setting | TF-IDF+LR accuracy | DistilBERT accuracy | Runtime (T4) |
| --- | --- | --- | --- |
| Dev (200K / 50K) | ~76.7% | ~83.4% | ~13 min |
| Paper (1.6M / 160K) | ~77.7% | ~84.8% | ~45–60 min |

---

## Reproducing specific results

| Result | Notebook section |
| --- | --- |
| TF-IDF+LR Twitter metrics | 2.5 |
| DistilBERT Twitter metrics | 3.6 |
| Preprocessing ablation (raw vs stemmed BERT) | 3.7 |
| TF-IDF+LR IMDB cross-domain | 4.2 |
| DistilBERT IMDB cross-domain | 4.3 |
| Results table (Table 1) | 5.1 |
| Figure 1: accuracy/F1 bar chart | 5.2 |
| Figure 2: confusion matrices | 5.3 |
| Domain degradation gap | 6.1 |
| Figure 3: precision/recall breakdown | 6.1b |
| McNemar's test — Twitter | 6.2b |
| McNemar's test — IMDB | 6.2b |
| Qualitative error examples | 6.3 |

---

## Key findings

**In-domain (Twitter):** DistilBERT achieves 84.8% accuracy vs 77.7% for TF-IDF+LR — a 7.1-point gap. Statistically significant (McNemar χ² = 1123.2, p < 0.001).

**Cross-domain (IMDB):** TF-IDF+LR drops 5.4 accuracy points moving to IMDB. DistilBERT drops 12.3 points — approximately 2.3× faster degradation. Despite this, the two models reach near-identical IMDB accuracy. The difference is **not statistically significant** (McNemar χ² = 1.526, p = 0.217) — a 7-point in-domain advantage is entirely absorbed by domain shift.

**The precision/recall asymmetry:** DistilBERT's precision drops 13.7 points on IMDB while recall drops only 7.8 points (a 1.8× ratio). The model keeps finding positive reviews (recall roughly intact) but over-triggers on positive-sentiment words that appear in the setup of negative reviews ("I wanted to love this film..."). TF-IDF+LR degrades more symmetrically because global word-count features are less sensitive to local trigger words.

**Practical implication:** In-domain accuracy alone is an unreliable basis for model selection when distribution shift is expected. The cheaper, GPU-free TF-IDF+LR pipeline is statistically indistinguishable from DistilBERT under domain shift.

---

## Datasets

### Sentiment140

* 1.6 million tweets, binary sentiment labels via emoticon distant supervision
* Go, Bhayani & Huang (2009)
* Downloaded automatically from Stanford NLP: `cs.stanford.edu/people/alecmgo/trainingandtestdata.zip`
* Saved to `data/raw/sentiment140/` (~80MB, gitignored)

### IMDB Movie Reviews

* 50,000 reviews, human-annotated, balanced classes (25K train / 25K test)
* Maas et al. (2011)
* Loaded via HuggingFace `datasets` library, cached to `data/raw/imdb/`

---

## References

* Go, A., Bhayani, R., & Huang, L. (2009). *Twitter Sentiment Classification using Distant Supervision.* Stanford Technical Report.
* Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL.
* Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). *DistilBERT, a distilled version of BERT.* NeurIPS Workshop on EMC².
* Maas, A. et al. (2011). *Learning Word Vectors for Sentiment Analysis.* ACL.
* Blitzer, J., McDonald, R., & Pereira, F. (2006). *Domain Adaptation with Structural Correspondence Learning.* EMNLP.
* Gururangan, S. et al. (2020). *Don't Stop Pretraining: Adapt Language Models to Domains and Tasks.* ACL.
* McNemar, Q. (1947). *Note on the Sampling Error of the Difference between Correlated Proportions or Percentages.* Psychometrika.

---

## Contributing

Contributions are welcome. Please open an issue before starting large changes.

1. Fork — [github.com/[USERNAME]/sentiment_analysis](https://github.com/[USERNAME]/sentiment_analysis)
2. Create a branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m "add: description of change"`
4. Push and open a pull request

**Good first contributions:**

* Run the preprocessing ablation (`RUN_ABLATION = True` in section 3.7) and report numbers
* Extend cross-domain evaluation to SST-2, Yelp, or Amazon Reviews
* Add domain-adaptive pre-training as a third model condition (Gururangan et al. 2020)
* Add token-level attribution analysis (integrated gradients or LIME) to validate the precision-collapse hypothesis
* Add a `scripts/download_data.py` to separate data download from the notebook

---

## Cite this work

If you use this code or findings in your research, please cite:

```bibtex
@misc{[CITATION_KEY],
  author       = {[AUTHOR]},
  title        = {Cross-Domain Sentiment Analysis: {TF-IDF} vs {DistilBERT}},
  year         = {2025},
  howpublished = {\url{https://github.com/[USERNAME]/sentiment_analysis}},
  note         = {Compares a classical TF-IDF + Logistic Regression pipeline against
                  fine-tuned DistilBERT on binary sentiment classification,
                  trained on Sentiment140 (Twitter) and evaluated cross-domain
                  on IMDB movie reviews without retraining.}
}
```

A machine-readable copy is also available in [`CITING.bib`](https://github.com/[USERNAME]/sentiment_analysis/blob/main/CITING.bib).

---

## License

MIT License — see [LICENSE](https://github.com/[USERNAME]/sentiment_analysis/blob/main/LICENSE) for full text.

Copyright (c) 2025 [AUTHOR]