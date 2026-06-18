# Contributing

Thanks for your interest in contributing. This is a research project so contributions that improve reproducibility, extend the experiments, or fix bugs are especially welcome.

## Before you start

Open an issue first if you're planning something substantial — it avoids duplicate work and lets us discuss the approach before you write code.

For small fixes (typos, doc improvements, bug fixes) just open a PR directly.

## Setup

```bash
git clone https://github.com/<your-org-or-username>/sentiment_analysis.git
cd sentiment_analysis
conda create -n sentiment_analysis python=3.10
conda activate sentiment_analysis
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.41.2 accelerate==0.30.0
pip install -r requirements.txt
```

## How to contribute

```bash
# 1. Fork the repo on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/sentiment_analysis.git
# 2. Create a branch
git checkout -b feature/your-feature-name
# 3. Make your changes
# 4. Commit
git commit -m "add: short description of what you did"
# 5. Push and open a pull request against main
git push origin feature/your-feature-name
```

## Commit message format

Keep it short and use a prefix:
- `add:` new feature or experiment
- `fix:` bug fix
- `update:` change to existing feature
- `docs:` documentation only
- `refactor:` code change with no behavior change

Examples:
```
add: SST-2 cross-domain evaluation
fix: path error on Windows for data/raw/
update: BERT training size to 160K for paper mode
docs: clarify Colab setup steps in README
```

## Good first contributions

These are well-scoped tasks that don't require deep knowledge of the full codebase:

- **Multi-seed variance** — re-run DistilBERT fine-tuning with seeds 1, 2, 3 (keeping `random_state=42` fixed in `train_test_split` so the data slice is identical across runs), and open a PR reporting mean ± std for the cross-domain metrics
- **Run the preprocessing ablation** — set `RUN_ABLATION = True` in section 3.7, report the raw vs stemmed F1 comparison
- **Add a third dataset** — SST-2, Yelp Reviews, or Amazon Reviews as an additional cross-domain test
- **Domain-adaptive pre-training** — add DistilBERT continued pre-training on unlabeled IMDB text before fine-tuning (Gururangan et al. 2020)
- **scripts/download_data.py** — separate the Sentiment140 download from the notebook into a standalone script
- **Reverse transfer direction** — evaluate both models trained on IMDB and tested zero-shot on Twitter

## Questions

Open an issue in the repository to discuss larger changes.

## Citing this work

```bibtex
@misc{anonymous2026crossdomain,
  title        = {Does Bigger Mean Better Everywhere? Cross-Domain Sentiment Analysis with TF-IDF and DistilBERT},
  year         = {2026},
  howpublished = {\url{https://anonymous.4open.science/r/sentiment_analysis-DBC8}},
  note         = {Cross-domain sentiment analysis with TF-IDF + Logistic Regression and fine-tuned DistilBERT on Sentiment140 and IMDB.}
}
```