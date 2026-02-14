# Native Language Identification on Chinese Learner Texts

NLI system for identifying the native language (L1) of Chinese-as-a-second-language writers using the JCLCv2 corpus. Compares three approaches: traditional ML with hand-crafted features, frozen-encoder probes, and fine-tuned transformers.

## Dataset

The [JCLCv2 corpus](https://github.com/JCLCv2) contains 8,739 Chinese learner essays annotated with the writer's native language (52 L1s), writing context, and gender. The class distribution is highly imbalanced (Indonesian: 3,381 vs. Nigerian: 7).

Place the corpus under `JCLCv2/` so that `JCLCv2/index.csv` and the text files exist.

## Setup

```bash
# Install uv if needed: https://docs.astral.sh/uv/
uv sync
```

Or with pip:

```bash
pip install torch transformers scikit-learn jieba pandas matplotlib seaborn numpy scipy tqdm
# Optional: spacy (for dependency features)
pip install spacy && python -m spacy download zh_core_web_sm
```

## Project Structure

```
NNP/
├── config.py                  # Paths, hyperparameters, model list
├── data_loader.py             # Load corpus, stratified train/val/test splits
├── evaluate.py                # Metrics, confusion matrices, classification reports
├── analysis.py                # EDA: distribution plots, text stats, LaTeX/TikZ export
├── features/
│   ├── __init__.py            # build_features() — combines all feature groups
│   ├── ngrams.py              # Character & word n-gram TF-IDF
│   ├── pos_tags.py            # POS tag distributions (jieba)
│   ├── pos_ngrams.py          # POS tag bigram/trigram patterns
│   ├── function_words.py      # Function word frequencies
│   ├── particles.py           # Grammatical particle context features
│   ├── discourse.py           # Discourse connective frequencies
│   ├── lexical_richness.py    # TTR, hapax ratio, Yule's K, etc.
│   ├── segmentation.py        # Word length distribution, OOV ratio
│   ├── radicals.py            # Kangxi radical usage from Unicode Unihan
│   └── dependency.py          # Dependency parse features (requires spaCy)
├── models/
│   ├── svm.py                 # LogReg, SGD, LinearSVC, MLP classifiers
│   ├── neural.py              # BiLSTM and TextCNN
│   └── transformer.py         # Transformer fine-tuning wrapper
├── train_baselines.py         # Majority, stratified random, uniform random
├── train_svm.py               # Train classifiers on hand-crafted features
├── train_neural.py            # Train BiLSTM / TextCNN
├── train_transformer.py       # Fine-tune transformer encoders
├── train_probe.py             # Frozen embeddings + MLP/LogReg probe (single model)
├── train_probe_all.py         # Run probes for all encoder models
├── notebooks/
│   ├── train_svm.ipynb        # Colab: traditional ML pipeline
│   ├── finetune_transformer.ipynb  # Colab: fine-tune with HF Trainer
│   └── probe_all.ipynb        # Colab: frozen-encoder probes
├── jobs/
│   ├── train_svm.sh           # SLURM job script
│   ├── train_neural.sh        # SLURM job script
│   ├── train_transformer.sh   # SLURM array job (one model per task)
│   └── train_probe.sh         # SLURM array job
└── results/                   # Output: CSVs, plots, LaTeX tables
```

## Usage

### Exploratory Data Analysis

```bash
python analysis.py
```

Generates distribution plots, text length histograms, and exports LaTeX tables and TikZ charts to `results/`.

### Baselines

```bash
python train_baselines.py
```

### Traditional ML (Hand-Crafted Features)

```bash
# Default: Logistic Regression
python train_svm.py

# Choose classifier: logreg | sgd | svm | mlp
python train_svm.py --model mlp

# Enable optional feature groups
python train_svm.py --radicals          # Kangxi radical features
python train_svm.py --dep               # Dependency parse features (needs spaCy)
python train_svm.py --gridsearch        # Grid search over C values for LinearSVC
```

Before using radical features, generate the mapping:

```bash
python -m features.radicals
```

### Frozen-Encoder Probes

Extract embeddings from a frozen transformer and train a classifier on top:

```bash
# Single model with MLP probe (default)
python train_probe.py --model google-bert/bert-base-chinese

# Switch to Logistic Regression
python train_probe.py --model hfl/chinese-roberta-wwm-ext --clf logreg

# Lower batch size for large models
python train_probe.py --model hfl/chinese-roberta-wwm-ext-large --batch-size 4

# Run all 20 encoder models
python train_probe_all.py
```

### Transformer Fine-Tuning

```bash
python train_transformer.py --model google-bert/bert-base-chinese
```

### Google Colab

For running on Colab when a GPU cluster is unavailable, see the notebooks in `notebooks/`. Each notebook mounts Google Drive, extracts the data locally for fast I/O, and runs the full pipeline.

1. Zip the project: `cd ~/Desktop/uni && zip -r NNP.zip NNP/ -x 'NNP/.venv/*' 'NNP/.git/*' 'NNP/results/*'`
2. Upload `NNP.zip` to Google Drive root
3. Open the desired notebook in Colab and run all cells

### SLURM Cluster

```bash
sbatch jobs/train_transformer.sh   # Array job: one model per task
sbatch jobs/train_probe.sh
sbatch jobs/train_svm.sh
```

## Encoder Models

The project evaluates 20 encoder models spanning BERT, RoBERTa, MacBERT, LERT, PERT, ALBERT, and several Chinese embedding models. See `config.py` for the full list.

## Features (Traditional ML)

| Group | Description | Dimensions |
|---|---|---|
| Character n-grams | TF-IDF over char 1/2/3-grams | up to 50k |
| Word n-grams | TF-IDF over jieba-segmented 1/2-grams | up to 50k |
| POS tags | POS tag frequency distribution | ~26 |
| POS n-grams | POS bigram & trigram patterns | ~700 |
| Function words | Frequency of 50 Chinese function words | 50 |
| Particle context | Usage patterns of 9 grammatical particles | 243 |
| Discourse connectives | Frequency of ~40 discourse markers | ~40 |
| Lexical richness | TTR, hapax ratio, Yule's K, etc. | 8 |
| Segmentation | Word length distribution, OOV ratio | ~12 |
| Radicals (opt.) | Kangxi radical usage distribution | 214 |
| Dependency (opt.) | Dependency relation frequencies | ~50 |

## Class Imbalance Handling

- **Stratified splits**: train/val/test maintain class proportions; rare classes (< 3 samples) are placed entirely in the training set
- **Traditional ML**: `class_weight="balanced"` for all sklearn classifiers
- **Transformer fine-tuning**: `WeightedTrainer` with inverse-frequency class weights in cross-entropy loss
