from pathlib import Path

DATA_DIR = Path(__file__).parent / "JCLCv2"
INDEX_CSV = DATA_DIR / "index.csv"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

SVM_CONFIG = {
    "C_values": [0.01, 0.1, 1.0, 10.0],
    "char_ngram_range": (1, 3),
    "word_ngram_range": (1, 2),
    "max_features": 50_000,
}

NEURAL_CONFIG = {
    "vocab_size": 8_000,
    "embed_dim": 128,
    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.3,
    "max_seq_len": 512,
    "batch_size": 64,
    "lr": 1e-3,
    "epochs": 30,
    "patience": 5,
}

TRANSFORMER_CONFIG = {
    "max_length": 512,
    "batch_size": 16,
    "lr": 2e-5,
    "epochs": 5,
    "warmup_ratio": 0.1,
    "patience": 2,
}

# Models to compare. "type" determines loading strategy:
#   "classifier" → AutoModelForSequenceClassification (has built-in head)
#   "embedding"  → AutoModel + linear classification head
ENCODER_MODELS = [
    {"name": "google-bert/bert-base-chinese", "type": "embedding"},
    {"name": "google-bert/bert-base-uncased", "type": "embedding"},
    {"name": "google-bert/bert-large-uncased", "type": "embedding"},
    {"name": "google-bert/bert-base-multilingual-cased", "type": "embedding"},
    {"name": "hfl/chinese-roberta-wwm-ext", "type": "embedding"},
    {"name": "hfl/chinese-bert-wwm", "type": "embedding"},
    {"name": "hfl/chinese-roberta-wwm-ext-large", "type": "embedding"},
    {"name": "hfl/chinese-macbert-base", "type": "embedding"},
    {"name": "hfl/chinese-lert-base", "type": "embedding"},
    {"name": "hfl/chinese-macbert-large", "type": "embedding"},
    {"name": "hfl/chinese-lert-large", "type": "embedding"},
    {"name": "hfl/chinese-pert-base", "type": "embedding"},
    {"name": "hfl/chinese-pert-large", "type": "embedding"},
    {"name": "voidful/albert_chinese_base", "type": "embedding"},
    {"name": "shibing624/text2vec-base-chinese", "type": "embedding"},
    {"name": "Octen/Octen-Embedding-0.6B", "type": "embedding"},
    {"name": "Kingsoft-LLM/QZhou-Embedding-Zh", "type": "embedding"},
    {"name": "IEITYuan/Yuan-embedding-2.0-zh", "type": "embedding"},
    {"name": "DMetaSoul/Dmeta-embedding-zh-small", "type": "embedding"},
    {"name": "OpenMOSS-Team/bart-base-chinese", "type": "embedding"},
]
