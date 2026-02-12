"""Chinese function word frequency features.

Function words carry little semantic content but reveal syntactic habits
transferred from the learner's L1 (e.g., overuse of aspect markers,
determiners, or prepositions).
"""

import numpy as np
from tqdm import tqdm

# Common Chinese function words grouped by category
FUNCTION_WORDS = [
    # Pronouns
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "自己", "大家",
    "这", "那", "这些", "那些", "这个", "那个",
    # Determiners / demonstratives
    "每", "各", "某", "其",
    # Prepositions
    "在", "从", "到", "向", "对", "把", "被", "给", "用", "按",
    "关于", "通过", "根据", "按照", "为了", "由于", "随着",
    # Conjunctions
    "和", "与", "或", "而", "但", "但是", "可是", "不过", "然而",
    "因为", "所以", "如果", "虽然", "尽管", "即使", "只要", "除非",
    "不但", "而且", "既", "又", "也", "还", "就", "才", "都", "只",
    # Auxiliary / modal verbs
    "是", "有", "没有", "能", "会", "可以", "应该", "必须", "要", "想",
    "得", "可能", "需要",
    # Aspect markers
    "了", "过", "着", "的", "地", "得",
    # Structural particles
    "吗", "呢", "吧", "啊", "嘛", "哦", "呀",
    # Adverbs (grammatical)
    "不", "没", "很", "太", "最", "更", "非常", "特别", "比较",
    "已经", "正在", "将", "曾", "刚", "一直", "常常", "经常", "往往",
    # Measure words (common)
    "个", "些", "位", "种", "件", "次",
]

_FW2IDX = {w: i for i, w in enumerate(FUNCTION_WORDS)}


def extract_function_word_features(texts: list[str]) -> np.ndarray:
    """Count function word frequencies (normalized per document)."""
    import jieba

    n = len(texts)
    features = np.zeros((n, len(FUNCTION_WORDS)), dtype=np.float64)

    for i, text in enumerate(tqdm(texts, desc="Function words")):
        words = list(jieba.cut(text))
        total = len(words)
        for w in words:
            idx = _FW2IDX.get(w)
            if idx is not None:
                features[i, idx] += 1
        if total > 0:
            features[i] /= total

    return features
