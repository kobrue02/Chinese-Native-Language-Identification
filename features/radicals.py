"""Radical-based features for Chinese characters.

Chinese characters are composed of radicals (semantic/phonetic components).
Learners often confuse characters sharing a radical (e.g., 请/情/清), and
the distribution of radicals used reveals L1-specific character knowledge.

Requires a radical mapping file at data/radical_map.json.
Generate it by running:  python -m features.radicals
"""

import json
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

_MAP_PATH = Path(__file__).parent.parent / "data" / "radical_map.json"
_N_KANGXI_RADICALS = 214
_RADICAL_MAP: dict[str, int] | None = None


def _load_radical_map() -> dict[str, int]:
    """Load character → Kangxi radical index mapping."""
    global _RADICAL_MAP
    if _RADICAL_MAP is not None:
        return _RADICAL_MAP
    if not _MAP_PATH.exists():
        raise FileNotFoundError(
            f"Radical map not found at {_MAP_PATH}. "
            "Generate it with: uv run python -m features.radicals"
        )
    with open(_MAP_PATH) as f:
        _RADICAL_MAP = json.load(f)
    return _RADICAL_MAP


def build_radical_map() -> None:
    """Download Unihan database and extract character → radical mapping.

    Saves to data/radical_map.json.
    """
    import urllib.request

    url = "https://www.unicode.org/Public/UCD/latest/ucd/Unihan.zip"
    print(f"Downloading Unihan database from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req)
    data = resp.read()

    print("Extracting kRSUnicode field...")
    radical_map = {}
    with zipfile.ZipFile(BytesIO(data)) as zf:
        with zf.open("Unihan_IRGSources.txt") as f:
            for line in f:
                line = line.decode("utf-8").strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 3 and parts[1] == "kRSUnicode":
                    # Format: U+XXXX  kRSUnicode  radical.strokes
                    codepoint = int(parts[0].replace("U+", ""), 16)
                    char = chr(codepoint)
                    # Take the first radical if multiple are listed
                    radical_str = parts[2].split()[0]
                    radical_num = int(radical_str.split(".")[0].rstrip("'"))
                    if 1 <= radical_num <= _N_KANGXI_RADICALS:
                        radical_map[char] = radical_num

    _MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_MAP_PATH, "w") as f:
        json.dump(radical_map, f, ensure_ascii=False)
    print(f"Saved {len(radical_map)} character mappings to {_MAP_PATH}")


def extract_radical_features(texts: list[str]) -> np.ndarray:
    """Extract radical frequency distribution (214 dims, L1-normalized)."""
    rmap = _load_radical_map()
    features = np.zeros((len(texts), _N_KANGXI_RADICALS), dtype=np.float64)

    for i, text in enumerate(tqdm(texts, desc="Radical features")):
        for char in text:
            rad = rmap.get(char)
            if rad is not None:
                features[i, rad - 1] += 1  # radicals are 1-indexed
        row_sum = features[i].sum()
        if row_sum > 0:
            features[i] /= row_sum

    return features


def texts_to_radical_sequences(
    texts: list[str], desc: str = "Radical sequences"
) -> list[str]:
    """Convert texts to space-separated radical index sequences for n-gram TF-IDF."""
    rmap = _load_radical_map()
    results = []
    for text in tqdm(texts, desc=desc):
        seq = []
        for char in text:
            rad = rmap.get(char)
            if rad is not None:
                seq.append(f"R{rad}")
        results.append(" ".join(seq))
    return results


def radical_ngram_vectorizer(
    ngram_range: tuple[int, int] = (2, 3),
    max_features: int = 10_000,
) -> TfidfVectorizer:
    """TF-IDF vectorizer over radical n-gram sequences."""
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
    )


if __name__ == "__main__":
    build_radical_map()
