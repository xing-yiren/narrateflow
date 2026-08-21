"""
语义相似度工具 — Stage 3 对齐用
TF-IDF 向量化 + 余弦相似度，轻量无 GPU 依赖。
"""
import re
import logging
from typing import List, Tuple, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ─── 中文分词 (轻量: jieba 可选, 无则用字符级别) ───────────

_HAS_JIEBA = False
try:
    import jieba
    _HAS_JIEBA = True
except ImportError:
    pass


def _tokenize(text: str) -> str:
    """中文文本分词/分字（空格分隔）"""
    if _HAS_JIEBA:
        return " ".join(jieba.cut(text))
    # 字符级回退
    tokens = []
    for ch in text:
        if ch.strip():
            tokens.append(ch)
    return " ".join(tokens)


# ─── TF-IDF 向量化 ─────────────────────────────────────────

class TextMatcher:
    """TF-IDF 文本匹配器"""

    def __init__(self, max_features: int = 500):
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x.split(),
            max_features=max_features,
            lowercase=False,
        )
        self._fitted = False
        self._doc_vectors: Optional[np.ndarray] = None
        self._docs: List[str] = []

    def fit(self, documents: List[str]):
        """对所有文档建 TF-IDF 索引"""
        tokenized = [_tokenize(doc) for doc in documents]
        self._docs = documents
        self._doc_vectors = self.vectorizer.fit_transform(tokenized)
        self._fitted = True
        logger.debug(f"TF-IDF fit: {len(documents)} docs, "
                     f"vocab={len(self.vectorizer.vocabulary_)}")

    def query_similarities(self, query: str) -> np.ndarray:
        """对已建索引的文档做相似度查询 (返回 [0..1] 数组)"""
        if not self._fitted:
            raise RuntimeError("TextMatcher not fitted")
        q_vec = self.vectorizer.transform([_tokenize(query)])
        sims = cosine_similarity(q_vec, self._doc_vectors)[0]
        return sims

    def pairwise_similarity(self, text_a: str, text_b: str) -> float:
        """两段文本的即时相似度（不建索引）"""
        tfidf = TfidfVectorizer(tokenizer=lambda x: x.split(),
                                max_features=200, lowercase=False)
        vecs = tfidf.fit_transform([_tokenize(text_a), _tokenize(text_b)])
        return float(cosine_similarity(vecs[0:1], vecs[1:2])[0][0])


# ─── 窗口语义预索引 ────────────────────────────────────────

def build_window_index(windows: List[dict]) -> TextMatcher:
    """为所有窗口的 visual_summary 建 TF-IDF 索引"""
    matcher = TextMatcher()
    docs = [w.get("vlm_output", {}).get("visual_summary", "") for w in windows]
    matcher.fit(docs)
    return matcher


# ─── 标签/OCR 重叠计算 ─────────────────────────────────────

def _token_overlap_score(narration_text: str, tags: List[str]) -> float:
    """旁白文本与标签列表的关键词重叠得分"""
    if not tags:
        return 0.0
    # 对旁白做简单分词
    nar_tokens = set(_tokenize(narration_text).split())
    # 标签也分词
    tag_tokens = set()
    for tag in tags:
        tag_tokens.update(_tokenize(tag).split())
    if not tag_tokens:
        return 0.0
    overlap = nar_tokens & tag_tokens
    return len(overlap) / len(tag_tokens)


def physical_tag_score(narration_text: str, window: dict) -> float:
    """旁白文本与画面 physical_tags 的重叠得分"""
    tags = window.get("vlm_output", {}).get("physical_tags", [])
    return _token_overlap_score(narration_text, tags)


def ocr_match_score(narration_text: str, window: dict) -> float:
    """旁白提及词是否出现在画面文字中"""
    texts = window.get("vlm_output", {}).get("visible_text", [])
    if not texts:
        return 0.0
    # 合并所有 OCR 文字
    ocr_combined = " ".join(texts)
    nar_tokens = set(_tokenize(narration_text).split())
    ocr_tokens = set(_tokenize(ocr_combined).split())
    if not ocr_tokens:
        return 0.0
    # 旁白词在 OCR 文字中的命中率
    hits = nar_tokens & ocr_tokens
    # 归一化：看旁白中有多少比例的词出现在 OCR 中
    return min(1.0, len(hits) / max(1, len(nar_tokens)) * 2)


def mood_match_score(narration_text: str, window: dict) -> float:
    """意境/氛围匹配：旁白情绪词与 mood_tags 重叠"""
    moods = window.get("vlm_output", {}).get("mood_tags", [])
    if isinstance(moods, str):
        moods = [t.strip() for t in moods.replace("、", ",").split(",")]
    return _token_overlap_score(narration_text, moods)


def action_match_score(narration_text: str, window: dict) -> float:
    """旁白动作描述与检测动作的匹配"""
    actions = window.get("vlm_output", {}).get("action_candidates", [])
    if not actions:
        return 0.0
    # 合并所有 action description
    action_text = " ".join([
        a.get("description", "") for a in actions
        if isinstance(a, dict)
    ])
    if not action_text.strip():
        return 0.0
    # 简单重叠
    return _token_overlap_score(narration_text, [action_text])
