"""Layer 4: HybridRouter — orchestrates the full enrichment pipeline."""

import logging
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from taxonomy import INACTIVE_L2, get_l1
from preprocessor import normalize
from entity_db import EntityDB, EntityMatch, get_db
from model import MerchantClassifier, CascadeClassifier, Prediction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Confidence thresholds
# ---------------------------------------------------------------------------

UNCLASSIFIABLE_THRESHOLD: float = 0.0   # disabled — tune after data review
LOW_CONFIDENCE_THRESHOLD: float = 0.60
AUTO_LEARN_THRESHOLD: float = 0.90
MIN_MERCHANT_LENGTH: int = 3


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EnrichmentResult:
    """
    Enrichment result for a single transaction.

    Fields:
        merchant_raw         : original input string, never modified
        merchant_normalized  : preprocessor output
        category_l2          : L2 label (None in l1-only inference mode)
        category_l1          : predicted L1 label, always filled
        confidence           : 0.0 – 1.0
        source               : "entity_db" | "ml_model" | "fallback"
        flag                 : None | "LOW_CONFIDENCE" | "ENTITY_DB_HIT"
        top_k                : runner-up predictions [(label, conf), ...]
    """
    merchant_raw: str
    merchant_normalized: str
    category_l2: str | None
    category_l1: str
    confidence: float
    source: str
    flag: str | None
    top_k: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "merchant_raw":        self.merchant_raw,
            "merchant_normalized": self.merchant_normalized,
            "category_l2":         self.category_l2,
            "category_l1":         self.category_l1,
            "confidence":          self.confidence,
            "source":              self.source,
            "flag":                self.flag,
            "top_k":               self.top_k,
        }

    def __repr__(self) -> str:
        flag_str = f" [{self.flag}]" if self.flag else ""
        l2_str   = self.category_l2 or "-"
        return (
            f"EnrichmentResult("
            f"'{self.merchant_raw}' → L1={self.category_l1} L2={l2_str} "
            f"({self.confidence:.2f}){flag_str})"
        )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class HybridRouter:
    """
    Pipeline orchestrator: preprocessor → entity_db → ml_model → flag assignment.

    inference_mode:
        "cascade" (default) — L1 → L1-specific L2 model
        "l2"                — single model, direct L2 prediction
        "l1"                — single model, direct L1 prediction

    Usage:
        router = HybridRouter(inference_mode="cascade")
        router.load_model("models/berturk_mode_cascade")
        result = router.enrich("*** STARBUCKS ISTANBUL ***")
        print(result.category_l1, result.category_l2, result.flag)
    """

    def __init__(
        self,
        entity_db: EntityDB | None = None,
        classifier: MerchantClassifier | CascadeClassifier | None = None,
        auto_learn: bool = True,
        inference_mode: str = "cascade",
    ):
        self._db: EntityDB = entity_db or get_db()
        self._clf: MerchantClassifier | CascadeClassifier | None = classifier
        self._auto_learn    = auto_learn
        self._inference_mode = inference_mode
        self._auto_learn_candidates: list[dict] = []

        self._stats = {
            "total":           0,
            "entity_db_hits":  0,
            "ml_predictions":  0,
            "unclassifiable":  0,
            "low_confidence":  0,
            "fallback":        0,
        }

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, model_dir: str | Path) -> None:
        """Load a model from the given directory.

        cascade mode → loads CascadeClassifier (requires cascade_meta.json).
        l1/l2 mode  → loads MerchantClassifier.
        """
        if self._inference_mode == "cascade":
            clf = CascadeClassifier()
            clf.load(model_dir)
            self._clf = clf
            logger.info(f"Cascade model loaded: {model_dir}")
        else:
            if self._clf is None or isinstance(self._clf, CascadeClassifier):
                self._clf = MerchantClassifier()
            self._clf.load(model_dir)
            logger.info(f"Model loaded [{self._inference_mode.upper()}]: {model_dir}")

    def set_classifier(self, clf: MerchantClassifier) -> None:
        self._clf = clf

    @property
    def _classifier(self) -> MerchantClassifier | CascadeClassifier | None:
        if self._clf is None:
            try:
                if self._inference_mode == "cascade":
                    self._clf = CascadeClassifier()
                else:
                    self._clf = MerchantClassifier()
            except ImportError:
                return None
        return self._clf

    # ------------------------------------------------------------------
    # Core enrichment
    # ------------------------------------------------------------------

    def enrich(self, merchant_raw: str, top_k: int = 3) -> EnrichmentResult:
        """
        Predict the category for a single merchant string.

        Args:
            merchant_raw: raw merchant name from POS terminal.
            top_k: number of runner-up predictions to include.

        Returns:
            EnrichmentResult with category_l1, category_l2, confidence, and flag.
        """
        self._stats["total"] += 1

        # Guard: empty / too-short string
        stripped = (merchant_raw or "").strip()
        if len(stripped) < MIN_MERCHANT_LENGTH:
            self._stats["fallback"] += 1
            return self._fallback(merchant_raw, "tanimlanamayan_isyeri")

        normalized = normalize(merchant_raw)
        if not normalized:
            self._stats["fallback"] += 1
            return self._fallback(merchant_raw, "tanimlanamayan_isyeri")

        # Layer 2: Entity DB
        db_match: EntityMatch | None = self._db.lookup(merchant_raw)
        if db_match:
            self._stats["entity_db_hits"] += 1
            l1 = get_l1(db_match.category_l2) or "diger"
            return EnrichmentResult(
                merchant_raw=merchant_raw,
                merchant_normalized=normalized,
                category_l2=db_match.category_l2,
                category_l1=l1,
                confidence=db_match.confidence,
                source="entity_db",
                flag="ENTITY_DB_HIT",
                top_k=[(db_match.category_l2, db_match.confidence)],
            )

        # Layer 3: ML model
        clf = self._classifier
        if clf is None or not clf.is_trained:
            logger.warning("ML model not yet trained; returning fallback.")
            self._stats["fallback"] += 1
            return self._fallback(merchant_raw, "tanimlanamayan_isyeri")

        prediction: Prediction = clf.predict(merchant_raw, top_k=top_k)
        self._stats["ml_predictions"] += 1

        # Layer 4: confidence routing
        if UNCLASSIFIABLE_THRESHOLD > 0.0 and prediction.confidence < UNCLASSIFIABLE_THRESHOLD:
            self._stats["unclassifiable"] += 1
            self._stats["fallback"] += 1
            return self._fallback(merchant_raw, "tanimlanamayan_isyeri")

        flag: str | None = None
        if prediction.confidence < LOW_CONFIDENCE_THRESHOLD:
            flag = "LOW_CONFIDENCE"
            self._stats["low_confidence"] += 1
        elif prediction.confidence >= AUTO_LEARN_THRESHOLD and self._auto_learn:
            if prediction.category_l2 is not None:
                self._auto_learn_candidates.append({
                    "normalized":  prediction.merchant_normalized,
                    "category_l2": prediction.category_l2,
                    "confidence":  prediction.confidence,
                })

        return EnrichmentResult(
            merchant_raw=merchant_raw,
            merchant_normalized=prediction.merchant_normalized,
            category_l2=prediction.category_l2,
            category_l1=prediction.category_l1,
            confidence=prediction.confidence,
            source="ml_model",
            flag=flag,
            top_k=prediction.top_k,
        )

    def enrich_batch(
        self,
        merchants: list[str],
        top_k: int = 1,
    ) -> list[EnrichmentResult]:
        return [self.enrich(m, top_k=top_k) for m in merchants]

    # ------------------------------------------------------------------
    # Auto-learn
    # ------------------------------------------------------------------

    def flush_auto_learn(self, min_occurrences: int = 2) -> int:
        """
        Promote high-confidence ML predictions to the Entity DB and persist to catalog.json.

        min_occurrences: minimum times the same merchant must appear before promotion.
        Conflict resolution: when multiple L2 labels qualify, the most frequent wins.
        Returns the number of new entries added.
        """
        # normalized → {l2 → count}
        label_counts: dict[str, Counter] = defaultdict(Counter)
        # normalized → {l2 → max_confidence}
        best_conf: dict[str, dict[str, float]] = defaultdict(dict)

        for c in self._auto_learn_candidates:
            norm = c["normalized"]
            l2   = c["category_l2"]
            conf = c["confidence"]
            label_counts[norm][l2] += 1
            best_conf[norm][l2] = max(best_conf[norm].get(l2, 0.0), conf)

        added = 0
        for normalized, counter in label_counts.items():
            best_l2, count = counter.most_common(1)[0]
            if count >= min_occurrences:
                self._db.add_correction(
                    normalized,
                    best_l2,
                    source="auto_learn",
                    confidence=best_conf[normalized][best_l2],
                )
                added += 1

        self._auto_learn_candidates.clear()

        if added > 0 and self._db._db_path:
            self._db.save()
            logger.info(f"Auto-learn: {added} entries written to catalog.json.")
        else:
            logger.info(f"Auto-learn: {added} entries added (no db_path, not persisted to disk).")

        return added

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback(merchant_raw: str, l2_flag: str = "tanimlanamayan_isyeri") -> EnrichmentResult:
        normalized = normalize(merchant_raw) if merchant_raw else ""
        return EnrichmentResult(
            merchant_raw=merchant_raw,
            merchant_normalized=normalized,
            category_l2=l2_flag,
            category_l1="diger",
            confidence=0.0,
            source="fallback",
            flag=None,
            top_k=[],
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        total = self._stats["total"]
        if total == 0:
            return self._stats.copy()
        return {
            **self._stats,
            "entity_db_hit_rate":  round(self._stats["entity_db_hits"]  / total, 3),
            "ml_rate":             round(self._stats["ml_predictions"]   / total, 3),
            "unclassifiable_rate": round(self._stats["unclassifiable"]   / total, 3),
            "low_confidence_rate": round(self._stats["low_confidence"]   / total, 3),
        }

    def reset_stats(self) -> None:
        for key in self._stats:
            self._stats[key] = 0

    def __repr__(self) -> str:
        is_trained = self._clf.is_trained if self._clf is not None else False
        mode_str   = self._inference_mode.upper() if is_trained else "untrained"
        return (
            f"HybridRouter("
            f"db={len(self._db)} entries, "
            f"model={mode_str}, "
            f"inference_mode={self._inference_mode})"
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_router: HybridRouter | None = None
_router_lock = threading.Lock()


def get_router(model_dir: str | Path | None = None) -> HybridRouter:
    """Thread-safe singleton using double-checked locking.

    Guards against two concurrent requests both seeing _default_router is None
    under multi-threaded WSGI servers (gunicorn, uWSGI).
    """
    global _default_router
    if _default_router is None:
        with _router_lock:
            if _default_router is None:
                r = HybridRouter()
                if model_dir and Path(model_dir).exists():
                    r.load_model(model_dir)
                _default_router = r
    return _default_router


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    router = HybridRouter()

    test_cases = [
        "*** STARBUCKS ISTANBUL ***",
        "MIGROS HYPERMARKET SUBE 5",
        "NETFLIX ABONELIK",
        "SHELL AKARYAKIT",
        "KARŞI KURUYEMİŞ",
        "MARMARAY GAZETE BAYİ",
        "",
        "AB",
    ]

    print(f"\n{router}\n")
    print(f"{'Raw String':<35} {'Source':<12} {'Conf':>5}  {'L1':<25} {'L2':<20} {'Flag'}")
    print("-" * 105)

    for t in test_cases:
        r = router.enrich(t)
        flag_str = r.flag or ""
        l2_str   = r.category_l2 or "-"
        print(
            f"  {t!r:<33} {r.source:<12} {r.confidence:>5.2f}  "
            f"{r.category_l1:<25} {l2_str:<20} {flag_str}"
        )

    print(f"\nStats: {router.get_stats()}")
