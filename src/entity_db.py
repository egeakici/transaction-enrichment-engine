"""Layer 2: deterministic merchant lookup with exact, prefix, and fuzzy matching."""

import datetime
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from preprocessor import normalize

logger = logging.getLogger(__name__)

try:
    from rapidfuzz import fuzz, process as rf_process
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    import difflib
    _RAPIDFUZZ_AVAILABLE = False
    logger.warning("rapidfuzz not found; using difflib fallback "
                   "for better performance: pip install rapidfuzz")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EntityMatch:
    merchant_raw: str
    merchant_normalized: str
    category_l2: str
    match_type: str        # "exact" | "prefix" | "fuzzy"
    confidence: float      # 0.0 – 1.0
    matched_key: str       # key matched in the catalog

    @property
    def category_l1(self) -> str:
        from taxonomy import get_l1
        return get_l1(self.category_l2) or "diger"


# ---------------------------------------------------------------------------
# Built-in merchant catalog
# Keys are preprocessor.normalize() outputs
# ---------------------------------------------------------------------------

_BUILTIN_CATALOG: dict[str, str] = {
    # Yeme & İçme
    "starbucks": "cafe_kahve",
    "caribou coffee": "cafe_kahve",
    "nero": "cafe_kahve",
    "gloria jeans": "cafe_kahve",
    "mcdonalds": "fast_food",
    "burger king": "fast_food",
    "kfc": "fast_food",
    "popeyes": "fast_food",
    "subway": "fast_food",
    "pizza hut": "restoran",
    "dominos pizza": "restoran",
    "little caesars": "restoran",
    "simit sarayi": "pastane_firin",
    "unlu mamuller": "pastane_firin",
    "yemeksepeti": "yemek_siparis",
    "getir yemek": "yemek_siparis",
    "migros yemek": "yemek_siparis",
    "trendyol yemek": "yemek_siparis",

    # Market & Gıda
    "migros": "supermarket",
    "carrefoursa": "supermarket",
    "carrefour": "supermarket",
    "sok market": "indirim_market",
    "bim": "indirim_market",
    "a 101": "indirim_market",
    "a101": "indirim_market",
    "hakmar": "indirim_market",
    "macro center": "supermarket",
    "file market": "supermarket",

    # Online Alışveriş
    "trendyol": "eticaret_platform",
    "hepsiburada": "eticaret_platform",
    "n11": "eticaret_platform",
    "gittigidiyor": "eticaret_platform",
    "amazon": "eticaret_platform",
    "ebay": "eticaret_platform",
    "aliexpress": "eticaret_platform",
    "aras kargo": "kargo_lojistik",
    "yurtici kargo": "kargo_lojistik",
    "mng kargo": "kargo_lojistik",
    "ptt kargo": "kargo_lojistik",
    "ups": "kargo_lojistik",
    "dhl": "kargo_lojistik",

    # Elektronik
    "mediamarkt": "elektronik_market",
    "teknosa": "elektronik_market",
    "vatan bilgisayar": "elektronik_market",
    "bimeks": "elektronik_market",
    "turkcell": "gsm_fatura",
    "vodafone": "gsm_fatura",
    "turk telekom": "internet_fatura",
    "superonline": "internet_fatura",

    # Ulaşım
    "ibb istanbulkart": "toplu_tasima",
    "istanbul kart": "toplu_tasima",
    "istanbulkart": "toplu_tasima",
    "ego kart": "toplu_tasima",
    "eshot": "toplu_tasima",
    "uber": "taksi_ozel_tasimacilik",
    "bitaksi": "taksi_ozel_tasimacilik",
    "in driver": "taksi_ozel_tasimacilik",
    "otobus bileti": "sehirlerarasi_ulasim",
    "obilet": "sehirlerarasi_ulasim",
    "ispark": "otopark",
    "ego otopark": "otopark",
    "otopark": "otopark",
    "hgs": "otoyol_kopru_gecis",
    "ogs": "otoyol_kopru_gecis",
    "kgm otoyol": "otoyol_kopru_gecis",
    "martı": "mikromobilite_hizmeti",

    # Seyahat
    "thy": "ucak_havayolu",
    "turk hava yollari": "ucak_havayolu",
    "turkish airlines": "ucak_havayolu",
    "pegasus": "ucak_havayolu",
    "anadolujet": "ucak_havayolu",
    "sunexpress": "ucak_havayolu",
    "booking.com": "otel_konaklama",
    "airbnb": "otel_konaklama",
    "trivago": "otel_konaklama",
    "jolly tur": "seyahat_acentesi_tur",
    "tatilsepeti": "seyahat_acentesi_tur",

    # Sağlık
    "eczane": "eczane",
    "dogtas": "mobilya",

    # Fatura & Abonelik
    "netflix": "dijital_abonelik",
    "spotify": "dijital_abonelik",
    "youtube premium": "dijital_abonelik",
    "apple": "dijital_abonelik",
    "google play": "dijital_abonelik",
    "platform": "dijital_abonelik",
    "tabii": "tv_yayin",
    "gain tv": "tv_yayin",
    "exxen": "tv_yayin",
    "blutv": "tv_yayin",
    "digiturk": "tv_yayin",

    # Akaryakıt
    "shell": "akaryakit_istasyonu",
    "opet": "akaryakit_istasyonu",
    "total": "akaryakit_istasyonu",
    "petrol ofisi": "akaryakit_istasyonu",
    "aytemiz": "akaryakit_istasyonu",
    "lukoil": "akaryakit_istasyonu",

    # Eğitim
    "udemy": "cevrimici_egitim",
    "coursera": "cevrimici_egitim",
    "dr&tr": "kitap_kirtasiye",
    "d&r": "kitap_kirtasiye",
    "kitapyurdu": "kitap_kirtasiye",
    "idefix": "kitap_kirtasiye",

    # Kişisel Bakım
    "sephora": "kozmetik_bakim",
    "gratis": "kozmetik_bakim",
    "watsons": "kozmetik_bakim",

    # Finans & Sigorta
    "atm": "nakit_cekim_transfer",
    "bankamatik": "nakit_cekim_transfer",
    "emeklilik": "bireysel_emeklilik_odeme",
    "anadolu hayat": "bireysel_emeklilik_odeme",

    # Eğlence & Kültür
    "cinemaximum": "sinema_tiyatro",
    "biletix": "konser_festival",
    "passo": "konser_festival",

    # Giyim & Aksesuar
    "zara": "giyim_magaza",
    "lc waikiki": "giyim_magaza",
    "mango": "giyim_magaza",
    "h&m": "giyim_magaza",
    "defacto": "giyim_magaza",
    "koton": "giyim_magaza",
    "bershka": "giyim_magaza",
    "stradivarius": "giyim_magaza",
    "pull&bear": "giyim_magaza",
    "adidas": "giyim_magaza",
    "nike": "giyim_magaza",
    "puma": "giyim_magaza",
    "reebok": "ayakkabi",
    "flo": "ayakkabi",
    "network": "giyim_magaza",
    "ipekyolu": "giyim_magaza",
    "boyner": "giyim_magaza",
}

# ---------------------------------------------------------------------------
# Fuzzy matching helpers
# ---------------------------------------------------------------------------

_FUZZY_THRESHOLD = 90  # similarity score 0–100


def _fuzzy_match(query: str, catalog: dict[str, str]) -> tuple[str, float] | None:
    """Return (key, confidence) for the best fuzzy match, or None if below threshold."""
    if _RAPIDFUZZ_AVAILABLE:
        result = rf_process.extractOne(
            query, catalog.keys(),
            scorer=fuzz.token_set_ratio,
            score_cutoff=_FUZZY_THRESHOLD,
        )
        if result:
            matched_key, score, _ = result
            confidence = round(0.80 + (score - _FUZZY_THRESHOLD) / (100 - _FUZZY_THRESHOLD) * 0.14, 4)
            return matched_key, confidence
    else:
        matches = difflib.get_close_matches(query, catalog.keys(), n=1, cutoff=_FUZZY_THRESHOLD / 100)
        if matches:
            key = matches[0]
            ratio = difflib.SequenceMatcher(None, query, key).ratio()
            confidence = round(0.80 + (ratio - _FUZZY_THRESHOLD / 100) * 0.14, 4)
            return key, confidence
    return None


# ---------------------------------------------------------------------------
# EntityDB class
# ---------------------------------------------------------------------------

class EntityDB:
    """
    Deterministic merchant lookup engine.

    Usage:
        db = EntityDB()
        match = db.lookup("*** STARBUCKS IST ***")
        if match:
            print(match.category_l2, match.confidence)
    """

    def __init__(self, db_path: Path | None = None):
        self._catalog: dict[str, str] = {
            normalize(k): v for k, v in _BUILTIN_CATALOG.items()
        }
        # Metadata only for catalog.json / add_correction entries; builtin catalog is excluded.
        # save() writes only this dict to disk.
        self._metadata: dict[str, dict] = {}
        self._db_path = db_path
        self._sorted_keys: list[str] = []
        self._rebuild_prefix_keys()
        if db_path and db_path.exists():
            self.load(db_path)

    # ------------------------------------------------------------------
    # Prefix key cache
    # ------------------------------------------------------------------

    def _rebuild_prefix_keys(self) -> None:
        self._sorted_keys = sorted(self._catalog.keys(), key=len, reverse=True)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(self, merchant_raw: str) -> EntityMatch | None:
        """Try a deterministic match against the catalog; return None on miss."""
        normalized = normalize(merchant_raw)
        if not normalized:
            return None

        # 1. Exact match
        if normalized in self._catalog:
            return EntityMatch(
                merchant_raw=merchant_raw,
                merchant_normalized=normalized,
                category_l2=self._catalog[normalized],
                match_type="exact",
                confidence=1.0,
                matched_key=normalized,
            )

        # 2. Prefix match — requires a word boundary after the key to avoid
        # "bp" matching "bpay istanbul".
        for key in self._sorted_keys:
            if normalized.startswith(key + " "):
                return EntityMatch(
                    merchant_raw=merchant_raw,
                    merchant_normalized=normalized,
                    category_l2=self._catalog[key],
                    match_type="prefix",
                    confidence=0.95,
                    matched_key=key,
                )

        # 3. Fuzzy match
        result = _fuzzy_match(normalized, self._catalog)
        if result:
            matched_key, conf = result
            return EntityMatch(
                merchant_raw=merchant_raw,
                merchant_normalized=normalized,
                category_l2=self._catalog[matched_key],
                match_type="fuzzy",
                confidence=conf,
                matched_key=matched_key,
            )

        return None

    # ------------------------------------------------------------------
    # Add / update entries
    # ------------------------------------------------------------------

    def add_correction(
        self,
        merchant_normalized: str,
        category_l2: str,
        source: str = "manual",
        confidence: float = 1.0,
    ) -> None:
        """Add or update an entry. Call save() to persist to disk."""
        from taxonomy import is_valid_l2
        if not is_valid_l2(category_l2):
            raise ValueError(f"Invalid L2 category: {category_l2}")
        key = normalize(merchant_normalized)
        self._catalog[key] = category_l2
        self._metadata[key] = {
            "l2_category": category_l2,
            "source":      source,
            "is_active":   True,
            "added_date":  datetime.date.today().isoformat(),
            "confidence":  round(confidence, 4),
        }
        self._rebuild_prefix_keys()
        logger.info(
            f"EntityDB updated: '{key}' → '{category_l2}' "
            f"(source={source}, conf={confidence:.2f})"
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | None = None) -> None:
        """Write catalog.json. Only _metadata entries are written (builtin catalog excluded)."""
        target = path or self._db_path
        if not target:
            raise ValueError("db_path not set; cannot save.")
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2, sort_keys=True)
        logger.info(f"EntityDB saved: {target} ({len(self._metadata)} entries)")

    def load(self, path: Path) -> None:
        """Load and merge entries from catalog.json; skips is_active=false records."""
        with open(path, encoding="utf-8") as f:
            raw: dict = json.load(f)

        loaded = 0
        for key, value in raw.items():
            if isinstance(value, dict):
                if not value.get("is_active", True):
                    continue
                l2 = value.get("l2_category", "")
                meta = value
            else:
                l2 = str(value)
                meta = {
                    "l2_category": l2,
                    "source":      "manual",
                    "is_active":   True,
                    "added_date":  "",
                    "confidence":  1.0,
                }
            if not l2:
                continue
            norm_key = normalize(key.upper())
            self._catalog[norm_key] = l2
            self._metadata[norm_key] = meta
            loaded += 1

        self._rebuild_prefix_keys()
        logger.info(f"EntityDB loaded: {path} ({loaded} entries)")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        from collections import Counter
        cat_counts = Counter(self._catalog.values())
        return {
            "total_entries": len(self._catalog),
            "top_categories": cat_counts.most_common(5),
        }

    def __len__(self) -> int:
        return len(self._catalog)

    def __repr__(self) -> str:
        return f"EntityDB(entries={len(self._catalog)})"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_db: EntityDB | None = None


def get_db(db_path: Path | None = None) -> EntityDB:
    """Return the default EntityDB singleton."""
    global _default_db
    if _default_db is None:
        resolved = Path(db_path) if db_path else Path(__file__).parent.parent / "data" / "catalog.json"
        _default_db = EntityDB(db_path=resolved)
    return _default_db


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    db = EntityDB()
    tests = [
        "*** STARBUCKS ISTANBUL ***",
        "MİGROS HYPERMARKET",
        "YEMEKSEPETI ODEME",
        "NETFLIX ABONELIK",
        "SHELL PETROL",
        "KARŞI KURUYEMİŞ",
        "MARMARAY GAZETE BAYİ",
        "STARBUCKSS KADIKOY",
    ]
    print(f"\nEntityDB ({len(db)} entries)\n")
    print(f"{'Raw String':<35} {'Type':<8} {'Conf':>5}  {'L2 Category'}")
    print("-" * 75)
    for t in tests:
        m = db.lookup(t)
        if m:
            print(f"  {t:<33} {m.match_type:<8} {m.confidence:>5.2f}  {m.category_l2}")
        else:
            print(f"  {t:<33} {'—':<8} {'—':>5}  (no match)")
