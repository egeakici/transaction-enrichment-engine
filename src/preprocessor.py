"""Layer 1: Turkish merchant string normalization pipeline."""

import re
import unicodedata
from functools import lru_cache

# ---------------------------------------------------------------------------
# Turkish-safe uppercase → lowercase mapping
# str.casefold() gets İ/I wrong for Turkish
# ---------------------------------------------------------------------------
_TR_UPPER_TO_LOWER: dict[str, str] = {
    "I": "ı",
    "İ": "i",
    "Ş": "ş",
    "Ğ": "ğ",
    "Ü": "ü",
    "Ö": "ö",
    "Ç": "ç",
}

_TR_LOWER_TABLE = str.maketrans(_TR_UPPER_TO_LOWER)


def _tr_lower(text: str) -> str:
    return text.translate(_TR_LOWER_TABLE).lower()


# ---------------------------------------------------------------------------
# OCR/encoding error corrections for POS terminal artifacts
# ---------------------------------------------------------------------------
_OCR_FIX: dict[str, str] = {
    # Latin-1 / Windows-1252 artifacts
    "ý": "ı",
    "þ": "ş",
    "ð": "ğ",
    "ü": "ü",
    "ö": "ö",
    "ç": "ç",
    "â": "a",
    "î": "i",
    "û": "u",
    "Â": "A",
    # IBM OEM 857 artifacts
    "\x9f": "i",
    "\x8e": "ş",
    "\x9e": "ş",
    "\x9c": "ü",
    "\x81": "ü",
    "\x94": "ö",
    "\x8d": "ı",
}

_OCR_TABLE = str.maketrans(_OCR_FIX)


# ---------------------------------------------------------------------------
# Prefix/suffix strip patterns — more specific patterns first
# ---------------------------------------------------------------------------
_PREFIX_PATTERNS: list[re.Pattern] = [
    # asterisk blocks: "*** STARBUCKS ***"
    re.compile(r"^\*+\s*"),
    re.compile(r"\s*\*+$"),
    # POS terminal prefix: "POS/123456 MIGROS"
    re.compile(r"^POS[/\s]?\d*\s*", re.IGNORECASE),
    # transaction number prefix: "ISL:00123 A101"
    re.compile(r"^ISL\s*:\s*\d+\s*", re.IGNORECASE),
    # city code prefix: "34-STARBUCKS", "IST MIGROS"
    re.compile(r"^\d{1,2}[-/]\s*"),
    re.compile(r"^(IST|ANK|IZM|BRS|ADN|KON|ANT)\s+", re.IGNORECASE),
    # card type suffix: "MIGROS VISA"
    re.compile(r"\s+(VISA|MASTERCARD|MAESTRO|TROY|AMEX)$", re.IGNORECASE),
    # branch suffix: "MIGROS SUBE 015"
    re.compile(r"\s+SUBE\s*\d*$", re.IGNORECASE),
    re.compile(r"\s+SUBESI\s*\d*$", re.IGNORECASE),
    # location number suffix: "ECZANE NO:4"
    re.compile(r"\s+(NO|NUM|NUMARA)\s*[:.]?\s*\d+$", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Known brand abbreviation → canonical name mappings
# ---------------------------------------------------------------------------
_NORMALIZATION_MAP: dict[str, str] = {
    "mcd": "mcdonalds",
    "mc donalds": "mcdonalds",
    "mc donald": "mcdonalds",
    "bk": "burger king",
    "kfc": "kfc",
    "sbux": "starbucks",
    "a101": "a 101",
    "bim": "bim",
}


def _apply_ocr_fix(text: str) -> str:
    return text.translate(_OCR_TABLE)


def _strip_prefixes(text: str) -> str:
    for pattern in _PREFIX_PATTERNS:
        text = pattern.sub("", text).strip()
    return text


def _clean_punctuation(text: str) -> str:
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[-/]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return text


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _apply_brand_normalization(text: str) -> str:
    for abbr, full in _NORMALIZATION_MAP.items():
        text = re.sub(rf"\b{re.escape(abbr)}\b", full, text)
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize(merchant_raw: str) -> str:
    """Normalize a raw merchant string for model input."""
    if not merchant_raw or not merchant_raw.strip():
        return ""
    canonical = _tr_lower(unicodedata.normalize("NFC", merchant_raw.strip()))
    return _normalize_impl(canonical)


@lru_cache(maxsize=4096)
def _normalize_impl(text: str) -> str:
    """Full normalization pipeline on a pre-lowercased NFC string."""
    # 1. fix OCR/encoding artifacts
    text = _apply_ocr_fix(text)
    # 2. strip POS prefixes / branch suffixes
    text = _strip_prefixes(text)
    # 3. clean punctuation
    text = _clean_punctuation(text)
    # 4. collapse whitespace
    text = _normalize_whitespace(text)
    # 5. expand brand abbreviations
    text = _apply_brand_normalization(text)
    # 6. merge ı→i (POS terminals mix İ and I inconsistently)
    text = text.replace("ı", "i")
    return text


def normalize_batch(merchants: list[str]) -> list[str]:
    return [normalize(m) for m in merchants]


# ---------------------------------------------------------------------------
# Debug utilities
# ---------------------------------------------------------------------------

def explain(merchant_raw: str) -> None:
    """Print each normalization step for debugging."""
    steps = {
        "0_raw": merchant_raw,
    }
    text = merchant_raw.strip()
    text = unicodedata.normalize("NFC", text)
    steps["1_unicode_nfc"] = text
    text = _apply_ocr_fix(text)
    steps["2_ocr_fix"] = text
    text = _tr_lower(text)
    steps["3_tr_lower"] = text
    text = _strip_prefixes(text)
    steps["4_strip_prefix"] = text
    text = _clean_punctuation(text)
    steps["5_clean_punct"] = text
    text = _normalize_whitespace(text)
    steps["6_whitespace"] = text
    text = _apply_brand_normalization(text)
    steps["7_brand_norm"] = text

    print(f"\n{'Adım':<22} {'Değer'}")
    print("-" * 60)
    for step, value in steps.items():
        print(f"  {step:<20} '{value}'")
    print()


if __name__ == "__main__":
    test_cases = [
        "*** STARBUCKS ISTANBUL ***",
        "POS/001 MİGROS HYPERMARKet",
        "ECZANE NO:4",
        "IST A101 SUBE 012",
        "MARMARAY GAZETE BAYİ",
        "KARŞI KURUYEMİŞ",
        "34-DOMINO'S PIZZA",
        "McD KADIKOY",
    ]
    for t in test_cases:
        result = normalize(t)
        print(f"  '{t}' → '{result}'")
