"""Smoke and unit tests for the enrichment pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from taxonomy import (
    get_l1, get_l2_list, is_valid_l2, is_active_l2,
    ACTIVE_L2, ALL_L2, INACTIVE_L2, TAXONOMY,
)
from preprocessor import normalize
from entity_db import EntityDB
from router import HybridRouter


# ---------------------------------------------------------------------------
# Taxonomy tests
# ---------------------------------------------------------------------------

def test_taxonomy_l2_to_l1_mapping():
    """Assert every L2 maps to a valid L1."""
    for l2 in ALL_L2:
        l1 = get_l1(l2)
        assert l1 is not None, f"L2 '{l2}' için L1 bulunamadı"
        assert l1 in TAXONOMY, f"L1 '{l1}' taxonomy'de yok"


def test_taxonomy_active_l2_count():
    """Assert ACTIVE_L2 count equals ALL_L2 minus INACTIVE_L2."""
    assert len(ACTIVE_L2) == len(ALL_L2) - len(INACTIVE_L2), (
        f"Aktif L2 sayısı beklenenden farklı: {len(ACTIVE_L2)}"
    )


def test_taxonomy_get_l2_list():
    assert "eczane" in get_l2_list("saglik")
    assert "cafe_kahve" in get_l2_list("yeme_icme")
    assert get_l2_list("var_olmayan_l1") == []


def test_taxonomy_is_valid():
    assert is_valid_l2("eczane") is True
    assert is_valid_l2("ECZANE") is False
    assert is_valid_l2("yok_olan_kategori") is False


def test_taxonomy_inactive_l2_not_active():
    for l2 in INACTIVE_L2:
        assert not is_active_l2(l2), f"'{l2}' aktif olmamalı"


# ---------------------------------------------------------------------------
# Preprocessor tests
# ---------------------------------------------------------------------------

def test_normalize_basic():
    assert normalize("MIGROS") == "migros"
    assert normalize("MİGROS") == "migros"
    assert normalize("  STARBUCKS  ") == "starbucks"


def test_normalize_turkish_chars():
    result = normalize("İSTANBUL KAFE")
    assert "istanbul" in result
    result2 = normalize("ŞELALE LOKANTASI")
    assert "şelale" in result2.lower() or "selale" in result2


def test_normalize_prefix_strip():
    result = normalize("*** STARBUCKS ISTANBUL ***")
    assert "starbucks" in result
    assert "***" not in result


def test_normalize_pos_strip():
    result = normalize("POS/001 MIGROS")
    assert "pos" not in result
    assert "gros" in result


def test_normalize_empty():
    assert normalize("") == ""
    assert normalize("   ") == ""


def test_normalize_consistency():
    """Same input must always produce the same output (cache-safe)."""
    r1 = normalize("BURGER KING KADIKOY")
    r2 = normalize("BURGER KING KADIKOY")
    assert r1 == r2


# ---------------------------------------------------------------------------
# EntityDB tests
# ---------------------------------------------------------------------------

def test_entity_db_exact_match():
    db = EntityDB()
    match = db.lookup("STARBUCKS")
    assert match is not None
    assert match.category_l2 == "cafe_kahve"
    assert match.match_type == "exact"
    assert match.confidence == 1.0


def test_entity_db_prefix_match():
    db = EntityDB()
    match = db.lookup("MIGROS HYPERMARKET SUBE 5")
    assert match is not None
    assert match.category_l2 == "supermarket"
    assert match.match_type in ("exact", "prefix", "fuzzy")


def test_entity_db_no_match():
    db = EntityDB()
    match = db.lookup("KAPADOKYA ŞELALE LOKANTASI")
    assert match is None


def test_entity_db_add_correction():
    db = EntityDB()
    db.add_correction("kapadokya selale lokantasi", "restoran")
    match = db.lookup("KAPADOKYA SELALE LOKANTASI")
    assert match is not None
    assert match.category_l2 == "restoran"


def test_entity_db_invalid_l2_raises():
    db = EntityDB()
    try:
        db.add_correction("test_merchant", "gecersiz_kategori")
        assert False, "Geçersiz L2 için ValueError yükseltilmeliydi"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Router tests
# ---------------------------------------------------------------------------

def test_router_entity_db_hit():
    router = HybridRouter()
    result = router.enrich("*** STARBUCKS ISTANBUL ***")
    assert result.source == "entity_db"
    assert result.flag == "ENTITY_DB_HIT"
    assert result.category_l2 == "cafe_kahve"
    assert result.confidence >= 0.9


def test_router_empty_string():
    router = HybridRouter()
    result = router.enrich("")
    assert result.source == "fallback"
    assert result.category_l2 == "yetersiz_islem_verisi"
    assert result.confidence == 0.0


def test_router_very_short_string():
    router = HybridRouter()
    result = router.enrich("AB")
    assert result.source == "fallback"


def test_router_unknown_no_model():
    router = HybridRouter()
    result = router.enrich("KAPADOKYA ŞELALE LOKANTASI")
    assert result.source in ("fallback", "ml_model")


def test_router_batch():
    router = HybridRouter()
    merchants = ["STARBUCKS", "MIGROS", "NETFLIX", ""]
    results = router.enrich_batch(merchants)
    assert len(results) == 4
    assert all(hasattr(r, "category_l2") for r in results)


def test_router_result_to_dict():
    router = HybridRouter()
    result = router.enrich("STARBUCKS")
    d = result.to_dict()
    required_keys = {
        "merchant_raw", "merchant_normalized", "category_l2",
        "category_l1", "confidence", "source", "flag", "top_k"
    }
    assert required_keys.issubset(d.keys())


def test_router_stats():
    router = HybridRouter()
    router.enrich("STARBUCKS")
    router.enrich("MIGROS")
    router.enrich("")
    stats = router.get_stats()
    assert stats["total"] == 3
    assert stats["entity_db_hits"] >= 2


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all():
    import traceback
    tests = [
        # Taxonomy
        test_taxonomy_l2_to_l1_mapping,
        test_taxonomy_active_l2_count,
        test_taxonomy_get_l2_list,
        test_taxonomy_is_valid,
        test_taxonomy_inactive_l2_not_active,
        # Preprocessor
        test_normalize_basic,
        test_normalize_turkish_chars,
        test_normalize_prefix_strip,
        test_normalize_pos_strip,
        test_normalize_empty,
        test_normalize_consistency,
        # EntityDB
        test_entity_db_exact_match,
        test_entity_db_prefix_match,
        test_entity_db_no_match,
        test_entity_db_add_correction,
        test_entity_db_invalid_l2_raises,
        # Router
        test_router_entity_db_hit,
        test_router_empty_string,
        test_router_very_short_string,
        test_router_unknown_no_model,
        test_router_batch,
        test_router_result_to_dict,
        test_router_stats,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✅  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ❌  {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{'='*40}")
    print(f"  {passed} geçti / {failed} başarısız / {len(tests)} toplam")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    print("\n🔬 TR Kredi Kartı Zenginleştirme — Test Paketi\n")
    run_all()
