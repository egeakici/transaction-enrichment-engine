"""Two-level category taxonomy for Turkish credit card transaction enrichment (16 L1 / 101 L2)."""

# ---------------------------------------------------------------------------
# L1 → L2 full listing
# ---------------------------------------------------------------------------

TAXONOMY: dict[str, list[str]] = {
    "yeme_icme": [
        "restoran",
        "fast_food",
        "pastane_firin",
        "bar_icki_servisi",
        "yemek_siparis",
        "cafe_kahve",
        "diger_yeme_icme",
    ],
    "market_gida": [
        "supermarket",
        "indirim_market",
        "bakkal_tekel",
        "kuruyemis_aktar",
        "manav_kasap_sarkuteri",
        "diger_market_gida",
    ],
    "giyim_aksesuar": [
        "giyim_magaza",
        "ayakkabi",
        "kuyumcu",
        "saat",
        "diger_giyim_aksesuar",
    ],
    "online_alisveris": [
        "eticaret_platform",
        "diger_online_alisveris",
        "kargo_lojistik",
    ],
    "elektronik": [
        "elektronik_market",
        "elektronik_servis",
        "beyaz_esya",
        "telefon_aksesuar",
        "diger_elektronik",
    ],
    "ev_yasam": [
        "mobilya",
        "dekorasyon",
        "yapi_market",
        "zuccaciye",
        "ev_hizmet",
        "oyuncak_cocuk",
        "spor_urun",
        "diger_ev_yasam",
    ],
    "saglik": [
        "hastane_klinik",
        "eczane",
        "goz_optisyen",
        "dis_hekimi",
        "veteriner",
        "diger_saglik",
    ],
    "kisisel_bakim": [
        "kuafor_berber_guzellik",
        "spor_fitness",
        "kozmetik_bakim",
        "kuru_temizleme",
        "terzi_dikis",
        "diger_kisisel_bakim",
    ],
    "ulasim": [
        "toplu_tasima",
        "taksi_ozel_tasimacilik",
        "sehirlerarasi_ulasim",
        "arac_bakim_servis",
        "otopark",
        "otoyol_kopru_gecis",
        "mikromobilite_hizmeti",
        "arac_kiralama",
        "otomotiv_aksesuar",
        "diger_ulasim",
    ],
    "seyahat": [
        "otel_konaklama",
        "seyahat_acentesi_tur",
        "duty_free",
        "ucak_havayolu",
        "diger_seyahat",
    ],
    "eglence_kultur": [
        "sinema_tiyatro",
        "konser_festival",
        "muze_sergi",
        "hayvanat_bahcesi_akvaryum",
        "sans_oyunlari_bahis",
        "spor_etkinlik",
        "eglence_parki",
        "diger_eglence_kultur",
    ],
    "fatura_abonelik": [
        "elektrik_fatura",
        "dogalgaz_fatura",
        "su_fatura",
        "tv_yayin",
        "aidat_odeme",
        "dijital_abonelik",
        "gsm_fatura",
        "internet_fatura",
        "diger_fatura_abonelik",
    ],
    "finans_sigorta": [
        "nakit_cekim_transfer",
        "yatirim",
        "bagis",
        "konut_kredisi_odeme",
        "tasit_kredisi_odeme",
        "ihtiyac_kredisi_odeme",
        "kart_ekstre_odemesi",
        "bireysel_emeklilik_odeme",
        "sigorta",
        "kamu_yukumluluk_odeme",
        "diger_finans_sigorta",
    ],
    "akaryakit": [
        "akaryakit_istasyonu",
        "elektrikli_arac_sarj",
        "diger_akaryakit",
    ],
    "egitim": [
        "okul_universite",
        "kurs_egitim_merkezi",
        "cevrimici_egitim",
        "kitap_kirtasiye",
        "sinav_sertifika",
        "diger_egitim",
    ],
    "diger": [
        "tanimlanamayan_isyeri",        # router/cascade flag — not learned by any model
        "iade_iptal",
    ],
}

# ---------------------------------------------------------------------------
# L2 labels not included in model training (router/cascade-generated flags)
# ---------------------------------------------------------------------------
INACTIVE_L2: frozenset[str] = frozenset({
    "tanimlanamayan_isyeri",
})

# ---------------------------------------------------------------------------
# Derived constants
# ---------------------------------------------------------------------------

# Reverse lookup: L2 → L1
L2_TO_L1: dict[str, str] = {
    l2: l1
    for l1, l2_list in TAXONOMY.items()
    for l2 in l2_list
}

# L2 labels the model is trained to predict
ACTIVE_L2: list[str] = [
    l2
    for l2_list in TAXONOMY.values()
    for l2 in l2_list
    if l2 not in INACTIVE_L2
]

ALL_L1: list[str] = list(TAXONOMY.keys())
ALL_L2: list[str] = [l2 for l2_list in TAXONOMY.values() for l2 in l2_list]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_l1(l2: str) -> str | None:
    """Return the L1 label for a given L2 label."""
    return L2_TO_L1.get(l2)


def get_l2_list(l1: str) -> list[str]:
    """Return all L2 labels for a given L1 label."""
    return TAXONOMY.get(l1, [])


def is_valid_l2(l2: str) -> bool:
    """Return True if l2 exists in the taxonomy."""
    return l2 in L2_TO_L1


def is_active_l2(l2: str) -> bool:
    """Return True if l2 is included in model training."""
    return l2 in ACTIVE_L2 and l2 not in INACTIVE_L2


def get_diger_l2(l1: str) -> str:
    """Return the fallback L2 for a given L1 (diger_<l1>, or tanimlanamayan_isyeri)."""
    candidate = f"diger_{l1}"
    if candidate in L2_TO_L1:
        return candidate
    return "tanimlanamayan_isyeri"


def taxonomy_summary() -> str:
    """Return a formatted summary table of the taxonomy."""
    lines = [f"{'L1':<25} {'L2 Sayısı':>10}"]
    lines.append("-" * 37)
    for l1, l2_list in TAXONOMY.items():
        active = [l for l in l2_list if l not in INACTIVE_L2]
        lines.append(f"{l1:<25} {len(active):>10}")
    lines.append("-" * 37)
    total_active = len(ACTIVE_L2)
    lines.append(f"{'TOPLAM Aktif L2':<25} {total_active:>10}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(taxonomy_summary())
    print(f"\nÖrnek: get_l1('eczane') → {get_l1('eczane')}")
    print(f"Örnek: get_l1('fast_food') → {get_l1('fast_food')}")
    print(f"Aktif L2 sayısı: {len(ACTIVE_L2)}")
