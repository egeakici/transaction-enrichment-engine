"""
routes/statement_routes.py — Kredi Kartı Ekstre Rotaları

Endpoint'ler:
    GET  /statement/           — Ekstre görüntüleme sayfası
    POST /statement/parse      — Metni ayrıştır, tarihe göre sırala
    POST /statement/classify   — Metni ayrıştır + ML ile L1/L2 sınıflandır
"""

import re
from datetime import datetime
from pathlib import Path

from flask import Blueprint, jsonify, render_template, request

ROOT_DIR   = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT_DIR / "models"

statement_bp = Blueprint("statement", __name__, template_folder="../templates")

# ---------------------------------------------------------------------------
# Tarih ayrıştırma yardımcıları
# ---------------------------------------------------------------------------

_DATE_PATTERNS = [
    # DD.MM.YYYY veya DD/MM/YYYY veya DD-MM-YYYY
    (re.compile(r"(\d{1,2})[.\/\-](\d{1,2})[.\/\-](\d{2,4})"), "%d %m %Y"),
    # YYYY-MM-DD
    (re.compile(r"(\d{4})[.\/\-](\d{1,2})[.\/\-](\d{1,2})"), "%Y %m %d"),
]


def _parse_date(raw: str) -> datetime | None:
    """Ham tarih string'inden datetime nesnesi döner; ayrıştırılamazsa None."""
    raw = raw.strip()
    for pattern, fmt in _DATE_PATTERNS:
        m = pattern.match(raw)
        if m:
            try:
                parts = list(m.groups())
                # 2 haneli yılı 4 haneye çevir
                if len(parts[2]) == 2:
                    parts[2] = "20" + parts[2]
                return datetime.strptime(" ".join(parts), fmt)
            except ValueError:
                continue
    return None


def _parse_amount(raw: str) -> float | None:
    """
    '1.234,56' veya '1234.56' veya '1,234.56' gibi formatlardaki tutarı float'a çevirir.
    Ekstre sütununda TL/$ işaretleri temizlenir.
    """
    raw = raw.strip().replace("TL", "").replace("₺", "").replace("$", "").replace("€", "").strip()
    # Türk formatı: nokta binlik ayırıcı, virgül ondalık
    if re.match(r"^\d{1,3}(\.\d{3})*(,\d+)?$", raw):
        raw = raw.replace(".", "").replace(",", ".")
    # İngiliz formatı: virgül binlik ayırıcı, nokta ondalık
    elif re.match(r"^\d{1,3}(,\d{3})*(\.\d+)?$", raw):
        raw = raw.replace(",", "")
    else:
        raw = raw.replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_statement_text(text: str) -> list[dict]:
    """
    Metin formatındaki ekstre satırlarını ayrıştırır.

    Beklenen format (tab veya çoklu boşluk ayırıcı):
        DD.MM.YYYY   İşlem Açıklaması   1.234,56

    Her satırı {date_raw, date, description, amount_raw, amount} dict'ine dönüştürür.
    Ayrıştırılamayan satırlar atlanır.
    """
    rows = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # Tab veya 2+ boşlukla bölün
        parts = re.split(r"\t| {2,}", line)
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) < 3:
            # Tek boşlukla da dene
            parts = line.split()

        if len(parts) < 2:
            continue

        # İlk parça tarih mi?
        date_obj = _parse_date(parts[0])
        if date_obj is None:
            continue

        # Son parça tutar mı?
        amount    = None
        amount_raw = ""
        desc_parts = []

        # Sondan başa doğru tutar ara
        for i in range(len(parts) - 1, 0, -1):
            amount = _parse_amount(parts[i])
            if amount is not None:
                amount_raw = parts[i]
                desc_parts = parts[1:i]
                break

        if amount is None:
            desc_parts = parts[1:]

        description = " ".join(desc_parts).strip() if desc_parts else (parts[1] if len(parts) > 1 else "")

        rows.append({
            "date_raw":    parts[0],
            "date":        date_obj,
            "description": description,
            "amount_raw":  amount_raw,
            "amount":      amount,
        })

    return rows


def _rows_to_json(rows: list[dict]) -> list[dict]:
    """datetime → string dönüşümü yaparak JSON-serileştirilebilir liste üretir."""
    return [
        {
            "date":        r["date"].strftime("%d.%m.%Y") if r["date"] else r["date_raw"],
            "description": r["description"],
            "amount":      r["amount"],
            "amount_raw":  r["amount_raw"],
        }
        for r in rows
    ]


def _get_router(inference_mode: str = "cascade"):
    """HybridRouter örneğini döndürür; ml_routes cache'ini kullanır."""
    try:
        from routes.ml_routes import _get_or_load_router
        return _get_or_load_router(inference_mode)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Rotalar
# ---------------------------------------------------------------------------

@statement_bp.route("/")
def index():
    return render_template("statement/index.html")


@statement_bp.route("/parse", methods=["POST"])
def parse():
    """
    Ekstre metnini ayrıştır ve tarihe göre sırala.
    Sınıflandırma yapılmaz — sadece ham satırlar döner.
    """
    data = request.get_json(force=True)
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "Ekstre metni boş olamaz"}), 400

    rows = _parse_statement_text(text)
    rows.sort(key=lambda r: r["date"] or datetime.min)

    return jsonify({
        "mode":   "sorted",
        "count":  len(rows),
        "rows":   _rows_to_json(rows),
    })


@statement_bp.route("/classify", methods=["POST"])
def classify():
    """
    Ekstre metnini ayrıştır, ML modeli ile L1/L2 sınıflandır,
    L1 kategorilerine göre grupla.
    """
    data           = request.get_json(force=True)
    text           = data.get("text", "")
    inference_mode = data.get("inference_mode", "cascade")

    if not text.strip():
        return jsonify({"error": "Ekstre metni boş olamaz"}), 400

    rows = _parse_statement_text(text)
    rows.sort(key=lambda r: r["date"] or datetime.min)

    router = _get_router(inference_mode)
    if router is None:
        return jsonify({"error": "Model bulunamadı. Önce eğitim yapın."}), 500

    # Batch tahmin
    merchants = [r["description"] for r in rows]
    results   = router.enrich_batch(merchants)

    # Satırları zenginleştir
    enriched = []
    for row, res in zip(rows, results):
        enriched.append({
            "date":        row["date"].strftime("%d.%m.%Y") if row["date"] else row["date_raw"],
            "description": row["description"],
            "amount":      row["amount"],
            "amount_raw":  row["amount_raw"],
            "category_l1": res.category_l1 or "diger",
            "category_l2": res.category_l2 or "",
            "confidence":  round(res.confidence, 3),
            "source":      res.source,
        })

    # L1'e göre grupla
    groups: dict[str, list] = {}
    for item in enriched:
        l1 = item["category_l1"]
        groups.setdefault(l1, []).append(item)

    # kart_ekstre_odeme işlemleri görünsün ama toplam hesabına katılmasın
    EXCLUDE_FROM_TOTAL = {"kart_ekstre_odeme"}

    # Her grubun toplam tutarını hesapla
    group_summaries = []
    for l1, items in sorted(groups.items()):
        total = sum(
            i["amount"] for i in items
            if i["amount"] is not None and i["category_l2"] not in EXCLUDE_FROM_TOTAL
        )
        group_summaries.append({
            "l1":    l1,
            "items": items,
            "total": round(total, 2),
            "count": len(items),
        })

    # Genel toplam (kart_ekstre_odeme hariç)
    grand_total = sum(
        i["amount"] for i in enriched
        if i["amount"] is not None and i["category_l2"] not in EXCLUDE_FROM_TOTAL
    )

    return jsonify({
        "mode":        "classified",
        "count":       len(enriched),
        "groups":      group_summaries,
        "grand_total": round(grand_total, 2),
    })
