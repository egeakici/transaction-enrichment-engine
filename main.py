"""
CLI entry point for the Turkish credit card transaction enrichment pipeline.

Commands:
    predict   — single merchant prediction
    batch     — bulk prediction from CSV
    train     — fine-tune BERTurk (--mode l1|l2|cascade)
    evaluate  — evaluate model on test set
    cv        — cross-validation (--mode l1|l2)
    tsne      — t-SNE embedding visualization
    taxonomy  — print taxonomy summary
    demo      — run sample predictions
    explain   — show preprocessor normalization steps

Training mode (--mode):
    cascade  — hierarchical: L1 model + per-L1 L2 models (default)
    l2       — single model, 101 L2 classes
    l1       — single model, 16 L1 classes

Inference mode (--inference-mode, for predict/batch/demo):
    cascade  — L1 → L1-specific L2 model (default)
    l2       — direct L2 prediction
    l1       — direct L1 prediction

Usage:
    python main.py train data/master.csv                  # cascade (default)
    python main.py train data/master.csv --mode l2        # single L2
    python main.py train data/master.csv --mode l1        # single L1
    python main.py predict "MIGROS HYPERMARKET"                        # cascade (default)
    python main.py predict "MIGROS HYPERMARKET" --inference-mode l2   # single L2
    python main.py evaluate data/master_test_oov.csv
    python main.py batch data/input.csv --output data/output.csv
    python main.py cv --mode l1
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from taxonomy import taxonomy_summary, ACTIVE_L2
from preprocessor import normalize, explain as explain_normalize
from entity_db import EntityDB, get_db
from model import (
    MerchantClassifier, CascadeClassifier,
    csv_to_training_file, build_training_file,
    detect_mode_from_model,
)
from router import HybridRouter, EnrichmentResult

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# Directory constants
# ---------------------------------------------------------------------------
ROOT_DIR        = Path(__file__).parent
DATA_DIR        = ROOT_DIR / "data"
MODELS_DIR      = ROOT_DIR / "models"
PROCESSED_DIR   = DATA_DIR / "processed"
EXPERIMENTS_DIR = MODELS_DIR / "experiments"
PLOTS_DIR       = MODELS_DIR / "plots"

for _d in [DATA_DIR, MODELS_DIR, PROCESSED_DIR, EXPERIMENTS_DIR, PLOTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

TRAIN_TXT    = PROCESSED_DIR / "train.txt"
TEST_TXT     = PROCESSED_DIR / "test.txt"
MODEL_DIR_L1 = MODELS_DIR / "berturk_mode_l1"
MODEL_DIR_L2 = MODELS_DIR / "berturk_mode_l2"
CASCADE_DIR  = MODELS_DIR / "berturk_mode_cascade"
MASTER_CSV   = DATA_DIR / "master.csv"


# ---------------------------------------------------------------------------
# Helper: mode detection
# ---------------------------------------------------------------------------

def _resolve_mode(args) -> str:
    """Return the inference mode from args or saved model metrics."""
    if hasattr(args, "mode") and args.mode:
        return args.mode
    model_dir = Path(args.model_dir) if hasattr(args, "model_dir") and args.model_dir else MODEL_DIR_L2
    return detect_mode_from_model(model_dir)


# ---------------------------------------------------------------------------
# Helper: pretty-print result
# ---------------------------------------------------------------------------

def _print_result(result: EnrichmentResult) -> None:
    flag_str    = f"  ⚑  {result.flag}" if result.flag else ""
    source_icon = {"entity_db": "🗄 ", "ml_model": "🤖", "fallback": "⚠ "}.get(result.source, "  ")
    l2_line     = f"       L2 : {result.category_l2}\n" if result.category_l2 else ""

    print(
        f"  {source_icon} '{result.merchant_raw}'\n"
        f"{l2_line}"
        f"       L1 : {result.category_l1}\n"
        f"     Güven: {result.confidence:.3f}{flag_str}"
    )
    if result.top_k and len(result.top_k) > 1:
        alts = "  |  ".join(f"{lbl} ({c:.2f})" for lbl, c in result.top_k[1:])
        print(f"      Alt : {alts}")
    print()


# ---------------------------------------------------------------------------
# Command: predict
# ---------------------------------------------------------------------------

def cmd_predict(args) -> None:
    router = _load_router(args)
    result = router.enrich(args.merchant, top_k=3)
    _print_result(result)


# ---------------------------------------------------------------------------
# Command: batch
# ---------------------------------------------------------------------------

def cmd_batch(args) -> None:
    input_path  = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".enriched.csv")
    raw_col     = args.col or "merchant_raw"
    router      = _load_router(args)

    merchants: list[str] = []
    rows: list[dict]     = []

    with open(input_path, encoding="utf-8", newline="") as f:
        reader     = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            merchants.append(row.get(raw_col, ""))
            rows.append(row)

    results = router.enrich_batch(merchants)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_fields    = fieldnames + [
        "merchant_normalized", "category_l2", "category_l1",
        "confidence", "source", "flag",
    ]
    seen          = set()
    unique_fields = [f for f in new_fields if not (f in seen or seen.add(f))]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=unique_fields, extrasaction="ignore")
        writer.writeheader()
        for row, res in zip(rows, results):
            row.update({
                "merchant_normalized": res.merchant_normalized,
                "category_l2":         res.category_l2 or "",
                "category_l1":         res.category_l1,
                "confidence":          res.confidence,
                "source":              res.source,
                "flag":                res.flag or "",
            })
            writer.writerow(row)

    stats = router.get_stats()
    print(f"\n✅ Tamamlandı: {len(results)} satır → {output_path}")
    print(f"   Entity DB hit : {stats.get('entity_db_hits', 0)}")
    print(f"   ML tahmini   : {stats.get('ml_predictions', 0)}")
    print(f"   Düşük güven  : {stats.get('low_confidence', 0)}")

    flushed = router.flush_auto_learn()
    if flushed:
        print(f"   Auto-learn   : {flushed} yeni işyeri catalog.json'a eklendi")


# ---------------------------------------------------------------------------
# Command: train
# ---------------------------------------------------------------------------

def cmd_train(args) -> None:
    mode      = args.mode or "l2"
    source    = Path(args.source) if args.source else MASTER_CSV
    _default_dirs = {"cascade": CASCADE_DIR, "l1": MODEL_DIR_L1, "l2": MODEL_DIR_L2}
    model_out = Path(args.model_dir) if args.model_dir else _default_dirs[mode]

    if not source.exists():
        print(f"❌ Kaynak bulunamadı: {source}")
        sys.exit(1)

    hp = {}
    if args.lr:
        hp["lr"] = float(args.lr)
    if args.epoch:
        hp["epoch"] = int(args.epoch)

    # CASCADE mode
    if mode == "cascade":
        print(f"\n🎯 Eğitim modu: CASCADE (L1 → L1'e özgü L2 hiyerarşisi)")
        print(f"   Kaynak: {source}")

        records: list[dict] = []
        raw_col   = args.raw_col   or "merchant_raw"
        label_col = args.label_col or "category_l2"
        with open(source, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw   = row.get(raw_col,   "").strip()
                label = row.get(label_col, "").strip()
                norm  = normalize(raw)
                if norm and label:
                    records.append({"merchant_normalized": norm, "category_l2": label})

        print(f"   {len(records)} örnek yüklendi.\n")

        cascade = CascadeClassifier(hyperparams=hp if hp else None)
        cascade.train(records)
        cascade.save(model_out)

        print(f"\n✅ Cascade model kaydedildi: {model_out}")
        print(f"   L2 model sayısı : {len(cascade._l2_clfs)}")
        print(f"   Toplam sınıf    : {cascade.n_labels}")
        return

    # L1 / L2 mode
    print(f"\n🎯 Eğitim modu: {mode.upper()} "
          f"({'103 L2 sınıfı' if mode == 'l2' else '16 L1 sınıfı'})")

    if source.suffix.lower() == ".csv":
        print(f"📄 CSV → Eğitim dosyası [{mode.upper()}] dönüşümü: {source}")
        n = csv_to_training_file(
            csv_path=source,
            output_path=TRAIN_TXT,
            raw_col=args.raw_col or "merchant_raw",
            label_col=args.label_col or "category_l2",
            mode=mode,
        )
        print(f"   {n} satır eğitim dosyasına yazıldı: {TRAIN_TXT}")
    else:
        print(f"📄 Mevcut train.txt kullanılıyor: {source}")
        print(f"   ⚠  Bu dosyanın {mode.upper()} etiketleri içerdiğinden emin olun.")

    train_file = TRAIN_TXT if source.suffix.lower() == ".csv" else source

    clf = MerchantClassifier(mode=mode, hyperparams=hp if hp else None)
    clf.train(train_file)
    clf.save(model_out)
    print(f"\n✅ Model kaydedildi [{mode.upper()}]: {model_out}")
    print(f"   Sınıf sayısı: {clf.n_labels}")

    if getattr(args, "curve", False):
        print(f"\n📈 Eğitim eğrisi çiziliyor...")
        from visualizer import plot_training_curve
        plot_training_curve(clf, plots_dir=PLOTS_DIR)
        print(f"   → {PLOTS_DIR}/training_curve.png")

    if TEST_TXT.exists():
        print(f"\n📊 Test seti üzerinde hızlı değerlendirme ({TEST_TXT})...")
        result = clf.evaluate(TEST_TXT)
        _print_evaluation_report(result, TEST_TXT.name, mode=mode, per_class=False)
        from visualizer import plot_evaluation
        plot_evaluation(result, PLOTS_DIR)
        _save_eval_result(result, "train_eval", mode=mode)


# ---------------------------------------------------------------------------
# Command: evaluate
# ---------------------------------------------------------------------------

def _print_evaluation_report(
    result, source_name: str, mode: str, per_class: bool = False
) -> None:
    """Print per-mode evaluation report.

    mode="l2": shows both L2 and L1 metrics side-by-side.
    mode="l1": shows L1 metrics only.
    """
    W = 64

    print(f"\n{'═' * W}")
    print(f"  📊  Değerlendirme Raporu [{mode.upper()}] — {source_name}")
    print(f"{'═' * W}")

    if mode in ("l2", "cascade"):
        # L2 and L1 side by side
        print(f"\n  {'Metrik':<22} {'L2 (İnce Tahmin)':>18}   {'L1 (Üst Kategori)':>18}")
        print(f"  {'─' * 62}")
        print(f"  {'Örnek sayısı':<22} {result.n_samples:>18}   {result.n_samples:>18}")
        print(f"  {'Accuracy':<22} {result.accuracy:>17.4f}   {result.accuracy_l1:>17.4f}")
        print(f"  {'F1 Macro':<22} {result.f1_macro:>17.4f}   {result.f1_macro_l1:>17.4f}")
        print(f"  {'F1 Weighted':<22} {result.f1_weighted:>17.4f}   {result.f1_weighted_l1:>17.4f}")

        # L1 per-category table
        if result.per_class_l1:
            print(f"\n  {'─' * W}")
            print(f"  L1 Kategori Performansı")
            print(f"  {'─' * W}")
            print(f"  {'L1 Kategori':<28} {'F1':>6}  {'Precision':>9}  {'Recall':>7}  {'n':>5}")
            print(f"  {'─' * W}")
            for l1, m in sorted(result.per_class_l1.items(),
                                 key=lambda x: x[1].get("f1-score", 0.0)):
                f1  = m.get("f1-score",  0.0)
                pre = m.get("precision", 0.0)
                rec = m.get("recall",    0.0)
                sup = int(m.get("support", 0))
                flag = "  ⚠" if f1 < 0.50 else ("  ✓" if f1 >= 0.80 else "")
                print(f"  {l1:<28} {f1:>6.3f}  {pre:>9.3f}  {rec:>7.3f}  {sup:>5}{flag}")

        # L2 per-category table (optional)
        if per_class and result.per_class:
            print(f"\n  {'─' * W}")
            print(f"  L2 Kategori Performansı")
            print(f"  {'─' * W}")
            print(f"  {'L2 Kategori':<38} {'F1':>6}  {'Precision':>9}  {'Recall':>7}  {'n':>5}")
            print(f"  {'─' * W}")
            for l2, m in sorted(result.per_class.items(),
                                 key=lambda x: x[1].get("f1-score", 0.0)):
                if l2 in ("accuracy", "macro avg", "weighted avg"):
                    continue
                f1  = m.get("f1-score",  0.0)
                pre = m.get("precision", 0.0)
                rec = m.get("recall",    0.0)
                sup = int(m.get("support", 0))
                flag = "  ⚠" if f1 < 0.40 else ("  ✓" if f1 >= 0.80 else "")
                print(f"  {l2:<38} {f1:>6.3f}  {pre:>9.3f}  {rec:>7.3f}  {sup:>5}{flag}")

    else:
        # L1 only
        print(f"\n  {'Metrik':<22} {'L1 (Doğrudan Tahmin)':>22}")
        print(f"  {'─' * 46}")
        print(f"  {'Örnek sayısı':<22} {result.n_samples:>22}")
        print(f"  {'Accuracy':<22} {result.accuracy:>21.4f}")
        print(f"  {'F1 Macro':<22} {result.f1_macro:>21.4f}")
        print(f"  {'F1 Weighted':<22} {result.f1_weighted:>21.4f}")

        if result.per_class_l1:
            print(f"\n  {'─' * W}")
            print(f"  L1 Kategori Performansı")
            print(f"  {'─' * W}")
            print(f"  {'L1 Kategori':<28} {'F1':>6}  {'Precision':>9}  {'Recall':>7}  {'n':>5}")
            print(f"  {'─' * W}")
            for l1, m in sorted(result.per_class_l1.items(),
                                 key=lambda x: x[1].get("f1-score", 0.0)):
                f1  = m.get("f1-score",  0.0)
                pre = m.get("precision", 0.0)
                rec = m.get("recall",    0.0)
                sup = int(m.get("support", 0))
                flag = "  ⚠" if f1 < 0.50 else ("  ✓" if f1 >= 0.80 else "")
                print(f"  {l1:<28} {f1:>6.3f}  {pre:>9.3f}  {rec:>7.3f}  {sup:>5}{flag}")

    print(f"\n{'═' * W}\n")


def cmd_evaluate(args) -> None:
    source    = Path(args.test) if args.test else TEST_TXT
    model_dir = Path(args.model_dir) if args.model_dir else None

    if not source.exists():
        print(f"❌ Test dosyası bulunamadı: {source}")
        sys.exit(1)

    # Detect cascade vs single model
    cascade_meta = CASCADE_DIR / CascadeClassifier.META_FILE
    use_cascade  = model_dir is None and cascade_meta.exists()

    if use_cascade:
        mode = "cascade"
        print(f"\n🎯 Algılanan model modu: CASCADE")
    else:
        resolved_dir = model_dir or (
            MODEL_DIR_L2 if (MODEL_DIR_L2 / "metrics.json").exists() else MODEL_DIR_L1
        )
        mode         = detect_mode_from_model(resolved_dir)
        print(f"\n🎯 Algılanan model modu: {mode.upper()}")

    # Convert test file to training format
    test_file = source
    if source.suffix.lower() == ".csv":
        ft_mode = "l2" if use_cascade else mode
        print(f"📄 Test CSV → Eğitim dosyası [{ft_mode.upper()}] dönüşümü yapılıyor...")
        test_file = PROCESSED_DIR / "test_converted.txt"
        csv_to_training_file(
            csv_path=source,
            output_path=test_file,
            raw_col="merchant_raw",
            label_col="category_l2",
            mode=ft_mode,
        )

    # Load model and evaluate
    if use_cascade:
        clf    = CascadeClassifier()
        clf.load(CASCADE_DIR)
        result = clf.evaluate(test_file)
        _print_evaluation_report(result, source.name, mode="cascade", per_class=args.per_class)
        _save_eval_result(result, "eval", mode="cascade")
    else:
        resolved_dir = model_dir or (MODEL_DIR_L1 if mode == "l1" else MODEL_DIR_L2)
        clf          = MerchantClassifier(mode=mode)
        clf.load(resolved_dir)
        result = clf.evaluate(test_file)
        _print_evaluation_report(result, source.name, mode=mode, per_class=args.per_class)
        _save_eval_result(result, "eval", mode=mode)

    # Plot results
    from visualizer import plot_evaluation, plot_training_curve
    plots_ok = plot_evaluation(result, PLOTS_DIR)
    if plots_ok:
        print(f"\n📈 Grafikler kaydedildi: {PLOTS_DIR}")
        print(f"   • confusion_matrix_l1.png")
        if result.per_class_l1:
            print(f"   • per_class_f1_l1.png")
    else:
        print(f"\n⚠  Grafik üretilemedi — matplotlib kurulu değil.")
        print(f"   Kurmak için: pip install matplotlib")

    # Training curve (single model only)
    if not use_cascade and TRAIN_TXT.exists():
        max_ep      = clf._hyperparams.get("epoch", 25)
        checkpoints = [e for e in [1, 5, 10, 25, 50, 75, 100] if e <= max_ep] or [max_ep]
        print(f"\n📈 Accuracy-epoch eğrisi hesaplanıyor (max epoch={max_ep}, checkpoints={checkpoints})...")
        curve_ok = plot_training_curve(clf, TRAIN_TXT, PLOTS_DIR, epoch_checkpoints=checkpoints)
        if curve_ok:
            print(f"   • training_curve.png → {PLOTS_DIR}")
        else:
            print(f"   ⚠  Epoch eğrisi üretilemedi — eğitim geçmişi bu modelde kayıtlı değil.")
            print(f"      Bu model eski bir sürümde eğitildi; training_history metrics.json'a yazılmıyordu.")
            print(f"      Grafik almak için modeli yeniden eğitin: python main.py train")


# ---------------------------------------------------------------------------
# Command: tsne
# ---------------------------------------------------------------------------

def cmd_tsne(args) -> None:
    model_dir  = Path(args.model_dir) if args.model_dir else MODEL_DIR_L2
    train_file = Path(args.train) if args.train else TRAIN_TXT
    mode       = detect_mode_from_model(model_dir)

    if not (model_dir / "metrics.json").exists():
        print(f"❌ Model bulunamadı: {model_dir}")
        print(f"   Önce eğitin: python main.py train --mode {mode}")
        sys.exit(1)

    if not train_file.exists():
        print(f"❌ Eğitim dosyası bulunamadı: {train_file}")
        sys.exit(1)

    clf = MerchantClassifier(mode=mode)
    clf.load(model_dir)

    n_samples  = int(args.samples)    if args.samples    else 1500
    perplexity = int(args.perplexity) if args.perplexity else 40

    print(f"\n🔍 t-SNE Görselleştirme [{mode.upper()}]")
    print(f"   Model      : {model_dir}")
    print(f"   Eğitim     : {train_file}")
    print(f"   Örneklem   : {n_samples} (kategori bazlı dengeli)")
    print(f"   Perplexity : {perplexity}")
    print(f"   Çıktı      : {PLOTS_DIR}/tsne_embeddings.png\n")

    from visualizer import plot_tsne
    ok = plot_tsne(
        classifier=clf,
        train_path=train_file,
        plots_dir=PLOTS_DIR,
        n_samples=n_samples,
        perplexity=perplexity,
    )

    if ok:
        print(f"\n✅ t-SNE grafiği kaydedildi: {PLOTS_DIR / 'tsne_embeddings.png'}")
        print(f"   • Ayrışmış kümeler → model bu kategorileri iyi ayırt ediyor")
        print(f"   • İç içe geçmiş kümeler → yüksek hata oranının nedeni bu kategoriler")
        print(f"   • ⚠  işaretli noktalar → düşük F1 kategorileri")
    else:
        print(f"\n❌ t-SNE üretilemedi.")
        print(f"   Gerekli paketler: pip install scikit-learn matplotlib")


# ---------------------------------------------------------------------------
# Command: cv
# ---------------------------------------------------------------------------

def cmd_cv(args) -> None:
    mode   = args.mode or "l2"
    source = Path(args.source) if args.source else MASTER_CSV
    if not source.exists():
        print(f"❌ Kaynak bulunamadı: {source}")
        sys.exit(1)

    records: list[dict] = []
    with open(source, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw   = row.get("merchant_raw", "").strip()
            label = row.get("category_l2",  "").strip()
            norm  = normalize(raw)
            if norm and label:
                records.append({"merchant_normalized": norm, "category_l2": label})

    n_splits = int(args.folds) if args.folds else 5
    print(
        f"\n🔁 {n_splits}-Fold Stratified Cross-Validation "
        f"[{mode.upper()}] ({len(records)} örnek)..."
    )

    clf     = MerchantClassifier(mode=mode)
    summary = clf.cross_validate(records, n_splits=n_splits)

    print(f"\n{'='*45}")
    print(f"  Mod              : {mode.upper()}")
    print(f"  F1-Macro (ort.)  : {summary['mean_f1_macro']:.4f}")
    print(f"  F1-Macro (std)   : {summary['std_f1_macro']:.4f}")
    print(f"  Accuracy (ort.)  : {summary['mean_accuracy']:.4f}")
    print(f"  Fold F1 skorları : {[f'{s:.3f}' for s in summary['fold_f1_scores']]}")
    print(f"{'='*45}\n")

    exp_file = EXPERIMENTS_DIR / "cv_results.jsonl"
    with open(exp_file, "a", encoding="utf-8") as f:
        import time
        summary["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        summary["source"]    = str(source)
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(f"  → Sonuç kaydedildi: {exp_file}")


# ---------------------------------------------------------------------------
# Command: taxonomy
# ---------------------------------------------------------------------------

def cmd_taxonomy(args) -> None:
    print("\n" + taxonomy_summary())
    print(f"\nToplam aktif L2: {len(ACTIVE_L2)}")


# ---------------------------------------------------------------------------
# Command: demo
# ---------------------------------------------------------------------------

def cmd_demo(args) -> None:
    router = _load_router(args)
    mode   = detect_mode_from_model(MODEL_DIR)

    demo_cases = [
        "KARŞI KURUYEMİŞ",
        "MARMARAY GAZETE BAYİ",
        "KAPADOKYA ŞELALE LOKANTASI",
        "*** STARBUCKS ISTANBUL ***",
        "POS/001 MİGROS HYPERMARKet",
        "NETFLIX ABONELIK",
        "SHELL PETROL SUBE",
        "YEMEKSEPETI ODEME",
        "",
        "AB",
    ]

    print("\n" + "=" * 70)
    print(f"  TR Kredi Kartı Zenginleştirme — DEMO  [{mode.upper()} MODEL]")
    print("=" * 70 + "\n")

    for merchant in demo_cases:
        result = router.enrich(merchant, top_k=3)
        _print_result(result)

    print("─" * 70)
    stats = router.get_stats()
    print(f"  Toplam   : {stats['total']}")
    print(f"  DB hit   : {stats['entity_db_hits']} ({stats.get('entity_db_hit_rate', 0):.0%})")
    print(f"  ML       : {stats['ml_predictions']} ({stats.get('ml_rate', 0):.0%})")
    print(f"  Düşük güv: {stats['low_confidence']}")


# ---------------------------------------------------------------------------
# Command: explain
# ---------------------------------------------------------------------------

def cmd_explain(args) -> None:
    explain_normalize(args.merchant)


# ---------------------------------------------------------------------------
# Helper: load router
# ---------------------------------------------------------------------------

def _load_router(args=None) -> HybridRouter:
    """Load the appropriate router based on args and available model files.

    Priority: explicit --inference-mode arg → cascade_meta.json → l1/l2 fallback.
    """
    from model import CascadeClassifier

    # Explicit model directory override
    explicit_model_dir = (
        Path(args.model_dir)
        if (args and hasattr(args, "model_dir") and args.model_dir)
        else None
    )

    # inference_mode: from arg or auto-detect
    inference_mode = (
        getattr(args, "inference_mode", None)
        or "cascade"
    )

    # Use cascade if cascade_meta.json exists
    cascade_meta = CASCADE_DIR / CascadeClassifier.META_FILE
    if inference_mode == "cascade" and explicit_model_dir is None:
        if cascade_meta.exists():
            router = HybridRouter(inference_mode="cascade")
            router.load_model(CASCADE_DIR)
            return router
        else:
            logger.warning(
                "Cascade model bulunamadı. Eski modele geçiliyor. "
                "Cascade eğitmek için: python main.py train --mode cascade"
            )
            if (MODEL_DIR_L2 / "metrics.json").exists():
                inference_mode = "l2"
            elif (MODEL_DIR_L1 / "metrics.json").exists():
                inference_mode = "l1"
            else:
                inference_mode = detect_mode_from_model(MODEL_DIR_L2)

    # L1 / L2 fallback
    model_dir = explicit_model_dir or (MODEL_DIR_L1 if inference_mode == "l1" else MODEL_DIR_L2)
    mode      = inference_mode if inference_mode in ("l1", "l2") else detect_mode_from_model(model_dir)
    router    = HybridRouter(inference_mode=mode)

    if model_dir.exists() and (model_dir / "metrics.json").exists():
        router.load_model(model_dir)
        logger.info(f"Model yüklendi [{mode.upper()}]: {model_dir}")
    else:
        logger.warning(
            "Eğitilmiş model bulunamadı. Sadece Entity DB kullanılacak. "
            "Eğitmek için: python main.py train [--mode cascade|l1|l2]"
        )
    return router


def _save_eval_result(result, tag: str, mode: str = "l2") -> None:
    import time
    out  = EXPERIMENTS_DIR / f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    data: dict = {"mode": mode, "n_samples": result.n_samples,
                  "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if mode == "l2":
        data["l2"] = {
            "accuracy": result.accuracy, "f1_macro": result.f1_macro,
            "f1_weighted": result.f1_weighted,
        }
        data["l1"] = {
            "accuracy": result.accuracy_l1, "f1_macro": result.f1_macro_l1,
            "f1_weighted": result.f1_weighted_l1,
        }
    else:
        data["l1"] = {
            "accuracy": result.accuracy, "f1_macro": result.f1_macro,
            "f1_weighted": result.f1_weighted,
        }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Değerlendirme sonucu kaydedildi: {out}")


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="TR Kredi Kartı İşlem Zenginleştirme Pipeline — L1/L2 Seçilebilir",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Mod seçimi:\n"
            "  train komutunda --mode ile seçilir (varsayılan: cascade)\n"
            "  Diğer komutlar modu kaydedilmiş metrics.json'dan otomatik algılar."
        ),
    )
    sub = parser.add_subparsers(dest="command", metavar="KOMUT")
    sub.required = True

    # predict
    p_pred = sub.add_parser("predict", help="Tek merchant tahmin et")
    p_pred.add_argument("merchant",    help="İşyeri ismi (tırnak içinde)")
    p_pred.add_argument("--model-dir", help="Model dizini")
    p_pred.add_argument(
        "--inference-mode", dest="inference_mode",
        choices=["cascade", "l1", "l2"], default=None,
        help="Tahmin mimarisi: 'cascade' (varsayılan), 'l2', 'l1'",
    )
    p_pred.set_defaults(func=cmd_predict)

    # batch
    p_batch = sub.add_parser("batch", help="CSV dosyasından toplu tahmin")
    p_batch.add_argument("input",      help="Girdi CSV dosyası")
    p_batch.add_argument("--output", "-o", help="Çıktı CSV dosyası")
    p_batch.add_argument("--col",      help="Merchant kolon adı (varsayılan: merchant_raw)")
    p_batch.add_argument("--model-dir",help="Model dizini")
    p_batch.add_argument(
        "--inference-mode", dest="inference_mode",
        choices=["cascade", "l1", "l2"], default=None,
        help="Tahmin mimarisi: 'cascade' (varsayılan), 'l2', 'l1'",
    )
    p_batch.set_defaults(func=cmd_batch)

    # train
    p_train = sub.add_parser("train", help="BERTurk modelini eğit")
    p_train.add_argument("source",    nargs="?", help="Eğitim verisi (CSV veya .txt)")
    p_train.add_argument(
        "--mode", choices=["l1", "l2", "cascade"], default="cascade",
        help=(
            "Eğitim modu: "
            "'cascade' → L1→L2 hiyerarşik (varsayılan), "
            "'l2' → 101 L2 tekli model, "
            "'l1' → 16 L1 tekli model"
        ),
    )
    p_train.add_argument("--model-dir",  help="Model çıktı dizini")
    p_train.add_argument("--raw-col",    help="CSV'de ham merchant kolon adı")
    p_train.add_argument("--label-col",  help="CSV'de etiket kolon adı (varsayılan: category_l2)")
    p_train.add_argument("--lr",         help="Öğrenme hızı override")
    p_train.add_argument("--epoch",      help="Epoch sayısı override")
    p_train.add_argument("--curve",      action="store_true", help="Epoch eğrisi grafiği üret")
    p_train.set_defaults(func=cmd_train)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Model değerlendirme")
    p_eval.add_argument("test",        nargs="?", help="Test dosyası (.txt veya .csv)")
    p_eval.add_argument("--model-dir", help="Model dizini")
    p_eval.add_argument("--per-class", action="store_true", help="Sınıf bazlı metrikler")
    p_eval.set_defaults(func=cmd_evaluate)

    # tsne
    p_tsne = sub.add_parser("tsne", help="Word embedding t-SNE görselleştirmesi")
    p_tsne.add_argument("--model-dir",  help="Model dizini")
    p_tsne.add_argument("--train",      help="Eğitim dosyası yolu")
    p_tsne.add_argument("--samples",    help="Kullanılacak örnek sayısı (varsayılan: 1500)")
    p_tsne.add_argument("--perplexity", help="t-SNE perplexity değeri (varsayılan: 40)")
    p_tsne.set_defaults(func=cmd_tsne)

    # cv
    p_cv = sub.add_parser("cv", help="Cross-validation")
    p_cv.add_argument("source",   nargs="?", help="master.csv yolu")
    p_cv.add_argument(
        "--mode", choices=["l1", "l2"], default="l2",
        help="Eğitim modu: 'l2' (varsayılan) veya 'l1'",
    )
    p_cv.add_argument("--folds",  help="Fold sayısı (varsayılan 5)")
    p_cv.set_defaults(func=cmd_cv)

    # taxonomy
    p_tax = sub.add_parser("taxonomy", help="Taxonomy özetini göster")
    p_tax.set_defaults(func=cmd_taxonomy)

    # demo
    p_demo = sub.add_parser("demo", help="Örnek tahminler")
    p_demo.add_argument("--model-dir", help="Model dizini")
    p_demo.add_argument(
        "--inference-mode", dest="inference_mode",
        choices=["cascade", "l1", "l2"], default=None,
        help="Tahmin mimarisi: 'cascade' (varsayılan), 'l2', 'l1'",
    )
    p_demo.set_defaults(func=cmd_demo)

    # explain
    p_exp = sub.add_parser("explain", help="Preprocessor adımlarını göster")
    p_exp.add_argument("merchant", help="İşyeri ismi")
    p_exp.set_defaults(func=cmd_explain)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)
