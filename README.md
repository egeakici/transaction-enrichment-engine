# Transaction Enrichment Engine

> A multi-layer NLP pipeline for classifying raw Turkish merchant strings from bank credit card statements into a two-level category taxonomy.

![Python](https://img.shields.io/badge/python-3.13%2B-blue?logo=python&logoColor=white)
![BERTurk](https://img.shields.io/badge/BERTurk-bert--base--turkish--uncased-orange?logo=huggingface&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B-ee4c2c?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Inference Modes](#inference-modes)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Taxonomy](#taxonomy)
- [Data Schema](#data-schema)
- [Performance](#performance)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Privacy](#privacy)

---

## Overview

Raw merchant strings from Turkish credit card statements — such as `"*** STARBUCKS ISTANBUL ***"` or `"POS/001 MIGROS HYPER SUBE 5"` — are noisy, inconsistently formatted, and often contain POS terminal artifacts. This pipeline cleans, normalizes, and classifies them into a two-level taxonomy: **16 L1 categories** and **100+ active L2 sub-categories**.

**Key features:**

- **4-layer hybrid pipeline** — rule-based + ML; each layer handles what the previous one misses
- **Three inference modes** — Cascade (recommended), direct L2, direct L1; switchable at runtime
- **Cascade architecture** — predicts L1 first, then routes to an L1-specific L2 model, reducing cross-category confusion
- **BERTurk backbone** — fine-tuned [`dbmdz/bert-base-turkish-uncased`](https://huggingface.co/dbmdz/bert-base-turkish-uncased) for Turkish-native contextualized representations
- **Entity DB** — deterministic dictionary lookup with exact, prefix, and fuzzy matching; no model inference needed for known merchants
- **Auto-learn** — high-confidence ML predictions (≥ 0.90) are automatically promoted to the Entity DB
- **Turkish-aware preprocessing** — handles İ/I casing, `POS/` prefixes, branch suffixes, OCR artifacts, and brand abbreviations

---

## Architecture

The pipeline routes each raw merchant string through four sequential layers. Faster, deterministic layers run first; the ML model is only invoked on cache misses.

```
Input: "POS/001 MIGROS HYPER SUBE 5"
      │
      ▼
┌──────────────────────────────────────────────┐
│  Layer 1 — Preprocessor                      │
│  · Unicode NFC normalization                 │
│  · Turkish-safe lowercasing  (İ→i, Ş→ş)     │
│  · OCR error correction      (ý→ı, þ→ş)     │
│  · Strip POS prefixes        (POS/001 → "")  │
│  · Remove branch suffixes    (SUBE 5 → "")   │
│  · Expand abbreviations      (mcd→mcdonalds) │
└────────────────────┬─────────────────────────┘
                     │  "migros hyper"
                     ▼
┌──────────────────────────────────────────────┐
│  Layer 2 — Entity DB  (catalog.json)         │
│  · Exact match      conf = 1.00              │
│  · Prefix match     conf = 0.95              │
│  · Fuzzy match      conf = 0.80–0.94         │
│    (rapidfuzz token_set_ratio)               │
└──────────┬───────────────────┬───────────────┘
        HIT │                   │ MISS
            ▼                   ▼
   ENTITY_DB_HIT        ┌─────────────────────────────────────┐
   L1 + L2 filled       │  Layer 3 — BERTurk ML Model         │
                         │                                     │
                         │  CASCADE MODE  (default)            │
                         │  ┌──────────────────────────────┐  │
                         │  │  BERTurk L1  (16 classes)    │  │
                         │  └────────────┬─────────────────┘  │
                         │               │ "market_gida"       │
                         │  ┌────────────▼─────────────────┐  │
                         │  │  BERTurk L2                  │  │
                         │  │  market_gida sub-classes only │  │
                         │  └──────────────────────────────┘  │
                         │                                     │
                         │  --inference-mode l2 | l1           │
                         │  also available                     │
                         └──────────────────┬──────────────────┘
                                            │
                                            ▼
                         ┌──────────────────────────────────────┐
                         │  Layer 4 — Router                    │
                         │  conf < 0.60  → LOW_CONFIDENCE flag  │
                         │  conf ≥ 0.90  → auto-learn to DB     │
                         │  empty/short  → fallback category    │
                         └──────────────────┬───────────────────┘
                                            │
                                            ▼
                                    EnrichmentResult
                             merchant_raw, merchant_normalized
                             category_l1, category_l2
                             confidence, source, flag, top_k
```

### Module responsibilities

| Layer | Module | Role |
|-------|--------|------|
| 1 | `src/preprocessor.py` | Text normalization — deterministic, zero ML dependencies |
| 2 | `src/entity_db.py` | Dictionary lookup — 250+ built-in merchants + dynamic catalog |
| 3 | `src/model.py` | BERTurk fine-tuning and inference (`MerchantClassifier`, `CascadeClassifier`) |
| 4 | `src/router.py` | Pipeline orchestration, confidence thresholds, auto-learning |
| — | `src/taxonomy.py` | Taxonomy definitions and L2→L1 mapping helpers |
| — | `src/visualizer.py` | Matplotlib plots (confusion matrix, per-class F1, training curves) |

---

## Inference Modes

Training (`--mode`) and inference (`--inference-mode`) are configured independently.

### Training modes

| `--mode` | Description | Model directory |
|----------|-------------|-----------------|
| `cascade` | L1 model + one L2 model per L1 group **(recommended)** | `models/berturk_mode_cascade/` |
| `l2` | Single BERTurk model over all L2 classes | `models/berturk_mode_l2/` |
| `l1` | Single BERTurk model over 16 L1 classes | `models/berturk_mode_l1/` |

### Inference modes

| `--inference-mode` | Flow | `category_l2` |
|--------------------|------|---------------|
| `cascade` *(default)* | L1 → L1-specific L2 model | Always filled |
| `l2` | Direct multi-class BERTurk | Filled |
| `l1` | Direct 16-class BERTurk | `None` |

### Cascade vs. direct L2

```
# Direct L2 (single model)
"STARBUCKS ISTANBUL"
    └─► BERTurk (100+ classes) ──► cafe_kahve  (conf: 0.82)

# Cascade (default)
"STARBUCKS ISTANBUL"
    └─► BERTurk L1 (16 classes) ──► yeme_icme    (conf: 0.94)
            └─► yeme_icme BERTurk L2 ──► cafe_kahve  (conf: 0.87)
                    └─► combined conf = √(0.94 × 0.87) = 0.904
```

Each L2 model in cascade mode is fine-tuned only on its own L1 group's sub-classes, eliminating cross-category noise at the fine-grained level.

---

## Installation

**Requirements:** Python 3.13+, GPU recommended for training (CPU works for inference)

```bash
git clone https://github.com/egeakici/transaction-enrichment-engine.git
cd transaction-enrichment-engine

python -m venv .venv

# Windows
.\.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

> **GPU note:** Training BERTurk requires substantially more compute than rule-based approaches. A CUDA-capable GPU reduces a 5-epoch fine-tuning run from hours to minutes. Inference on CPU is fast enough for batch workloads.

> **Data & models:** Training data and trained model weights are not included in this repository. Place your labeled CSV in `data/` and run `python main.py train` to train from scratch.

> **First training run:** `python main.py train` downloads the base BERTurk model (`dbmdz/bert-base-turkish-uncased`, ~440 MB) from HuggingFace Hub and caches it at `~/.cache/huggingface/`. An internet connection is required only for this initial download; all subsequent runs (training and inference) use the local cache.

---

## Quick Start

```bash
# 1. Train the cascade model (recommended)
python main.py train data/master_train.csv --mode cascade

# 2. Predict a single merchant
python main.py predict "MIGROS HYPERMARKET"

# 3. Run the built-in demo
python main.py demo

# 4. Trace preprocessing steps on a raw string
python main.py explain "POS/001 MİGROS HYPERMARKet SUBE 5"
```

**Example output:**

```
'MIGROS HYPERMARKET'
     L2 : supermarket
     L1 : market_gida
   conf : 0.891
  top-k : indirim_market (0.06)  |  diger_market_gida (0.02)
 source : ml_model
```

---

## CLI Reference

All commands go through the single entry point `python main.py <command> [options]`.

---

### `train` — Fine-tune a model

```bash
python main.py train [source] [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `source` | `data/master.csv` | Training data CSV |
| `--mode` | `cascade` | `cascade` \| `l2` \| `l1` |
| `--epoch` | `5` | Fine-tuning epochs |
| `--lr` | `2e-5` | AdamW learning rate |
| `--model-dir` | auto | Override output directory |
| `--curve` | — | Save accuracy-vs-epoch plot |

```bash
# Cascade — recommended
python main.py train data/master_train.csv --mode cascade

# Single L2 model with custom hyperparameters
python main.py train data/master_train.csv --mode l2 --epoch 8 --lr 3e-5

# L1 model with training curve
python main.py train data/master_train.csv --mode l1 --curve
```

---

### `predict` — Single merchant prediction

```bash
python main.py predict "MERCHANT NAME" [--inference-mode cascade|l2|l1]
```

```bash
python main.py predict "STARBUCKS ISTANBUL"
python main.py predict "NETFLIX ABONELIK"  --inference-mode l2
python main.py predict "SHELL PETROL"      --inference-mode l1
```

---

### `batch` — Bulk prediction from CSV

```bash
python main.py batch input.csv [--output output.csv] [--col column] [--inference-mode ...]
```

```bash
python main.py batch data/master_gercek.csv --output results/enriched.csv
python main.py batch data/input.csv --inference-mode l2 --col merchant_name
```

Appended output columns: `merchant_normalized`, `category_l2`, `category_l1`, `confidence`, `source`, `flag`

---

### `evaluate` — Evaluate on a test set

```bash
python main.py evaluate [test_file] [--per-class]
```

```bash
# Default held-out test set
python main.py evaluate

# Labeled CSV with per-class breakdown
python main.py evaluate data/master_test.csv --per-class
```

Saves: `models/plots/confusion_matrix_l1.png`, `models/plots/per_class_f1_l1.png`, and a timestamped JSON to `models/experiments/`.

---

### `cv` — Cross-validation

```bash
python main.py cv [source] --mode l2|l1 --folds N
```

```bash
python main.py cv data/master_train.csv --mode l2 --folds 5
```

Results appended to `models/experiments/cv_results.jsonl`.

> **Note:** Each fold requires a full BERTurk fine-tuning run. 5-fold CV on a GPU takes roughly 5× the time of a single training run.

---

### `tsne` — Embedding visualization

Generates a t-SNE plot of `[CLS]` token embeddings colored by L2 category.

```bash
python main.py tsne [--samples 1500] [--perplexity 40]
```

---

### `demo` — Sample predictions

```bash
python main.py demo [--inference-mode cascade|l2|l1]
```

Runs a curated set of representative merchant strings through the full pipeline and prints the enrichment table.

---

### `explain` — Preprocessing trace

Prints each normalization step applied to a raw string.

```bash
python main.py explain "POS/001 İSTANBUL MİGROS SUBE 5"
```

---

### `taxonomy` — List full taxonomy

```bash
python main.py taxonomy
```

---

## Taxonomy

16 L1 categories × 100+ active L2 sub-categories.

| L1 Category | L2 Sub-categories |
|-------------|-------------------|
| `yeme_icme` | restoran, fast_food, pastane_firin, bar_icki_servisi, yemek_siparis, cafe_kahve, diger_yeme_icme |
| `market_gida` | supermarket, indirim_market, bakkal_tekel, kuruyemis_aktar, manav_kasap_sarkuteri, diger_market_gida |
| `giyim_aksesuar` | giyim_magaza, ayakkabi, kuyumcu, saat, diger_giyim_aksesuar |
| `online_alisveris` | eticaret_platform, yurt_disi_eticaret, kargo_lojistik, diger_online_alisveris |
| `elektronik` | elektronik_market, elektronik_servis, beyaz_esya, telefon_aksesuar, diger_elektronik |
| `ev_yasam` | mobilya, dekorasyon, yapi_market, zuccaciye, ev_hizmet, oyuncak_cocuk, spor_urun, diger_ev_yasam |
| `saglik` | eczane, hastane_klinik, optik, veteriner, saglik_urun, diger_saglik |
| `kisisel_bakim` | berber_kuafor, guzellik_estetik, parfumeri, spa_masaj, kozmetik, diger_kisisel_bakim |
| `ulasim` | taksi_uber, toplu_tasima, otopark, rent_a_car, akaryakit_yol, feribot, otoyol_kopru, bisiklet_scooter, tamir_servis, diger_ulasim |
| `seyahat` | otel_konaklama, ucak_bileti, tur_acentesi, tatil_rezervasyon, diger_seyahat |
| `eglence_kultur` | sinema_tiyatro, muzik_konser, oyun_oyuncak, spor_etkinlik, kitab_dergi, muzeum_gezi, dijital_oyun, diger_eglence |
| `fatura_abonelik` | elektrik_su_dogalgaz, internet_telefon, sigorta_odeme, dijital_abonelik, okul_harci, uyelik_karti, vergi_harc, muhasebe_fintech, diger_fatura |
| `finans_sigorta` | banka_islem, atm_nakit, doviz_bozma, yatirim, kripto, sigorta_porice, emeklilik, kredi_odeme, havale_eft, odeme_aracilari, diger_finans |
| `akaryakit` | akaryakit_istasyonu, elektrikli_arac_sarj, diger_akaryakit |
| `egitim` | okul_universite, kurs_dershane, kitap_kirtasiye, egitim_yazilim, spor_kurs, diger_egitim |
| `diger` | diger_isyeri, tanimlanamayan_isyeri |

Run `python main.py taxonomy` for the authoritative listing.

---

## Data Schema

### Training data (`master_*.csv`)

| Column | Type | Required | Notes |
|--------|------|----------|-------|
| `merchant_raw` | str | Yes | Raw POS terminal string — never modified |
| `category_l2` | str | Yes | Training target (L2 label) |
| `category_l1` | str | No | Derived from taxonomy mapping at runtime |
| `amount` | float | No | Reserved for future amount-aware features |
| `transaction_date` | date | No | Reserved for future temporal features |

### Entity DB entry (`data/catalog.json`)

```json
{
  "starbucks": {
    "l2_category": "cafe_kahve",
    "source": "manual",
    "is_active": true,
    "added_date": "2026-03-27",
    "confidence": 1.0
  }
}
```

`source` is either `"manual"` (hand-labeled) or `"auto_learn"` (promoted from high-confidence ML predictions).

### Model metadata (`metrics.json`)

```json
{
  "mode": "cascade",
  "trained_at": 1775671005.4,
  "train_samples": 18279,
  "n_labels": 83,
  "hyperparams": {
    "lr": 2e-05,
    "epoch": 5,
    "batch_size": 32,
    "max_length": 64
  },
  "label2id": { "cafe_kahve": 10 },
  "id2label": { "10": "cafe_kahve" }
}
```

### `EnrichmentResult` — pipeline output

```python
@dataclass
class EnrichmentResult:
    merchant_raw: str                   # Original input, never modified
    merchant_normalized: str            # Preprocessor output
    category_l2: str | None             # L2 label (None in l1 mode)
    category_l1: str                    # L1 label, always filled
    confidence: float                   # 0.0 – 1.0
    source: str                         # "entity_db" | "ml_model" | "fallback"
    flag: str | None                    # "LOW_CONFIDENCE" | "ENTITY_DB_HIT" | None
    top_k: list[tuple[str, float]]      # Runner-up predictions with probabilities
```

---

## Performance

Evaluation on a held-out test set, single L2 BERTurk model:

| Metric | L2 (fine-grained) | L1 (coarse) |
|--------|-------------------|-------------|
| Accuracy | 0.569 | 0.716 |
| F1 Macro | 0.468 | 0.669 |
| F1 Weighted | 0.518 | 0.709 |

### Training curve

L1 model fine-tuned on 14K samples (BERTurk, representative checkpoints):

| Epoch | Train Acc | Val Acc | Gap |
|-------|-----------|---------|-----|
| 1 | 0.016 | 0.015 | — |
| 5 | 0.844 | 0.786 | 0.058 |
| 10 | 0.974 | 0.887 | 0.087 |
| 25 | 0.992 | 0.903 | 0.089 |

### Targets

| Stage | Metric | Target |
|-------|--------|--------|
| Single L2 (current) | F1 Macro | ~0.47 |
| Cascade | F1 Macro | ~0.60+ |
| Cascade + 50K samples | F1 Macro | ~0.70+ |
| Production | L1 Accuracy | 90%+ |

The cascade architecture is expected to close the gap between L1 and L2 accuracy by confining each sub-model to its own category space.

---

## Project Structure

Only source files are tracked; data, models, and the frontend are excluded (see below).

```
transaction-enrichment-engine/
├── main.py                    # Single CLI entry point
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── taxonomy.py            # 16 L1 / 100+ L2 taxonomy definitions + helpers
│   ├── preprocessor.py        # Layer 1 — text normalization
│   ├── entity_db.py           # Layer 2 — dictionary lookup (exact/prefix/fuzzy)
│   ├── model.py               # Layer 3 — MerchantClassifier + CascadeClassifier
│   ├── router.py              # Layer 4 — HybridRouter + confidence logic
│   └── visualizer.py          # Matplotlib plot generation
│
└── tests/
    └── test_pipeline.py       # Unit and smoke tests
```

### Not included in this repository

| Path | Reason |
|------|--------|
| `data/` | Training CSVs and `catalog.json` — not distributed |
| `models/` | Trained BERTurk weights — too large for git |
| `frontend/` | Flask web UI — lives in a separate repository |
| `venv/` / `.venv/` | Virtual environment |

Model directory layout after training:

```
models/
├── berturk_mode_cascade/      # Cascade model (recommended)
│   ├── l1_model/
│   ├── l2_models/
│   │   ├── yeme_icme/
│   │   ├── market_gida/
│   │   └── ...
│   └── cascade_meta.json
├── berturk_mode_l2/           # Single-stage L2 model
├── berturk_mode_l1/           # Single-stage L1 model
├── experiments/
│   ├── eval_*.json            # Timestamped evaluation results
│   └── cv_results.jsonl       # Cross-validation history
└── plots/
    ├── confusion_matrix_l1.png
    ├── per_class_f1_l1.png
    └── training_curve.png
```

---

## Testing

```bash
# Run the full test suite
python -m pytest tests/ -v

# Run without pytest installed
python tests/test_pipeline.py
```

| Area | What is tested |
|------|----------------|
| Taxonomy | L2→L1 mapping correctness, active class validation |
| Preprocessor | Turkish character handling, POS prefix stripping, branch suffix removal |
| Entity DB | Exact match, fuzzy match, correction application |
| Router | Entity DB hits, ML fallback, LOW_CONFIDENCE flagging, batch processing |

---

## Privacy

- All bank statement data must be anonymized before being passed to the pipeline.
- `merchant_raw` must not contain personal names, IBANs, account numbers, or national ID numbers.
- Auto-learned entries written to `catalog.json` store only normalized merchant strings — no personal data is persisted.
