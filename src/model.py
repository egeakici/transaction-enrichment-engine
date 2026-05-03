"""Layer 3: BERTurk fine-tuning and inference (l1 / l2 / cascade modes)."""

import csv
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BERTurk imports
# ---------------------------------------------------------------------------

BERTURK_MODEL_NAME = "dbmdz/bert-base-turkish-uncased"

try:
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification
    _BERTURK_AVAILABLE = True
except ImportError:
    _BERTURK_AVAILABLE = False
    logger.error(
        "torch or transformers not found. "
        "Install with: pip install torch transformers"
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_PREFIX             = "__label__"
DEFAULT_METRICS_FILENAME = "metrics.json"
VALID_MODES              = ("l1", "l2")

CASCADE_L1_CONFIDENCE_THRESHOLD: float = 0.50
CASCADE_L2_CONFIDENCE_THRESHOLD: float = 0.40

DEFAULT_HYPERPARAMS: dict = {
    "lr":           2e-5,
    "epoch":        5,
    "batch_size":   32,
    "max_length":   64,     # merchant strings are short, 64 tokens is sufficient
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Prediction:
    merchant_raw: str
    merchant_normalized: str
    category_l2: str | None   # None in l1 mode
    category_l1: str
    confidence: float
    top_k: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class EvaluationResult:
    # Primary metrics:
    #   mode="l2" → L2 metrics
    #   mode="l1" → L1 metrics
    accuracy: float
    f1_macro: float
    f1_weighted: float
    n_samples: int
    per_class: dict[str, dict]
    confusion_matrix: dict
    # L1 metrics:
    #   mode="l2" → derived from L2 predictions via taxonomy
    #   mode="l1" → same as primary (backward compat)
    accuracy_l1: float = 0.0
    f1_macro_l1: float = 0.0
    f1_weighted_l1: float = 0.0
    per_class_l1: dict[str, dict] = None
    # Raw L1 label lists for visualizer.plot_evaluation()
    _true_l1: list = None
    _pred_l1: list = None

    def __post_init__(self):
        if self.per_class_l1 is None:
            self.per_class_l1 = {}
        if self._true_l1 is None:
            self._true_l1 = []
        if self._pred_l1 is None:
            self._pred_l1 = []


# ---------------------------------------------------------------------------
# Training file helpers
# ---------------------------------------------------------------------------

def to_training_line(merchant_normalized: str, label: str) -> str:
    return f"{LABEL_PREFIX}{label} {merchant_normalized}"


def build_training_file(
    records: list[dict],
    output_path: Path,
    text_col: str  = "merchant_normalized",
    label_col: str = "category_l2",
    mode: str      = "l2",
) -> int:
    """
    Write a training file from a list of records.

    Format: __label__<label> <normalized_text>

    mode="l2": writes L2 labels directly.
    mode="l1": maps L2 → L1 via taxonomy before writing.

    Returns the number of lines written.
    """
    from taxonomy import is_active_l2, L2_TO_L1

    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode: '{mode}'. Valid values: {VALID_MODES}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            text     = rec.get(text_col,  "").strip()
            l2_label = rec.get(label_col, "").strip()
            if not text or not l2_label:
                skipped += 1
                continue
            if not is_active_l2(l2_label):
                skipped += 1
                continue
            label = L2_TO_L1.get(l2_label, "diger") if mode == "l1" else l2_label
            f.write(to_training_line(text, label) + "\n")
            written += 1

    logger.info(
        f"Training file ({mode.upper()}): {output_path} "
        f"({written} lines, {skipped} skipped)"
    )
    return written


def csv_to_training_file(
    csv_path: Path,
    output_path: Path,
    raw_col: str   = "merchant_raw",
    label_col: str = "category_l2",
    mode: str      = "l2",
) -> int:
    """Convert a labeled CSV to a training file, applying preprocessor normalization."""
    from preprocessor import normalize
    csv_path = Path(csv_path)
    records  = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw   = row.get(raw_col,   "")
            label = row.get(label_col, "")
            norm  = normalize(raw)
            if norm and label:
                records.append({"merchant_normalized": norm, "category_l2": label})
    return build_training_file(records, output_path, mode=mode)


# ---------------------------------------------------------------------------
# MerchantClassifier — BERTurk single-stage (l1 or l2)
# ---------------------------------------------------------------------------

class MerchantClassifier:
    """
    BERTurk-based merchant classifier.
    Base model: dbmdz/bert-base-turkish-uncased (Hugging Face)

    Supports both l1 (16 classes) and l2 (100+ classes) modes.

    Example:
        clf = MerchantClassifier(mode="l2")
        clf.train("data/processed/train.txt")
        pred = clf.predict("STARBUCKS ISTANBUL")
    """

    def __init__(self, mode: str = "l2", hyperparams: dict | None = None):
        if not _BERTURK_AVAILABLE:
            raise ImportError(
                "torch or transformers not installed. "
                "pip install torch transformers"
            )
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode: '{mode}'. Valid values: {VALID_MODES}")

        self._mode: str                    = mode
        self._model                        = None
        self._tokenizer                    = None
        self._label2id: dict[str, int]     = {}
        self._id2label: dict[int, str]     = {}
        self._hyperparams: dict            = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
        self._trained_at: float | None     = None
        self._train_samples: int           = 0
        self._training_history: list[dict] = []
        self._label_prefix_len: int        = len(LABEL_PREFIX)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, train_path: str | Path, **override_params) -> None:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from transformers import (
            BertTokenizer, BertForSequenceClassification,
            get_linear_schedule_with_warmup,
        )
        from torch.optim import AdamW

        params     = {**self._hyperparams, **override_params}
        train_path = Path(train_path)
        prefix_len = len(LABEL_PREFIX)

        logger.info(f"BERTurk training started [{self._mode.upper()}]: {train_path}")
        logger.info(f"Hyperparameters: {params}")
        t0 = time.time()

        texts: list[str]      = []
        raw_labels: list[str] = []
        with open(train_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) < 2:
                    continue
                label = parts[0][prefix_len:]
                text  = parts[1].strip()
                if label and text:
                    texts.append(text)
                    raw_labels.append(label)

        if not texts:
            raise ValueError(f"No valid lines found in training file: {train_path}")

        unique_labels  = sorted(set(raw_labels))
        self._label2id = {l: i for i, l in enumerate(unique_labels)}
        self._id2label = {i: l for i, l in enumerate(unique_labels)}
        label_ids      = [self._label2id[l] for l in raw_labels]

        logger.info(
            f"Training data: {len(texts)} samples, {len(unique_labels)} classes"
        )

        logger.info(f"Loading BERTurk: {BERTURK_MODEL_NAME}")
        tokenizer = BertTokenizer.from_pretrained(BERTURK_MODEL_NAME)
        model     = BertForSequenceClassification.from_pretrained(
            BERTURK_MODEL_NAME,
            num_labels=len(unique_labels),
            id2label=self._id2label,
            label2id=self._label2id,
        )

        # Dynamic per-batch padding: merchant strings are short and vary widely
        # in length, so padding to longest-in-batch wastes less compute than
        # padding all sequences to a fixed max_length.
        logger.info(f"Building DataLoader: {len(texts)} samples, dynamic padding enabled")

        _max_length = params["max_length"]
        _tokenizer  = tokenizer

        class _LabeledDataset(Dataset):
            def __init__(self, txts, ids):
                self.texts  = txts
                self.labels = ids

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                return self.texts[idx], self.labels[idx]

        def _collate_fn(batch):
            batch_texts, batch_ids = zip(*batch)
            enc = _tokenizer(
                list(batch_texts),
                truncation=True,
                padding="longest",
                max_length=_max_length,
                return_tensors="pt",
            )
            enc["labels"] = torch.tensor(batch_ids, dtype=torch.long)
            return enc

        dataset = _LabeledDataset(texts, label_ids)
        loader  = DataLoader(
            dataset,
            batch_size=params["batch_size"],
            shuffle=True,
            num_workers=0,
            collate_fn=_collate_fn,
        )

        device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model       = model.to(device)
        optimizer   = AdamW(
            model.parameters(),
            lr=params["lr"],
            weight_decay=params["weight_decay"],
        )
        total_steps  = len(loader) * params["epoch"]
        warmup_steps = max(1, int(total_steps * params["warmup_ratio"]))
        scheduler    = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        logger.info(
            f"Device: {device} | "
            f"Batch: {params['batch_size']} | "
            f"Epoch: {params['epoch']} | "
            f"Steps: {total_steps}"
        )

        try:
            from tqdm import tqdm as _tqdm
        except ImportError:
            _tqdm = None

        self._training_history = []
        for ep in range(params["epoch"]):
            model.train()
            ep_loss    = 0.0
            ep_correct = 0
            ep_total   = 0

            batches = (
                _tqdm(loader, desc=f"Epoch {ep+1}/{params['epoch']}", unit="batch")
                if _tqdm else loader
            )
            for batch in batches:
                batch   = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss    = outputs.loss
                logits  = outputs.logits

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                ep_loss    += loss.item()
                preds       = logits.argmax(dim=-1)
                ep_correct += (preds == batch["labels"]).sum().item()
                ep_total   += len(batch["labels"])

            epoch_acc  = ep_correct / ep_total if ep_total else 0.0
            epoch_loss = ep_loss / len(loader)
            self._training_history.append({
                "epoch":    ep + 1,
                "loss":     round(epoch_loss, 4),
                "accuracy": round(epoch_acc, 4),
            })
            logger.info(
                f"  Epoch {ep + 1}/{params['epoch']}: "
                f"loss={epoch_loss:.4f}  acc={epoch_acc:.4f}"
            )

        elapsed              = time.time() - t0
        self._model          = model
        self._tokenizer      = tokenizer
        self._trained_at     = time.time()
        self._train_samples  = len(texts)

        logger.info(
            f"BERTurk training complete [{self._mode.upper()}]: "
            f"{elapsed:.1f}s | {self._train_samples} samples | "
            f"{len(unique_labels)} classes"
        )

    # ------------------------------------------------------------------
    # Single-sample prediction
    # ------------------------------------------------------------------

    def predict(self, merchant_raw: str, top_k: int = 3) -> Prediction:
        """
        Predict category for a single raw merchant string.

        mode="l2": category_l2 filled, category_l1 derived from taxonomy.
        mode="l1": category_l2=None, category_l1 is the direct model output.
        """
        if self._model is None:
            raise RuntimeError("Model not yet trained. Call train() first.")

        import torch
        from preprocessor import normalize
        from taxonomy import get_l1

        normalized = normalize(merchant_raw)
        if not normalized:
            return Prediction(
                merchant_raw=merchant_raw,
                merchant_normalized=normalized,
                category_l2=None if self._mode == "l1" else "tanimlanamayan_isyeri",
                category_l1="diger",
                confidence=0.0,
                top_k=[],
            )

        device   = next(self._model.parameters()).device
        encoding = self._tokenizer(
            normalized,
            truncation=True,
            padding=True,
            max_length=self._hyperparams.get("max_length", 64),
            return_tensors="pt",
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}

        self._model.eval()
        with torch.no_grad():
            outputs = self._model(**encoding)
            probs   = torch.softmax(outputs.logits[0], dim=-1)

        k           = min(top_k, len(self._id2label))
        topk_result = torch.topk(probs, k)
        top_results = [
            (self._id2label[idx.item()], round(prob.item(), 4))
            for idx, prob in zip(topk_result.indices, topk_result.values)
        ]
        best_label, best_conf = top_results[0]

        if self._mode == "l1":
            return Prediction(
                merchant_raw=merchant_raw,
                merchant_normalized=normalized,
                category_l2=None,
                category_l1=best_label,
                confidence=best_conf,
                top_k=top_results,
            )
        else:
            return Prediction(
                merchant_raw=merchant_raw,
                merchant_normalized=normalized,
                category_l2=best_label,
                category_l1=get_l1(best_label) or "diger",
                confidence=best_conf,
                top_k=top_results,
            )

    def predict_batch(self, merchants: list[str], top_k: int = 1) -> list[Prediction]:
        """Batch inference via _infer_batch; N/batch_size forward passes instead of N."""
        if self._model is None:
            raise RuntimeError("Model not yet trained. Call train() first.")

        import torch
        from preprocessor import normalize
        from taxonomy import get_l1

        normalized_list = [normalize(m) for m in merchants]

        non_empty_idx   = [i for i, t in enumerate(normalized_list) if t]
        non_empty_texts = [normalized_list[i] for i in non_empty_idx]

        all_top: list[list[tuple[str, float]]] = [[] for _ in merchants]

        if non_empty_texts:
            batch_results = self._infer_batch(non_empty_texts, top_k=top_k)
            for idx, top in zip(non_empty_idx, batch_results):
                all_top[idx] = top

        predictions: list[Prediction] = []
        for merchant_raw, normalized, top_results in zip(merchants, normalized_list, all_top):
            if not normalized or not top_results:
                predictions.append(Prediction(
                    merchant_raw=merchant_raw,
                    merchant_normalized=normalized,
                    category_l2=None if self._mode == "l1" else "tanimlanamayan_isyeri",
                    category_l1="diger",
                    confidence=0.0,
                    top_k=[],
                ))
                continue

            best_label, best_conf = top_results[0]
            if self._mode == "l1":
                predictions.append(Prediction(
                    merchant_raw=merchant_raw,
                    merchant_normalized=normalized,
                    category_l2=None,
                    category_l1=best_label,
                    confidence=best_conf,
                    top_k=top_results,
                ))
            else:
                predictions.append(Prediction(
                    merchant_raw=merchant_raw,
                    merchant_normalized=normalized,
                    category_l2=best_label,
                    category_l1=get_l1(best_label) or "diger",
                    confidence=best_conf,
                    top_k=top_results,
                ))

        return predictions

    # ------------------------------------------------------------------
    # Internal batch inference
    # ------------------------------------------------------------------

    def _infer_batch(
        self, texts: list[str], top_k: int = 1
    ) -> list[list[tuple[str, float]]]:
        """Run inference on a list of normalized texts; return top_k (label, conf) per input."""
        import torch

        device     = next(self._model.parameters()).device
        batch_size = self._hyperparams.get("batch_size", 32)
        k          = min(top_k, len(self._id2label))
        results: list[list[tuple[str, float]]] = []

        try:
            from tqdm import tqdm as _tqdm
        except ImportError:
            _tqdm = None

        n_batches = (len(texts) + batch_size - 1) // batch_size
        indices   = range(0, len(texts), batch_size)
        bar       = (
            _tqdm(indices, total=n_batches, desc="Evaluating", unit="batch")
            if _tqdm else indices
        )

        self._model.eval()
        with torch.no_grad():
            for i in bar:
                batch_texts = texts[i : i + batch_size]
                encoding    = self._tokenizer(
                    batch_texts,
                    truncation=True,
                    padding="longest",
                    max_length=self._hyperparams.get("max_length", 64),
                    return_tensors="pt",
                )
                encoding = {key: v.to(device) for key, v in encoding.items()}
                outputs  = self._model(**encoding)
                probs    = torch.softmax(outputs.logits, dim=-1)

                for row in probs:
                    topk_result = torch.topk(row, k)
                    top = [
                        (self._id2label[idx.item()], round(prob.item(), 4))
                        for idx, prob in zip(topk_result.indices, topk_result.values)
                    ]
                    results.append(top)

        return results

    # ------------------------------------------------------------------
    # Sentence vector — visualizer interface
    # ------------------------------------------------------------------

    def get_sentence_vector(self, text: str) -> list[float]:
        """Return the BERTurk last-layer CLS token embedding for a text."""
        if self._model is None:
            raise RuntimeError("Model not trained.")

        import torch

        device   = next(self._model.parameters()).device
        encoding = self._tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self._hyperparams.get("max_length", 64),
            return_tensors="pt",
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}

        self._model.eval()
        with torch.no_grad():
            outputs = self._model(**encoding, output_hidden_states=True)
            cls_vec = outputs.hidden_states[-1][:, 0, :]

        return cls_vec[0].cpu().tolist()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, test_path: str | Path) -> EvaluationResult:
        """
        Compute F1 and accuracy on a labeled test file.

        mode="l2": expects L2 labels; primary = L2 metrics, L1 derived via taxonomy.
        mode="l1": expects L1 labels; primary = L1 metrics.
        """
        if self._model is None:
            raise RuntimeError("Model not trained.")

        from collections import defaultdict
        from taxonomy import get_l1 as _get_l1

        test_path  = Path(test_path)
        prefix_len = len(LABEL_PREFIX)

        true_labels: list[str]    = []
        texts_to_infer: list[str] = []

        with open(test_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) < 2:
                    continue
                true_labels.append(parts[0][prefix_len:])
                texts_to_infer.append(parts[1])

        logger.info(f"Evaluation: running inference on {len(true_labels)} samples...")
        batch_preds = self._infer_batch(texts_to_infer, top_k=1)
        pred_labels = [top[0][0] for top in batch_preds]

        # Primary metrics
        try:
            from sklearn.metrics import (
                accuracy_score, f1_score, classification_report,
            )
            acc         = accuracy_score(true_labels, pred_labels)
            f1_macro    = f1_score(true_labels, pred_labels, average="macro",    zero_division=0)
            f1_weighted = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
            report      = classification_report(
                true_labels, pred_labels, output_dict=True, zero_division=0
            )
            per_class   = {
                k: v for k, v in report.items()
                if k not in ("accuracy", "macro avg", "weighted avg")
            }
        except ImportError:
            correct     = sum(t == p for t, p in zip(true_labels, pred_labels))
            acc         = correct / len(true_labels) if true_labels else 0.0
            f1_macro    = acc
            f1_weighted = acc
            per_class   = {}

        confusion: dict = defaultdict(lambda: defaultdict(int))
        for t, p in zip(true_labels, pred_labels):
            confusion[t][p] += 1

        # L1 metrics
        if self._mode == "l2":
            true_l1 = [_get_l1(l) or "diger" for l in true_labels]
            pred_l1 = [_get_l1(l) or "diger" for l in pred_labels]
        else:
            true_l1 = true_labels
            pred_l1 = pred_labels

        try:
            from sklearn.metrics import (
                accuracy_score as _acc,
                f1_score as _f1,
                classification_report as _report,
            )
            acc_l1         = _acc(true_l1, pred_l1)
            f1_macro_l1    = _f1(true_l1, pred_l1, average="macro",    zero_division=0)
            f1_weighted_l1 = _f1(true_l1, pred_l1, average="weighted", zero_division=0)
            rep_l1         = _report(true_l1, pred_l1, output_dict=True, zero_division=0)
            per_class_l1   = {
                k: v for k, v in rep_l1.items()
                if k not in ("accuracy", "macro avg", "weighted avg")
            }
        except ImportError:
            correct_l1     = sum(t == p for t, p in zip(true_l1, pred_l1))
            acc_l1         = correct_l1 / len(true_l1) if true_l1 else 0.0
            f1_macro_l1    = acc_l1
            f1_weighted_l1 = acc_l1
            per_class_l1   = {}

        result = EvaluationResult(
            accuracy=round(acc, 4),
            f1_macro=round(f1_macro, 4),
            f1_weighted=round(f1_weighted, 4),
            n_samples=len(true_labels),
            per_class=per_class,
            confusion_matrix={k: dict(v) for k, v in confusion.items()},
            accuracy_l1=round(acc_l1, 4),
            f1_macro_l1=round(f1_macro_l1, 4),
            f1_weighted_l1=round(f1_weighted_l1, 4),
            per_class_l1=per_class_l1,
            _true_l1=true_l1,
            _pred_l1=pred_l1,
        )

        if self._mode == "l2":
            logger.info(
                f"Evaluation [L2]: n={result.n_samples} | "
                f"L2 Acc={result.accuracy:.3f} F1={result.f1_macro:.3f} | "
                f"L1 Acc={result.accuracy_l1:.3f} F1={result.f1_macro_l1:.3f}"
            )
        else:
            logger.info(
                f"Evaluation [L1]: n={result.n_samples} | "
                f"Accuracy={result.accuracy:.3f} | F1-Macro={result.f1_macro:.3f}"
            )
        return result

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        records: list[dict],
        n_splits: int = 5,
        tmp_dir: Path | None = None,
    ) -> dict:
        """
        Stratified k-fold cross-validation.
        Each fold runs a full BERTurk fine-tuning; GPU strongly recommended.
        """
        import tempfile
        import shutil

        tmp_dir      = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(prefix="cv_"))
        fold_results: list[EvaluationResult] = []

        try:
            from sklearn.model_selection import StratifiedKFold
            labels_for_split = [r["category_l2"] for r in records]
            skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = list(skf.split(records, labels_for_split))
        except ImportError:
            n         = len(records)
            fold_size = n // n_splits
            splits    = []
            for i in range(n_splits):
                test_idx  = list(range(i * fold_size, min((i + 1) * fold_size, n)))
                train_idx = [j for j in range(n) if j not in test_idx]
                splits.append((train_idx, test_idx))

        logger.info(
            f"Cross-validation [{self._mode.upper()}]: "
            f"{n_splits}-fold, {len(records)} samples"
        )

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            train_file = tmp_dir / f"fold_{fold_idx}_train.txt"
            test_file  = tmp_dir / f"fold_{fold_idx}_test.txt"
            build_training_file(
                [records[i] for i in train_idx], train_file, mode=self._mode
            )
            build_training_file(
                [records[i] for i in test_idx],  test_file,  mode=self._mode
            )
            fold_clf = MerchantClassifier(mode=self._mode, hyperparams=self._hyperparams)
            fold_clf.train(train_file)
            result = fold_clf.evaluate(test_file)
            fold_results.append(result)
            logger.info(
                f"  Fold {fold_idx + 1}/{n_splits}: "
                f"F1-Macro={result.f1_macro:.3f} | Accuracy={result.accuracy:.3f}"
            )

        mean_f1  = sum(r.f1_macro for r in fold_results) / len(fold_results)
        mean_acc = sum(r.accuracy for r in fold_results) / len(fold_results)
        std_f1   = (
            sum((r.f1_macro - mean_f1) ** 2 for r in fold_results) / len(fold_results)
        ) ** 0.5

        summary = {
            "mode":           self._mode,
            "n_folds":        n_splits,
            "n_samples":      len(records),
            "mean_f1_macro":  round(mean_f1, 4),
            "std_f1_macro":   round(std_f1, 4),
            "mean_accuracy":  round(mean_acc, 4),
            "fold_f1_scores": [r.f1_macro for r in fold_results],
        }
        logger.info(
            f"CV Summary [{self._mode.upper()}]: "
            f"F1-Macro = {mean_f1:.3f} ± {std_f1:.3f}"
        )
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return summary

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, model_dir: str | Path) -> None:
        """Save model, tokenizer, and metadata in HuggingFace format."""
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("No model to save.")

        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        self._model.save_pretrained(str(model_dir))
        self._tokenizer.save_pretrained(str(model_dir))

        meta = {
            "mode":             self._mode,
            "trained_at":       self._trained_at,
            "train_samples":    self._train_samples,
            "n_labels":         len(self._label2id),
            "hyperparams":      self._hyperparams,
            "label2id":         self._label2id,
            "id2label":         {str(k): v for k, v in self._id2label.items()},
            "training_history": self._training_history,
        }
        with open(model_dir / DEFAULT_METRICS_FILENAME, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info(
            f"BERTurk model saved [{self._mode.upper()}]: {model_dir}"
        )

    def load(self, model_dir: str | Path) -> None:
        """Load a saved BERTurk model; updates self._mode from metrics.json."""
        import torch
        from transformers import BertTokenizer, BertForSequenceClassification

        model_dir    = Path(model_dir)
        metrics_path = model_dir / DEFAULT_METRICS_FILENAME

        if not metrics_path.exists():
            raise FileNotFoundError(
                f"Metrics file not found: {metrics_path}\n"
                f"Train the model with: python main.py train"
            )

        with open(metrics_path, encoding="utf-8") as f:
            meta = json.load(f)

        saved_mode = meta.get("mode", "l2")
        if saved_mode != self._mode:
            logger.info(
                f"Model mode updated: {self._mode} → {saved_mode} "
                f"(read from metrics.json)"
            )
            self._mode = saved_mode

        self._label2id          = meta["label2id"]
        self._id2label          = {int(k): v for k, v in meta["id2label"].items()}
        self._training_history  = meta.get("training_history", [])

        self._tokenizer = BertTokenizer.from_pretrained(str(model_dir))
        self._model     = BertForSequenceClassification.from_pretrained(str(model_dir))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = self._model.to(device)
        self._model.eval()

        logger.info(
            f"BERTurk model loaded [{self._mode.upper()}]: "
            f"{model_dir} ({len(self._label2id)} classes)"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    @property
    def n_labels(self) -> int:
        return len(self._label2id)

    def __repr__(self) -> str:
        status = f"trained ({self._mode.upper()}), {self.n_labels} labels" \
                 if self.is_trained else "untrained"
        return f"MerchantClassifier[BERTurk]({status})"


# ---------------------------------------------------------------------------
# Mode detection helper — used by main.py
# ---------------------------------------------------------------------------

def detect_mode_from_model(model_dir: str | Path) -> str:
    """Read the training mode from a saved model's metrics.json; defaults to 'l2'."""
    metrics_path = Path(model_dir) / DEFAULT_METRICS_FILENAME
    if metrics_path.exists():
        try:
            with open(metrics_path, encoding="utf-8") as f:
                return json.load(f).get("mode", "l2")
        except Exception:
            pass
    return "l2"


# ---------------------------------------------------------------------------
# CascadeClassifier — hierarchical L1 → L2 prediction
# ---------------------------------------------------------------------------

class CascadeClassifier:
    """
    Two-stage hierarchical classifier.

    Stage 1: L1 prediction (16 classes) — single BERTurk model.
    Stage 2: L2 prediction (L1-specific subset) — one BERTurk model per L1.

        Input → L1 prediction → L1-specific L2 model → L2 prediction

    Saved layout:
        <model_dir>/
            l1_model/
            l2_models/
                yeme_icme/
                market_gida/
                ...
            l2_flat_model/    — fallback when L1 confidence is low
            cascade_meta.json

    Confidence score: (conf_l1 × conf_l2)^0.5  (geometric mean)
    """

    L1_SUBDIR      = "l1_model"
    L2_SUBDIR      = "l2_models"
    L2_FLAT_SUBDIR = "l2_flat_model"
    META_FILE      = "cascade_meta.json"

    def __init__(self, hyperparams: dict | None = None):
        self._hyperparams: dict                      = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
        self._l1_clf: MerchantClassifier | None      = None
        self._l2_clfs: dict[str, MerchantClassifier] = {}
        self._l2_flat_clf: MerchantClassifier | None = None
        self._trained_at: float | None               = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, records: list[dict], tmp_dir: Path | None = None) -> None:
        """
        Train the full cascade: L1 model + per-L1 L2 models + flat L2 fallback.

        Args:
            records : [{"merchant_normalized": ..., "category_l2": ...}, ...]
            tmp_dir : directory for temporary training files (auto-created if None)
        """
        import tempfile
        import shutil
        from collections import defaultdict
        from taxonomy import L2_TO_L1, is_active_l2

        use_auto_tmp = tmp_dir is None
        tmp_dir = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(prefix="cascade_"))
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Stage 1: L1 model
            l1_train = tmp_dir / "l1_train.txt"
            build_training_file(records, l1_train, mode="l1")
            self._l1_clf = MerchantClassifier(mode="l1", hyperparams=self._hyperparams)
            self._l1_clf.train(l1_train)
            logger.info("Cascade → L1 model trained.")

            # Stage 2: per-L1 L2 models
            l1_groups: dict[str, list[dict]] = defaultdict(list)
            for rec in records:
                l2 = rec.get("category_l2", "")
                if not is_active_l2(l2):
                    continue
                l1 = L2_TO_L1.get(l2, "diger")
                l1_groups[l1].append(rec)

            self._l2_clfs = {}
            for l1_label, l1_recs in sorted(l1_groups.items()):
                l2_train = tmp_dir / f"l2_{l1_label}.txt"
                written  = build_training_file(l1_recs, l2_train, mode="l2")
                if written == 0:
                    logger.warning(f"Cascade → {l1_label}: no lines to write, skipped.")
                    continue
                l2_clf = MerchantClassifier(mode="l2", hyperparams=self._hyperparams)
                l2_clf.train(l2_train)
                self._l2_clfs[l1_label] = l2_clf
                logger.info(
                    f"Cascade → L2 model [{l1_label}]: "
                    f"{written} samples, {l2_clf.n_labels} classes"
                )

            # Stage 3: flat L2 fallback (used when L1 confidence is too low)
            l2_flat_train = tmp_dir / "l2_flat_train.txt"
            build_training_file(records, l2_flat_train, mode="l2")
            self._l2_flat_clf = MerchantClassifier(mode="l2", hyperparams=self._hyperparams)
            self._l2_flat_clf.train(l2_flat_train)
            logger.info("Cascade → Flat L2 model trained (L1 uncertainty fallback).")

            self._trained_at = time.time()
            logger.info(
                f"Cascade training complete: 1 L1 + {len(self._l2_clfs)} L2 + 1 flat L2 model"
            )
        finally:
            if use_auto_tmp:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, merchant_raw: str, top_k: int = 3) -> Prediction:
        """
        Hierarchical prediction with three-tier confidence logic:

        Step 1 — L1 prediction.
          conf_l1 >= L1_THRESHOLD → Step 2 (cascade continues)
          conf_l1 <  L1_THRESHOLD → Step 3 (flat L2 fallback)

        Step 2 — L1-specific L2 prediction.
          conf_l2 >= L2_THRESHOLD → return (l1, l2)
          conf_l2 <  L2_THRESHOLD → return (l1, diger_<l1>)

        Step 3 — Flat L2 model (all ~100 L2 classes).
          conf >= L2_THRESHOLD → return (taxonomy-derived l1, l2)
          conf <  L2_THRESHOLD → return ("diger", "tanimlanamayan_isyeri")

        Combined confidence (cascade path): (conf_l1 × conf_l2)^0.5
        """
        if self._l1_clf is None:
            raise RuntimeError("Cascade model not trained. Call train() first.")

        from preprocessor import normalize
        from taxonomy import get_diger_l2, get_l1

        normalized = normalize(merchant_raw)
        if not normalized:
            return Prediction(
                merchant_raw=merchant_raw,
                merchant_normalized=normalized,
                category_l2="tanimlanamayan_isyeri",
                category_l1="diger",
                confidence=0.0,
                top_k=[],
            )

        # Step 1: L1 prediction
        l1_pred  = self._l1_clf.predict(merchant_raw, top_k=1)
        l1_label = l1_pred.category_l1
        l1_conf  = l1_pred.confidence

        if l1_conf >= CASCADE_L1_CONFIDENCE_THRESHOLD:
            # Step 2: L1-specific L2 prediction
            l2_clf = self._l2_clfs.get(l1_label)
            if l2_clf is not None:
                l2_pred = l2_clf.predict(merchant_raw, top_k=top_k)
                if l2_pred.confidence >= CASCADE_L2_CONFIDENCE_THRESHOLD:
                    combined = round((l1_conf * l2_pred.confidence) ** 0.5, 4)
                    return Prediction(
                        merchant_raw=merchant_raw,
                        merchant_normalized=normalized,
                        category_l2=l2_pred.category_l2,
                        category_l1=l1_label,
                        confidence=combined,
                        top_k=l2_pred.top_k,
                    )
                logger.debug(
                    f"Cascade L2 confidence low ({l2_pred.confidence:.2f}): "
                    f"'{l1_label}' → {get_diger_l2(l1_label)}"
                )
            else:
                logger.debug(f"Cascade: no L2 model found for '{l1_label}'.")

            return Prediction(
                merchant_raw=merchant_raw,
                merchant_normalized=normalized,
                category_l2=get_diger_l2(l1_label),
                category_l1=l1_label,
                confidence=round(l1_conf, 4),
                top_k=l1_pred.top_k,
            )

        # Step 3: low L1 confidence → flat L2 fallback
        logger.debug(
            f"Cascade L1 confidence low ({l1_conf:.2f}): trying flat L2 fallback."
        )
        if self._l2_flat_clf is not None:
            flat_pred = self._l2_flat_clf.predict(merchant_raw, top_k=top_k)
            if flat_pred.confidence >= CASCADE_L2_CONFIDENCE_THRESHOLD:
                return Prediction(
                    merchant_raw=merchant_raw,
                    merchant_normalized=normalized,
                    category_l2=flat_pred.category_l2,
                    category_l1=flat_pred.category_l1,
                    confidence=flat_pred.confidence,
                    top_k=flat_pred.top_k,
                )

        return Prediction(
            merchant_raw=merchant_raw,
            merchant_normalized=normalized,
            category_l2="tanimlanamayan_isyeri",
            category_l1="diger",
            confidence=0.0,
            top_k=[],
        )

    def predict_batch(self, merchants: list[str], top_k: int = 1) -> list[Prediction]:
        return [self.predict(m, top_k=top_k) for m in merchants]

    def get_sentence_vector(self, text: str) -> list[float]:
        """Delegate to the L1 model for visualizer compatibility."""
        if self._l1_clf is None:
            raise RuntimeError("Cascade model not trained.")
        return self._l1_clf.get_sentence_vector(text)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, test_path: str | Path) -> EvaluationResult:
        """
        Evaluate on a test file using full cascade inference.

        Expected format: __label__<l2_label> <normalized_text>
        """
        if self._l1_clf is None:
            raise RuntimeError("Cascade model not trained.")

        from collections import defaultdict
        from taxonomy import get_l1 as _get_l1

        test_path   = Path(test_path)
        prefix_len  = len(LABEL_PREFIX)
        true_labels: list[str] = []
        pred_labels: list[str] = []

        with open(test_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) < 2:
                    continue
                true_l2 = parts[0][prefix_len:]
                text    = parts[1]

                pred    = self.predict(text, top_k=1)
                pred_l2 = pred.category_l2 or "tanimlanamayan_isyeri"

                true_labels.append(true_l2)
                pred_labels.append(pred_l2)

        # Primary (L2) metrics
        try:
            from sklearn.metrics import (
                accuracy_score, f1_score, classification_report,
            )
            acc         = accuracy_score(true_labels, pred_labels)
            f1_macro    = f1_score(true_labels, pred_labels, average="macro",    zero_division=0)
            f1_weighted = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
            report      = classification_report(
                true_labels, pred_labels, output_dict=True, zero_division=0
            )
            per_class   = {
                k: v for k, v in report.items()
                if k not in ("accuracy", "macro avg", "weighted avg")
            }
        except ImportError:
            correct     = sum(t == p for t, p in zip(true_labels, pred_labels))
            acc         = correct / len(true_labels) if true_labels else 0.0
            f1_macro    = acc
            f1_weighted = acc
            per_class   = {}

        confusion: dict = defaultdict(lambda: defaultdict(int))
        for t, p in zip(true_labels, pred_labels):
            confusion[t][p] += 1

        # L1 metrics derived from L2 predictions
        true_l1 = [_get_l1(l) or "diger" for l in true_labels]
        pred_l1 = [_get_l1(l) or "diger" for l in pred_labels]

        try:
            from sklearn.metrics import (
                accuracy_score as _acc,
                f1_score as _f1,
                classification_report as _report,
            )
            acc_l1         = _acc(true_l1, pred_l1)
            f1_macro_l1    = _f1(true_l1, pred_l1, average="macro",    zero_division=0)
            f1_weighted_l1 = _f1(true_l1, pred_l1, average="weighted", zero_division=0)
            rep_l1         = _report(true_l1, pred_l1, output_dict=True, zero_division=0)
            per_class_l1   = {
                k: v for k, v in rep_l1.items()
                if k not in ("accuracy", "macro avg", "weighted avg")
            }
        except ImportError:
            correct_l1     = sum(t == p for t, p in zip(true_l1, pred_l1))
            acc_l1         = correct_l1 / len(true_l1) if true_l1 else 0.0
            f1_macro_l1    = acc_l1
            f1_weighted_l1 = acc_l1
            per_class_l1   = {}

        result = EvaluationResult(
            accuracy=round(acc, 4),
            f1_macro=round(f1_macro, 4),
            f1_weighted=round(f1_weighted, 4),
            n_samples=len(true_labels),
            per_class=per_class,
            confusion_matrix={k: dict(v) for k, v in confusion.items()},
            accuracy_l1=round(acc_l1, 4),
            f1_macro_l1=round(f1_macro_l1, 4),
            f1_weighted_l1=round(f1_weighted_l1, 4),
            per_class_l1=per_class_l1,
            _true_l1=true_l1,
            _pred_l1=pred_l1,
        )
        logger.info(
            f"Cascade Evaluation: n={result.n_samples} | "
            f"L2 Acc={result.accuracy:.3f} F1={result.f1_macro:.3f} | "
            f"L1 Acc={result.accuracy_l1:.3f} F1={result.f1_macro_l1:.3f}"
        )
        return result

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, model_dir: str | Path) -> None:
        """Save L1 model, all L2 models, flat L2 model, and cascade_meta.json."""
        if self._l1_clf is None:
            raise RuntimeError("No model to save.")
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        self._l1_clf.save(model_dir / self.L1_SUBDIR)

        l2_base = model_dir / self.L2_SUBDIR
        for l1_label, clf in self._l2_clfs.items():
            clf.save(l2_base / l1_label)

        has_flat = self._l2_flat_clf is not None
        if has_flat:
            self._l2_flat_clf.save(model_dir / self.L2_FLAT_SUBDIR)

        meta = {
            "mode":          "cascade",
            "trained_at":    self._trained_at,
            "l1_labels":     sorted(self._l2_clfs.keys()),
            "n_l2_models":   len(self._l2_clfs),
            "has_flat_l2":   has_flat,
            "l1_threshold":  CASCADE_L1_CONFIDENCE_THRESHOLD,
            "l2_threshold":  CASCADE_L2_CONFIDENCE_THRESHOLD,
        }
        with open(model_dir / self.META_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        flat_info = " + 1 flat L2" if has_flat else ""
        logger.info(
            f"Cascade model saved: {model_dir} "
            f"(1 L1 + {len(self._l2_clfs)} L2{flat_info})"
        )

    def load(self, model_dir: str | Path) -> None:
        """Load a saved cascade model from disk."""
        model_dir = Path(model_dir)
        meta_path = model_dir / self.META_FILE

        if not meta_path.exists():
            raise FileNotFoundError(
                f"Cascade meta file not found: {meta_path}\n"
                f"Train cascade with: python main.py train --mode cascade"
            )

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        self._l1_clf = MerchantClassifier(mode="l1")
        self._l1_clf.load(model_dir / self.L1_SUBDIR)

        l2_base = model_dir / self.L2_SUBDIR
        self._l2_clfs = {}
        for l1_label in meta.get("l1_labels", []):
            l2_dir = l2_base / l1_label
            if l2_dir.exists():
                clf = MerchantClassifier(mode="l2")
                clf.load(l2_dir)
                self._l2_clfs[l1_label] = clf
            else:
                logger.warning(f"Cascade: L2 model directory not found → {l2_dir}")

        self._l2_flat_clf = None
        flat_dir = model_dir / self.L2_FLAT_SUBDIR
        if flat_dir.exists():
            self._l2_flat_clf = MerchantClassifier(mode="l2")
            self._l2_flat_clf.load(flat_dir)
            flat_info = " + 1 flat L2"
        else:
            flat_info = " (no flat L2)"
            logger.warning(
                "Cascade: flat L2 model not found — L1 uncertainty fallback disabled. "
                "Retrain with: python main.py train --mode cascade"
            )

        logger.info(
            f"Cascade model loaded: {model_dir} "
            f"(1 L1 + {len(self._l2_clfs)} L2{flat_info})"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return "cascade"

    @property
    def is_trained(self) -> bool:
        return self._l1_clf is not None and self._l1_clf.is_trained

    @property
    def n_labels(self) -> int:
        return sum(c.n_labels for c in self._l2_clfs.values())

    def __repr__(self) -> str:
        status = (
            f"trained, 1 L1 + {len(self._l2_clfs)} L2 models, {self.n_labels} total labels"
            if self.is_trained else "untrained"
        )
        return f"CascadeClassifier[BERTurk]({status})"


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    print("BERTurk available:", _BERTURK_AVAILABLE)
    if not _BERTURK_AVAILABLE:
        sys.exit(1)

    print(f"Model: {BERTURK_MODEL_NAME}")
    print(f"Default hyperparameters: {DEFAULT_HYPERPARAMS}")
    print("For smoke test: python main.py demo")
