"""Visualization layer: all plot generation isolated from model.py."""

import logging
import random
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib not found. Install with: pip install matplotlib"
        )


def _require_sklearn():
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        raise ImportError(
            "scikit-learn not found. Install with: pip install scikit-learn"
        )


# ---------------------------------------------------------------------------
# 1. plot_evaluation
# ---------------------------------------------------------------------------

def plot_evaluation(result, plots_dir: str | Path) -> bool:
    """
    Generate confusion matrix and per-class F1 bar chart from an EvaluationResult.

    Outputs:
        {plots_dir}/confusion_matrix_l1.png
        {plots_dir}/per_class_f1_l1.png

    Returns True if both plots were produced successfully.
    """
    try:
        plt = _require_matplotlib()
    except ImportError as e:
        logger.warning(str(e))
        return False

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    true_l1 = result._true_l1
    pred_l1 = result._pred_l1

    if not true_l1:
        logger.warning("Raw L1 labels are empty; cannot generate plot.")
        return False

    # Confusion matrix
    labels = sorted(set(true_l1) | set(pred_l1))
    n      = len(labels)
    idx    = {l: i for i, l in enumerate(labels)}
    matrix = [[0] * n for _ in range(n)]
    for t, p in zip(true_l1, pred_l1):
        if t in idx and p in idx:
            matrix[idx[t]][idx[p]] += 1

    # Row-normalize (recall perspective)
    norm_matrix = []
    for row in matrix:
        total = sum(row) or 1
        norm_matrix.append([v / total for v in row])

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(norm_matrix, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall (row-normalized)")

    short = [l.replace("_", "\n") for l in labels]
    ax.set_xticks(range(n)); ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(short, fontsize=7)
    ax.set_xlabel("Predicted L1", fontsize=11, labelpad=10)
    ax.set_ylabel("True L1",      fontsize=11, labelpad=10)
    ax.set_title(
        f"L1 Confusion Matrix — n={len(true_l1):,} | "
        f"Acc={result.accuracy:.3f}  F1={result.f1_macro:.3f}",
        fontsize=12, pad=14,
    )
    for i in range(n):
        for j in range(n):
            v = norm_matrix[i][j]
            if v >= 0.05:
                color = "white" if v > 0.55 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.5, color=color,
                        fontweight="bold" if i == j else "normal")

    fig.tight_layout()
    out_cm = plots_dir / "confusion_matrix_l1.png"
    fig.savefig(out_cm, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix kaydedildi: {out_cm}")

    # Per-class F1 bar chart
    if not result.per_class_l1:
        logger.warning("per_class_l1 boş; F1 bar chart üretilemiyor.")
        return True

    items = [
        (l1, m.get("f1-score", 0.0), int(m.get("support", 0)))
        for l1, m in result.per_class_l1.items()
    ]
    items.sort(key=lambda x: x[1])
    cats   = [x[0] for x in items]
    f1s    = [x[1] for x in items]
    counts = [x[2] for x in items]

    colors = [
        "#E24B4A" if f < 0.40 else
        "#EF9F27" if f < 0.65 else
        "#185FA5"
        for f in f1s
    ]

    fig, ax = plt.subplots(figsize=(10, 0.55 * len(cats) + 1.8))
    bars = ax.barh(cats, f1s, color=colors, height=0.65, edgecolor="none")

    for bar, f1, cnt in zip(bars, f1s, counts):
        ax.text(
            min(f1 + 0.01, 0.97), bar.get_y() + bar.get_height() / 2,
            f"{f1:.3f}  (n={cnt})", va="center", fontsize=9,
            color="#2C2C2A" if f1 < 0.55 else "white" if f1 > 0.72 else "#2C2C2A",
        )

    ax.axvline(0.60, color="#1D9E75", linestyle="--", linewidth=1.2, alpha=0.8,
               label="LOW_CONFIDENCE eşiği (0.60)")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1 Score", fontsize=11)
    ax.set_title(
        f"L1 Kategori F1 Performansı — F1 Macro={result.f1_macro:.3f}",
        fontsize=12, pad=14,
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_f1 = plots_dir / "per_class_f1_l1.png"
    fig.savefig(out_f1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Per-class F1 grafiği kaydedildi: {out_f1}")
    return True


# ---------------------------------------------------------------------------
# 2. plot_training_curve
# ---------------------------------------------------------------------------

def plot_training_curve(
    classifier,
    train_path: str | Path | None = None,
    plots_dir: str | Path = "models/plots",
    epoch_checkpoints: list[int] | None = None,
    val_ratio: float = 0.20,
) -> bool:
    """
    Plot loss and accuracy curves from classifier._training_history.

    Returns True if the plot was saved, False if matplotlib is missing
    or training history is empty.
    """
    try:
        plt = _require_matplotlib()
    except ImportError as e:
        logger.warning(str(e))
        return False

    history = getattr(classifier, "_training_history", [])
    if not history:
        logger.warning(
            "classifier._training_history boş; eğitim eğrisi üretilemiyor. "
            "Grafik için modeli train() ile eğitin."
        )
        return False

    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    epochs     = [h["epoch"]    for h in history]
    losses     = [h["loss"]     for h in history]
    accuracies = [h["accuracy"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss curve
    ax1.plot(epochs, losses, "o-", color="#D85A30", linewidth=2,
             markersize=6, label="Train loss")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss (Cross-Entropy)", fontsize=12)
    ax1.set_title("Eğitim Loss Eğrisi", fontsize=13, pad=12)
    ax1.set_xticks(epochs)
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)

    best_loss_ep = epochs[losses.index(min(losses))]
    ax1.axvline(best_loss_ep, color="#1D9E75", linestyle=":", linewidth=1.5, alpha=0.8,
                label=f"Min loss (epoch {best_loss_ep})")

    # Accuracy curve
    ax2.plot(epochs, accuracies, "s-", color="#185FA5", linewidth=2,
             markersize=6, label="Train accuracy")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Eğitim Accuracy Eğrisi", fontsize=13, pad=12)
    ax2.set_ylim(0, 1.05)
    ax2.set_xticks(epochs)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)

    best_acc     = max(accuracies)
    best_acc_ep  = epochs[accuracies.index(best_acc)]
    ax2.axvline(best_acc_ep, color="#1D9E75", linestyle=":", linewidth=1.5, alpha=0.8)
    ax2.annotate(
        f"Peak acc\nepoch={best_acc_ep}\n{best_acc:.3f}",
        xy=(best_acc_ep, best_acc),
        xytext=(best_acc_ep + max(epochs) * 0.05, best_acc - 0.08),
        fontsize=9, color="#1D9E75",
        arrowprops=dict(arrowstyle="->", color="#1D9E75", lw=1.2),
    )

    mode_str = getattr(classifier, "_mode", "?").upper()
    fig.suptitle(
        f"BERTurk Fine-Tuning Eğrisi — [{mode_str}] | {len(epochs)} Epoch",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()

    out = plots_dir / "training_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Eğitim eğrisi kaydedildi: {out}")
    return True


# ---------------------------------------------------------------------------
# 3. plot_tsne
# ---------------------------------------------------------------------------

# Fixed color palette for 16 L1 categories
_L1_COLORS: dict[str, str] = {
    "yeme_icme":        "#E63946",
    "market_gida":      "#2A9D8F",
    "giyim_aksesuar":   "#E9C46A",
    "online_alisveris": "#F4A261",
    "elektronik":       "#264653",
    "ev_yasam":         "#A8DADC",
    "saglik":           "#457B9D",
    "kisisel_bakim":    "#1D3557",
    "ulasim":           "#6A4C93",
    "seyahat":          "#1982C4",
    "eglence_kultur":   "#8AC926",
    "fatura_abonelik":  "#FF595E",
    "finans_sigorta":   "#FFCA3A",
    "akaryakit":        "#6A994E",
    "egitim":           "#BC6C25",
    "diger":            "#ADB5BD",
}

_DEFAULT_COLOR = "#999999"


def plot_tsne(
    classifier,
    train_path: str | Path,
    plots_dir: str | Path,
    n_samples: int = 1500,
    perplexity: int = 40,
    random_state: int = 42,
    highlight_weak: bool = True,
) -> bool:
    """
    Project BERTurk CLS embeddings to 2D with t-SNE and save a scatter plot.

    Each point is a training example colored by L1 category.
    Overlapping clusters indicate categories the model confuses.

    Args:
        classifier   : trained MerchantClassifier
        train_path   : training file with __label__<l1> <text> lines
        plots_dir    : output directory
        n_samples    : max samples (balanced per category); t-SNE is O(n²)
        perplexity   : t-SNE perplexity (5–50 recommended)
        random_state : reproducibility seed
        highlight_weak: mark low-F1 categories with a triangle marker

    Returns True if the plot was saved successfully.
    """
    try:
        plt = _require_matplotlib()
    except ImportError as e:
        logger.warning(str(e))
        return False

    try:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.warning("scikit-learn bulunamadı; t-SNE üretilemiyor. pip install scikit-learn")
        return False

    if not classifier.is_trained:
        logger.error("Model eğitilmemiş; t-SNE üretilemiyor.")
        return False

    train_path = Path(train_path)
    plots_dir  = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"t-SNE: eğitim dosyası okunuyor → {train_path}")
    samples: list[tuple[str, str]] = []
    prefix_len = len("__label__")

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
                samples.append((text, label))

    if not samples:
        logger.error("Eğitim dosyasında geçerli satır bulunamadı.")
        return False

    logger.info(f"t-SNE: toplam {len(samples)} örnek okundu.")

    # Balanced per-category sampling to avoid majority-class visual dominance
    from collections import defaultdict
    by_label: dict[str, list[str]] = defaultdict(list)
    for text, label in samples:
        by_label[label].append(text)

    n_cats     = len(by_label)
    per_cat    = max(1, n_samples // n_cats)
    random.seed(random_state)

    selected_texts:  list[str] = []
    selected_labels: list[str] = []

    for label, texts in by_label.items():
        chosen = random.sample(texts, min(per_cat, len(texts)))
        selected_texts.extend(chosen)
        selected_labels.extend([label] * len(chosen))

    total = len(selected_texts)
    logger.info(
        f"t-SNE: {total} örnek seçildi "
        f"({n_cats} kategori × ~{per_cat} örnek)"
    )

    # Extract BERTurk CLS token embeddings
    logger.info("t-SNE: BERTurk CLS embedding'leri hesaplanıyor...")
    vectors = []
    for text in selected_texts:
        vec = classifier.get_sentence_vector(text)
        vectors.append(vec)

    import numpy as np
    X = np.array(vectors, dtype=np.float32)

    # StandardScaler before t-SNE: raw CLS magnitudes vary across layers
    X = StandardScaler().fit_transform(X)

    logger.info(
        f"t-SNE: boyut indirgeme başlıyor "
        f"(n={total}, perplexity={perplexity})..."
    )
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, total - 1),
        random_state=random_state,
        max_iter=1000,
        learning_rate="auto",
        init="pca",       # PCA init → more stable convergence
        verbose=0,
    )
    X_2d = tsne.fit_transform(X)
    logger.info("t-SNE: boyut indirgeme tamamlandı.")

    # Categories known to have low F1 (visual diagnostic aid)
    weak_cats = {"online_alisveris", "elektronik", "ev_yasam", "ulasim", "seyahat"}

    all_labels  = sorted(set(selected_labels))
    fig, ax     = plt.subplots(figsize=(14, 11))
    legend_handles = []

    for label in all_labels:
        mask   = [i for i, l in enumerate(selected_labels) if l == label]
        xs     = X_2d[mask, 0]
        ys     = X_2d[mask, 1]
        color  = _L1_COLORS.get(label, _DEFAULT_COLOR)
        is_weak= highlight_weak and label in weak_cats
        size   = 55 if is_weak else 28
        alpha  = 0.85 if is_weak else 0.65
        marker = "^" if is_weak else "o"
        zorder = 4 if is_weak else 3

        sc = ax.scatter(
            xs, ys,
            c=color, s=size, alpha=alpha,
            marker=marker, zorder=zorder,
            edgecolors="white" if is_weak else "none",
            linewidths=0.4,
        )
        legend_label = f"{label}  ⚠" if is_weak else label
        legend_handles.append(
            plt.Line2D(
                [0], [0],
                marker=marker, color="w",
                markerfacecolor=color,
                markersize=8 if is_weak else 6,
                label=f"{legend_label}  (n={len(mask)})",
            )
        )

    ax.set_title(
        f"BERTurk CLS Embedding'leri — t-SNE Projeksiyonu\n"
        f"n={total} örnek | {n_cats} L1 kategori | perplexity={perplexity}\n"
        f"⚠  işareti düşük F1 (<0.45) kategorileri gösterir",
        fontsize=12, pad=16, linespacing=1.6,
    )
    ax.set_xlabel("t-SNE Boyut 1", fontsize=10)
    ax.set_ylabel("t-SNE Boyut 2", fontsize=10)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)

    ax.legend(
        handles=legend_handles,
        title="L1 Kategori",
        title_fontsize=9,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        frameon=True,
        framealpha=0.9,
        ncol=1,
    )

    fig.text(
        0.5, -0.01,
        "Birbirine yakın kümeler → model bu kategorileri karıştırıyor.  "
        "Ayrışmış kümeler → güçlü sınıf ayrımı.",
        ha="center", fontsize=8.5, color="#555555", style="italic",
    )

    out = plots_dir / "tsne_embeddings.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"t-SNE grafiği kaydedildi: {out}")
    return True


# ---------------------------------------------------------------------------
# Convenience: generate all plots in one call
# ---------------------------------------------------------------------------

def plot_all(
    result,
    classifier,
    train_path: str | Path,
    plots_dir: str | Path,
    tsne_samples: int = 1500,
) -> dict[str, bool]:
    """
    Run all plot functions and return a dict of {plot_name: success}.

    Args:
        result      : EvaluationResult (skips evaluation plots if None)
        classifier  : trained MerchantClassifier
        train_path  : training file for t-SNE
        plots_dir   : output directory
        tsne_samples: sample count for t-SNE
    """
    status = {}

    if result is not None:
        status["confusion_matrix + f1_bar"] = plot_evaluation(result, plots_dir)

    status["tsne"] = plot_tsne(
        classifier, train_path, plots_dir, n_samples=tsne_samples
    )

    return status


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("visualizer.py — bağımsız test için main.py üzerinden çalıştırın:")
    print("  python main.py tsne")
    print("  python main.py evaluate data/master_test_oov.csv")
