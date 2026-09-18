"""
routes/ml_routes.py — ML Operasyonları Rotaları

Endpoint'ler:
    GET  /ml/              — Dashboard sayfası
    GET  /ml/datasets      — data/ klasöründeki CSV listesi (JSON)
    GET  /ml/experiments   — models/experiments/ sonuçları (JSON)
    POST /ml/predict       — Tek merchant tahmini
    POST /ml/train         — Model eğitimi (subprocess, streaming SSE)
    POST /ml/evaluate      — Model değerlendirmesi
    POST /ml/tsne          — t-SNE görselleştirmesi (base64 PNG döner)
    GET  /ml/plot/<name>   — models/plots/ altındaki PNG'yi serve et
"""

import json
import subprocess
import sys
from pathlib import Path

from flask import (
    Blueprint, Response, jsonify, render_template,
    request, send_file, stream_with_context,
)

# ── Model önbelleği — sunucu ömrü boyunca tek seferlik yükleme ──
_router_cache: dict = {}   # key: inference_mode → HybridRouter

ROOT_DIR   = Path(__file__).parent.parent.parent
DATA_DIR   = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
PLOTS_DIR  = MODELS_DIR / "plots"
EXPERIMENTS_DIR = MODELS_DIR / "experiments"
MAIN_PY    = ROOT_DIR / "main.py"

# .venv veya venv klasörünü dene; yoksa mevcut python
_VENV_PY = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
if not _VENV_PY.exists():
    _VENV_PY = ROOT_DIR / "venv" / "Scripts" / "python.exe"
PYTHON_EXE = str(_VENV_PY) if _VENV_PY.exists() else sys.executable

ml_bp = Blueprint("ml", __name__, template_folder="../templates")


# ---------------------------------------------------------------------------
# Yardımcılar
# ---------------------------------------------------------------------------

def _list_csv_files() -> list[dict]:
    """data/ klasöründeki CSV dosyalarını döner."""
    files = []
    for p in sorted(DATA_DIR.glob("*.csv")):
        files.append({"name": p.name, "path": str(p), "size_kb": round(p.stat().st_size / 1024, 1)})
    return files


def _list_experiments() -> list[dict]:
    """models/experiments/ klasöründeki JSON sonuçlarını döner."""
    results = []
    if not EXPERIMENTS_DIR.exists():
        return results
    for p in sorted(EXPERIMENTS_DIR.glob("eval_*.json"), reverse=True):
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            results.append({
                "filename": p.name,
                "timestamp": data.get("timestamp", p.stem.replace("eval_", "")),
                "mode": data.get("mode", "?"),
                "source": data.get("source", "?"),
                "accuracy_l1": data.get("accuracy_l1", data.get("accuracy")),
                "accuracy_l2": data.get("accuracy_l2"),
                "macro_f1_l1": data.get("macro_f1_l1", data.get("macro_f1")),
                "macro_f1_l2": data.get("macro_f1_l2"),
                "n_samples": data.get("n_samples"),
            })
        except Exception:
            pass
    return results


def _get_or_load_router(mode: str = "cascade"):
    """
    Router'ı önbellekten döner; yoksa yükler.
    BERTurk modelleri: berturk_cascade (cascade) veya berturk_v1 (l1/l2).
    """
    if mode in _router_cache:
        return _router_cache[mode]

    import json as _json
    from router import HybridRouter

    cascade_dir  = MODELS_DIR / "berturk_mode_cascade"
    berturk_l2   = MODELS_DIR / "berturk_mode_l2"
    berturk_l1   = MODELS_DIR / "berturk_mode_l1"

    if mode == "cascade" and (cascade_dir / "cascade_meta.json").exists():
        router = HybridRouter(inference_mode="cascade")
        router.load_model(str(cascade_dir))
    elif mode == "l1" and (berturk_l1 / "metrics.json").exists():
        router = HybridRouter(inference_mode="l1")
        router.load_model(str(berturk_l1))
    elif (berturk_l2 / "metrics.json").exists():
        router = HybridRouter(inference_mode="l2")
        router.load_model(str(berturk_l2))
    else:
        return None

    _router_cache[mode] = router
    return router


def _invalidate_router_cache():
    """Eğitim sonrası cache'i temizle — modeller yeniden yüklensin."""
    _router_cache.clear()


def _run_subprocess(cmd: list[str]) -> tuple[str, str, int]:
    """main.py komutunu çalıştırır; (stdout, stderr, returncode) döner."""
    try:
        env = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}
        result = subprocess.run(
            [PYTHON_EXE] + cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(ROOT_DIR),
            env=env,
        )
        return result.stdout or "", result.stderr or "", result.returncode
    except Exception as exc:
        return "", str(exc), 1


# ---------------------------------------------------------------------------
# Rotalar
# ---------------------------------------------------------------------------

@ml_bp.route("/")
def dashboard():
    datasets    = _list_csv_files()
    experiments = _list_experiments()
    plots       = [p.name for p in PLOTS_DIR.glob("*.png")] if PLOTS_DIR.exists() else []
    return render_template(
        "ml/dashboard.html",
        datasets=datasets,
        experiments=experiments,
        plots=plots,
    )


@ml_bp.route("/datasets")
def datasets():
    return jsonify(_list_csv_files())


@ml_bp.route("/experiments")
def experiments():
    return jsonify(_list_experiments())


@ml_bp.route("/predict", methods=["POST"])
def predict():
    data     = request.get_json(force=True)
    merchant = (data.get("merchant") or "").strip()
    mode     = data.get("inference_mode", "cascade")

    if not merchant:
        return jsonify({"error": "merchant boş olamaz"}), 400

    try:
        router = _get_or_load_router(mode)
        if router is None:
            return jsonify({"error": "Model eğitilmemiş. Lütfen önce 'Eğitim' sekmesinden modeli eğitin."}), 500

        result = router.enrich(merchant, top_k=5)

        return jsonify({
            "merchant_raw":        result.merchant_raw,
            "merchant_normalized": result.merchant_normalized,
            "category_l1":         result.category_l1,
            "category_l2":         result.category_l2,
            "confidence":          round(result.confidence, 4),
            "source":              result.source,
            "flag":                result.flag,
            "top_k": [
                {"label": lbl, "confidence": round(conf, 4)}
                for lbl, conf in (result.top_k or [])
            ],
        })
    except FileNotFoundError:
        return jsonify({"error": "Model eğitilmemiş. Lütfen önce 'Eğitim' sekmesinden modeli eğitin."}), 500
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@ml_bp.route("/train/stream", methods=["POST"])
def train_stream():
    """SSE ile train komutunu çalıştırıp çıktıyı adım adım gönderir."""
    data      = request.get_json(force=True)
    dataset   = data.get("dataset", "")
    mode      = data.get("mode", "cascade")
    lr        = data.get("lr", "")
    epoch     = data.get("epoch", "")

    cmd = [str(MAIN_PY), "train", dataset, "--mode", mode]
    if lr:
        cmd += ["--lr", str(lr)]
    if epoch:
        cmd += ["--epoch", str(epoch)]

    def generate():
        import os as _os
        _env = {**_os.environ, "PYTHONIOENCODING": "utf-8"}
        process = subprocess.Popen(
            [PYTHON_EXE] + cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(ROOT_DIR),
            env=_env,
        )
        for line in iter(process.stdout.readline, ""):
            yield f"data: {json.dumps({'line': line.rstrip()})}\n\n"
        process.wait()
        yield f"data: {json.dumps({'done': True, 'returncode': process.returncode})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@ml_bp.route("/evaluate", methods=["POST"])
def evaluate():
    try:
        data    = request.get_json(force=True) or {}
        dataset = data.get("dataset", "")
        mode    = data.get("mode", "")

        if not dataset:
            return jsonify({"error": "dataset seçilmedi"}), 400

        cmd = [str(MAIN_PY), "evaluate", dataset]
        # main.py berturk_v1 ve berturk_cascade'i otomatik algılar; model-dir gerekmez

        stdout, stderr, rc = _run_subprocess(cmd)
        output = stdout + (("\n[STDERR]\n" + stderr) if stderr.strip() else "")

        experiments = _list_experiments()
        latest = experiments[0] if experiments else None

        return jsonify({
            "output":            output,
            "success":           rc == 0,
            "latest_experiment": latest,
        })
    except Exception as exc:
        return jsonify({"error": str(exc), "output": "", "success": False}), 500


@ml_bp.route("/tsne", methods=["POST"])
def tsne():
    data    = request.get_json(force=True)
    dataset = data.get("dataset", "")
    n       = data.get("n", 500)

    cmd = [str(MAIN_PY), "tsne", "--samples", str(n)]
    if dataset:
        cmd += ["--train", dataset]

    stdout, stderr, rc = _run_subprocess(cmd)

    # Oluşturulan PNG'yi base64 olarak döndür
    tsne_png = PLOTS_DIR / "tsne_embeddings.png"
    img_b64  = None
    if tsne_png.exists():
        import base64
        with open(tsne_png, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

    return jsonify({
        "output":  stdout + (("\n[STDERR]\n" + stderr) if stderr.strip() else ""),
        "success": rc == 0,
        "image":   img_b64,
    })


@ml_bp.route("/plot/<filename>")
def serve_plot(filename: str):
    """models/plots/ altındaki PNG'leri serve eder."""
    path = PLOTS_DIR / filename
    if not path.exists() or not path.suffix == ".png":
        return "", 404
    return send_file(str(path), mimetype="image/png")


@ml_bp.route("/cache/clear", methods=["POST"])
def cache_clear():
    """Eğitim sonrası model önbelleğini temizle."""
    _invalidate_router_cache()
    return jsonify({"ok": True})


@ml_bp.route("/taxonomy")
def taxonomy():
    """Taxonomy özetini JSON olarak döndür."""
    try:
        from taxonomy import TAXONOMY, get_l1
        result = {}
        for l1, l2_list in TAXONOMY.items():
            result[l1] = l2_list
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
