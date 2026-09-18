"""
Hızlı başlatma: python frontend/run.py
Tarayıcıda açar: http://127.0.0.1:5051

NOT: .venv/Scripts/python.exe ile çalıştırılmazsa
     otomatik olarak kendini .venv Python ile yeniden başlatır.
"""
import sys
import os
import subprocess
from pathlib import Path

ROOT_DIR   = Path(__file__).parent.parent
VENV_PY    = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
if not VENV_PY.exists():
    VENV_PY = ROOT_DIR / "venv" / "Scripts" / "python.exe"
THIS_FILE  = Path(__file__).resolve()

# venv Python'u varsa ve şu an onunla çalışmıyorsak, yeniden başlat
if VENV_PY.exists() and Path(sys.executable).resolve() != VENV_PY.resolve():
    import subprocess
    print(f"  venv Python kullanılıyor: {VENV_PY}")
    result = subprocess.run([str(VENV_PY), str(THIS_FILE)] + sys.argv[1:])
    sys.exit(result.returncode)

# ── Buraya geldiysek doğru Python'dayız ──
import webbrowser
import threading

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from app import create_app

def open_browser():
    webbrowser.open("http://127.0.0.1:5051")

if __name__ == "__main__":
    app = create_app()
    threading.Timer(1.2, open_browser).start()
    print("\n  Transaction Engine Lab")
    print("  ─────────────────────────────────")
    print("  http://127.0.0.1:5051\n")
    app.run(host="127.0.0.1", port=5051, debug=False)
