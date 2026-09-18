"""
frontend/app.py — Flask Uygulama Fabrikası

Çalıştırmak için:
    cd frontend
    flask run --port 5050
    # veya:
    python app.py
"""

import sys
from pathlib import Path

# src/ modüllerini import edebilmek için proje kökünü sys.path'e ekle
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))


from flask import Flask, redirect, url_for
from routes.ml_routes import ml_bp
from routes.statement_routes import statement_bp


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")

    app.register_blueprint(ml_bp, url_prefix="/ml")
    app.register_blueprint(statement_bp, url_prefix="/statement")

    @app.route("/")
    def index():
        return redirect(url_for("ml.dashboard"))

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(debug=True, port=5051)
