/**
 * main.js — ML Paneli JavaScript Mantığı
 * Tab yönetimi, Predict, Train (SSE), Evaluate, t-SNE
 */

"use strict";

// ──────────────────────────────────────────────────────────────────
// Tab Yönetimi
// ──────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });
  // URL hash'e göre tab aç
  const hash = location.hash.replace("#", "");
  if (hash && document.getElementById(`tab-${hash}`)) {
    switchTab(hash);
  }
});

function switchTab(tabName) {
  document.querySelectorAll(".tab-btn").forEach((b) =>
    b.classList.toggle("active", b.dataset.tab === tabName)
  );
  document.querySelectorAll(".tab-panel").forEach((p) =>
    p.classList.toggle("active", p.id === `tab-${tabName}`)
  );
  history.replaceState(null, "", `#${tabName}`);
}

// ──────────────────────────────────────────────────────────────────
// Tahmin
// ──────────────────────────────────────────────────────────────────

function quickPredict(merchant) {
  document.getElementById("predictMerchant").value = merchant;
  runPredict();
}

function clearPredict() {
  document.getElementById("predictMerchant").value = "";
  document.getElementById("predictResult").classList.add("hidden");
  document.getElementById("predictEmpty").classList.remove("hidden");
}

async function runPredict() {
  const merchant = document.getElementById("predictMerchant").value.trim();
  const mode = document.getElementById("predictMode").value;

  if (!merchant) {
    showAlert("predictResult", "warning", t("predict.err_empty"));
    return;
  }

  const loading = document.getElementById("predictLoading");
  const result = document.getElementById("predictResult");
  const empty = document.getElementById("predictEmpty");

  loading.classList.remove("hidden");
  result.classList.add("hidden");
  empty.classList.add("hidden");

  try {
    const res = await fetch("/ml/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ merchant, inference_mode: mode }),
    });
    const data = await res.json();
    loading.classList.add("hidden");

    if (!res.ok || data.error) {
      result.innerHTML = renderAlert("error", data.error || t("predict.err_server"));
      result.classList.remove("hidden");
      return;
    }

    result.innerHTML = renderPredictResult(data);
    result.classList.remove("hidden");
  } catch (err) {
    loading.classList.add("hidden");
    result.innerHTML = renderAlert("error", t("predict.err_connection") + err.message);
    result.classList.remove("hidden");
  }
}

function renderPredictResult(d) {
  const confPct = Math.round(d.confidence * 100);
  const confClass = confPct >= 85 ? "high" : confPct >= 60 ? "mid" : "low";

  let topkHtml = "";
  if (d.top_k && d.top_k.length > 1) {
    const items = d.top_k
      .slice(1)
      .map(
        (t) => `
      <div class="topk-item">
        <span class="topk-item__label">${escHtml(t.label)}</span>
        <div class="confidence-bar">
          <div class="confidence-bar__track">
            <div class="confidence-bar__fill ${
              t.confidence >= 0.85 ? "high" : t.confidence >= 0.6 ? "mid" : "low"
            }" style="width:${Math.round(t.confidence * 100)}%"></div>
          </div>
          <span class="confidence-bar__val">${(t.confidence * 100).toFixed(1)}%</span>
        </div>
      </div>`
      )
      .join("");
    topkHtml = `<div class="topk-list"><p class="text-muted mb-1" style="font-size:.78rem;text-transform:uppercase;letter-spacing:.05em;" data-i18n="predict.alternatives">${t("predict.alternatives")}</p>${items}</div>`;
  }

  const flagHtml = d.flag
    ? `<span class="badge badge--warning" style="margin-left:6px;">${escHtml(d.flag)}</span>`
    : "";

  return `
  <div class="predict-result">
    <div class="predict-result__merchant">"${escHtml(d.merchant_raw)}"${flagHtml}</div>
    ${
      d.merchant_normalized !== d.merchant_raw
        ? `<div class="predict-result__row"><span class="predict-result__key" data-i18n="predict.key_normalized">${t("predict.key_normalized")}</span><span class="text-muted">${escHtml(d.merchant_normalized)}</span></div>`
        : ""
    }
    <div class="predict-result__row">
      <span class="predict-result__key" data-i18n="predict.key_l1">${t("predict.key_l1")}</span>
      <span class="badge badge--l1">${escHtml(d.category_l1 || "—")}</span>
    </div>
    ${
      d.category_l2
        ? `<div class="predict-result__row"><span class="predict-result__key" data-i18n="predict.key_l2">${t("predict.key_l2")}</span><span class="badge badge--l2">${escHtml(d.category_l2)}</span></div>`
        : ""
    }
    <div class="predict-result__row">
      <span class="predict-result__key" data-i18n="predict.key_confidence">${t("predict.key_confidence")}</span>
      <div class="confidence-bar">
        <div class="confidence-bar__track">
          <div class="confidence-bar__fill ${confClass}" style="width:${confPct}%"></div>
        </div>
        <span class="confidence-bar__val">${confPct}%</span>
      </div>
    </div>
    <div class="predict-result__row">
      <span class="predict-result__key" data-i18n="predict.key_source">${t("predict.key_source")}</span>
      <span class="badge badge--source">${escHtml(d.source)}</span>
    </div>
    ${topkHtml}
  </div>`;
}

// Enter tuşu ile tahmin
document.addEventListener("DOMContentLoaded", () => {
  const inp = document.getElementById("predictMerchant");
  if (inp) inp.addEventListener("keydown", (e) => { if (e.key === "Enter") runPredict(); });
});

// ──────────────────────────────────────────────────────────────────
// Eğitim (SSE ile streaming)
// ──────────────────────────────────────────────────────────────────

let _trainEventSource = null;

function runTrain() {
  const dataset = document.getElementById("trainDataset").value;
  const mode = document.getElementById("trainMode").value;
  const lr = document.getElementById("trainLr").value;
  const epoch = document.getElementById("trainEpoch").value;

  if (!dataset) { alert("Veri seti seçin."); return; }

  const terminal = document.getElementById("trainTerminal");
  const statusBadge = document.getElementById("trainStatus");
  const btn = document.getElementById("btnTrain");

  terminal.innerHTML = "";
  statusBadge.textContent = "Çalışıyor...";
  statusBadge.className = "badge badge--warning";
  btn.disabled = true;

  // Önceki SSE varsa kapat
  if (_trainEventSource) _trainEventSource.close();

  // SSE isteği için body göndermek gerektiğinden fetch + ReadableStream kullanıyoruz
  fetch("/ml/train/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset, mode, lr, epoch }),
  }).then((res) => {
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    function read() {
      reader.read().then(({ done, value }) => {
        if (done) {
          statusBadge.textContent = "Tamamlandı";
          statusBadge.className = "badge badge--success";
          btn.disabled = false;
          return;
        }
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop(); // tamamlanmamış son satırı sakla

        lines.forEach((line) => {
          if (line.startsWith("data: ")) {
            try {
              const payload = JSON.parse(line.slice(6));
              if (payload.line !== undefined) {
                terminal.innerHTML += colorizeTerminalLine(escHtml(payload.line)) + "\n";
                terminal.scrollTop = terminal.scrollHeight;
              }
              if (payload.done) {
                const ok = payload.returncode === 0;
                if (ok) fetch("/ml/cache/clear", { method: "POST" }); // yeni modeli yükle
                statusBadge.textContent = ok ? "Başarılı" : "Hata";
                statusBadge.className = `badge badge--${ok ? "success" : "error"}`;
                btn.disabled = false;
              }
            } catch (_) {}
          }
        });
        read();
      });
    }
    read();
  }).catch((err) => {
    terminal.innerHTML += `\n<span class="t-error">Bağlantı hatası: ${escHtml(err.message)}</span>`;
    statusBadge.textContent = "Hata";
    statusBadge.className = "badge badge--error";
    btn.disabled = false;
  });
}

function clearTerminal(id) {
  const t = document.getElementById(id);
  if (t) t.innerHTML = "";
}

// ──────────────────────────────────────────────────────────────────
// Değerlendirme
// ──────────────────────────────────────────────────────────────────

async function runEvaluate() {
  const dataset = document.getElementById("evalDataset").value;
  const mode = document.getElementById("evalMode").value;

  if (!dataset) { alert("Test veri seti seçin."); return; }

  const terminal = document.getElementById("evalTerminal");
  const loading = document.getElementById("evalLoading");
  const metricsDiv = document.getElementById("evalMetrics");
  const statusBadge = document.getElementById("evalStatus");
  const btn = document.getElementById("btnEval");

  terminal.innerHTML = "Değerlendirme çalıştırılıyor...";
  loading.classList.remove("hidden");
  metricsDiv.classList.add("hidden");
  statusBadge.textContent = "Çalışıyor...";
  statusBadge.className = "badge badge--warning";
  btn.disabled = true;

  try {
    const res = await fetch("/ml/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset, mode }),
    });
    const data = await res.json();

    loading.classList.add("hidden");
    btn.disabled = false;

    terminal.innerHTML = colorizeTerminalOutput(escHtml(data.output || ""));

    if (data.success) {
      statusBadge.textContent = "Başarılı";
      statusBadge.className = "badge badge--success";

      if (data.latest_experiment) {
        metricsDiv.innerHTML = renderExpMetrics(data.latest_experiment);
        metricsDiv.classList.remove("hidden");
      }
    } else {
      statusBadge.textContent = "Hata";
      statusBadge.className = "badge badge--error";
    }
  } catch (err) {
    loading.classList.add("hidden");
    terminal.innerHTML = `<span class="t-error">Bağlantı hatası: ${escHtml(err.message)}</span>`;
    btn.disabled = false;
    statusBadge.textContent = "Hata";
    statusBadge.className = "badge badge--error";
  }
}

function renderExpMetrics(exp) {
  const rows = [
    ["L1 Accuracy", exp.accuracy_l1, true],
    ["L2 Accuracy", exp.accuracy_l2, true],
    ["L1 Macro-F1", exp.macro_f1_l1, false],
    ["L2 Macro-F1", exp.macro_f1_l2, false],
  ]
    .filter(([, v]) => v !== null && v !== undefined)
    .map(
      ([label, val, isPct]) => `
    <div class="stat-card stat-card--navy">
      <div class="stat-card__value">${isPct ? (val * 100).toFixed(1) + "%" : val.toFixed(3)}</div>
      <div class="stat-card__label">${label}</div>
    </div>`
    )
    .join("");
  return `<div class="stat-grid">${rows}</div>`;
}

async function refreshExperiments() {
  await fetch("/ml/experiments");
  location.reload();
}

// ──────────────────────────────────────────────────────────────────
// t-SNE
// ──────────────────────────────────────────────────────────────────

async function runTsne() {
  const dataset = document.getElementById("tsneDataset").value;
  const n = document.getElementById("tsneN").value || 500;

  const loading = document.getElementById("tsneLoading");
  const imageDiv = document.getElementById("tsneImage");
  const emptyDiv = document.getElementById("tsneEmpty");
  const terminal = document.getElementById("tsneTerminal");
  const statusBadge = document.getElementById("tsneStatus");
  const btn = document.getElementById("btnTsne");

  loading.classList.remove("hidden");
  imageDiv.classList.add("hidden");
  emptyDiv.classList.add("hidden");
  terminal.innerHTML = "t-SNE hesaplanıyor...";
  statusBadge.textContent = "Çalışıyor...";
  statusBadge.className = "badge badge--warning";
  btn.disabled = true;

  try {
    const res = await fetch("/ml/tsne", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset, n: parseInt(n) }),
    });
    const data = await res.json();

    loading.classList.add("hidden");
    btn.disabled = false;
    terminal.innerHTML = colorizeTerminalOutput(escHtml(data.output || ""));

    if (data.success && data.image) {
      document.getElementById("tsneImg").src = "data:image/png;base64," + data.image;
      imageDiv.classList.remove("hidden");
      statusBadge.textContent = "Hazır";
      statusBadge.className = "badge badge--success";
    } else {
      emptyDiv.classList.remove("hidden");
      statusBadge.textContent = "Hata";
      statusBadge.className = "badge badge--error";
    }
  } catch (err) {
    loading.classList.add("hidden");
    btn.disabled = false;
    terminal.innerHTML += `\n<span class="t-error">Hata: ${escHtml(err.message)}</span>`;
    emptyDiv.classList.remove("hidden");
    statusBadge.textContent = "Hata";
    statusBadge.className = "badge badge--error";
  }
}

// ──────────────────────────────────────────────────────────────────
// İmaj Modal
// ──────────────────────────────────────────────────────────────────

function openImageModal(src) {
  const modal = document.getElementById("imgModal");
  if (!modal) return;
  document.getElementById("imgModalSrc").src = src;
  modal.classList.remove("hidden");
  modal.style.display = "flex";
}

function closeImageModal() {
  const modal = document.getElementById("imgModal");
  if (!modal) return;
  modal.classList.add("hidden");
  modal.style.display = "none";
}

// ──────────────────────────────────────────────────────────────────
// Yardımcılar
// ──────────────────────────────────────────────────────────────────

function escHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function colorizeTerminalLine(line) {
  if (/✅|başarı|success|tamamlandı/i.test(line)) return `<span class="t-success">${line}</span>`;
  if (/❌|hata|error|fail/i.test(line)) return `<span class="t-error">${line}</span>`;
  if (/⚠|warn|uyarı/i.test(line)) return `<span class="t-warn">${line}</span>`;
  if (/🎯|📊|📈|ℹ|INFO/i.test(line)) return `<span class="t-info">${line}</span>`;
  if (/^\s*#|^---/i.test(line)) return `<span class="t-dim">${line}</span>`;
  return line;
}

function colorizeTerminalOutput(text) {
  return text
    .split("\n")
    .map(colorizeTerminalLine)
    .join("\n");
}

function renderAlert(type, msg) {
  const icons = {
    error:   '<path d="M10 2a8 8 0 100 16A8 8 0 0010 2zm1 11H9v-2h2v2zm0-4H9V5h2v4z"/>',
    warning: '<path d="M10 2a8 8 0 100 16A8 8 0 0010 2zm1 11H9v-2h2v2zm0-4H9V5h2v4z"/>',
    success: '<path d="M9 12l-3-3 1.4-1.4L9 9.2l4.6-4.6L15 6l-6 6z"/>',
    info:    '<path d="M10 2a8 8 0 100 16A8 8 0 0010 2zm1 11H9v-2h2v2zm0-4H9V5h2v4z"/>',
  };
  return `<div class="alert alert--${type}">
    <svg viewBox="0 0 20 20" fill="currentColor">${icons[type] || icons.info}</svg>
    <span>${escHtml(msg)}</span>
  </div>`;
}

function showAlert(containerId, type, msg) {
  const el = document.getElementById(containerId);
  if (!el) return;
  el.innerHTML = renderAlert(type, msg);
  el.classList.remove("hidden");
}
