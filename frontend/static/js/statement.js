/**
 * statement.js — Ekstre Görüntüleyici JavaScript Mantığı
 * Mod seçimi, parse/classify, tablo render
 */

"use strict";

// Son render edilen veriyi sakla — dil değişince yeniden çizim için
let _lastData = null;
let _lastMode = null;

// i18n.js applyTranslations() tarafından çağrılır
window.reRenderStatement = function () {
  if (!_lastData || !_lastMode) return;
  if (_lastMode === "sorted") renderSortedTable(_lastData);
  else renderClassifiedTable(_lastData);
};

// ──────────────────────────────────────────────────────────────────
// Mod Seçimi
// ──────────────────────────────────────────────────────────────────

function selectMode(mode) {
  document.getElementById("selectedMode").value = mode;

  const cardSorted = document.getElementById("modeCardSorted");
  const cardClass  = document.getElementById("modeCardClassified");
  const btnText    = document.getElementById("btnAnalyzeText");

  const modelSelector = document.getElementById("inferenceModelSelector");
  if (mode === "sorted") {
    cardSorted.classList.add("selected");
    cardClass.classList.remove("selected");
    btnText.textContent = typeof t === "function" ? t("stmt.btn_analyze_sort") : "Sırala ve Göster";
    modelSelector.classList.add("hidden");
  } else {
    cardClass.classList.add("selected");
    cardSorted.classList.remove("selected");
    btnText.textContent = typeof t === "function" ? t("stmt.btn_analyze_ai") : "Sınıflandır ve Göster";
    modelSelector.classList.remove("hidden");
  }
}

// ──────────────────────────────────────────────────────────────────
// Ana Analiz Fonksiyonu
// ──────────────────────────────────────────────────────────────────

async function analyzeStatement() {
  const text = document.getElementById("statementText").value.trim();
  const mode = document.getElementById("selectedMode").value;

  if (!text) {
    alert("Ekstre metni boş olamaz.");
    return;
  }

  showLoading(t(mode === "classified" ? "stmt.loading_classify" : "stmt.loading_parse"));

  const endpoint = mode === "classified" ? "/statement/classify" : "/statement/parse";
  const inferenceMode = document.getElementById("inferenceMode")?.value || "cascade";

  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, inference_mode: inferenceMode }),
    });
    const data = await res.json();
    hideLoading();

    if (!res.ok || data.error) {
      renderError(data.error || "Sunucu hatası oluştu.");
      return;
    }

    if (mode === "sorted") {
      renderSortedTable(data);
    } else {
      renderClassifiedTable(data);
    }
  } catch (err) {
    hideLoading();
    renderError("Bağlantı hatası: " + err.message);
  }
}

// ──────────────────────────────────────────────────────────────────
// Tablo Render: Tarihe Göre Sıralı
// ──────────────────────────────────────────────────────────────────

function renderSortedTable(data) {
  _lastData = data; _lastMode = "sorted";
  const rows = data.rows || [];

  // İstatistikler
  const total = rows.reduce((s, r) => s + (r.amount || 0), 0);
  renderStats([
    { value: rows.length, label: t("stmt.stat_tx_count"), cls: "stat-card--navy" },
    { value: formatAmount(total), label: t("stmt.stat_total"), cls: "stat-card--yellow" },
  ]);

  if (rows.length === 0) {
    renderError(t("stmt.err_no_rows"));
    return;
  }

  const tbody = rows.map((r) => `
    <tr>
      <td style="white-space:nowrap;font-size:.85rem;">${escHtml(r.date)}</td>
      <td>${escHtml(r.description)}</td>
      <td class="amount-cell">${r.amount !== null ? formatAmount(r.amount) : escHtml(r.amount_raw)}</td>
    </tr>`).join("");

  setResult(`
    <div class="card">
      <div class="card__header">
        <span class="card__title">
          <svg viewBox="0 0 20 20" fill="currentColor" style="width:16px;height:16px;"><path d="M4 4h12v2H4V4zm0 4h12v2H4V8zm0 4h8v2H4v-2z"/></svg>
          ${t("stmt.mode_sorted")} (${rows.length} ${t("stmt.tx_suffix")})
        </span>
      </div>
      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>${t("stmt.col_date")}</th>
              <th>${t("stmt.col_desc")}</th>
              <th style="text-align:right;">${t("stmt.col_amount")}</th>
            </tr>
          </thead>
          <tbody>${tbody}</tbody>
          <tfoot>
            <tr style="background:var(--app-yellow-light);">
              <td colspan="2" style="padding:10px 14px;font-weight:700;font-size:.82rem;color:var(--app-navy);text-transform:uppercase;letter-spacing:.04em;">
                ${t("stmt.tfoot_total")}
              </td>
              <td class="amount-cell" style="font-size:1rem;font-weight:800;color:var(--app-navy);">${formatAmount(total)}</td>
            </tr>
          </tfoot>
        </table>
      </div>
    </div>`);
}

// ──────────────────────────────────────────────────────────────────
// Tablo Render: Sınıflandırılmış (L1 Gruplar)
// ──────────────────────────────────────────────────────────────────

function renderClassifiedTable(data) {
  _lastData = data; _lastMode = "classified";
  const groups = data.groups || [];

  // İstatistikler
  renderStats([
    { value: data.count, label: t("stmt.stat_tx_count"), cls: "stat-card--navy" },
    { value: groups.length, label: t("stmt.stat_category"), cls: "stat-card--navy" },
    { value: formatAmount(data.grand_total), label: t("stmt.stat_total"), cls: "stat-card--yellow" },
  ]);

  if (groups.length === 0) {
    renderError(t("stmt.err_no_classify"));
    return;
  }

  const html = groups.map((g) => {
    const rows = g.items.map((item) => `
      <tr>
        <td style="white-space:nowrap;font-size:.85rem;">${escHtml(item.date)}</td>
        <td>${escHtml(item.description)}</td>
        <td class="amount-cell">${item.amount !== null ? formatAmount(item.amount) : escHtml(item.amount_raw)}</td>
        <td><span class="badge badge--l2">${escHtml(item.category_l2 || "—")}</span></td>
      </tr>`).join("");

    return `
      <div class="group-card">
        <div class="group-summary-bar">
          <div>
            <span class="group-summary-bar__name">${l1Label(g.l1)}</span>
            <span class="group-summary-bar__meta">${g.count} ${t("stmt.tx_suffix")}</span>
          </div>
          <span class="group-summary-bar__total">${formatAmount(g.total)}</span>
        </div>
        <table>
          <thead>
            <tr>
              <th>${t("stmt.col_date")}</th>
              <th>${t("stmt.col_desc")}</th>
              <th style="text-align:right;">${t("stmt.col_amount")}</th>
              <th>${t("stmt.col_l2")}</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>`;
  }).join("");

  setResult(`
    <div>
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;">
        <h3 style="font-size:1rem;font-weight:700;color:var(--app-navy);">
          ${t("stmt.grouped_title")}
        </h3>
        <span class="badge badge--l1">${t("stmt.badge_ai")}</span>
      </div>
      ${html}
      <div style="background:var(--app-white);color:var(--app-black);padding:14px 18px;border-radius:var(--radius-md);margin-top:16px;display:flex;justify-content:space-between;align-items:center;border:1px solid var(--app-gray-200);">
        <span style="font-weight:700;font-size:.9rem;">${t("stmt.grand_total")} (${data.count} ${t("stmt.tx_suffix")})</span>
        <span style="font-weight:800;font-size:1.1rem;color:var(--app-yellow);">${formatAmount(data.grand_total)}</span>
      </div>
    </div>`);

  // setResult DOM'u hazırladıktan sonra pasta grafiği çiz
  requestAnimationFrame(() => drawPieChart(groups));
}

// ──────────────────────────────────────────────────────────────────
// Örnek Ekstre Verisi
// ──────────────────────────────────────────────────────────────────

function loadSampleStatement() {
  document.getElementById("statementText").value = `01.03.2024\tMIGROS HYPERMARKET\t312,50
02.03.2026\tSTARBUCKS ISTINYE PARK\t180,00
03.03.2026\tSHELL İZMİR 6701\t1750,00
04.03.2026\tAMAZON TR\t199,90
05.03.2026\tCIRAGAN ECZANE\t65,00
06.03.2026\tLCWAIKIKI OPTIMUM AVM\t245,00
07.03.2026\tNETFLIX.com\t139,00
08.03.2026\tBIM MARKET\t78,40
09.03.2026\tTURKCELL\t349,00
10.03.2026\tMARKT A101\t124,75
11.03.2026\tDENIZ YILDIZI RESTAURANT\t2880,00
12.03.2026\tCARREFOURSA MARKET POS 6788\t415,00
13.03.2026\tVENİ VİDİ VİCİ OPTİK\t5550,50
14.03.2026\tMACFIT ATASEŞEHİR\t1450,00
14.03.2026\BKM BEŞİKTAŞ TİYATRO SİNEMA\t599,00
15.03.2026\tIKEA BURSA ANATOLIUM\t1.250,00
16.03.2026\tMCDONALDS KAGITHANE ISTANBUL\t495,00
17.03.2026\tGETIRYEMEK/Teşekkür Ederiz\t375,50
18.03.2026\tHEPSIBURADA\t824,90
19.03.2026\tPTT KARGO\t28,00
20.03.2026\tMEHMET SARAÇ BARBERS CLUB\t750,00`;
}

function clearStatement() {
  _lastData = null; _lastMode = null;
  document.getElementById("statementText").value = "";
  document.getElementById("stmtStats").classList.add("hidden");
  document.getElementById("stmtResult").classList.add("hidden");
  document.getElementById("stmtEmpty").classList.remove("hidden");
  clearPieChart();
}

// ──────────────────────────────────────────────────────────────────
// Pasta Grafiği
// ──────────────────────────────────────────────────────────────────

const PIE_COLORS = [
  "#F5A623","#D4891A","#E8C068","#F7C96B","#FDD890",
  "#888888","#AAAAAA","#C0C0C0","#555555","#D0A84E",
  "#3A3A3A","#C8960F","#6B6B6B","#FFE4A0","#444444","#B8860B",
];

// Pasta grafik state — hover için
let _pieSlices   = [];   // [{startAngle, endAngle, group, color, pct}]
let _pieTotalVal = 0;
let _pieCtxRef   = null;
let _pieCxRef    = 0, _pieCyRef = 0, _pieRRef = 0;

function _redrawPie(hoveredIdx) {
  const ctx = _pieCtxRef;
  if (!ctx) return;
  const W = ctx.canvas.width, H = ctx.canvas.height;
  ctx.clearRect(0, 0, W, H);

  _pieSlices.forEach((s, i) => {
    const isHovered = i === hoveredIdx;
    const rr = isHovered ? _pieRRef + 10 : _pieRRef;
    ctx.beginPath();
    ctx.moveTo(_pieCxRef, _pieCyRef);
    ctx.arc(_pieCxRef, _pieCyRef, rr, s.startAngle, s.endAngle);
    ctx.closePath();
    ctx.fillStyle = s.color;
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 2;
    ctx.stroke();
  });

  // Delik
  ctx.beginPath();
  ctx.arc(_pieCxRef, _pieCyRef, _pieRRef * 0.45, 0, 2 * Math.PI);
  ctx.fillStyle = "#fff";
  ctx.fill();

  // Ortada tutar
  ctx.fillStyle = "#1A1A1A";
  ctx.font = "bold 13px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(new Intl.NumberFormat("tr-TR", { style:"currency", currency:"TRY", maximumFractionDigits:0 }).format(_pieTotalVal), _pieCxRef, _pieCyRef);
}

function drawPieChart(groups) {
  const canvas  = document.getElementById("pieChart");
  const legend  = document.getElementById("pieLegend");
  const empty   = document.getElementById("chartEmpty");
  if (!canvas || !legend || !empty) return;
  const ctx     = canvas.getContext("2d");
  const total   = groups.reduce((s, g) => s + (g.total || 0), 0);
  if (!total) return;

  canvas.style.display = "block";
  legend.style.display = "block";
  empty.style.display  = "none";

  const W = canvas.width, H = canvas.height;
  const cx = W / 2, cy = H / 2, r = Math.min(W, H) / 2 - 8;

  // Büyükten küçüğe sırala, negatif toplamları pasta dışında tut
  const sorted    = [...groups].sort((a, b) => b.total - a.total);
  const pieGroups = sorted.filter(g => g.total > 0);
  const pieTotal  = pieGroups.reduce((s, g) => s + g.total, 0);

  // Dilim verilerini hesapla ve sakla
  _pieSlices   = [];
  _pieTotalVal = total;
  _pieCtxRef   = ctx;
  _pieCxRef = cx; _pieCyRef = cy; _pieRRef = r;

  let startAngle = -Math.PI / 2;
  pieGroups.forEach((g, i) => {
    const slice = (g.total / pieTotal) * 2 * Math.PI;
    _pieSlices.push({
      startAngle,
      endAngle: startAngle + slice,
      group: g,
      color: PIE_COLORS[i % PIE_COLORS.length],
      pct: ((g.total / pieTotal) * 100).toFixed(1),
    });
    startAngle += slice;
  });

  _redrawPie(-1);

  // Tooltip elementi
  let tooltip = document.getElementById("pieTooltip");
  if (!tooltip) {
    tooltip = document.createElement("div");
    tooltip.id = "pieTooltip";
    tooltip.style.cssText = "position:absolute;pointer-events:none;background:#1A1A1A;color:#fff;padding:8px 12px;border-radius:8px;font-size:.75rem;line-height:1.6;display:none;z-index:100;white-space:nowrap;box-shadow:0 4px 12px rgba(0,0,0,.2);";
    canvas.parentElement.style.position = "relative";
    canvas.parentElement.appendChild(tooltip);
  }

  // Hover dinleyicileri — önce eskilerini temizle
  canvas.onmousemove = (e) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width  / rect.width;
    const scaleY = canvas.height / rect.height;
    const mx = (e.clientX - rect.left) * scaleX;
    const my = (e.clientY - rect.top)  * scaleY;

    const dx = mx - cx, dy = my - cy;
    const dist  = Math.sqrt(dx * dx + dy * dy);
    const angle = Math.atan2(dy, dx);

    // Donut iç dairesinin dışında mı?
    if (dist < r * 0.45 || dist > r + 12) {
      _redrawPie(-1);
      tooltip.style.display = "none";
      canvas.style.cursor = "default";
      return;
    }

    // Hangi dilim?
    const hit = _pieSlices.findIndex(s => {
      let a = angle;
      let sa = s.startAngle, ea = s.endAngle;
      // -PI/2'den başladığı için normalize et
      if (a < sa - 0.001) a += 2 * Math.PI;
      return a >= sa - 0.001 && a <= ea + 0.001;
    });

    if (hit >= 0) {
      _redrawPie(hit);
      canvas.style.cursor = "pointer";
      const s = _pieSlices[hit];
      tooltip.innerHTML = `<strong>${escHtml(l1Label(s.group.l1))}</strong><br>${formatAmount(s.group.total)} &nbsp;·&nbsp; %${s.pct}`;
      // Canvas üzerinde konumlandır
      const contRect = canvas.parentElement.getBoundingClientRect();
      tooltip.style.display = "block";
      let tx = e.clientX - contRect.left + 12;
      let ty = e.clientY - contRect.top  - 40;
      if (tx + tooltip.offsetWidth > contRect.width - 8) tx -= tooltip.offsetWidth + 24;
      tooltip.style.left = tx + "px";
      tooltip.style.top  = ty + "px";
    } else {
      _redrawPie(-1);
      tooltip.style.display = "none";
      canvas.style.cursor = "default";
    }
  };

  canvas.onmouseleave = () => {
    _redrawPie(-1);
    tooltip.style.display = "none";
    canvas.style.cursor = "default";
  };

  // Legend — büyükten küçüğe sıralı, yüzde + tutar
  legend.innerHTML = sorted.map((g) => {
    const pieIdx = pieGroups.indexOf(g);
    const color  = pieIdx >= 0 ? PIE_COLORS[pieIdx % PIE_COLORS.length] : "#CCCCCC";
    const pct    = pieIdx >= 0
      ? ((g.total / pieTotal) * 100).toFixed(1)
      : null;
    return `
      <div style="display:flex;align-items:center;justify-content:space-between;padding:4px 0;border-bottom:1px solid var(--app-gray-100);">
        <div style="display:flex;align-items:center;gap:8px;">
          <span style="width:11px;height:11px;border-radius:50%;background:${color};flex-shrink:0;display:inline-block;"></span>
          <span style="font-size:.78rem;font-weight:500;">${escHtml(l1Label(g.l1))}</span>
        </div>
        <div style="display:flex;align-items:center;gap:10px;white-space:nowrap;margin-left:8px;">
          <span style="font-size:.75rem;font-weight:400;color:var(--app-gray-500);">${formatAmount(g.total)}</span>
          ${pct !== null ? `<span style="font-size:.73rem;font-weight:600;color:var(--app-black);min-width:36px;text-align:right;">%${pct}</span>` : `<span style="min-width:36px;"></span>`}
        </div>
      </div>`;
  }).join("");
}

function clearPieChart() {
  const canvas = document.getElementById("pieChart");
  const legend = document.getElementById("pieLegend");
  const empty  = document.getElementById("chartEmpty");
  canvas.style.display = "none";
  legend.style.display = "none";
  legend.innerHTML = "";
  empty.style.display = "block";
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// ──────────────────────────────────────────────────────────────────
// UI Yardımcıları
// ──────────────────────────────────────────────────────────────────

function showLoading(msg) {
  document.getElementById("stmtLoading").classList.remove("hidden");
  const msgEl = document.getElementById("loadingMsg");
  if (msgEl) msgEl.textContent = msg;
  document.getElementById("stmtEmpty").classList.add("hidden");
  document.getElementById("stmtResult").classList.add("hidden");
  document.getElementById("stmtStats").classList.add("hidden");
  document.getElementById("btnAnalyze").disabled = true;
}

function hideLoading() {
  document.getElementById("stmtLoading").classList.add("hidden");
  document.getElementById("btnAnalyze").disabled = false;
}

function renderStats(stats) {
  const grid = document.getElementById("stmtStats");
  grid.innerHTML = stats.map((s) => `
    <div class="stat-card ${s.cls || ''}">
      <div class="stat-card__value">${escHtml(String(s.value))}</div>
      <div class="stat-card__label">${escHtml(s.label)}</div>
    </div>`).join("");
  grid.classList.remove("hidden");
}

function setResult(html) {
  const el = document.getElementById("stmtResult");
  el.innerHTML = html;
  el.classList.remove("hidden");
  document.getElementById("stmtEmpty").classList.add("hidden");
}

function renderError(msg) {
  setResult(`<div class="alert alert--error">
    <svg viewBox="0 0 20 20" fill="currentColor"><path d="M10 2a8 8 0 100 16A8 8 0 0010 2zm1 11H9v-2h2v2zm0-4H9V5h2v4z"/></svg>
    <span>${escHtml(msg)}</span>
  </div>`);
}

// ──────────────────────────────────────────────────────────────────
// Biçimlendirme Yardımcıları
// ──────────────────────────────────────────────────────────────────

function formatAmount(val) {
  if (val === null || val === undefined) return "—";
  return new Intl.NumberFormat("tr-TR", {
    style: "currency",
    currency: "TRY",
    minimumFractionDigits: 2,
  }).format(val);
}

function l1Label(key) {
  const labels = {
    yeme_icme:          "Yeme & İçme",
    market_gida:        "Market & Gıda",
    giyim_aksesuar:     "Giyim & Aksesuar",
    online_alisveris:   "Online Alışveriş",
    elektronik:         "Elektronik",
    ev_yasam:           "Ev & Yaşam",
    saglik:             "Sağlık",
    kisisel_bakim:      "Kişisel Bakım",
    ulasim:             "Ulaşım",
    akaryakit:          "Akaryakıt",
    egitim:             "Eğitim",
    eglence_kultur:     "Eğlence & Kültür",
    seyahat:            "Seyahat",
    fatura_abonelik:    "Fatura & Abonelik",
    finans_sigorta:     "Finans & Sigorta",
    diger:              "Diğer",
  };
  return labels[key] || key;
}

function escHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
