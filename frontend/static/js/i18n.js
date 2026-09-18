/**
 * i18n.js — Türkçe / İngilizce dil desteği
 */

"use strict";

const TRANSLATIONS = {
  tr: {
    // Topbar
    "nav.home":              "Ana Sayfa",
    // Header nav
    "nav.analysis":          "Analiz",
    "nav.statements":        "Ekstreler",
    // Footer
    "footer.copy":           "Transaction Enrichment & Classifier \u00a9 2026",

    // ML Dashboard — sayfa başlığı
    "ml.title":              "Analiz",
    "ml.subtitle":           "Model eğitimi, değerlendirme, görselleştirme ve tahmin işlemleri",

    // Sekmeler
    "tab.predict":           "Tahmin",
    "tab.train":             "Eğitim",
    "tab.evaluate":          "Değerlendirme",
    "tab.tsne":              "t-SNE",
    "tab.experiments":       "Deneyler",

    // Tahmin sekmesi
    "predict.card_title":    "Merchant Tahmini",
    "predict.label_merchant":"İşyeri Adı",
    "predict.placeholder":   "Örn: MIGROS HYPERMARKET, STARBUCKS...",
    "predict.label_mode":    "Çıkarım Modu",
    "predict.mode_cascade":  "Cascade (L1 → L2 hiyerarşik)",
    "predict.mode_l2":       "L2 Tekli Model",
    "predict.mode_l1":       "Sadece L1",
    "predict.btn_predict":   "Tahmin Et",
    "predict.btn_clear":     "Temizle",
    "predict.quick_label":   "Hızlı örnekler:",
    "predict.result_title":  "Sonuç",
    "predict.empty":         "Bir işyeri adı girin ve \"Tahmin Et\"e basın.",
    "predict.loading":       "Sınıflandırılıyor...",
    "predict.key_normalized":"Normalize",
    "predict.key_l1":        "L1 Kategori",
    "predict.key_l2":        "L2 Kategori",
    "predict.key_confidence":"Güven",
    "predict.key_source":    "Kaynak",
    "predict.alternatives":  "Alternatif tahminler",
    "predict.err_empty":     "İşyeri adı boş olamaz.",
    "predict.err_server":    "Sunucu hatası",
    "predict.err_connection":"Bağlantı hatası: ",

    // Eğitim sekmesi
    "train.card_title":      "Eğitim Parametreleri",
    "train.label_dataset":   "Veri Seti",
    "train.label_mode":      "Eğitim Modu",
    "train.mode_cascade":    "Cascade (önerilen)",
    "train.mode_l2":         "L2 Tekli",
    "train.mode_l1":         "L1 Tekli",
    "train.label_lr":        "Learning Rate",
    "train.label_epoch":     "Epoch Sayısı",
    "train.btn_start":       "Eğitimi Başlat",
    "train.btn_clear":       "Temizle",
    "train.output_title":    "Eğitim Çıktısı",
    "train.status_waiting":  "Bekliyor",
    "train.terminal_empty":  "Eğitim başlatıldığında çıktı burada görünür...",

    // Değerlendirme sekmesi
    "eval.card_title":       "Değerlendirme Ayarları",
    "eval.label_dataset":    "Test Veri Seti",
    "eval.label_mode":       "Mod (isteğe bağlı)",
    "eval.mode_auto":        "Otomatik (model dosyasından oku)",
    "eval.btn_run":          "Değerlendirmeyi Çalıştır",
    "eval.result_title":     "Sonuçlar",
    "eval.status_waiting":   "Bekliyor",
    "eval.loading":          "Değerlendiriliyor...",
    "eval.terminal_empty":   "Değerlendirme çıktısı burada görünür...",
    "eval.plots_title":      "Kayıtlı Grafikler",

    // t-SNE sekmesi
    "tsne.card_title":       "t-SNE Parametreleri",
    "tsne.label_dataset":    "Veri Seti (isteğe bağlı)",
    "tsne.default_dataset":  "Varsayılan (train.txt)",
    "tsne.label_n":          "Örnek Sayısı (N)",
    "tsne.info":             "t-SNE hesaplama birkaç dakika sürebilir. Sabırlı olun.",
    "tsne.btn_run":          "t-SNE Çalıştır",
    "tsne.result_title":     "Görselleştirme",
    "tsne.status_waiting":   "Bekliyor",
    "tsne.loading":          "t-SNE hesaplanıyor, lütfen bekleyin...",
    "tsne.empty":            "t-SNE çalıştırıldığında görsel burada belirir.",
    "tsne.output_title":     "Çıktı",
    "tsne.terminal_empty":   "t-SNE çıktısı burada görünür...",

    // Deneyler sekmesi
    "exp.card_title":        "Değerlendirme Sonuçları (models/experiments/)",
    "exp.btn_refresh":       "Yenile",
    "exp.col_time":          "Zaman",
    "exp.col_mode":          "Mod",
    "exp.col_source":        "Kaynak",
    "exp.col_l1acc":         "L1 Accuracy",
    "exp.col_l2acc":         "L2 Accuracy",
    "exp.col_l1f1":          "L1 Macro-F1",
    "exp.col_l2f1":          "L2 Macro-F1",
    "exp.col_n":             "Örnek",
    "exp.empty":             "Henüz değerlendirme yapılmamış.",

    // Ekstre sayfası
    "stmt.title":            "Ekstreler",
    "stmt.subtitle":         "Kredi kartı ekstrenizi yapıştırın; tarih bazlı sıralama veya AI sınıflandırması uygulayın",
    "stmt.card_title":       "Ekstre Metni",
    "stmt.btn_sample":       "Örnek Yükle",
    "stmt.info":             "Her satır: <code style=\"background:var(--app-gray-100);padding:1px 5px;border-radius:3px;\">Tarih &nbsp; İşlem Açıklaması &nbsp; Tutar</code> formatında olmalıdır.",
    "stmt.mode_sorted":      "Tarihe Göre Sıralı",
    "stmt.mode_sorted_desc": "Sınıflandırma yapılmaz. Tüm işlemler tarih sırasıyla listelenir.",
    "stmt.mode_ai":          "AI Sınıflandırması",
    "stmt.mode_ai_desc":     "İşlemler L1 kategorilerine göre gruplandırılır; L2 kategorisi eklenir.",
    "stmt.model_label":      "Kullanılacak Model",
    "stmt.mode_cascade":     "Cascade (L1 → L2 hiyerarşik)",
    "stmt.mode_l2":          "L2 Tekli Model",
    "stmt.mode_l1":          "Sadece L1",
    "stmt.btn_analyze_sort": "Sırala ve Göster",
    "stmt.btn_analyze_ai":   "Sınıflandır ve Göster",
    "stmt.btn_clear":        "Temizle",
    "stmt.chart_title":      "Kategori Dağılımı",
    "stmt.chart_empty":      "AI Sınıflandırması çalıştırın",
    "stmt.empty":            "Ekstre metnini girin ve analiz edin.",

    // Ekstre — dinamik JS metinleri
    "stmt.loading_classify": "Model ile sınıflandırılıyor...",
    "stmt.loading_parse":    "Satırlar ayrıştırılıyor...",
    "stmt.err_no_rows":      "Hiçbir satır ayrıştırılamadı. Formatı kontrol edin.",
    "stmt.err_no_classify":  "Hiçbir satır sınıflandırılamadı.",
    "stmt.stat_tx_count":    "İşlem Sayısı",
    "stmt.stat_total":       "Toplam Tutar",
    "stmt.stat_category":    "Kategori",
    "stmt.tx_suffix":        "işlem",
    "stmt.col_date":         "İşlem Tarihi",
    "stmt.col_desc":         "Açıklama",
    "stmt.col_amount":       "Tutar (TL)",
    "stmt.col_l2":           "L2 Kategorisi",
    "stmt.tfoot_total":      "TOPLAM",
    "stmt.grand_total":      "GENEL TOPLAM",
    "stmt.grouped_title":    "L1 Kategorilerine Göre Gruplandırılmış Ekstre",
    "stmt.badge_ai":         "AI Sınıflandırması",
  },

  en: {
    "nav.home":              "Home",
    "nav.analysis":          "Analysis",
    "nav.statements":        "Statements",
    "footer.copy":           "Transaction Enrichment & Classifier \u00a9 2026",

    "ml.title":              "Analysis",
    "ml.subtitle":           "Model training, evaluation, visualization and prediction",

    "tab.predict":           "Predict",
    "tab.train":             "Training",
    "tab.evaluate":          "Evaluation",
    "tab.tsne":              "t-SNE",
    "tab.experiments":       "Experiments",

    "predict.card_title":    "Merchant Prediction",
    "predict.label_merchant":"Merchant Name",
    "predict.placeholder":   "E.g: MIGROS HYPERMARKET, STARBUCKS...",
    "predict.label_mode":    "Inference Mode",
    "predict.mode_cascade":  "Cascade (L1 → L2 hierarchical)",
    "predict.mode_l2":       "L2 Single Model",
    "predict.mode_l1":       "L1 Only",
    "predict.btn_predict":   "Predict",
    "predict.btn_clear":     "Clear",
    "predict.quick_label":   "Quick examples:",
    "predict.result_title":  "Result",
    "predict.empty":         "Enter a merchant name and press \"Predict\".",
    "predict.loading":       "Classifying...",
    "predict.key_normalized":"Normalized",
    "predict.key_l1":        "L1 Category",
    "predict.key_l2":        "L2 Category",
    "predict.key_confidence":"Confidence",
    "predict.key_source":    "Source",
    "predict.alternatives":  "Alternative predictions",
    "predict.err_empty":     "Merchant name cannot be empty.",
    "predict.err_server":    "Server error",
    "predict.err_connection":"Connection error: ",

    "train.card_title":      "Training Parameters",
    "train.label_dataset":   "Dataset",
    "train.label_mode":      "Training Mode",
    "train.mode_cascade":    "Cascade (recommended)",
    "train.mode_l2":         "L2 Single",
    "train.mode_l1":         "L1 Single",
    "train.label_lr":        "Learning Rate",
    "train.label_epoch":     "Epoch Count",
    "train.btn_start":       "Start Training",
    "train.btn_clear":       "Clear",
    "train.output_title":    "Training Output",
    "train.status_waiting":  "Waiting",
    "train.terminal_empty":  "Output will appear here when training starts...",

    "eval.card_title":       "Evaluation Settings",
    "eval.label_dataset":    "Test Dataset",
    "eval.label_mode":       "Mode (optional)",
    "eval.mode_auto":        "Auto (read from model file)",
    "eval.btn_run":          "Run Evaluation",
    "eval.result_title":     "Results",
    "eval.status_waiting":   "Waiting",
    "eval.loading":          "Evaluating...",
    "eval.terminal_empty":   "Evaluation output will appear here...",
    "eval.plots_title":      "Saved Plots",

    "tsne.card_title":       "t-SNE Parameters",
    "tsne.label_dataset":    "Dataset (optional)",
    "tsne.default_dataset":  "Default (train.txt)",
    "tsne.label_n":          "Sample Count (N)",
    "tsne.info":             "t-SNE computation may take a few minutes. Please be patient.",
    "tsne.btn_run":          "Run t-SNE",
    "tsne.result_title":     "Visualization",
    "tsne.status_waiting":   "Waiting",
    "tsne.loading":          "Computing t-SNE, please wait...",
    "tsne.empty":            "Visualization will appear here after running t-SNE.",
    "tsne.output_title":     "Output",
    "tsne.terminal_empty":   "t-SNE output will appear here...",

    "exp.card_title":        "Evaluation Results (models/experiments/)",
    "exp.btn_refresh":       "Refresh",
    "exp.col_time":          "Time",
    "exp.col_mode":          "Mode",
    "exp.col_source":        "Source",
    "exp.col_l1acc":         "L1 Accuracy",
    "exp.col_l2acc":         "L2 Accuracy",
    "exp.col_l1f1":          "L1 Macro-F1",
    "exp.col_l2f1":          "L2 Macro-F1",
    "exp.col_n":             "Samples",
    "exp.empty":             "No evaluations yet.",

    "stmt.title":            "Statements",
    "stmt.subtitle":         "Paste your credit card statement; apply date-based sorting or AI classification",
    "stmt.card_title":       "Statement Text",
    "stmt.btn_sample":       "Load Sample",
    "stmt.info":             "Each line: <code style=\"background:var(--app-gray-100);padding:1px 5px;border-radius:3px;\">Date &nbsp; Description &nbsp; Amount</code> format.",
    "stmt.mode_sorted":      "Sort by Date",
    "stmt.mode_sorted_desc": "No classification. All transactions listed by date.",
    "stmt.mode_ai":          "AI Classification",
    "stmt.mode_ai_desc":     "Transactions grouped by L1 category; L2 category added.",
    "stmt.model_label":      "Model to Use",
    "stmt.mode_cascade":     "Cascade (L1 → L2 hierarchical)",
    "stmt.mode_l2":          "L2 Single Model",
    "stmt.mode_l1":          "L1 Only",
    "stmt.btn_analyze_sort": "Sort and Show",
    "stmt.btn_analyze_ai":   "Classify and Show",
    "stmt.btn_clear":        "Clear",
    "stmt.chart_title":      "Category Distribution",
    "stmt.chart_empty":      "Run AI Classification",
    "stmt.empty":            "Enter statement text and analyze.",

    // Statement — dynamic JS strings
    "stmt.loading_classify": "Classifying with model...",
    "stmt.loading_parse":    "Parsing rows...",
    "stmt.err_no_rows":      "No rows could be parsed. Check the format.",
    "stmt.err_no_classify":  "No rows could be classified.",
    "stmt.stat_tx_count":    "Transaction Count",
    "stmt.stat_total":       "Total Amount",
    "stmt.stat_category":    "Category",
    "stmt.tx_suffix":        "transactions",
    "stmt.col_date":         "Transaction Date",
    "stmt.col_desc":         "Description",
    "stmt.col_amount":       "Amount (TL)",
    "stmt.col_l2":           "L2 Category",
    "stmt.tfoot_total":      "TOTAL",
    "stmt.grand_total":      "GRAND TOTAL",
    "stmt.grouped_title":    "Statement Grouped by L1 Categories",
    "stmt.badge_ai":         "AI Classification",
  }
};

let currentLang = "en";

function t(key) {
  return TRANSLATIONS[currentLang][key] || TRANSLATIONS["en"][key] || key;
}

function applyTranslations() {
  document.querySelectorAll("[data-i18n]").forEach(el => {
    const key = el.getAttribute("data-i18n");
    const val = t(key);
    if (el.tagName === "INPUT" && el.hasAttribute("placeholder")) {
      el.placeholder = val;
    } else if (el.hasAttribute("data-i18n-html")) {
      el.innerHTML = val;
    } else {
      el.textContent = val;
    }
  });

  // Dil toggle butonunu güncelle
  const btn = document.getElementById("langToggleLabel");
  if (btn) btn.textContent = currentLang === "tr" ? "EN" : "TR";

  // Dinamik olarak render edilmiş ekstre içeriğini yeniden çiz
  if (typeof window.reRenderStatement === "function") {
    window.reRenderStatement();
  }
}

function toggleLanguage() {
  currentLang = currentLang === "tr" ? "en" : "tr";
  localStorage.setItem("app_lang", currentLang);
  applyTranslations();
}

document.addEventListener("DOMContentLoaded", applyTranslations);
