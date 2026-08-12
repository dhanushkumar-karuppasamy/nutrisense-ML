/* ==========================================================================
   NutriSense - Client-Side App Controller & Chart.js Integration
   ========================================================================== */

let currentPredictionData = null;
let shapChart = null;
let globalShapChart = null;

document.addEventListener("DOMContentLoaded", () => {
  loadBenchmarkData();
  // Auto-run initial prediction with default form values
  runPrediction();
});

// Tab Switching Handler
function switchTab(tabId) {
  document.querySelectorAll(".tab-content").forEach(el => el.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach(el => el.classList.remove("active"));

  const targetTab = document.getElementById(tabId);
  if (targetTab) {
    targetTab.classList.add("active");
  }

  const btn = Array.from(document.querySelectorAll(".tab-btn")).find(b => b.getAttribute("onclick")?.includes(tabId));
  if (btn) {
    btn.classList.add("active");
  }

  // Refresh charts if entering SHAP or Benchmark tab
  if (tabId === "explainabilityTab" && currentPredictionData) {
    fetchShapExplanation(currentPredictionData);
  } else if (tabId === "counterfactualTab" && currentPredictionData) {
    fetchCounterfactuals(currentPredictionData);
  }
}

// Preset Sample Loader
async function loadPresetProfile(presetId) {
  if (!presetId) return;
  try {
    const res = await fetch("/api/sample_children");
    const data = await res.json();
    if (data.success) {
      const sample = data.samples.find(s => s.id === presetId);
      if (sample) {
        populateForm(sample.data);
        runPrediction();
      }
    }
  } catch (err) {
    console.error("Error loading sample profile:", err);
  }
}

function populateForm(formData) {
  for (const [key, val] of Object.entries(formData)) {
    const field = document.getElementById(key);
    if (field) {
      field.value = val;
    }
  }
}

// Extract Form Inputs
function getFormData() {
  const form = document.getElementById("riskForm");
  const data = {};
  const elements = form.querySelectorAll("input, select");
  elements.forEach(el => {
    if (el.name) {
      data[el.name] = isNaN(el.value) ? el.value : parseFloat(el.value);
    }
  });
  return data;
}

// Execute Prediction
async function runPrediction() {
  const payload = getFormData();
  currentPredictionData = payload;

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const result = await res.json();

    if (result.success) {
      updateGauge(result.stunting_risk_pct, result.risk_tier, result.badge_class, result.action_summary);
      
      // Async fetch explanations
      fetchShapExplanation(payload);
      fetchCounterfactuals(payload);
    }
  } catch (err) {
    console.error("Prediction API error:", err);
  }
}

// Update Animated Gauge Meter
function updateGauge(riskPct, tier, badgeClass, summaryText) {
  const scoreVal = document.getElementById("riskScoreVal");
  const riskBadge = document.getElementById("riskBadge");
  const tierTitle = document.getElementById("riskTierTitle");
  const summaryBox = document.getElementById("riskActionSummary");
  const gaugeFill = document.getElementById("gaugeFill");

  scoreVal.textContent = `${riskPct.toFixed(1)}%`;
  riskBadge.textContent = tier;
  riskBadge.className = `badge badge-${badgeClass}`;
  tierTitle.textContent = `${tier} Stunting Trajectory`;
  summaryBox.textContent = summaryText;

  // Arc Circumference ~ 125.6px
  const maxDash = 125.6;
  const offset = maxDash - (riskPct / 100) * maxDash;
  gaugeFill.style.strokeDashoffset = offset;

  // Set Color based on risk level
  if (riskPct < 15) gaugeFill.style.stroke = "var(--accent-green)";
  else if (riskPct < 35) gaugeFill.style.stroke = "var(--accent-warning)";
  else if (riskPct < 60) gaugeFill.style.stroke = "var(--accent-danger)";
  else gaugeFill.style.stroke = "var(--accent-critical)";
}

// Fetch & Render SHAP Breakdown
async function fetchShapExplanation(payload) {
  try {
    const res = await fetch("/api/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    if (data.success && data.explanation) {
      renderShapChart(data.explanation.top_features);
      renderShapLists(data.explanation.risk_factors, data.explanation.protective_factors);
    }
  } catch (err) {
    console.error("SHAP API error:", err);
  }
}

function renderShapChart(features) {
  const ctx = document.getElementById("shapWaterfallChart")?.getContext("2d");
  if (!ctx) return;

  const labels = features.map(f => f.feature_name);
  const values = features.map(f => f.impact_pct);
  const backgroundColors = values.map(v => v >= 0 ? "rgba(248, 113, 113, 0.85)" : "rgba(52, 211, 153, 0.85)");

  if (shapChart) {
    shapChart.destroy();
  }

  shapChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "Risk Contribution (% SHAP Impact)",
        data: values,
        backgroundColor: backgroundColors,
        borderRadius: 6
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const val = ctx.raw;
              return `${val >= 0 ? '+' : ''}${val.toFixed(2)}% Stunting Risk Impact`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: { color: "rgba(0,0,0,0.08)" },
          ticks: { color: "#64748b" }
        },
        y: {
          grid: { display: false },
          ticks: { color: "#0f172a" }
        }
      }
    }
  });
}

function renderShapLists(riskFactors, protectiveFactors) {
  const quickList = document.getElementById("quickDriversList");
  const riskList = document.getElementById("shapRiskList");
  const shieldList = document.getElementById("shapProtectiveList");

  if (quickList) {
    quickList.innerHTML = riskFactors.slice(0, 4).map(f => `
      <li>
        <span>${f.feature_name}</span>
        <span class="impact-tag positive">+${f.impact_pct.toFixed(1)}% Risk</span>
      </li>
    `).join("") || '<li class="empty-msg">No major risk factors.</li>';
  }

  if (riskList) {
    riskList.innerHTML = riskFactors.map(f => `
      <li>
        <span>${f.feature_name}</span>
        <strong style="color: var(--accent-danger)">+${f.impact_pct.toFixed(2)}%</strong>
      </li>
    `).join("") || '<li class="empty-msg">None detected.</li>';
  }

  if (shieldList) {
    shieldList.innerHTML = protectiveFactors.map(f => `
      <li>
        <span>${f.feature_name}</span>
        <strong style="color: var(--accent-green)">${f.impact_pct.toFixed(2)}%</strong>
      </li>
    `).join("") || '<li class="empty-msg">None detected.</li>';
  }
}

// Fetch & Render DiCE Counterfactual Packages
async function fetchCounterfactuals(payload) {
  try {
    const res = await fetch("/api/counterfactual", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const data = await res.json();

    if (data.success && data.interventions) {
      renderInterventions(data.interventions);
    }
  } catch (err) {
    console.error("Counterfactual API error:", err);
  }
}

function renderInterventions(interventions) {
  const baseRisk = interventions.baseline_risk_pct;
  document.getElementById("simBaselineRisk").textContent = `${baseRisk.toFixed(1)}%`;

  const packagesGrid = document.getElementById("interventionPackagesGrid");
  const pkgs = interventions.packages;

  if (pkgs.length > 0) {
    // Default show best package target
    const bestPkg = pkgs[pkgs.length - 1];
    document.getElementById("simTargetRisk").textContent = `${bestPkg.simulated_risk_pct.toFixed(1)}%`;
    document.getElementById("simDeltaRisk").textContent = `-${bestPkg.risk_reduction_pct.toFixed(1)}%`;

    packagesGrid.innerHTML = pkgs.map(pkg => `
      <div class="package-card" onclick="selectPackageTarget(${pkg.simulated_risk_pct}, ${pkg.risk_reduction_pct})">
        <div class="package-header">
          <div class="package-icon"><i class="fa-solid fa-${pkg.icon}"></i></div>
          <div>
            <h4>${pkg.package_name}</h4>
            <span class="badge badge-success">Simulated Risk: ${pkg.simulated_risk_pct.toFixed(1)}% (-${pkg.risk_reduction_pct.toFixed(1)}%)</span>
          </div>
        </div>

        <ul class="actions-list">
          ${pkg.key_actions.map(a => `<li><i class="fa-solid fa-circle-check"></i> ${a}</li>`).join("")}
        </ul>
      </div>
    `).join("");
  }
}

function selectPackageTarget(simRisk, delta) {
  document.getElementById("simTargetRisk").textContent = `${simRisk.toFixed(1)}%`;
  document.getElementById("simDeltaRisk").textContent = `-${delta.toFixed(1)}%`;
}

// Benchmark Ladder & Global Importance
async function loadBenchmarkData() {
  try {
    const res = await fetch("/api/benchmark");
    const data = await res.json();

    if (data.success) {
      renderBenchmarkTable(data.benchmark_results);
      renderGlobalShapChart(data.global_shap_importance);
    }
  } catch (err) {
    console.error("Benchmark API error:", err);
  }
}

async function runBenchmarkRetest() {
  const btn = document.getElementById("btnRetestBenchmark");
  if(btn) {
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Running...';
    btn.disabled = true;
  }
  try {
    const res = await fetch("/api/benchmark/retest", { method: "POST" });
    const data = await res.json();
    if (data.success) {
      renderBenchmarkTable(data.benchmark_results);
      renderGlobalShapChart(data.global_shap_importance);
    }
  } catch (err) {
    console.error("Benchmark Retest API error:", err);
  }
  if(btn) {
    btn.innerHTML = '<i class="fa-solid fa-rotate-right"></i> Run Retest';
    btn.disabled = false;
  }
}

function renderBenchmarkTable(results) {
  const tbody = document.getElementById("benchmarkTableBody");
  if (!tbody || !results) return;

  const formatVal = (val, decimals = 4, pct = false) => {
    if (val === null || val === undefined) return "N/A";
    if (pct) return `${(val * 100).toFixed(1)}%`;
    return val.toFixed(decimals);
  };

  tbody.innerHTML = results.map((row, idx) => `
    <tr class="${idx === 0 ? 'highlight-row' : ''}">
      <td><strong>${row.Model}</strong> ${idx === 0 ? '<span class="badge badge-success">Selected</span>' : ''}</td>
      <td><strong>${formatVal(row.CV_ROC_AUC)}</strong> ± ${formatVal(row.CV_ROC_AUC_Std)}</td>
      <td>${formatVal(row.Test_ROC_AUC)}</td>
      <td>${formatVal(row.Test_Recall, 1, true)}</td>
      <td>${formatVal(row.Test_Precision, 1, true)}</td>
      <td>${formatVal(row.Test_F1, 3)}</td>
      <td>${formatVal(row.Test_Accuracy, 1, true)}</td>
      <td>${formatVal(row.Brier_Score)}</td>
    </tr>
  `).join("");
}

function renderGlobalShapChart(globalShap) {
  const ctx = document.getElementById("globalShapChart")?.getContext("2d");
  if (!ctx || !globalShap) return;

  const labels = globalShap.map(f => f.feature_name);
  const values = globalShap.map(f => f.importance_score);

  if (globalShapChart) {
    globalShapChart.destroy();
  }

  globalShapChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [{
        label: "Mean |SHAP| Importance Score",
        data: values,
        backgroundColor: "rgba(59, 130, 246, 0.75)",
        borderRadius: 4
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: "rgba(0,0,0,0.08)" }, ticks: { color: "#64748b" } },
        y: { grid: { display: false }, ticks: { color: "#0f172a" } }
      }
    }
  });
}

// ASHA Field Worker Quick Screen
function toggleFieldMode() {
  switchTab('fieldTab');
}

function setFieldVal(fieldId, val) {
  document.getElementById(fieldId).value = val;
  const parent = document.getElementById(fieldId).parentElement;
  parent.querySelectorAll(".field-opt-btn").forEach(b => b.classList.remove("active"));

  event.target.classList.add("active");
}

async function runFieldPrediction() {
  const age = parseFloat(document.getElementById("field_age").value);
  const anc = parseFloat(document.getElementById("field_anc").value);
  const toilet = document.getElementById("field_toilet").value;
  const water = document.getElementById("field_water").value;
  const diarrhea = parseFloat(document.getElementById("field_diarrhea").value);

  const payload = {
    child_age_months: age,
    anc_visits: anc,
    toilet_type: toilet,
    water_source: water,
    diarrhea_recent: diarrhea,
    child_sex: 0,
    wealth_index: 2,
    mother_education: 1
  };

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const result = await res.json();

    if (result.success) {
      const box = document.getElementById("fieldResultBox");
      box.classList.remove("hidden");
      document.getElementById("fieldRiskScore").textContent = `${result.stunting_risk_pct.toFixed(1)}% Stunting Risk`;
      document.getElementById("fieldAdviceText").textContent = `${result.risk_tier}: ${result.action_summary}`;
    }
  } catch (err) {
    console.error("Field Prediction API error:", err);
  }
}
