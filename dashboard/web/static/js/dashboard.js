"use strict";

const API_KEY = "dev-api-key-12345";
const HEADERS = { "X-API-Key": API_KEY };

async function fetchJSON(url) {
  const response = await fetch(url, { headers: HEADERS });
  if (!response.ok) throw new Error(`${response.status}`);
  return response.json();
}

function el(id) { return document.getElementById(id); }
function ts(t) { return t ? t.slice(11, 19) : "--"; }

// ---- Overview cards ----
async function refreshOverview() {
  try {
    const stats = await fetchJSON("/api/v1/statistics");
    el("metric-total-alerts").textContent = stats.alerts?.total ?? 0;
    el("metric-critical").textContent = stats.alerts?.by_severity?.critical ?? 0;
    el("metric-high").textContent = stats.alerts?.by_severity?.high ?? 0;
    el("metric-medium").textContent = stats.alerts?.by_severity?.medium ?? 0;
    el("metric-devices").textContent = stats.devices?.total ?? 0;
  } catch (e) { console.error("overview", e); }
}

// ---- Severity bars ----
async function refreshRiskBars() {
  try {
    const stats = await fetchJSON("/api/v1/statistics");
    const sev = stats.alerts?.by_severity || {};
    const total = Math.max((sev.critical||0)+(sev.high||0)+(sev.medium||0)+(sev.low||0), 1);
    const colors = { critical: "var(--critical)", high: "var(--high)", medium: "var(--medium)", low: "var(--low)" };
    const container = el("risk-bars");
    container.innerHTML = "";
    for (const level of ["critical","high","medium","low"]) {
      const count = sev[level] || 0;
      const width = ((count/total)*100).toFixed(1);
      const row = document.createElement("div");
      row.className = "bar-row";
      row.innerHTML =
        `<span class="bar-label">${level.toUpperCase()}</span>` +
        `<div class="bar-track"><div class="bar-fill" style="width:${width}%;background:${colors[level]}"></div></div>` +
        `<span>${count.toLocaleString()}</span>`;
      container.appendChild(row);
    }
  } catch (e) { console.error("risk bars", e); }
}

// ---- Model performance bars ----
async function refreshModelBars() {
  try {
    const data = await fetchJSON("/api/v1/dashboard/models");
    const container = el("model-bars");
    container.innerHTML = "";
    for (const m of data.models || []) {
      const pct = ((m.accuracy ?? 0) * 100).toFixed(1);
      const row = document.createElement("div");
      row.className = "model-bar";
      row.innerHTML =
        `<span class="model-name">${m.name}</span>` +
        `<div class="score-track"><div class="score-fill" style="width:${pct}%"></div></div>` +
        `<span class="score-val">${pct}%</span>`;
      container.appendChild(row);
    }
  } catch (e) { console.error("model bars", e); }
}

// ---- Live alerts feed ----
async function refreshLiveTable() {
  try {
    const data = await fetchJSON("/api/v1/alerts?limit=50");
    const tbody = document.querySelector("#live-table tbody");
    tbody.innerHTML = "";
    for (const a of data.alerts || []) {
      const sev = (a.severity || "unknown").toLowerCase();
      const tr = document.createElement("tr");
      tr.innerHTML =
        `<td>${ts(a.timestamp)}</td>` +
        `<td>${a.device_id || ""}</td>` +
        `<td>${a.alert_type || ""}</td>` +
        `<td><span class="badge badge-${sev}"><span class="severity-dot dot-${sev}"></span>${(a.severity||"").toUpperCase()}</span></td>` +
        `<td>${(a.risk_score ?? 0).toFixed(4)}</td>` +
        `<td>${a.status || ""}</td>`;
      tbody.appendChild(tr);
    }
    el("system-status").textContent = `Last update: ${new Date().toLocaleTimeString()}`;
  } catch (e) { console.error("live table", e); }
}

// ---- Physical anomalies ----
async function refreshAnomalies() {
  try {
    const data = await fetchJSON("/api/v1/anomalies?limit=30");
    const tbody = document.querySelector("#anomaly-table tbody");
    tbody.innerHTML = "";
    el("metric-anomalies").textContent = data.total ?? 0;
    for (const a of data.anomalies || []) {
      const tr = document.createElement("tr");
      tr.innerHTML =
        `<td>${ts(a.timestamp)}</td>` +
        `<td>${a.device_id || ""}</td>` +
        `<td>${(a.combined_score ?? 0).toFixed(4)}</td>` +
        `<td>${(a.confidence ?? 0).toFixed(2)}</td>`;
      tbody.appendChild(tr);
    }
  } catch (e) { console.error("anomalies", e); }
}

// ---- Cyber threats ----
async function refreshThreats() {
  try {
    const data = await fetchJSON("/api/v1/threats?limit=30");
    const tbody = document.querySelector("#threat-table tbody");
    tbody.innerHTML = "";
    el("metric-threats").textContent = data.total ?? 0;
    for (const t of data.threats || []) {
      const rf = t.rf_prediction || "N/A";
      const xgb = t.xgb_prediction || "N/A";
      const rfCls = rf.toLowerCase() === "attack" ? "badge-attack" : "badge-benign";
      const tr = document.createElement("tr");
      tr.innerHTML =
        `<td>${ts(t.timestamp)}</td>` +
        `<td>${t.device_id || ""}</td>` +
        `<td><span class="badge ${rfCls}">${rf}</span></td>` +
        `<td><span class="badge ${xgb.toLowerCase()==='attack'?'badge-attack':'badge-benign'}">${xgb}</span></td>` +
        `<td>${(t.combined_threat_score ?? 0).toFixed(4)}</td>`;
      tbody.appendChild(tr);
    }
  } catch (e) { console.error("threats", e); }
}

// ---- System health ----
async function refreshHealth() {
  try {
    const data = await fetchJSON("/api/v1/dashboard/health");
    const grid = el("health-grid");
    grid.innerHTML = "";
    for (const [name, value] of Object.entries(data.services || {})) {
      const cls = value === "ONLINE" || value === "LOADED" ? "status-online"
        : value === "OFFLINE" ? "status-offline" : "status-idle";
      const item = document.createElement("div");
      item.className = "health-item";
      item.innerHTML = `<span class="name">${name}</span><span class="${cls}">${value}</span>`;
      grid.appendChild(item);
    }
  } catch (e) { console.error("health", e); }
}

// ---- Master refresh ----
async function refreshAll() {
  await Promise.allSettled([
    refreshOverview(),
    refreshRiskBars(),
    refreshModelBars(),
    refreshLiveTable(),
    refreshAnomalies(),
    refreshThreats(),
    refreshHealth(),
  ]);
}

setInterval(refreshAll, 3000);
refreshAll();
