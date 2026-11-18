/**
 * PathSnap UI Evaluator Application
 * Communicates with FastAPI backend (Laptop 1) over LAN/Hotspot
 * Coordinates strictly in [lat, lon] order.
 */

// Application State
const state = {
  // When this page is served by the API itself (http://<laptop1>:8000/web/), always talk to that same server.
  // A saved address from an earlier attempt (e.g. localhost) must not override it.
  apiBaseUrl: window.location.pathname.startsWith('/web')
    ? window.location.origin
    : (localStorage.getItem('pathsnap_api_url') || 'http://localhost:8000'),
  trips: [],
  selectedTripId: null,
  currentTripData: null,
  predictionResult: null,
  abortController: null,
  layers: {
    rawGps: null,
    network: null,
    hmmRoute: null,
    classifierRoute: null,
    pointAccuracy: null
  }
};

// DOM Elements
const el = {
  apiBaseUrl: document.getElementById('apiBaseUrl'),
  btnTestConn: document.getElementById('btnTestConn'),
  connBadge: document.getElementById('connBadge'),
  tripSelect: document.getElementById('tripSelect'),
  tripCountBadge: document.getElementById('tripCountBadge'),
  tripMeta: document.getElementById('tripMeta'),
  metaGpsPoints: document.getElementById('metaGpsPoints'),
  metaDuration: document.getElementById('metaDuration'),
  networkBuffer: document.getElementById('networkBuffer'),
  bufferVal: document.getElementById('bufferVal'),
  btnRunPredict: document.getElementById('btnRunPredict'),
  predictSpinner: document.getElementById('predictSpinner'),
  alertsContainer: document.getElementById('alertsContainer'),
  alertBox: document.getElementById('alertBox'),
  alertIcon: document.getElementById('alertIcon'),
  alertTitle: document.getElementById('alertTitle'),
  alertMessages: document.getElementById('alertMessages'),
  btnAlertClose: document.getElementById('btnAlertClose'),
  resultsDrawer: document.getElementById('resultsDrawer'),
  resTripBadge: document.getElementById('resTripBadge'),
  resLatency: document.getElementById('resLatency'),
  hmmAcc: document.getElementById('hmmAcc'),
  clfAcc: document.getElementById('clfAcc'),
  accDelta: document.getElementById('accDelta'),
  hmmCont: document.getElementById('hmmCont'),
  clfCont: document.getElementById('clfCont'),
  contDelta: document.getElementById('contDelta'),
  hmmF1: document.getElementById('hmmF1'),
  clfF1: document.getElementById('clfF1'),
  f1Delta: document.getElementById('f1Delta'),
  hmmBreaks: document.getElementById('hmmBreaks'),
  resTotalArcs: document.getElementById('resTotalArcs'),
  breaksBadge: document.getElementById('breaksBadge'),
  // Layer Toggles
  layerRawGps: document.getElementById('layerRawGps'),
  layerNetwork: document.getElementById('layerNetwork'),
  layerHmm: document.getElementById('layerHmm'),
  layerClassifier: document.getElementById('layerClassifier'),
  layerPointAccuracy: document.getElementById('layerPointAccuracy')
};

// Initialize Leaflet Map
let map;
let canvasRenderer;

function initMap() {
  canvasRenderer = L.canvas({ padding: 0.5 });
  
  map = L.map('map', {
    zoomControl: true,
    preferCanvas: true
  }).setView([56.25, 54.96], 13);

  // High contrast basemap suitable for GPS & road networks
  // (CARTO's free raster tiles now return an "API KEY REQUIRED" watermark, so use OpenStreetMap's standard tiles)
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 19
  }).addTo(map);

  // Initialize Layer Groups
  state.layers.rawGps = L.layerGroup().addTo(map);
  state.layers.network = L.layerGroup().addTo(map);
  state.layers.hmmRoute = L.layerGroup().addTo(map);
  state.layers.classifierRoute = L.layerGroup().addTo(map);
  state.layers.pointAccuracy = L.layerGroup().addTo(map);
}

// Format seconds into MM:SS or HH:MM:SS
function formatDuration(sec) {
  if (!sec || isNaN(sec)) return '0s';
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  if (m >= 60) {
    const h = Math.floor(m / 60);
    const remM = m % 60;
    return `${h}h ${remM}m ${s}s`;
  }
  return `${m}m ${s}s`;
}

// Show Alerts / Warnings Banner
function showAlert(title, messages, isError = false) {
  el.alertTitle.textContent = title;
  el.alertBox.className = isError ? 'alert-box error' : 'alert-box';
  el.alertIcon.textContent = isError ? '🚨' : '⚠️';
  
  el.alertMessages.innerHTML = '';
  if (Array.isArray(messages)) {
    const ul = document.createElement('ul');
    ul.style.paddingLeft = '18px';
    messages.forEach(msg => {
      const li = document.createElement('li');
      li.textContent = msg;
      ul.appendChild(li);
    });
    el.alertMessages.appendChild(ul);
  } else {
    el.alertMessages.textContent = messages;
  }
  
  el.alertsContainer.style.display = 'block';
}

function hideAlert() {
  el.alertsContainer.style.display = 'none';
}

// Clean Base URL helper
function getCleanUrl() {
  let url = el.apiBaseUrl.value.trim();
  if (url.endsWith('/')) {
    url = url.slice(0, -1);
  }
  return url;
}

// Update Badge State
function updateBadge(connected, modelName = '', tripCount = 0, errMsg = '') {
  if (connected) {
    el.connBadge.className = 'status-badge connected';
    el.connBadge.querySelector('.status-text').textContent = `Connected (${modelName})`;
    el.tripCountBadge.textContent = `${tripCount} Trips`;
  } else {
    el.connBadge.className = 'status-badge disconnected';
    el.connBadge.querySelector('.status-text').textContent = errMsg || 'Disconnected';
  }
}

// Test API Connection (/health) & CORS
async function testConnection(showSuccessAlert = true) {
  const url = getCleanUrl();
  localStorage.setItem('pathsnap_api_url', url);
  state.apiBaseUrl = url;

  const btnText = el.btnTestConn.querySelector('.btn-text');
  const spinner = el.btnTestConn.querySelector('.spinner-sm');
  btnText.textContent = 'Pinging...';
  spinner.classList.remove('hidden');
  el.btnTestConn.disabled = true;

  try {
    const startTime = performance.now();
    const resp = await fetch(`${url}/health`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      signal: AbortSignal.timeout(5000)
    });

    const elapsed = Math.round(performance.now() - startTime);

    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status} ${resp.statusText}`);
    }

    const data = await resp.json();
    updateBadge(true, data.model || 'API v1', data.n_trips || 10);
    
    if (showSuccessAlert) {
      showAlert('API Connected Successfully', `Connected to ${url} in ${elapsed}ms. CORS verified.`);
      setTimeout(hideAlert, 4000);
    }
    
    // Refresh trips list
    await loadTrips();
    return true;
  } catch (err) {
    console.error('Connection test failed:', err);
    updateBadge(false, '', 0, 'Connection Failed');
    showAlert('Connection Error', `Failed to connect to ${url}. Check LAN/hotspot IP, backend port 8000, and firewall settings. (${err.message})`, true);
    return false;
  } finally {
    btnText.textContent = 'Test Ping';
    spinner.classList.add('hidden');
    el.btnTestConn.disabled = false;
  }
}

// Load Trips List (/trips)
async function loadTrips() {
  const url = getCleanUrl();
  try {
    const resp = await fetch(`${url}/trips`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      signal: AbortSignal.timeout(8000)
    });

    if (!resp.ok) {
      throw new Error(`Failed to fetch trips: HTTP ${resp.status}`);
    }

    const data = await resp.json();
    state.trips = data.trips || [];

    el.tripSelect.innerHTML = '<option value="">-- Choose an Evaluation Trip --</option>';
    state.trips.forEach(t => {
      const opt = document.createElement('option');
      opt.value = t.trip_id;
      opt.textContent = `Trip ${t.trip_id} (${t.n_gps_points.toLocaleString()} pts, ${formatDuration(t.duration_s)})`;
      el.tripSelect.appendChild(opt);
    });

    el.tripSelect.disabled = false;
    
    // If not selected, select the first trip
    if (!state.selectedTripId && state.trips.length > 0) {
      el.tripSelect.value = state.trips[0].trip_id;
      onTripSelected(state.trips[0].trip_id);
    }
  } catch (err) {
    console.error('Error loading trips:', err);
    showAlert('Trips Load Error', `Could not fetch evaluation trips: ${err.message}`, true);
  }
}

// On Trip Selected: Call /preview and /network
async function onTripSelected(tripId) {
  if (!tripId) return;
  state.selectedTripId = tripId;
  
  // Find trip meta
  const trip = state.trips.find(t => t.trip_id === tripId);
  if (trip) {
    el.metaGpsPoints.textContent = trip.n_gps_points.toLocaleString();
    el.metaDuration.textContent = formatDuration(trip.duration_s);
    el.tripMeta.style.display = 'flex';
  }

  // Clear previous layers
  clearAllMapLayers();
  el.resultsDrawer.style.display = 'none';
  hideAlert();

  // Load preview (raw GPS track)
  await loadTripPreview(tripId);

  // Lazily load network
  const bufferM = parseFloat(el.networkBuffer.value);
  loadTripNetwork(tripId, bufferM);

  // Enable Run Predict button
  el.btnRunPredict.disabled = false;
}

// Clear all map layers
function clearAllMapLayers() {
  state.layers.rawGps.clearLayers();
  state.layers.network.clearLayers();
  state.layers.hmmRoute.clearLayers();
  state.layers.classifierRoute.clearLayers();
  state.layers.pointAccuracy.clearLayers();
}

// GET /trips/{trip_id}/preview
async function loadTripPreview(tripId) {
  const url = getCleanUrl();
  try {
    const resp = await fetch(`${url}/trips/${tripId}/preview`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      signal: AbortSignal.timeout(15000)
    });

    if (!resp.ok) {
      const errData = await resp.json().catch(() => ({}));
      throw new Error(errData?.error?.message || `HTTP ${resp.status}`);
    }

    const data = await resp.json();
    state.currentTripData = data;

    // Coordinate order: ALWAYS [lat, lon]
    const gpsPoints = data.gps; // array of [lat, lon]
    const gpsTimes = data.gps_t || [];

    // Render raw GPS points as blue dots
    gpsPoints.forEach((pt, idx) => {
      const t = gpsTimes[idx] !== undefined ? `${gpsTimes[idx]}s` : `#${idx}`;
      const marker = L.circleMarker(pt, {
        renderer: canvasRenderer,
        radius: 3.5,
        fillColor: '#38bdf8',
        fillOpacity: 0.85,
        color: '#0284c7',
        weight: 1
      }).bindPopup(`<strong>Point #${idx}</strong><br>Time: ${t}<br>Lat: ${pt[0].toFixed(6)}<br>Lon: ${pt[1].toFixed(6)}`);
      
      state.layers.rawGps.addLayer(marker);
    });

    // Fit map bounds
    if (data.bbox && data.bbox.length === 2) {
      // bbox is [[south, west], [north, east]]
      map.fitBounds(data.bbox, { padding: [40, 40] });
    }
  } catch (err) {
    console.error('Error fetching preview:', err);
    showAlert('Preview Error', `Failed to load GPS track preview for Trip ${tripId}: ${err.message}`, true);
  }
}

// GET /trips/{trip_id}/network?buffer_m=...
async function loadTripNetwork(tripId, bufferM = 150.0) {
  const url = getCleanUrl();
  state.layers.network.clearLayers();

  try {
    const resp = await fetch(`${url}/trips/${tripId}/network?buffer_m=${bufferM}`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      signal: AbortSignal.timeout(20000)
    });

    if (!resp.ok) return;

    const data = await resp.json();
    const segments = data.segments || [];

    // Draw all road segments as thin grey layer on canvas
    segments.forEach(seg => {
      // seg is [[lat1, lon1], [lat2, lon2]]
      const line = L.polyline(seg, {
        renderer: canvasRenderer,
        color: '#94a3b8',
        weight: 1.2,
        opacity: 0.55
      });
      state.layers.network.addLayer(line);
    });
  } catch (err) {
    console.warn('Network layer fetch failed:', err);
  }
}

// POST /predict (Main Pipeline Execution)
async function runPrediction() {
  const tripId = state.selectedTripId;
  if (!tripId) return;

  const url = getCleanUrl();
  
  // UI Loading State: disable button, show spinner
  el.btnRunPredict.disabled = true;
  el.btnRunPredict.querySelector('.btn-text').textContent = 'Running Pipeline...';
  el.predictSpinner.classList.remove('hidden');
  hideAlert();

  // 120s timeout per API Contract
  state.abortController = new AbortController();
  const timeoutId = setTimeout(() => state.abortController.abort(), 120000);

  try {
    const startTime = performance.now();
    const resp = await fetch(`${url}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      },
      body: JSON.stringify({ trip_id: tripId }),
      signal: state.abortController.signal
    });

    clearTimeout(timeoutId);
    const clientElapsedMs = Math.round(performance.now() - startTime);

    if (!resp.ok) {
      const errData = await resp.json().catch(() => ({}));
      const msg = errData?.error?.message || `Pipeline failed with HTTP ${resp.status}`;
      throw new Error(msg);
    }

    const data = await resp.json();
    state.predictionResult = data;

    // Render Routes on Map
    renderPredictionResults(data);

    // Render Metrics Panel
    renderMetricsPanel(data, clientElapsedMs);

    // Handle Warnings
    if (data.warnings && data.warnings.length > 0) {
      showAlert(`Notice: GPS Anomalies Detected (${data.warnings.length})`, data.warnings, false);
    }

  } catch (err) {
    console.error('Prediction failed:', err);
    if (err.name === 'AbortError') {
      showAlert('Pipeline Timeout', 'Inference exceeded the 120-second timeout limit. Please retry.', true);
    } else {
      showAlert('Pipeline Execution Error', `${err.message}`, true);
    }
  } finally {
    el.btnRunPredict.disabled = false;
    el.btnRunPredict.querySelector('.btn-text').textContent = 'Run Map-Matching';
    el.predictSpinner.classList.add('hidden');
  }
}

// Render Prediction Results to Leaflet
function renderPredictionResults(data) {
  // Clear previous prediction layers
  state.layers.hmmRoute.clearLayers();
  state.layers.classifierRoute.clearLayers();
  state.layers.pointAccuracy.clearLayers();

  // 1. Draw HMM Predicted Route:
  // Note from Contract: pred_route.paths is a list of polylines, not one line!
  // Draw each path as a separate L.polyline, and never connect the end of one to start of next!
  if (data.pred_route && Array.isArray(data.pred_route.paths)) {
    data.pred_route.paths.forEach((path, idx) => {
      const poly = L.polyline(path, {
        renderer: canvasRenderer,
        color: '#10b981', // Solid Emerald Green
        weight: 5,
        opacity: 0.95,
        lineCap: 'round',
        lineJoin: 'round'
      }).bindPopup(`<strong>HMM Predicted Segment #${idx + 1}</strong><br>Points in subpath: ${path.length}`);
      state.layers.hmmRoute.addLayer(poly);
    });
  }

  // 2. Draw Classifier-Only Route (dashed red/crimson)
  if (data.classifier_only_route && Array.isArray(data.classifier_only_route.paths)) {
    data.classifier_only_route.paths.forEach((path, idx) => {
      const poly = L.polyline(path, {
        renderer: canvasRenderer,
        color: '#ef4444', // Red
        weight: 3.5,
        dashArray: '6, 8',
        opacity: 0.85
      }).bindPopup(`<strong>Classifier-Only Segment #${idx + 1}</strong>`);
      state.layers.classifierRoute.addLayer(poly);
    });
  }

  // 3. Point Correctness: Green/Red dots on raw GPS
  if (state.currentTripData && Array.isArray(data.point_correct)) {
    const gpsPts = state.currentTripData.gps;
    data.point_correct.forEach((isCorrect, idx) => {
      if (idx < gpsPts.length) {
        const pt = gpsPts[idx];
        const marker = L.circleMarker(pt, {
          renderer: canvasRenderer,
          radius: 4,
          fillColor: isCorrect ? '#10b981' : '#ef4444',
          fillOpacity: 0.95,
          color: isCorrect ? '#065f46' : '#991b1b',
          weight: 1.5
        }).bindPopup(`<strong>Point #${idx}</strong><br>Status: <span style="color: ${isCorrect ? '#10b981' : '#ef4444'}; font-weight: bold;">${isCorrect ? 'Correct' : 'Mismatch'}</span><br>Lat: ${pt[0].toFixed(6)}<br>Lon: ${pt[1].toFixed(6)}`);
        state.layers.pointAccuracy.addLayer(marker);
      }
    });
  }
}

// Render Metrics & Results Drawer
function renderMetricsPanel(data, clientElapsedMs) {
  const m = data.metrics || {};
  el.resTripBadge.textContent = `Trip ${data.trip_id}`;
  
  const serverTime = m.inference_time_ms ? `${m.inference_time_ms} ms` : `${clientElapsedMs} ms`;
  el.resLatency.textContent = `${serverTime} (roundtrip: ${clientElapsedMs} ms)`;

  // All numbers come from the API; null (no reference route on the server) is shown as n/a.
  const pct = v => (v === null || v === undefined) ? 'n/a' : (v * 100).toFixed(1) + '%';
  const num = v => (v === null || v === undefined) ? 'n/a' : v.toFixed(3);
  const delta = (a, b, scale, digits, suffix) =>
    (a === null || a === undefined || b === null || b === undefined) ? 'n/a'
      : `${(a - b) >= 0 ? '+' : ''}${((a - b) * scale).toFixed(digits)}${suffix}`;

  // Accuracy
  el.hmmAcc.textContent = pct(m.hmm_pointwise_accuracy);
  el.clfAcc.textContent = pct(m.classifier_only_pointwise_accuracy);
  el.accDelta.textContent = delta(m.hmm_pointwise_accuracy, m.classifier_only_pointwise_accuracy, 100, 1, '%');

  // Continuity
  el.hmmCont.textContent = pct(m.route_continuity);
  el.clfCont.textContent = pct(m.classifier_only_route_continuity);
  el.contDelta.textContent = delta(m.route_continuity, m.classifier_only_route_continuity, 100, 1, '%');

  // Macro F1
  el.hmmF1.textContent = num(m.hmm_macro_f1);
  el.clfF1.textContent = num(m.classifier_only_macro_f1);
  el.f1Delta.textContent = delta(m.hmm_macro_f1, m.classifier_only_macro_f1, 1, 3, '');

  // Breaks
  const breaks = m.n_disconnected_breaks !== undefined ? m.n_disconnected_breaks : '-';
  el.hmmBreaks.textContent = breaks;
  el.resTotalArcs.textContent = data.pred_route?.n_arcs || '-';
  el.breaksBadge.textContent = `${breaks} ${breaks === 1 ? 'break' : 'breaks'}`;

  el.resultsDrawer.style.display = 'block';
}

// Layer Toggle Handlers
function setupLayerToggles() {
  el.layerRawGps.addEventListener('change', e => {
    if (e.target.checked) map.addLayer(state.layers.rawGps);
    else map.removeLayer(state.layers.rawGps);
  });

  el.layerNetwork.addEventListener('change', e => {
    if (e.target.checked) map.addLayer(state.layers.network);
    else map.removeLayer(state.layers.network);
  });

  el.layerHmm.addEventListener('change', e => {
    if (e.target.checked) map.addLayer(state.layers.hmmRoute);
    else map.removeLayer(state.layers.hmmRoute);
  });

  el.layerClassifier.addEventListener('change', e => {
    if (e.target.checked) map.addLayer(state.layers.classifierRoute);
    else map.removeLayer(state.layers.classifierRoute);
  });

  el.layerPointAccuracy.addEventListener('change', e => {
    if (e.target.checked) map.addLayer(state.layers.pointAccuracy);
    else map.removeLayer(state.layers.pointAccuracy);
  });
}

// Setup Event Listeners
function setupEvents() {
  el.apiBaseUrl.value = state.apiBaseUrl;

  el.btnTestConn.addEventListener('click', () => testConnection(true));

  el.apiBaseUrl.addEventListener('keypress', e => {
    if (e.key === 'Enter') testConnection(true);
  });

  el.tripSelect.addEventListener('change', e => {
    onTripSelected(e.target.value);
  });

  el.networkBuffer.addEventListener('input', e => {
    el.bufferVal.textContent = `${e.target.value} m`;
  });

  el.networkBuffer.addEventListener('change', e => {
    if (state.selectedTripId) {
      loadTripNetwork(state.selectedTripId, parseFloat(e.target.value));
    }
  });

  el.btnRunPredict.addEventListener('click', runPrediction);

  el.btnAlertClose.addEventListener('click', hideAlert);

  setupLayerToggles();
}

// Main Boot
window.addEventListener('DOMContentLoaded', () => {
  initMap();
  setupEvents();
  // Auto test connection on launch
  testConnection(false);
});
