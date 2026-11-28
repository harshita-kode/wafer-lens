'use strict';

const COLORS = { pass: '#3553E6', pass2: '#2A43C4', fail: '#F2603A', edge: '#F6E04A' };
const NAMES = {
  'center': 'Center', 'donut': 'Donut', 'edge-loc': 'Edge-Loc', 'edge-ring': 'Edge-Ring',
  'loc': 'Loc', 'near-full': 'Near-full', 'random': 'Random', 'scratch': 'Scratch', 'none': 'none',
};
const FIRST_SAMPLE = 'edge-ring';
const MAX_UPLOAD_BYTES = 2_000_000;

const $ = (id) => document.getElementById(id);
const state = { busy: false };

/* ---------- talking to the API ---------- */

function errorMessage(body, status) {
  if (body && typeof body.detail === 'string') return body.detail;
  if (body && Array.isArray(body.detail)) return body.detail.map((d) => d.msg).join('; ');
  return `Request failed (${status})`;
}

async function api(path, options) {
  const response = await fetch(path, options);
  let body = null;
  try { body = await response.json(); } catch { /* not JSON */ }
  if (!response.ok) throw new Error(errorMessage(body, response.status));
  return body;
}

function showError(message) {
  const toast = $('error');
  toast.textContent = message;
  toast.hidden = false;
  clearTimeout(showError.timer);
  showError.timer = setTimeout(() => { toast.hidden = true; }, 6000);
}

/* ---------- drawing wafers ---------- */

function drawWafer(canvas, map) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!map || !map.length) return;

  const rows = map.length, cols = map[0].length;
  const cell = Math.floor(canvas.width / Math.max(rows, cols));
  const gap = cell >= 6 ? Math.max(1, Math.round(cell * 0.12)) : 0;
  const x0 = Math.floor((canvas.width - cols * cell) / 2);
  const y0 = Math.floor((canvas.height - rows * cell) / 2);

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const v = map[i][j];
      if (v === 0) continue;
      let color;
      if (v === 1) {
        color = (i * 7 + j * 13) % 5 === 0 ? COLORS.pass2 : COLORS.pass;
      } else {
        const touchesPass = [[i - 1, j], [i + 1, j], [i, j - 1], [i, j + 1]]
          .some(([a, b]) => map[a]?.[b] === 1);
        color = touchesPass ? COLORS.edge : COLORS.fail;
      }
      ctx.fillStyle = color;
      ctx.fillRect(x0 + j * cell, y0 + i * cell, cell - gap, cell - gap);
    }
  }
}

function showWafer(map, caption) {
  $('wafer-empty').hidden = true;
  drawWafer($('wafer'), map);
  $('wafer-caption').textContent = `Wafer map · ${caption}`;
  $('wafer').setAttribute('aria-label', `Wafer map from ${caption}, ${map.length} by ${map[0].length} dies`);
}

function clearWafer(message) {
  drawWafer($('wafer'), null);
  $('wafer-empty').textContent = message;
  $('wafer-empty').hidden = false;
}

/* ---------- reading .npy files in the browser (preview only) ---------- */

function parseNpy(buffer) {
  const bytes = new Uint8Array(buffer);
  const view = new DataView(buffer);
  if (bytes[0] !== 0x93 || String.fromCharCode(...bytes.slice(1, 6)) !== 'NUMPY') {
    throw new Error('Not a .npy file');
  }
  const major = bytes[6];
  const headerLen = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const start = major === 1 ? 10 : 12;
  const header = new TextDecoder().decode(bytes.slice(start, start + headerLen));

  const descr = header.match(/'descr':\s*'([^']+)'/)?.[1];
  const shape = (header.match(/'shape':\s*\(([^)]*)\)/)?.[1] || '')
    .split(',').map((s) => s.trim()).filter(Boolean).map(Number);
  if (/'fortran_order':\s*True/.test(header) || shape.length !== 2) throw new Error('Unsupported layout');

  const Types = { '|u1': Uint8Array, '|i1': Int8Array, '<i2': Int16Array, '<i4': Int32Array,
                  '<i8': BigInt64Array, '<f4': Float32Array, '<f8': Float64Array };
  const T = Types[descr];
  if (!T) throw new Error(`Unsupported dtype ${descr}`);

  const dataStart = start + headerLen;
  const values = new T(buffer.slice(dataStart, dataStart + shape[0] * shape[1] * T.BYTES_PER_ELEMENT));
  const rows = [];
  for (let i = 0; i < shape[0]; i++) {
    rows.push(Array.from(values.slice(i * shape[1], (i + 1) * shape[1]), Number));
  }
  return rows;
}

/* ---------- turning the report into safe HTML ---------- */

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function formatInline(text) {
  return escapeHtml(text)                                   // 1. make it safe
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')        // 2. then add our own tags
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/\[(S\d+)\]/g, '<a class="chip mono" href="#src-$1">$1</a>');
}

function renderReportBody(markdown) {
  const sections = [];
  let current = null;
  for (const raw of markdown.split('\n')) {
    const line = raw.trim();
    if (!line) continue;
    const heading = line.match(/^\*\*(.+?):\*\*\s*(.*)$/);
    if (heading) {
      current = { title: heading[1], items: heading[2] ? [heading[2]] : [] };
      sections.push(current);
      continue;
    }
    if (!current) { current = { title: 'Notes', items: [] }; sections.push(current); }
    current.items.push(line.replace(/^[-*]\s+/, ''));
  }
  if (!sections.length) return '<p class="muted">No report text was returned.</p>';
  return '<div class="sections">' + sections.map((s) =>
    `<div class="section"><h3 class="mono">${escapeHtml(s.title)}</h3>` +
    s.items.map((item) => `<p class="item">${formatInline(item)}</p>`).join('') +
    '</div>').join('') + '</div>';
}

/* ---------- filling the page ---------- */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

const pct = (x) => `${Math.round(x * 100)}%`;
const formatMs = (ms) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`);

function renderPrediction(result) {
  const p = result.prediction;
  const inv = result.investigation;
  const unresolved = p.resolved_by === 'unresolved';

  $('result').dataset.uncertain = String(p.uncertain);
  $('pattern').textContent = unresolved ? `${p.top_k[0].label} / ${p.top_k[1].label}` : p.label;
  $('pattern').classList.toggle('pair', unresolved);

  $('resolved-badge').hidden = p.resolved_by !== 'investigation';
  $('uncertain-badge').hidden = !p.uncertain;
  $('uncertain-badge').textContent = unresolved ? 'STILL UNCERTAIN AFTER INVESTIGATION' : 'UNCERTAIN · BELOW 70%';

  $('conf-label').textContent = inv ? 'Evidence score' : 'Confidence';
  $('conf-text').textContent = pct(p.confidence);

  const filled = Math.round(p.confidence * 20);
  $('blocks').replaceChildren(...Array.from({ length: 20 }, (_, i) => el('div', i < filled ? 'block on' : 'block')));
  $('blocks').setAttribute('aria-label', `${inv ? 'Evidence score' : 'Confidence'} ${pct(p.confidence)}`);

  $('topk').replaceChildren(...p.top_k.map((t) => {
    const row = el('div', 'topk-row');
    const bar = el('div', 'bar');
    const fill = el('span');
    fill.style.width = `${Math.max(2, Math.round(t.confidence * 100))}%`;
    bar.append(fill);
    row.append(el('span', 'topk-name', t.label), bar, el('span', 'topk-pct mono', pct(t.confidence)));
    return row;
  }));

  renderInvestigation(inv);
}

function renderInvestigation(inv) {
  $('investigation').hidden = !inv;
  if (!inv) return;

  const said = `The CNN alone said ${inv.cnn_label} (${pct(inv.cnn_confidence)}).`;
  $('inv-note').textContent = inv.resolved_by === 'investigation'
    ? `${said} Rotated views and similar past wafers point to ${inv.label}.`
    : `${said} The extra evidence was not strong enough to decide.`;

  $('inv-rows').replaceChildren(...inv.candidates.map((c) => {
    const row = el('tr');
    const name = el('th', '', c);
    name.scope = 'row';
    row.append(name, el('td', '', pct(inv.tta[c])), el('td', '', pct(inv.neighbours[c])), el('td', '', pct(inv.scores[c])));
    return row;
  }));
}

function renderFacts(facts) {
  $('facts').hidden = !facts;
  if (!facts) return;

  $('facts-text').textContent = `${facts.failed_dies} of ${facts.dies} dies failed (${pct(facts.fail_rate)})`;
  const zones = [['center', 'Centre'], ['middle', 'Middle'], ['edge', 'Edge']];
  $('zone-bar').replaceChildren(...zones.map(([key]) => {
    const part = el('span', `zone-${key}`);
    part.style.width = `${Math.round((facts.failure_share_by_zone[key] || 0) * 100)}%`;
    return part;
  }));
  $('zone-legend').replaceChildren(...zones.map(([key, name]) =>
    el('span', '', `${name} ${pct(facts.failure_share_by_zone[key] || 0)}`)));
}

function renderSteps(steps, timings) {
  $('agent-panel').hidden = !steps.length;
  if (!steps.length) return;

  $('agent-total').textContent = `${steps.length} STEPS · ${formatMs(timings.total)}`;
  $('steps').replaceChildren(...steps.map((step, i) => {
    const item = el('li', 'step');
    item.append(
      el('span', 'step-num mono', String(i + 1).padStart(2, '0')),
      el('span', `step-node px kind-${step.node}`, step.node),
      el('span', 'step-summary', step.summary),
      el('span', 'step-ms mono', formatMs(step.ms)),
    );
    return item;
  }));
}

function renderReport(result) {
  $('report-meta').textContent = result.generated_by === 'llm'
    ? 'WRITTEN BY LLM · ONLY FROM RETRIEVED NOTES'
    : 'TEMPLATE REPORT · LLM UNAVAILABLE';
  $('report-body').innerHTML = renderReportBody(result.report_body || '');

  const warnings = $('warnings');
  warnings.hidden = !result.warnings.length;
  warnings.textContent = result.warnings.join(' · ');

  $('sources-wrap').hidden = !result.sources.length;
  $('sources').replaceChildren(...result.sources.map((s) => {
    const li = el('li');
    li.id = `src-${s.label}`;
    li.append(el('span', 'chip mono', s.label), `${s.authors} (${s.year}). ${s.title}. `);
    if (/^https?:\/\//.test(s.url)) {
      const link = el('a', '', 'link');
      link.href = s.url;
      link.target = '_blank';
      link.rel = 'noopener';
      li.append(link);
    }
    return li;
  }));
}

async function renderSimilar(similar, predicted) {
  const same = similar.filter((w) => w.label === predicted).length;
  $('similar-summary').textContent = similar.length
    ? `${same} of the ${similar.length} most similar past wafers were labelled ${predicted}`
    : 'Similar-wafer search is not available';

  const figures = similar.map((w) => {
    const figure = el('figure', 'thumb');
    const canvas = el('canvas');
    canvas.width = canvas.height = 140;
    canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label', `Past wafer labelled ${w.label}`);
    const caption = el('figcaption');
    caption.append(el('span', 'px', w.label), el('span', 'mono', w.similarity.toFixed(2)));
    figure.append(canvas, caption);
    return { figure, canvas, index: w.index };
  });
  $('thumbs').replaceChildren(...figures.map((f) => f.figure));

  await Promise.all(figures.map(async (f) => {
    try {
      const past = await api(`/past-wafers/${f.index}`);
      drawWafer(f.canvas, past.wafer_map);
    } catch {
      f.canvas.title = 'Map not available';
    }
  }));
}

function renderTimings(t) {
  const parts = Object.entries(t)
    .filter(([key]) => key !== 'total')
    .map(([key, ms]) => `${key.toUpperCase()} ${formatMs(ms)}`);
  if (t.total !== undefined) parts.push(`TOTAL ${formatMs(t.total)}`);
  $('timings').textContent = parts.join(' · ');
}

function renderResult(result) {
  renderPrediction(result);
  renderFacts(result.wafer_facts);
  renderReport(result);
  renderSteps(result.agent_steps || [], result.timings_ms);
  renderTimings(result.timings_ms);
  renderSimilar(result.similar_wafers, result.prediction.label);
}

function renderResult(result) {
  renderPrediction(result);
  renderFacts(result.wafer_facts);
  renderReport(result);
  renderSteps(result.agent_steps || [], result.timings_ms);
  renderTimings(result.timings_ms);
  renderSimilar(result.similar_wafers, result.prediction.label);
}

/* ---------- user actions ---------- */

function setBusy(on) {
  state.busy = on;
  $('result').dataset.busy = String(on);
  document.querySelectorAll('.sample').forEach((b) => { b.disabled = on; });
  $('upload-input').disabled = on;
  if (on) {
    $('pattern').textContent = 'ANALYSING';
    $('report-meta').textContent = 'ANALYSING…';
  }
}

async function runAnalysis(request) {
  setBusy(true);
  try {
    renderResult(await request());
  } catch (e) {
    resetResult();
    showError(e.message);
  } finally {
    setBusy(false);
  }
}

async function pickSample(name, button) {
  if (state.busy) return;
  document.querySelectorAll('.sample').forEach((b) => b.setAttribute('aria-pressed', String(b === button)));
  $('window-file').textContent = `samples/${name}.npy`;
  try {
    const sample = await api(`/samples/${encodeURIComponent(name)}`);
    showWafer(sample.wafer_map, `${name}.npy`);
    await runAnalysis(() => api('/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ wafer_map: sample.wafer_map }),
    }));
  } catch (e) {
    showError(e.message);
  }
}

async function uploadFile(file) {
  if (!file || state.busy) return;
  if (!file.name.toLowerCase().endsWith('.npy')) return showError('Please choose a .npy file');
  if (file.size > MAX_UPLOAD_BYTES) return showError('File too large (max 2 MB)');

  document.querySelectorAll('.sample').forEach((b) => b.setAttribute('aria-pressed', 'false'));
  $('window-file').textContent = file.name;
  try {
    showWafer(parseNpy(await file.arrayBuffer()), file.name);
  } catch {
    clearWafer('Preview unavailable for this file; the server will still check it');
  }

  const form = new FormData();
  form.append('file', file);
  await runAnalysis(() => api('/analyze/file', { method: 'POST', body: form }));
}

/* ---------- start-up ---------- */

async function loadHealth() {
  const status = $('status');
  try {
    const h = await api('/health');
    status.dataset.state = h.status;
    const missing = [];
    if (!h.llm) missing.push('NO LLM');
    if (!h.similar_index) missing.push('NO SIMILAR');    
    const mode = h.mode === 'agent' ? 'AGENT' : 'PIPELINE';
    $('status-text').textContent = h.status === 'ok'
      ? `API OK · LLM ON · ${mode}`
      : `DEGRADED · ${missing.join(' · ')} · ${mode}`;
  } catch {
    status.dataset.state = 'down';
    $('status-text').textContent = 'API OFFLINE';
  }
}

async function loadSamples() {
  try {
    const { samples } = await api('/samples');
    const buttons = samples.map((name) => {
      const button = el('button', 'sample', NAMES[name] || name);
      button.type = 'button';
      button.setAttribute('aria-pressed', 'false');
      button.addEventListener('click', () => pickSample(name, button));
      return button;
    });
    $('samples').replaceChildren(...buttons);
    const first = samples.indexOf(FIRST_SAMPLE);
    if (first >= 0) pickSample(FIRST_SAMPLE, buttons[first]);
  } catch (e) {
    showError(`Could not load samples: ${e.message}`);
  }
}

$('upload-input').addEventListener('change', (event) => {
  uploadFile(event.target.files[0]);
  event.target.value = '';            // allow picking the same file again
});

loadHealth();
loadSamples();