// СТОП-РАК — основной модуль: навигация, загрузка данных, рабочая станция, вьюеры, сегментация.
import { S, app } from './state.js';
import { loadVol, loadLabels } from './nifti.js';
import { VR3D } from './vr.js';
import { SliceView, RGB, planeGeom } from './mpr.js';
import * as M from './model.js';
import { ic, $, $$, esc, toast, openModal, closeModal, confirmDialog, rangeFill } from './ui.js';
import * as P from './panels.js';
import { initUpload } from './upload.js';
import { initAI } from './ai.js';
import { openReport } from './report.js';

const DEMO_CASES = ['sub-BO16', 'sub-BO90', 'sub-BO102'];
const cache = {};                       // id -> {part: Promise}
const views = [];                       // активные SliceView
let vr = null, hero = null, heroData = null, vrCell = null, activeView = null;
app.stroke = new Map(); app.spaceDown = false;

/* ------------------------------------------------------------ загрузка данных */
// Данные читаются через fetch (веб-сервер) либо, при открытии index.html двойным кликом (file://), из data/*.js со встроенным base64
function loadScriptData(url) {
  window.__SR = window.__SR || {};
  return new Promise((res, rej) => {
    if (window.__SR[url]) return res(window.__SR[url]);
    const sc = document.createElement('script'); sc.src = url + '.js';
    sc.onload = () => (window.__SR[url] ? res(window.__SR[url]) : rej(new Error('Пустой файл данных ' + url)));
    sc.onerror = () => rej(new Error('Не удалось загрузить ' + url)); document.head.appendChild(sc);
  });
}
const b64blob = (b64) => { const bin = atob(b64); const u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return new Blob([u]); };
async function fetchBlob(url) {
  if (location.protocol !== 'file:') { try { const r = await fetch(url); if (r.ok) return r.blob(); } catch (e) { /* пробуем встроенные данные */ } }
  return b64blob(await loadScriptData(url));
}
function part(id, name) {
  cache[id] = cache[id] || {};
  if (!cache[id][name]) {
    const base = `data/${id}/`;
    cache[id][name] = name === 'meta' ? fetchBlob(base + 'case.json').then((b) => b.text()).then(JSON.parse)
      : name === 'seg' ? fetchBlob(base + 'seg.nii.gz').then(loadLabels)
      : name.startsWith('depth') ? fetchBlob(base + name + '.nii.gz').then(loadLabels)
      : fetchBlob(base + name + '.nii.gz').then((b) => loadVol(b, { name }));
  }
  return cache[id][name];
}
const loadAll = (id) => Promise.all(['meta', 't1', 't1ce', 't2', 'flair', 'seg', 'depth_all', 'depth_enh'].map((n) => part(id, n)));

/* ------------------------------------------------------------ главная */
function landing() {
  document.body.classList.add('landing-scroll');
  $$('[data-act=open-demo]').forEach((b) => b.addEventListener('click', () => openDemo(DEMO_CASES[0])));
  $$('[data-act=upload]').forEach((b) => b.addEventListener('click', () => app.openUpload()));
  const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add('is-in'); io.unobserve(e.target); } }), { threshold: 0.12 });
  $$('.lp-section .lp-wrap > *, .step, .role, .tcard, .prov-col, .zone-list li').forEach((el) => { el.classList.add('reveal'); io.observe(el); });
  initHero();
  $$('.lp-nav a').forEach((a) => a.addEventListener('click', (e) => { const t = $(a.getAttribute('href')); if (t) { e.preventDefault(); $('#screen-landing').scrollTo({ top: t.offsetTop - 60, behavior: 'smooth' }); } }));
}

async function initHero() {
  const id = DEMO_CASES[0];
  try {
    const [t1, seg] = await Promise.all([part(id, 't1'), part(id, 'seg')]);
    heroData = { t1, seg };
    hero = new VR3D($('#heroCanvas'), { autoRotate: true, rotateSpeed: 1.1, quality: 0.72, maxDpr: 1.5 });
    hero.setData({ t1, mod: t1, seg: seg.data, focus: [83, 127, 80] });
    hero.setBrain({ opacity: 0.3, thr: 0.2 });
    hero.setView('oblique');
    heroLayers();
    $('#heroLoader').classList.add('is-done');
    $$('#heroToggles .tg').forEach((b) => b.addEventListener('click', () => {
      const k = b.dataset.layer; const on = !b.classList.contains('is-on');
      if (k === 'all') { $$('#heroToggles .tg').forEach((x) => { x.classList.toggle('is-on', on); x.setAttribute('aria-pressed', on); }); }
      else { b.classList.toggle('is-on', on); b.setAttribute('aria-pressed', on); const rest = $$('#heroToggles .tg:not([data-layer=all])'); $('#heroToggles [data-layer=all]').classList.toggle('is-on', rest.some((x) => x.classList.contains('is-on'))); }
      heroLayers();
    }));
    // метрики на главной — реальные, из маски
    Promise.all([part(id, 'meta'), part(id, 'depth_all'), part(id, 'depth_enh')]).then(([meta, da, de]) => {
      const m = M.computeMetrics(seg.data, t1, meta.affine); const fm = M.simulateMask(seg.data, da.data, de.data, null, 2.4); const fv = M.countVols(fm, t1.spacing);
      const row = (n, a, b) => `<tr><td>${n}</td><td>${M.nf(a)} см³</td><td>${M.nf(b)} см³</td><td class="${b < a ? 'down' : 'up'}">${M.sgn(M.pct(b, a))}%</td></tr>`;
      $('#lpMetrics tbody').innerHTML = row('Вся опухолевая область', m.V.whole, fv.whole) + row('Активная зона', m.V.enh, fv.enh) + row('Некротическая зона', m.V.nec, fv.nec) + row('FLAIR-гиперинтенсивная зона', m.V.ede, fv.ede);
    }).catch(() => {});
    // фоновая подгрузка остальных серий
    setTimeout(() => loadAll(id).catch(() => {}), 2500);
  } catch (e) { $('#heroLoader').innerHTML = `<span>Не удалось загрузить демонстрационные данные: ${esc(e.message)}.<br>Запустите сайт через веб-сервер: <code>python3 -m http.server</code></span>`; }
}
function heroLayers() {
  const on = (k) => $(`#heroToggles [data-layer=${k}]`).classList.contains('is-on');
  hero.setLayer('nec', { show: on('nec'), opacity: 0.85 }); hero.setLayer('ede', { show: on('ede'), opacity: 0.26 }); hero.setLayer('enh', { show: on('enh'), opacity: 0.8 });
}

/* ------------------------------------------------------------ навигация */
function goTo(name) {
  $$('.screen').forEach((s) => s.classList.remove('is-active')); $('#screen-' + name).classList.add('is-active');
  document.body.classList.toggle('landing-scroll', name === 'landing');
  if (hero) hero.visible = name === 'landing';
  if (vr) vr.visible = name === 'ws';
}
app.goTo = goTo;

/* ------------------------------------------------------------ состояние случая */
function resetCaseState() {
  Object.assign(S, { seg: null, segSrc: null, segDone: false, segFade: 1, segRunning: false, follow: null, tp: 0, sim: null, simOn: false, simInfo: null,
    confirmed: false, maskEdited: false, reportStale: false, metrics: null, baseMetrics: null, log: [], versions: [], undo: [], redo: [], reportText: '', reportSigned: false, consilium: [],
    hiClass: 0, cine: false, returned: null, layout: 'quad', treat: { view: 'home', params: null, sel: { kps: '', excluded: {}, cmp: [] } } });
  S.layers = { 1: { show: true, opacity: 0.5, style: 'both' }, 2: { show: true, opacity: 0.32, style: 'fill' }, 3: { show: true, opacity: 0.55, style: 'both' } };
  S.clip = { on: false, axis: 'x', pos: 0.5, flip: false };
}
export function logEvent(t) { S.log.unshift({ t: new Date(), text: t }); app.log = logEvent; P.refresh('seg'); }
app.log = logEvent;

async function openDemo(id) {
  goTo('ws'); showLoading(true, 'Загрузка реального МРТ…');
  try {
    const [meta, t1, t1ce, t2, flair, seg, da, de] = await loadAll(id);
    resetCaseState();
    S.mode = 'demo'; S.caseId = id; S.meta = meta;
    S.vols = { t1, t1ce, t2, flair }; S.segSrc = seg.data; S.depthAll = da.data; S.depthEnh = de.data;
    S.seg = new Uint8Array(seg.data.length);
    S.mod = 't1ce';
    S.patient = { id: 'DEMO-GBM-' + id.replace('sub-', ''), age: meta.age, mgmt: meta.mgmt, field: meta.field, scanner: meta.scanner, affine: meta.affine };
    S.wl = {}; for (const k of Object.keys(S.vols)) S.wl[k] = { lo: 0, hi: 225 };
    // курсор — на центре очага (исследование открыто на срезе интереса)
    let sx = 0, sy = 0, sz = 0, n = 0; const d = seg.data, { nx, ny } = t1;
    for (let c = 0; c < d.length; c++) if (d[c] === 3) { sx += c % nx; sy += ((c / nx) | 0) % ny; sz += (c / (nx * ny)) | 0; n++; }
    S.cursor = n ? [Math.round(sx / n), Math.round(sy / n), Math.round(sz / n)] : [nx >> 1, ny >> 1, t1.nz >> 1];
    buildWorkstation();
    toast('Демонстрационный случай открыт. Запустите сегментацию на вкладке «Сегментация».', '');
  } catch (e) { console.error(e); toast('Ошибка загрузки данных: ' + e.message, 'err'); }
  showLoading(false);
}
app.openDemo = openDemo;

/** Открыть локальные пользовательские данные: {vols:{key:Vol}, seg?:Uint8Array, meta} */
async function openUser({ vols, seg, meta, names }) {
  goTo('ws'); showLoading(true, 'Подготовка просмотра…');
  resetCaseState();
  S.mode = 'user'; S.caseId = 'local'; S.meta = meta || {};
  S.vols = vols; const keys = Object.keys(vols); S.mod = keys[0];
  S.modNames = { ...S.modNames, ...names };
  S.segSrc = null; S.seg = new Uint8Array(vols[keys[0]].size); S.depthAll = S.depthEnh = null;
  if (seg) { S.seg = seg; S.segSrc = seg.slice(); S.segDone = true; S.metrics = M.computeMetrics(seg, vols[keys[0]], null, vols); }
  S.patient = { id: 'Локальный файл', age: null, mgmt: null, field: meta?.field, scanner: '', affine: null };
  S.wl = {}; for (const k of keys) S.wl[k] = { lo: 0, hi: 225 };
  const v = vols[keys[0]]; S.cursor = [v.nx >> 1, v.ny >> 1, v.nz >> 1];
  buildWorkstation();
  showLoading(false);
}
app.openUser = openUser;

function showLoading(on, text) {
  let o = $('#wsLoading');
  if (!o) { o = document.createElement('div'); o.id = 'wsLoading'; o.className = 'cell-empty'; o.style.cssText = 'position:absolute;inset:56px 0 0 0;z-index:40;background:rgba(7,11,16,.85);flex-direction:column;gap:12px'; o.innerHTML = '<div class="spinner"></div><span></span>'; $('#ws').appendChild(o); o.style.display = 'flex'; o.style.alignItems = 'center'; o.style.justifyContent = 'center'; }
  o.hidden = !on; if (text) o.lastElementChild.textContent = text;
}

/* ------------------------------------------------------------ построение интерфейса */
let uiBuilt = false;
function buildWorkstation() {
  if (!uiBuilt) { initChrome(); uiBuilt = true; }
  const v0 = S.vols[S.mod];
  if (!vr) {
    vrCell = document.createElement('div'); vrCell.className = 'cell vr-cell';
    vrCell.innerHTML = `<div class="vr-host" style="position:absolute;inset:0"></div>
      <div class="vr-hud"><b>3D-реконструкция</b><span id="vrHudSub"></span></div>
      <div class="cell-tools"><button class="icon-btn" data-vr="rot" title="Автовращение" aria-label="Автовращение">${ic('rot')}</button><button class="icon-btn" data-vr="shot" title="Снимок 3D" aria-label="Снимок">${ic('camera')}</button><button class="icon-btn" data-vr="max" title="На весь экран" aria-label="Развернуть">${ic('max')}</button></div>
      <div class="vr-views">${['oblique:Ракурс', 'front:Спереди', 'back:Сзади', 'left:Слева', 'right:Справа', 'top:Сверху'].map((s) => `<button data-view="${s.split(':')[0]}">${s.split(':')[1]}</button>`).join('')}</div>
      <div class="vr-legend" id="vrLegend"></div>`;
    vr = new VR3D($('.vr-host', vrCell), { quality: 0.9, preserve: true, onPick: (p) => { if (p) { S.cursor = [p.i, p.j, p.k]; if (p.label) setHi(p.label, true); onCursor(); } }, onHover: (p, e) => hover3D(p, e) });
    vrCell.addEventListener('click', (e) => {
      const b = e.target.closest('[data-view],[data-vr]'); if (!b) return;
      if (b.dataset.view) { vr.setView(b.dataset.view); $$('.vr-views button', vrCell).forEach((x) => x.classList.toggle('is-on', x === b)); }
      else if (b.dataset.vr === 'rot') { vr.setAutoRotate(!vr.controls.autoRotate); }
      else if (b.dataset.vr === 'shot') { const a = document.createElement('a'); a.href = vr.screenshot(); a.download = 'stoprak-3d.png'; a.click(); toast('Снимок 3D сохранён (без персональных данных)', 'ok'); }
      else if (b.dataset.vr === 'max') { setLayout(S.layout === 'vr' ? 'quad' : 'vr'); }
    });
  }
  vr.setData({ t1: S.vols.t1 || v0, mod: v0, seg: S.segDone ? activeMask() : null, spacing: v0.spacing, focus: S.mode === 'demo' ? S.cursor : null });
  vr.setSim(null); vr.setClip({ ...S.clip }); vr.setView('oblique'); vr.setAutoRotate(false); $$('.vr-views button', vrCell).forEach((x) => x.classList.remove('is-on'));
  syncVR();
  buildHeader(); buildRail(); buildCenterBar(); buildToolbar();
  setLayout('quad'); P.refreshAll(); updateBadge(); updateReadout();
  app.aiReset?.();
  applyAnon();
}

let chromeDone = false;
function initChrome() {
  if (chromeDone) return; chromeDone = true;
  $('#btnBack').innerHTML = ic('back'); $('#btnBack').onclick = () => { S.cine = false; goTo('landing'); };
  $('#btnAI').innerHTML = `${ic('ai', 15)} ИИ-помощник`; $('#btnSettings').innerHTML = ic('gear');
  $('#aiClose').innerHTML = ic('x');
  $$('[data-close]').forEach((b) => (b.innerHTML = ic('x'), b.addEventListener('click', () => { const m = b.closest('.modal'); m._cancel?.(); m.hidden = true; })));
  $$('.modal').forEach((m) => m.addEventListener('mousedown', (e) => { if (e.target === m) { m._cancel?.(); m.hidden = true; } }));
  $('#btnCompare').onclick = () => openCompare();
  $('#btnReport').onclick = () => openReport();
  $('#btnExport').onclick = (e) => exportMenu(e.currentTarget);
  $('#btnSettings').onclick = () => toggleSettings();
  initTabs($('#tabsLeft'), '[data-pane-left]', 'paneLeft'); initTabs($('#tabsRight'), '[data-pane-right]', 'paneRight');
  initAI(); initUpload();
  initKeys(); initCmdk();
  window.addEventListener('resize', () => views.forEach((v) => v.resize()));
}
function initTabs(bar, sel, attr) {
  bar.addEventListener('click', (e) => {
    const b = e.target.closest('.tab'); if (!b) return; selectTab(bar, sel, attr, b.dataset.tab);
  });
}
function selectTab(bar, sel, attr, key) {
  $$('.tab', bar).forEach((t) => t.classList.toggle('is-active', t.dataset.tab === key));
  $$(sel).forEach((p) => { p.hidden = p.dataset[attr] !== key; });
  if (attr === 'paneRight') P.refresh(key);
}
app.openTab = (key) => { selectTab($('#tabsRight'), '[data-pane-right]', 'paneRight', key); };

function buildHeader() {
  const p = S.patient;
  $('#ptName').textContent = S.mode === 'demo' ? p.id : 'Локальный файл';
  $('#ptTag').textContent = S.mode === 'demo' ? 'ДЕМО · ОБЕЗЛИЧЕННОЕ МРТ' : 'ПРОСМОТР · ЛОКАЛЬНО';
  $('#ptMeta').textContent = S.mode === 'demo'
    ? `${p.age} лет · Глиобластома (первичная) · MGMT: ${p.mgmt ? 'метилирован' : 'не метилирован'} · ${p.field} Тл · ${p.scanner} · врач-рентгенолог: не назначен`
    : `${S.meta?.fileName || ''} · ${S.meta?.dims || ''} · файлы обрабатываются локально`;
  const tp = $('#tpChips'); tp.innerHTML = '';
  const chips = S.mode === 'demo' ? [
    { k: 0, t: 'До операции', sub: 'реальное МРТ' }, { k: 'x1', t: 'Раннее п/о', off: 'В наборе данных только дооперационные исследования' },
    { k: 'x2', t: 'Перед ХЛТ', off: 'В наборе данных только дооперационные исследования' }, { k: 1, t: 'Контроль +3 мес', sub: 'синтетика' },
  ] : [{ k: 0, t: 'Загруженное исследование' }];
  for (const c of chips) {
    const b = document.createElement('button'); b.className = 'chip' + (c.k === S.tp ? ' is-active' : ''); b.disabled = !!c.off; if (c.off) b.title = c.off;
    b.innerHTML = c.t + (c.sub ? ` <span class="src-tag ${c.sub === 'синтетика' ? 'ed' : ''}">${c.sub}</span>` : '');
    b.dataset.k = c.k; b.onclick = () => setTimepoint(c.k); tp.appendChild(b);
  }
  $('#btnCompare').disabled = S.mode !== 'demo'; $('#btnCompare').title = S.mode !== 'demo' ? 'Для загруженного файла доступен только один момент времени' : '';
}
function setTimepoint(k) {
  if (k === 1 && !S.follow) { toast('Сначала выполните сегментацию базового исследования — контрольная точка строится по её результату', 'warn'); app.openTab('seg'); return; }
  S.tp = k; $$('#tpChips .chip').forEach((c) => c.classList.toggle('is-active', c.dataset.k == k));
  if (vr && S.segDone) vr.setSeg(activeMask());
  S.metrics = k === 1 && S.follow ? S.follow.metrics : S.baseMetrics; renderAll(); P.refreshAll();
  if (k === 1) toast('Показана СИНТЕТИЧЕСКАЯ контрольная точка: демонстрационная модель применена к базовому снимку', 'warn');
}
export const activeMask = () => (S.tp === 1 && S.follow ? S.follow.mask : S.seg);

/* --- левая панель --- */
function buildRail() {
  const box = $('#paneSeries'); box.dataset.paneLeft = 'series'; box.dataset.paneLeft = 'series';
  const keys = Object.keys(S.vols); const v = S.vols[keys[0]];
  const cases = S.mode === 'demo' ? `<div class="block"><div class="block-head"><span class="panel-title">Демо-случай</span></div><select class="sel" id="caseSel" aria-label="Демо-случай">${DEMO_CASES.map((c) => `<option value="${c}" ${c === S.caseId ? 'selected' : ''}>${'DEMO-GBM-' + c.replace('sub-', '')}</option>`).join('')}</select><p class="fine" style="margin-top:8px">Реальные обезличённые МРТ глиобластомы из открытого набора OpenNeuro ds007045 (CC0).</p></div>` : '';
  box.innerHTML = cases + keys.map((k) => `<div class="series-item ${k === S.mod ? 'is-active' : ''}" data-mod="${k}" tabindex="0" role="button"><div class="series-thumb"><canvas width="52" height="52"></canvas></div><div class="series-info"><span class="series-name">${esc(S.modNames[k])}</span><span class="series-sub">${v.spacing.map((x) => x.toFixed(1)).join('×')} мм · ${v.nx}×${v.ny}×${v.nz}</span></div></div>`).join('') +
    (S.mode === 'demo' ? ['DWI / ADC', 'Перфузия (rCBV)'].map((n) => `<div class="series-item is-off" title="Серия отсутствует в наборе данных"><div class="series-thumb"></div><div class="series-info"><span class="series-name">${n}</span><span class="series-sub">нет в исследовании</span></div></div>`).join('') : '') +
    routeBlock() +
    `<div class="block"><div class="block-head"><span class="panel-title">Качество данных</span></div>${qualityRows().map(([a, b, c]) => `<div class="kv"><span>${a}</span><b class="${c || ''}">${b}</b></div>`).join('')}</div>`;
  $$('.series-item[data-mod]', box).forEach((it) => { const k = it.dataset.mod; drawThumb($('canvas', it), S.vols[k]); it.onclick = () => setModality(k); it.onkeydown = (e) => { if (e.key === 'Enter') setModality(k); }; });
  $('#caseSel')?.addEventListener('change', (e) => openDemo(e.target.value));
  $('#btnReturn')?.addEventListener('click', returnStudy);
  // история
  const h = $('#paneHistory');
  h.innerHTML = S.mode === 'demo' ? `<div class="tl">
    <div class="tl-item is-real"><div class="tl-date">дата не раскрывается</div><div class="tl-title">МРТ до операции</div><div class="tl-desc">T1, T1-Gd, T2, FLAIR · ${S.meta.field} Тл</div><span class="badge badge-ok">реальное МРТ</span></div>
    <div class="tl-item is-empty"><div class="tl-date">—</div><div class="tl-title">Операция</div><div class="tl-desc">нет данных в наборе</div></div>
    <div class="tl-item is-empty"><div class="tl-date">—</div><div class="tl-title">Раннее п/о МРТ</div><div class="tl-desc">нет данных в наборе</div></div>
    <div class="tl-item is-empty"><div class="tl-date">—</div><div class="tl-title">Химиолучевая терапия</div><div class="tl-desc">не проводилась в демо; параметры задаются в разделе «Лечение»</div></div>
    <div class="tl-item is-synth"><div class="tl-date">условно +3 мес</div><div class="tl-title">Контроль</div><div class="tl-desc">маска рассчитана демонстрационной моделью по базовому снимку</div><span class="badge badge-warn">синтетика</span></div>
    <p class="fine" style="margin-top:6px">Прошлые этапы лечения в этом наборе отсутствуют — они показаны как пустые, а не как факты.</p></div>`
    : `<div class="tl"><div class="tl-item is-real"><div class="tl-date">локально</div><div class="tl-title">Загруженное исследование</div><div class="tl-desc">${esc(S.meta?.fileName || '')}</div></div></div>`;
}
const ROUTE = ['Запланировано', 'Сканирование выполнено', 'Данные загружены', 'Техническая проверка', 'Ожидает врача-рентгенолога', 'На анализе', 'Сегментация требует проверки', 'Заключение подписано', 'Доступно лечащему врачу'];
function routeIdx() { if (S.returned) return 3; if (S.reportSigned) return 8; if (S.segRunning) return 5; if (S.segDone) return S.confirmed ? 7 : 6; return 4; }
function routeBlock() {
  if (S.mode !== 'demo') return '';
  const cur = routeIdx(); const show = S.returned ? ['Данные загружены', 'Требуется техническое уточнение'] : ROUTE.slice(2);
  const idx = S.returned ? 1 : cur - 2;
  return `<div class="block" id="routeBlock"><div class="block-head"><span class="panel-title">Статус исследования</span></div><div class="route">${show.map((t, i) => `<div class="st ${i < idx ? 'is-done' : i === idx ? 'is-cur' : ''}"><i></i>${t}</div>`).join('')}</div>
    <button class="btn btn-sm btn-ghost btn-block" style="margin-top:10px" id="btnReturn" ${S.returned ? 'disabled' : ''}>Вернуть на техническое уточнение</button></div>`;
}
async function returnStudy() {
  const reasons = ['Отсутствует серия', 'Выраженные артефакты', 'Неверный пациент', 'Проблемы с контрастированием', 'Неполная загрузка'];
  $('#cfTitle').textContent = 'Вернуть на техническое уточнение';
  $('#cfBody').innerHTML = `<div class="field"><label>Причина</label><select class="sel" id="rtReason">${reasons.map((r) => `<option>${r}</option>`).join('')}</select></div><div class="field"><label>Комментарий рентгенолаборанту</label><textarea class="txt" id="rtNote" rows="3" placeholder="Что нужно исправить"></textarea></div><p class="fine">Повторная загрузка исправляет то же исследование; повторное сканирование создаёт новую временную точку.</p><div class="row" style="justify-content:flex-end;margin-top:14px"><button class="btn btn-ghost" id="rtNo">Отмена</button><button class="btn btn-primary" id="rtYes">Вернуть</button></div>`;
  const m = openModal('confirmModal'); $('#rtNo').onclick = () => closeModal('confirmModal');
  $('#rtYes').onclick = () => { const r = $('#rtReason').value; closeModal('confirmModal'); S.returned = r; logEvent('Исследование возвращено на техническое уточнение: ' + r); toast('Исследование возвращено рентгенолаборанту: ' + r, 'warn'); buildRail(); updateBadge(); };
}
function qualityRows() {
  const v = Object.values(S.vols)[0]; const n = Object.keys(S.vols).length; const iso = v.spacing.every((s) => Math.abs(s - v.spacing[0]) < 0.05);
  const rows = [['Комплектность', S.mode === 'demo' ? `${n}/4 · без DWI и перфузии` : `${n} серия`, S.mode === 'demo' ? 'warn' : ''], ['Разрешение', `${v.spacing.map((x) => x.toFixed(1)).join('×')} мм`, ''], ['Геометрия', iso ? 'изотропная' : 'анизотропная', ''], ['Ориентация', 'RAS+ (приведена)', ''], ['Артефакты', 'автооценка не выполнялась', 'warn']];
  if (S.mode === 'demo') rows.splice(3, 0, ['Предобработка', 'MNI152 · N4 · без черепа', '']);
  return rows;
}
function drawThumb(cv, vol) {
  const g = planeGeom('axial', vol); const ctx = cv.getContext('2d'); const img = ctx.createImageData(52, 52); const k = Math.min(vol.nz - 1, Math.max(0, S.cursor[2]));
  for (let y = 0; y < 52; y++) for (let x = 0; x < 52; x++) { const c = Math.floor((x / 52) * g.w), r = Math.floor((y / 52) * g.h); const v = vol.data[g.idx(c, r, k)]; const o = (y * 52 + x) * 4; img.data[o] = img.data[o + 1] = img.data[o + 2] = Math.min(255, v * 1.15); img.data[o + 3] = 255; }
  ctx.putImageData(img, 0, 0);
}
function setModality(k) {
  S.mod = k; $$('.series-item[data-mod]').forEach((i) => i.classList.toggle('is-active', i.dataset.mod === k));
  $$('#modCtl button').forEach((b) => b.classList.toggle('is-active', b.dataset.mod === k));
  vr.setModality(S.vols[k]); syncVR(); renderAll();
}
app.setModality = setModality;

/* --- центральная панель --- */
function buildCenterBar() {
  const lc = $('#layoutCtl'); const L = [['quad', 'quad', '2×2 · плоскости + 3D', '1'], ['single', 'single', 'Один вьюер', '2'], ['compare', 'compare', 'Сравнение дат', '3'], ['vr', 'cube', '3D на весь экран', '4']];
  lc.innerHTML = L.map(([k, i, t, key]) => `<button data-layout="${k}" title="${t} (${key})" aria-label="${t}">${ic(i)}</button>`).join('') + `<button data-cine title="Кино: автоматическая прокрутка срезов" aria-label="Кино">${ic('play')}<span>Кино</span></button>`;
  lc.onclick = (e) => { const b = e.target.closest('button'); if (!b) return; if (b.dataset.cine !== undefined) toggleCine(); else if (b.dataset.layout) setLayout(b.dataset.layout); };
  const mc = $('#modCtl'); mc.innerHTML = Object.keys(S.vols).map((k) => `<button data-mod="${k}" class="${k === S.mod ? 'is-active' : ''}">${esc(S.modNames[k])}</button>`).join('');
  mc.onclick = (e) => { const b = e.target.closest('button'); if (b) setModality(b.dataset.mod); };
  const pc = $('#presetCtl'); const pr = { std: ['Стандарт', 0, 225], con: ['Контраст', 40, 185], soft: ['Мягкие', 10, 140] };
  pc.innerHTML = Object.entries(pr).map(([k, v]) => `<button data-p="${k}" title="Пресет окна">${v[0]}</button>`).join('');
  pc.onclick = (e) => { const b = e.target.closest('button'); if (!b) return; const v = pr[b.dataset.p]; S.wl[S.mod] = { lo: v[1], hi: v[2] }; onWL(); renderAll(); };
}
function toggleCine() {
  S.cine = !S.cine; const b = $('#layoutCtl [data-cine]'); b.classList.toggle('is-active', S.cine); b.firstElementChild.outerHTML = ic(S.cine ? 'pause' : 'play');
  clearInterval(app._cine); if (S.cine) app._cine = setInterval(() => { const g = planeGeom('axial', S.vols[S.mod]); S.cursor = [S.cursor[0], S.cursor[1], (S.cursor[2] + 1) % g.n]; onCursor(); }, 90);
}
function onWL() { const w = S.wl[S.mod]; vr?.setWindow(Math.max(0, w.lo / 255), Math.min(1, w.hi / 255)); }
app.onWL = onWL;

/* --- раскладки --- */
export function setLayout(l) {
  S.layout = l; const grid = $('#viewerGrid');
  views.splice(0).forEach((v) => v.destroy());
  if (vrCell.parentNode) vrCell.remove();
  grid.innerHTML = ''; grid.className = 'viewer-grid l-' + (l === 'single' ? 'single' : l);
  const mk = (plane, opts = {}) => {
    const c = document.createElement('div'); c.className = 'cell'; grid.appendChild(c);
    const v = new SliceView(c, { S, app, plane, ...opts }); views.push(v); c.dataset.plane = plane;
    const t = document.createElement('div'); t.className = 'cell-tools'; t.innerHTML = `<button class="icon-btn" title="Развернуть / свернуть (двойной клик)" aria-label="Развернуть">${ic('max')}</button>`; t.firstElementChild.onclick = () => maximize(v); c.appendChild(t);
    return v;
  };
  if (l === 'quad') { mk('axial'); mk('coronal'); mk('sagittal'); grid.appendChild(vrCell); }
  else if (l === 'single') { const k = S.singleKind || 'axial'; if (k === 'vr') grid.appendChild(vrCell); else mk(k); }
  else if (l === 'compare') {
    if (S.mode !== 'demo' || !S.follow) { toast(S.mode !== 'demo' ? 'Для загруженного файла сравнение недоступно' : 'Для сравнения сначала выполните сегментацию', 'warn'); return setLayout('quad'); }
    mk('axial', { mask: 'base', label: 'Базовое · реальное МРТ' }); mk('axial', { mask: 'follow', label: 'Контроль +3 мес · синтетика' }); mk('axial', { mask: 'diff', label: 'Разность масок' });
    const c = document.createElement('div'); c.className = 'cell'; c.innerHTML = `<div class="cmp-table">${P.compareTable()}</div>`; grid.appendChild(c);
  } else if (l === 'vr') grid.appendChild(vrCell);
  $$('#layoutCtl [data-layout]').forEach((b) => b.classList.toggle('is-active', b.dataset.layout === l));
  views.forEach((v) => v.resize()); setTool(S.tool); updateVRHud(); vr.visible = true; vr.resize(); vr.dirty = true;
  if (views.length) setActive(views[0]);
}
app.setLayout = setLayout;
function maximize(v) {
  if (S.layout === 'single' && S.singleKind === v.plane) { S.singleKind = null; setLayout('quad'); } else { S.singleKind = v.plane; setLayout('single'); }
}
app.maximize = maximize;
function setActive(v) { activeView = v; views.forEach((x) => { x.active = x === v; x.canvas.parentNode.classList.toggle('is-active', x === v); }); }
app.setActive = setActive;

/* --- нижняя панель инструментов --- */
const TOOLS = [
  [['cross', 'cross', 'Перекрестие / навигация', 'C'], ['pan', 'pan', 'Панорама (Space)', 'Space'], ['zoom', 'zoom', 'Масштаб (Z)', 'Z'], ['wl', 'wl', 'Яркость / контраст (W)', 'W']],
  [['ruler', 'ruler', 'Линейка (M)', 'M'], ['roi', 'roi', 'ROI: эллипс, статистика', 'R']],
  [['brush', 'brush', 'Кисть (B)', 'B'], ['eraser', 'eraser', 'Ластик (E)', 'E'], ['cc', 'cc', 'Удалить связную область', 'X']],
];
function buildToolbar() {
  const tb = $('#toolBar');
  tb.innerHTML = TOOLS.map((g) => `<div class="tool-group">${g.map(([k, i, t, key]) => `<button class="tool" data-tool="${k}" title="${t}" aria-label="${t}">${ic(i)}<kbd>${key.length === 1 ? key : ''}</kbd></button>`).join('')}</div>` ).join('') +
    `<div class="tool-group"><button class="tool" data-a="undo" title="Отменить (Ctrl+Z)" aria-label="Отменить">${ic('undo')}</button><button class="tool" data-a="redo" title="Повторить (Ctrl+Shift+Z)" aria-label="Повторить">${ic('redo')}</button></div>` +
    `<div class="tool-group"><button class="tool" data-a="invert" title="Инверсия" aria-label="Инверсия">${ic('invert')}</button><button class="tool is-active" data-a="cross" title="Показать перекрестие" aria-label="Перекрестие">${ic('cross')}</button><button class="tool" data-a="reset" title="Сбросить вид" aria-label="Сбросить вид">${ic('reset')}</button><button class="tool" data-a="snap" title="Снимок активного вида" aria-label="Снимок">${ic('camera')}</button><button class="tool" data-a="anon" title="Скрыть персональные данные" aria-label="Скрыть данные">${ic('eyeoff')}</button></div>` +
    `<div class="readout" id="readout"></div>`;
  tb.onclick = (e) => {
    const b = e.target.closest('.tool'); if (!b) return;
    if (b.dataset.tool) return setTool(b.dataset.tool);
    const a = b.dataset.a;
    if (a === 'undo') undo(); else if (a === 'redo') redo();
    else if (a === 'invert') { S.invert = !S.invert; b.classList.toggle('is-active', S.invert); renderAll(); }
    else if (a === 'cross') { S.crosshair = !S.crosshair; b.classList.toggle('is-active', S.crosshair); renderAll(); }
    else if (a === 'reset') { views.forEach((v) => { v.zoom = 1; v.pan = [0, 0]; }); S.wl[S.mod] = { lo: 0, hi: 225 }; onWL(); vr.setView('oblique'); renderAll(); }
    else if (a === 'snap') { const v = activeView; if (!v) return; const l = document.createElement('a'); l.href = v.canvas.toDataURL('image/png'); l.download = `stoprak-${v.plane}.png`; l.click(); toast('Снимок вида сохранён', 'ok'); }
    else if (a === 'anon') setAnon(!S.anon);
  };
  const brushable = S.mode === 'demo' || S.segDone;
  $$('.tool[data-tool=brush],.tool[data-tool=eraser],.tool[data-tool=cc]', tb).forEach((b) => (b.disabled = !brushable));
  updateUndoBtns(); setTool(S.tool);
}
export function setTool(t) {
  S.tool = t; $$('#toolBar [data-tool]').forEach((b) => b.classList.toggle('is-active', b.dataset.tool === t));
  const cur = { cross: 'crosshair', pan: 'grab', zoom: 'zoom-in', wl: 'ns-resize', ruler: 'crosshair', roi: 'crosshair', brush: 'none', eraser: 'none', cc: 'pointer' }[t];
  views.forEach((v) => (v.canvas.style.cursor = cur)); P.refresh('seg');
}
app.setTool = setTool;
function updateUndoBtns() { const u = $('#toolBar [data-a=undo]'), r = $('#toolBar [data-a=redo]'); if (u) { u.disabled = !S.undo.length; r.disabled = !S.redo.length; } }

/* ------------------------------------------------------------ синхронизация */
export function renderAll() { views.forEach((v) => v.render()); }
app.renderAll = renderAll;
export function onCursor() { renderAll(); vr.setCursor(...S.cursor, S.crosshair); updateReadout(); }
app.onCursor = onCursor;
function updateReadout() {
  const ro = $('#readout'); if (!ro) return; const v = S.vols[S.mod]; const [i, j, k] = S.cursor; const val = v.data[v.idx(i, j, k)];
  const lab = S.segDone ? activeMask()[v.idx(i, j, k)] : 0; const a = S.patient?.affine;
  const mni = a ? [0, 1, 2].map((r) => Math.round(a[r][0] * i + a[r][1] * j + a[r][2] * k + a[r][3])) : null;
  ro.innerHTML = `<span>вокс. <b>${i}, ${j}, ${k}</b></span>${mni ? `<span>MNI <b>${mni.join(', ')}</b> мм</span>` : ''}<span>${esc(S.modNames[S.mod])} <b>${val}</b></span>${lab ? `<span><b style="color:${M.CLASSES[lab].color}">${M.CLASSES[lab].short}</b></span>` : ''}`;
}
app.onHover = (h) => { const tip = $('#hoverTip'); tip.hidden = true; };
function hover3D(p, e) {
  const tip = $('#hoverTip'); if (!p || !p.label || !S.metrics) { tip.hidden = true; return; }
  const c = M.CLASSES[p.label]; tip.hidden = false; tip.style.left = e.clientX + 14 + 'px'; tip.style.top = e.clientY + 14 + 'px';
  tip.innerHTML = `<b style="color:${c.color}">${c.short}</b><br>${M.nf(S.metrics.V[c.key])} см³`;
}
function activeMaskFor3D() { return S.segDone ? activeMask() : null; }
export function syncVR() {
  if (!vr) return; const fade = S.segFade;
  for (const [l, key] of [[1, 'nec'], [2, 'ede'], [3, 'enh']]) vr.setLayer(key, { show: S.layers[l].show && S.segDone, opacity: S.vrOpacity[l] * fade });
  vr.setBrain({ opacity: S.brainOpacity, thr: S.brainThr }); vr.setClip(S.clip); onWL();
  vr.setHighlight(S.hiClass); updateVRHud();
}
app.syncVR = syncVR;
function updateVRHud() {
  const sub = $('#vrHudSub'); if (sub) sub.textContent = S.segDone ? (S.tp === 1 ? 'контроль · синтетика' : 'базовое исследование') : 'сегментация не запущена';
  const lg = $('#vrLegend'); if (!lg) return; lg.innerHTML = [3, 1, 2].map((l) => `<div><i class="dot" style="--c:${M.CLASSES[l].color};${S.layers[l].show ? '' : 'opacity:.3'}"></i>${M.CLASSES[l].short}</div>`).join('') + (S.simOn ? '<div><i class="sim"></i>Модельный сценарий</div>' : '');
  lg.style.display = S.segDone ? '' : 'none';
}
function setHi(c, pulse) { S.hiClass = S.hiClass === c && !pulse ? 0 : c; vr.setHighlight(S.hiClass); renderAll(); P.markHi(); if (pulse) setTimeout(() => { S.hiClass = 0; vr.setHighlight(0); renderAll(); P.markHi(); }, 1600); }
app.setHi = setHi;

/* ------------------------------------------------------------ статус */
export function updateBadge() {
  const b = $('#stateBadge'); let cls = 'badge-mute', t = 'Данные загружены';
  if (S.returned) { cls = 'badge-err'; t = 'Возвращено на уточнение'; }
  else if (S.mode === 'user' && !S.segDone) t = 'Локальный просмотр';
  else if (S.segRunning) { cls = 'badge-info'; t = 'Анализ выполняется'; }
  else if (S.reportSigned) { cls = 'badge-ok'; t = 'Заключение подписано'; }
  else if (S.segDone && S.confirmed) { cls = 'badge-ok'; t = 'Подтверждено врачом'; }
  else if (S.segDone && S.maskEdited) { cls = 'badge-warn'; t = 'Правки не подтверждены'; }
  else if (S.segDone) { cls = 'badge-warn'; t = 'Требуется проверка врача'; }
  b.className = 'badge ' + cls; b.textContent = t;
  const rb = $('#routeBlock'); if (rb) { rb.outerHTML = routeBlock(); $('#btnReturn')?.addEventListener('click', returnStudy); }
}
app.updateBadge = updateBadge;

/* ------------------------------------------------------------ сегментация */
const STAGES = ['Подготовка данных', 'Регистрация и предобработка', 'Сегментация', 'Расчёт показателей', 'Контроль качества', 'Готово'];
app.STAGES = STAGES;
export async function runSegmentation() {
  if (S.mode !== 'demo' || S.segRunning || S.segDone) return;
  S.segRunning = true; updateBadge(); S.segProgress = { stage: 0, pct: 0 }; P.refresh('seg');
  const dur = [800, 1300, 2300, 900, 800, 350]; const total = dur.reduce((a, b) => a + b, 0); let elapsed = 0;
  const quick = document.body.classList.contains('no-anim');
  for (let s = 0; s < STAGES.length; s++) {
    S.segProgress.stage = s; P.refresh('seg', true);
    const step = quick ? 60 : dur[s]; const t0 = performance.now();
    await new Promise((res) => { const tick = () => { const f = Math.min(1, (performance.now() - t0) / step); S.segProgress.pct = Math.round(((elapsed + f * dur[s]) / total) * 100); P.progress(); f < 1 ? requestAnimationFrame(tick) : res(); }; tick(); });
    elapsed += dur[s];
  }
  S.segRunning = false; S.segDone = true; S.seg = S.segSrc.slice();
  S.baseMetrics = S.metrics = M.computeMetrics(S.seg, S.vols.t1ce, S.patient.affine, S.vols);
  computeFollow();
  S.versions = [{ v: 1, who: 'Модель SegResNet (BraTS)', when: new Date(), status: 'требует проверки', edited: false }];
  logEvent('Сегментация завершена · модель SegResNet (BraTS), маски набора ds007045');
  vr.setSeg(S.seg); S.segFade = 0; syncVR();
  const t0 = performance.now(); const fadeTick = () => { S.segFade = Math.min(1, (performance.now() - t0) / (document.body.classList.contains('no-anim') ? 1 : 900)); syncVR(); renderAll(); S.segFade < 1 ? requestAnimationFrame(fadeTick) : null; }; fadeTick();
  updateBadge(); buildToolbar(); P.refreshAll(); app.openTab('metrics');
  toast('Сегментация завершена — требуется проверка врача', 'warn');
  app.aiNote?.('Сегментация завершена. Маска ещё не подтверждена врачом.');
}
app.runSegmentation = runSegmentation;
function computeFollow() {
  const mask = M.simulateMask(S.seg, S.depthAll, S.depthEnh, null, 2.4, 3); const V = M.countVols(mask, S.vols.t1.spacing);
  S.follow = { mask, V, metrics: M.computeMetrics(mask, S.vols.t1ce, S.patient.affine, S.vols) };
}

/* ------------------------------------------------------------ правки маски */
app.beginStroke = () => { app.stroke = new Map(); };
app.endStroke = () => {
  const m = app.stroke; if (!m.size) return; const idx = new Int32Array(m.size), old = new Uint8Array(m.size), neu = new Uint8Array(m.size); let i = 0;
  for (const [id, o] of m) { idx[i] = id; old[i] = o; neu[i] = S.seg[id]; i++; }
  S.undo.push({ idx, old, neu }); if (S.undo.length > 40) S.undo.shift(); S.redo = []; app.stroke = new Map(); onMaskEdited(); updateUndoBtns();
};
function undo() { const r = S.undo.pop(); if (!r) return; for (let i = 0; i < r.idx.length; i++) S.seg[r.idx[i]] = r.old[i]; S.redo.push(r); onMaskEdited(true); updateUndoBtns(); }
function redo() { const r = S.redo.pop(); if (!r) return; for (let i = 0; i < r.idx.length; i++) S.seg[r.idx[i]] = r.neu[i]; S.undo.push(r); onMaskEdited(true); updateUndoBtns(); }
app.undo = undo; app.redo = redo;
let editTimer = null;
function onMaskEdited(quiet) {
  const first = !S.maskEdited; S.maskEdited = true; S.confirmed = false; S.reportStale = true; S.reportSigned = false;
  if (first) { S.versions.unshift({ v: 2, who: 'Правки врача (ручная коррекция)', when: new Date(), status: 'черновик', edited: true }); logEvent('Начата ручная коррекция маски → версия v2 (черновик). Исходная маска модели сохранена'); }
  clearTimeout(editTimer); editTimer = setTimeout(() => {
    S.baseMetrics = M.computeMetrics(S.seg, S.vols.t1ce, S.patient.affine, S.vols); if (S.tp === 0) S.metrics = S.baseMetrics; computeFollow(); vr.setSeg(activeMask()); P.refreshAll(); renderAll();
    toast('Изменение маски пересчитало показатели и сделало черновик заключения устаревшим', 'warn');
  }, 450);
  updateBadge(); P.refresh('seg'); renderAll();
}
app.removeComponent = (view, p) => {
  if (!S.segDone) return toast('Сначала запустите сегментацию', 'warn');
  const g = p.L.g; if (p.c < 0 || p.r < 0 || p.c >= g.w || p.r >= g.h) return; const slice = g.slice(...S.cursor); const start = g.idx(p.c, p.r, slice); const lab = S.seg[start]; if (!lab) return toast('Кликните по области маски, которую нужно удалить', 'warn');
  const v = S.vols[S.mod]; const { nx, ny, nz } = v; const st = [start]; const seen = new Set([start]); const idx = [], old = [];
  while (st.length) { const c = st.pop(); idx.push(c); old.push(S.seg[c]); const i = c % nx, j = ((c / nx) | 0) % ny, k = (c / (nx * ny)) | 0;
    for (const q of [i > 0 ? c - 1 : -1, i < nx - 1 ? c + 1 : -1, j > 0 ? c - nx : -1, j < ny - 1 ? c + nx : -1, k > 0 ? c - nx * ny : -1, k < nz - 1 ? c + nx * ny : -1]) if (q >= 0 && !seen.has(q) && S.seg[q] === lab) { seen.add(q); st.push(q); } }
  const neu = new Uint8Array(idx.length); const ix = Int32Array.from(idx); for (const c of ix) S.seg[c] = 0;
  S.undo.push({ idx: ix, old: Uint8Array.from(old), neu }); S.redo = []; onMaskEdited(); updateUndoBtns(); logEvent(`Удалена связная область «${M.CLASSES[lab].short}» (${idx.length} вокс.)`);
};
app.revertMask = async () => {
  if (!S.maskEdited) return; const ok = await confirmDialog({ title: 'Отменить изменения', html: 'Вернуть маску к результату модели (v1)? Ручные правки будут отброшены; журнал сохранится.', ok: 'Вернуть к v1', danger: true }); if (!ok) return;
  S.seg.set(S.segSrc); S.undo = []; S.redo = []; S.maskEdited = false; S.confirmed = false; S.versions = S.versions.filter((v) => !v.edited); logEvent('Правки отменены — маска возвращена к версии модели v1');
  S.baseMetrics = M.computeMetrics(S.seg, S.vols.t1ce, S.patient.affine, S.vols); if (S.tp === 0) S.metrics = S.baseMetrics; computeFollow(); vr.setSeg(activeMask()); updateBadge(); P.refreshAll(); renderAll(); updateUndoBtns();
};
app.confirmMask = async () => {
  const ok = await confirmDialog({ title: 'Подтвердить результат', html: `Вы подтверждаете, что маска проверена врачом и может использоваться для расчёта показателей и подбора терапии.<br><br><span class="fine">Будет сохранена версия ${S.maskEdited ? 'v2 (с правками)' : 'v1 (без правок)'}; исходная маска модели остаётся в журнале.</span>`, ok: 'Проверено и подтверждено врачом' });
  if (!ok) return; S.confirmed = true; const cur = S.versions.find((v) => v.edited) || S.versions[0]; cur.status = 'подтверждено'; logEvent('Результат подтверждён врачом (демо-подпись)'); updateBadge(); P.refreshAll(); toast('Сегментация подтверждена врачом', 'ok');
};

/* ------------------------------------------------------------ сравнение / отчёты / действия */
export function openCompare() {
  if (S.mode !== 'demo') return; if (!S.segDone) { toast('Для сравнения сначала запустите сегментацию', 'warn'); app.openTab('seg'); return; }
  setLayout(S.layout === 'compare' ? 'quad' : 'compare');
}
app.openCompare = openCompare;
app.doAction = (act) => {
  if (act === 'segment') app.openTab('seg');
  else if (act === 'compare') openCompare();
  else if (act === 'report') openReport();
  else if (act === 'tab-seg') app.openTab('seg');
  else if (act === 'goto-largest') { const c = S.metrics?.comps?.[0]; if (c) { S.cursor = c.c.map(Math.round); onCursor(); setHi(3, true); vr.setView('oblique'); toast('Курсор перенесён в центр крупнейшего очага', 'ok'); } }
  else if (act === '3d-enh') { S.layers[1].show = false; S.layers[2].show = false; S.layers[3].show = true; setLayout('vr'); syncVR(); renderAll(); P.refreshAll(); }
};
app.openCompare = openCompare;

/* ------------------------------------------------------------ настройки / приватность */
function setAnon(v) {
  S.anon = v; applyAnon(); $('#toolBar [data-a=anon]')?.classList.toggle('is-active', v); renderAll();
}
function applyAnon() {
  if (!S.patient) return;
  $('#ptName').textContent = S.anon ? '••••••••' : (S.mode === 'demo' ? S.patient.id : 'Локальный файл'); $('#ptMeta').style.filter = S.anon ? 'blur(5px)' : '';
}
function toggleSettings() {
  const pop = $('#settingsPop'); if (!pop.hidden) { pop.hidden = true; return; }
  const sw = (id, label, on) => `<div class="row"><span>${label}</span><button class="switch" id="${id}" role="switch" aria-checked="${on}"></button></div>`;
  pop.innerHTML = `<div class="panel-title" style="margin-bottom:12px">Настройки просмотра</div>${sw('swAnim', 'Отключить анимацию', document.body.classList.contains('no-anim'))}${sw('swCvd', 'Режим для дальтонизма', document.body.classList.contains('cvd'))}${sw('swAnon', 'Скрыть персональные данные', S.anon)}${sw('swRail', 'Свернуть левую панель', $('#wsBody').classList.contains('rail-off'))}${sw('swSide', 'Свернуть правую панель', $('#wsBody').classList.contains('side-off'))}
    <div class="field" style="margin-top:6px"><label>Яркость интерфейса</label><input type="range" id="rgBright" min="70" max="130" value="${Math.round((parseFloat(document.body.style.getPropertyValue('--bright')) || 1) * 100)}"></div>
    <button class="btn btn-sm btn-ghost btn-block" id="btnHelp">Горячие клавиши (?)</button>
    <p class="fine" style="margin-top:10px">Демо-режим: условия просмотра не проверялись — экран не аттестован как диагностический монитор.</p>`;
  pop.hidden = false;
  const bind = (id, fn) => { const b = $(id, pop); b.onclick = () => { const v = b.getAttribute('aria-checked') !== 'true'; b.setAttribute('aria-checked', v); fn(v); }; };
  bind('#swAnim', (v) => document.body.classList.toggle('no-anim', v));
  bind('#swCvd', (v) => setCvd(v)); bind('#swAnon', (v) => setAnon(v));
  bind('#swRail', (v) => { $('#wsBody').classList.toggle('rail-off', v); setTimeout(() => views.forEach((x) => x.resize()), 300); });
  bind('#swSide', (v) => { $('#wsBody').classList.toggle('side-off', v); setTimeout(() => views.forEach((x) => x.resize()), 300); });
  const rg = $('#rgBright', pop); rangeFill(rg); rg.oninput = () => { rangeFill(rg); document.body.style.filter = `brightness(${rg.value / 100})`; };
  $('#btnHelp', pop).onclick = () => { pop.hidden = true; showHelp(); };
}
function setCvd(v) {
  document.body.classList.toggle('cvd', v);
  const pal = v ? { 1: [0x78, 0x5E, 0xF0], 2: [0x64, 0x8F, 0xFF], 3: [0xFF, 0xB0, 0x00] } : { 1: [0xF5, 0xA6, 0x23], 2: [0x38, 0xBD, 0xF8], 3: [0xFF, 0x5B, 0x61] };
  for (const l of [1, 2, 3]) { RGB[l][0] = pal[l][0]; RGB[l][1] = pal[l][1]; RGB[l][2] = pal[l][2]; M.CLASSES[l].color = '#' + pal[l].map((x) => x.toString(16).padStart(2, '0')).join(''); }
  const set = (u, l) => u.value.setRGB(pal[l][0] / 255, pal[l][1] / 255, pal[l][2] / 255);
  for (const inst of [vr, hero]) if (inst) { set(inst.U.uColNec, 1); set(inst.U.uColEde, 2); set(inst.U.uColEnh, 3); inst.dirty = true; }
  document.documentElement.style.setProperty('--tumor', M.CLASSES[3].color); document.documentElement.style.setProperty('--necro', M.CLASSES[1].color); document.documentElement.style.setProperty('--edema', M.CLASSES[2].color);
  renderAll(); P.refreshAll(); updateVRHud();
}
document.addEventListener('mousedown', (e) => { const pop = $('#settingsPop'); if (!pop.hidden && !pop.contains(e.target) && e.target.id !== 'btnSettings' && !$('#btnSettings').contains(e.target)) pop.hidden = true; const ex = $('#exportPop'); if (ex && !ex.contains(e.target)) ex.remove(); });

/* ------------------------------------------------------------ экспорт */
function exportMenu(btn) {
  $('#exportPop')?.remove(); const pop = document.createElement('div'); pop.id = 'exportPop'; pop.className = 'settings-pop'; pop.style.right = '14px';
  const items = [['png', 'Снимок 3D (PNG)', true], ['txt', 'Заключение (TXT)', S.segDone && S.mode === 'demo'], ['json', 'Показатели (JSON)', S.segDone], ['nii', 'Маска (NIfTI, .nii.gz)', S.segDone]];
  pop.innerHTML = `<div class="panel-title" style="margin-bottom:10px">Экспорт</div>${items.map(([k, t, on]) => `<button class="btn btn-sm btn-block" style="margin-bottom:6px;justify-content:flex-start" data-ex="${k}" ${on ? '' : 'disabled'}>${ic('download', 14)} ${t}</button>`).join('')}<p class="fine">Файлы не содержат персональных данных; сохраняются локально на ваше устройство — получатель: вы.</p>`;
  $('#ws').appendChild(pop);
  pop.onclick = async (e) => { const b = e.target.closest('[data-ex]'); if (!b) return; pop.remove(); await doExport(b.dataset.ex); };
}
async function doExport(k) {
  const dl = (blob, name) => { const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 4000); toast('Файл сохранён: ' + name, 'ok'); };
  if (k === 'png') { const a = document.createElement('a'); a.href = vr.screenshot(); a.download = 'stoprak-3d.png'; a.click(); }
  else if (k === 'txt') dl(new Blob([S.reportText || (await import('./model.js')).buildReport(S)], { type: 'text/plain;charset=utf-8' }), 'zaklyuchenie-chernovik.txt');
  else if (k === 'json') dl(new Blob([JSON.stringify({ case: S.patient.id, status: S.confirmed ? 'confirmed' : 'draft', maskEdited: S.maskEdited, volumes_cm3: S.metrics.V, shares: S.metrics.share, max_diameter_mm: S.metrics.diam, foci: S.metrics.foci, localisation: S.metrics.loc?.text, model: 'SegResNet (BraTS) · masks from OpenNeuro ds007045', note: 'Демонстрационные данные. Не является клиническим документом.' }, null, 2)], { type: 'application/json' }), 'stoprak-metrics.json');
  else if (k === 'nii') dl(await maskToNifti(), `stoprak-mask-${S.maskEdited ? 'v2' : 'v1'}.nii.gz`);
}
async function maskToNifti() {
  const v = S.vols.t1, n = v.size; const buf = new ArrayBuffer(352 + n); const dv = new DataView(buf); const u = new Uint8Array(buf);
  dv.setInt32(0, 348, true); dv.setInt16(40, 3, true); dv.setInt16(42, v.nx, true); dv.setInt16(44, v.ny, true); dv.setInt16(46, v.nz, true); for (let i = 4; i < 8; i++) dv.setInt16(40 + 2 * i, 1, true);
  dv.setInt16(70, 2, true); dv.setInt16(72, 8, true); dv.setFloat32(76, 1, true); v.spacing.forEach((s, i) => dv.setFloat32(80 + 4 * i, s, true)); dv.setFloat32(108, 352, true);
  dv.setInt16(252, 0, true); const A = S.patient.affine || [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]; dv.setInt16(254, 1, true); for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) dv.setFloat32(280 + 16 * r + 4 * c, A[r][c], true);
  u.set([0x6e, 0x2b, 0x31, 0], 344); u.set(S.seg, 352);
  const cs = new Blob([buf]).stream().pipeThrough(new CompressionStream('gzip')); return new Response(cs).blob();
}

/* ------------------------------------------------------------ клавиатура и палитра команд */
function typing(e) { const t = e.target; return t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable); }
function initKeys() {
  window.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); showCmdk(); return; }
    if (e.key === 'Escape') { $$('.modal').forEach((m) => { if (!m.hidden) { m._cancel?.(); m.hidden = true; } }); $('#cmdk').hidden = true; $('#settingsPop').hidden = true; $('#exportPop')?.remove(); return; }
    if (!$('#screen-ws').classList.contains('is-active') || typing(e)) return;
    if (e.key === ' ') { app.spaceDown = true; e.preventDefault(); return; }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); e.shiftKey ? redo() : undo(); return; }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key.toLowerCase();
    const map = { w: 'wl', z: 'zoom', m: 'ruler', b: 'brush', e: 'eraser', c: 'cross', r: 'roi', x: 'cc' };
    if (map[k]) setTool(map[k]);
    else if (k >= '1' && k <= '4') setLayout(['quad', 'single', 'compare', 'vr'][+k - 1]);
    else if (k === 'f') { document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen?.(); }
    else if (k === '[') { S.brush.r = Math.max(1, S.brush.r - 1); P.refresh('seg'); } else if (k === ']') { S.brush.r = Math.min(30, S.brush.r + 1); P.refresh('seg'); }
    else if (k === '/') { e.preventDefault(); showCmdk(); } else if (k === '?') showHelp();
  });
  window.addEventListener('keyup', (e) => { if (e.key === ' ') app.spaceDown = false; });
}
function showHelp() {
  $('#helpBody').innerHTML = [['Space', 'панорамирование (удерживать)'], ['W', 'окно / уровень'], ['Z', 'масштаб'], ['M', 'линейка'], ['R', 'ROI'], ['B / E', 'кисть / ластик'], ['[ / ]', 'размер кисти'], ['X', 'удалить связную область'], ['Ctrl/⌘ + Z', 'отмена (+Shift — повтор)'], ['1 – 4', 'раскладка: 2×2, один, сравнение, 3D'], ['F', 'полный экран'], ['Ctrl/⌘ + K  или  /', 'палитра команд'], ['колесо', 'прокрутка срезов; Ctrl + колесо — масштаб'], ['двойной клик', 'развернуть вьюер'], ['?', 'эта справка']].map(([a, b]) => `<kbd>${a}</kbd><span>${b}</span>`).join('');
  openModal('helpModal');
}
let cmdSel = 0;
function commands() {
  const c = [
    ['Открыть демонстрационный случай', () => openDemo(DEMO_CASES[0])], ['Загрузить своё МРТ', () => app.openUpload()], ['На главную', () => goTo('landing')],
    ['Запустить сегментацию', () => { if (S.mode === 'demo') { app.openTab('seg'); runSegmentation(); } }], ['Раскладка: 2×2', () => setLayout('quad')], ['Раскладка: 3D на весь экран', () => setLayout('vr')], ['Сравнить исследования', () => openCompare()],
    ['Открыть заключение', () => openReport()], ['Открыть ИИ-помощника', () => app.toggleAI(true)], ['Вкладка: Слои', () => app.openTab('layers')], ['Вкладка: Показатели', () => app.openTab('metrics')], ['Вкладка: Сегментация', () => app.openTab('seg')], ['Вкладка: Лечение', () => app.openTab('treat')],
    ['Последовательность: T1-Gd', () => S.vols.t1ce && setModality('t1ce')], ['Последовательность: FLAIR', () => S.vols.flair && setModality('flair')], ['Горячие клавиши', showHelp],
  ];
  return c;
}
function showCmdk() { const el = $('#cmdk'); el.hidden = false; const inp = $('#cmdkInput'); inp.value = ''; inp.focus(); cmdSel = 0; renderCmd(''); }
function renderCmd(q) {
  const list = commands().filter(([n]) => n.toLowerCase().includes(q.toLowerCase())); cmdSel = Math.min(cmdSel, Math.max(0, list.length - 1));
  $('#cmdkList').innerHTML = list.map(([n], i) => `<div class="cmdk-item ${i === cmdSel ? 'is-sel' : ''}" data-i="${i}">${esc(n)}</div>`).join('') || '<div class="cmdk-item">Ничего не найдено</div>';
  $$('#cmdkList .cmdk-item[data-i]').forEach((it) => (it.onclick = () => { $('#cmdk').hidden = true; list[+it.dataset.i][1](); })); $('#cmdkList')._list = list;
}
function initCmdk() {
  const inp = $('#cmdkInput'); inp.oninput = () => { cmdSel = 0; renderCmd(inp.value); };
  inp.onkeydown = (e) => { const l = $('#cmdkList')._list || []; if (e.key === 'ArrowDown') { cmdSel = Math.min(l.length - 1, cmdSel + 1); renderCmd(inp.value); e.preventDefault(); } else if (e.key === 'ArrowUp') { cmdSel = Math.max(0, cmdSel - 1); renderCmd(inp.value); e.preventDefault(); } else if (e.key === 'Enter' && l[cmdSel]) { $('#cmdk').hidden = true; l[cmdSel][1](); } };
  $('#cmdk').addEventListener('mousedown', (e) => { if (e.target.id === 'cmdk') $('#cmdk').hidden = true; });
}

/* ------------------------------------------------------------ экспорт для панелей */
app.updateVRHud = updateVRHud; app.vr = () => vr; app.views = () => views; app.activeMask = activeMask;
app.updateUndoBtns = updateUndoBtns; app.setAnon = setAnon; app.cases = DEMO_CASES;

landing();
window.__stoprak = { S, app };
