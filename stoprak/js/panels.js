// Правая панель: Слои · Показатели · Сегментация · Лечение
import { S, app } from './state.js';
import * as M from './model.js';
import { ic, $, $$, esc, md, toast, rangeFill, confirmDialog } from './ui.js';

const KEYS = ['layers', 'metrics', 'seg', 'treat'];
const dirty = new Set(KEYS);
const pane = (k) => $(`#pane${k[0].toUpperCase()}${k.slice(1)}`);
const R = { layers: renderLayers, metrics: renderMetrics, seg: renderSeg, treat: renderTreat };
const CL = M.CLASSES; const ORDER = [3, 1, 2];
const key = (l) => CL[l].key;

export function refresh(k, soft) { const p = pane(k); if (!p) return; if (p.hidden) { dirty.add(k); return; } const st = p.parentElement.scrollTop; R[k](p); p.parentElement.scrollTop = st; bindPane(p); }
export function refreshAll() { KEYS.forEach((k) => refresh(k)); }
export function markHi() { $$('.layer,.mcard').forEach((el) => el.classList.toggle('is-hi', +el.dataset.l === S.hiClass && S.hiClass > 0)); }
export function progress() { const b = $('#segBar'); if (b) b.style.width = (S.segProgress?.pct || 0) + '%'; const t = $('#segPct'); if (t) t.textContent = (S.segProgress?.pct || 0) + ' %'; }
const noAnalysis = () => S.mode === 'user';
const userNote = `<div class="hint warn">${ic('info')}<div><b>Просмотр доступен; автоматический анализ для этого файла пока не подключен.</b><br>Подключённая модель сегментации обучена на предобработанных МРТ глиобластомы (T1, T1-Gd, T2, FLAIR). Персональную сегментацию и расчёты лечения система не имитирует.</div></div>`;

const bound = new WeakSet();
function bindPane(p) {
  if (bound.has(p)) return; bound.add(p);
  p.addEventListener('click', (e) => { const b = e.target.closest('[data-act]'); if (!b || b.disabled || b.tagName === 'INPUT' && b.type === 'range') return; act(b.dataset.act, b, e); });
  p.addEventListener('input', (e) => { const b = e.target.closest('[data-in]'); if (b) input(b.dataset.in, b, e); });
  p.addEventListener('change', (e) => { const b = e.target.closest('[data-ch]'); if (b) change(b.dataset.ch, b, e); });
}

/* =========================================================== СЛОИ */
function volOf(l) { const m = S.metrics; return m ? m.V[key(l)] : 0; }
function deltaOf(l) { if (S.tp !== 1 || !S.baseMetrics || !S.metrics) return null; return M.pct(S.metrics.V[key(l)], S.baseMetrics.V[key(l)]); }
const dCls = (d) => (d > 0.5 ? 'up' : d < -0.5 ? 'down' : 'flat');
const dTxt = (d) => `${d > 0.05 ? '▲' : d < -0.05 ? '▼' : '■'} ${M.sgn(d)} %`;
const swClass = { 3: 'p-contour', 1: 'p-hatch', 2: '' };
function renderLayers(p) {
  let h = '';
  if (noAnalysis() && !S.segDone) h += userNote;
  else if (!S.segDone) h += `<div class="hint">${ic('info')}<div>Слои опухоли появятся после сегментации. Пока доступен просмотр серий в 2D и 3D.</div></div><button class="btn btn-primary btn-block" data-act="goseg" style="margin-bottom:14px">${ic('play', 14)} Перейти к сегментации</button>`;
  else if (S.tp === 1) h += `<div class="hint warn">${ic('warn')}<div>Показана <b>синтетическая</b> контрольная точка (демо-модель). Это не результат реального повторного МРТ.</div></div>`;
  h += `<div class="sec-title"><span class="panel-title">Опухолевые зоны</span>${S.segDone ? `<span class="badge ${S.confirmed ? 'badge-ok' : 'badge-warn'}">${S.confirmed ? 'подтверждено' : 'не подтверждено'}</span>` : ''}</div>`;
  for (const l of ORDER) {
    const L = S.layers[l]; const on = S.segDone; const d = deltaOf(l);
    h += `<div class="layer ${S.hiClass === l ? 'is-hi' : ''}" data-l="${l}" style="--c:${CL[l].color}">
      <div class="layer-top"><input type="checkbox" class="cb" data-ch="show" data-l="${l}" ${L.show ? 'checked' : ''} ${on ? '' : 'disabled'} aria-label="Показать: ${esc(CL[l].short)}" style="--c:${CL[l].color}">
      <span class="layer-swatch ${swClass[l]}"></span><div class="layer-name" data-act="hi" data-l="${l}" style="cursor:pointer">${CL[l].short}<small>${esc(CL[l].name)}</small></div>
      <div style="text-align:right"><div class="layer-vol">${on ? M.nf(volOf(l)) + ' см³' : '—'}</div>${d !== null ? `<div class="mcard d ${dCls(d)}" style="font-size:10.5px">${dTxt(d)}</div>` : ''}</div></div>
      <div class="layer-bot"><input type="range" min="0.1" max="1" step="0.01" value="${L.opacity}" data-in="op" data-l="${l}" ${on ? '' : 'disabled'} style="--c:${CL[l].color}" aria-label="Прозрачность">
      <div class="mini-seg" role="group">${[['fill', 'заливка'], ['contour', 'контур'], ['both', 'оба']].map(([k, t]) => `<button data-act="style" data-l="${l}" data-s="${k}" class="${L.style === k ? 'is-active' : ''}" ${on ? '' : 'disabled'}>${t}</button>`).join('')}</div></div>
      <div class="row sp" style="margin-top:8px"><span class="fine">${on ? (S.confirmed ? 'проверено врачом' : 'создано моделью · требует проверки') : 'ожидает сегментации'}</span><button class="btn btn-sm btn-ghost" data-act="edit" data-l="${l}" ${on && S.mode === 'demo' && S.tp === 0 ? '' : 'disabled'}>Редактировать</button></div></div>`;
  }
  h += `<p class="fine" style="margin:-2px 0 4px">Цвет дублируется формой: активная — контур, некроз — штриховка, отёк — заливка.</p>`;
  if (S.simInfo) h += `<div class="row sp layer" style="--c:#fff"><div><b style="font-size:13px">Модельный сценарий</b><div class="fine">пунктир и штриховка · не факт</div></div><button class="switch" data-act="simtoggle" role="switch" aria-checked="${S.simOn}" aria-label="Показать модельный сценарий"></button></div>`;
  h += `</div>`;
  h += `<div class="sec" style="margin:14px -14px 0;border-top:1px solid var(--line);border-bottom:none"><div class="sec-title"><span class="panel-title">3D-визуализация</span></div>
    <div class="field"><label>Прозрачность мозга <span class="range-out" id="vBo">${Math.round(S.brainOpacity * 100)}%</span></label><input type="range" min="0.04" max="1" step="0.01" value="${S.brainOpacity}" data-in="bo" aria-label="Прозрачность мозга"></div>
    <div class="field"><label>Порог поверхности <span class="range-out" id="vThr">${S.brainThr.toFixed(2)}</span></label><input type="range" min="0.06" max="0.6" step="0.01" value="${S.brainThr}" data-in="thr" aria-label="Порог поверхности мозга"></div>
    <div class="row sp" style="margin-bottom:8px"><span style="font-size:13px;font-weight:600">Секущая плоскость</span><button class="switch" data-act="clip" role="switch" aria-checked="${S.clip.on}" aria-label="Секущая плоскость"></button></div>
    <div class="seg-ctl" style="margin-bottom:8px;width:100%"><button data-act="clipaxis" data-a="x" class="${S.clip.axis === 'x' ? 'is-active' : ''}" style="flex:1;justify-content:center">Сагиттальная</button><button data-act="clipaxis" data-a="y" class="${S.clip.axis === 'y' ? 'is-active' : ''}" style="flex:1;justify-content:center">Корональная</button><button data-act="clipaxis" data-a="z" class="${S.clip.axis === 'z' ? 'is-active' : ''}" style="flex:1;justify-content:center">Аксиальная</button></div>
    <div class="row"><input type="range" min="0.02" max="0.98" step="0.005" value="${S.clip.pos}" data-in="clip" ${S.clip.on ? '' : 'disabled'} aria-label="Положение плоскости"><button class="btn btn-sm btn-ghost" data-act="clipflip" ${S.clip.on ? '' : 'disabled'} title="Убрать другую сторону">⇄</button></div>
    <p class="fine" style="margin-top:8px">Разрез показывает настоящий срез выбранной серии. Клик по 3D — переход к точке в 2D-срезах. Наведение показывает зону и объём.</p></div>`;
  p.innerHTML = h; $$('input[type=range]', p).forEach(rangeFill);
}

/* =========================================================== ПОКАЗАТЕЛИ */
export function compareTable() {
  const a = S.baseMetrics, b = S.follow?.metrics; if (!a || !b) return '<p class="fine">Нет данных для сравнения.</p>';
  const rows = [['Вся опухолевая область', 'whole'], ['Активная зона', 'enh'], ['Некротическая зона', 'nec'], ['FLAIR-гиперинтенсивная зона', 'ede']];
  return `<div class="row sp" style="margin-bottom:12px"><span class="panel-title">Изменение показателей</span><span class="demo-tag">СИНТЕТИЧЕСКАЯ ДЕМО-ДИНАМИКА</span></div>
   <table class="mtable"><thead><tr><th>Показатель</th><th>Базовое</th><th>Контроль*</th><th>Δ</th></tr></thead><tbody>${rows.map(([n, k]) => { const d = M.pct(b.V[k], a.V[k]); return `<tr><td>${n}</td><td>${M.nf(a.V[k])} см³</td><td>${M.nf(b.V[k])} см³</td><td class="${d > 0.5 ? 'up' : d < -0.5 ? 'down' : ''}">${M.sgn(d)}%</td></tr>`; }).join('')}</tbody></table>
   <div class="hint warn" style="margin-top:14px">${ic('warn')}<div>Результат требует проверки врача. Изменения могут отражать как ответ опухоли, так и посттерапевтические эффекты. *Контрольная точка построена демонстрационной моделью по маске базового исследования.</div></div>
   <div class="panel-title" style="margin:14px 0 8px">Проверка по критериям</div>
   <div class="check"><span class="st miss">?</span><div>Оценка по RANO ограничена<small>Нет данных о кортикостероидах и клиническом статусе; версия критериев: RANO 2010 (Wen et al.)</small></div></div>
   <div class="check"><span class="st ok">✓</span><div>Активная зона: ${M.sgn(M.pct(b.V.enh, a.V.enh))} % (${M.nf(a.V.enh)} → ${M.nf(b.V.enh)} см³)<small>Расчёт по объёмам маски; ортогональные размеры — на вкладке «Показатели»</small></div></div>
   <div class="check"><span class="st miss">?</span><div>Возможны посттерапевтические изменения<small>Требуется клиническая корреляция</small></div></div>
   <div class="row" style="margin-top:12px;gap:14px"><span class="row"><i class="dot" style="--c:#FF5B61"></i><span class="fine">рост</span></span><span class="row"><i class="dot" style="--c:#22C997"></i><span class="fine">уменьшение</span></span><span class="row"><i class="dot" style="--c:#A0AAB6"></i><span class="fine">без изменений</span></span></div>`;
}
function chartSvg() {
  const pts = []; const a = S.baseMetrics?.V.whole; if (a == null) return '';
  pts.push({ x: 0, y: a, t: 'Базовое', real: true });
  if (S.follow) pts.push({ x: 3, y: S.follow.metrics.V.whole, t: 'Контроль*', real: false });
  if (S.simInfo) pts.push({ x: S.simInfo.params.horizon, y: S.simInfo.V.mid.whole, t: 'Модель', real: false, sim: true, lo: S.simInfo.V.lo.whole, hi: S.simInfo.V.hi.whole });
  const W = 300, H = 110, pl = 30, pr = 12, pt = 12, pb = 22; const xmax = Math.max(6, ...pts.map((q) => q.x)); const ys = pts.flatMap((q) => [q.y, q.lo ?? q.y, q.hi ?? q.y]); const y0 = 0, y1 = Math.max(...ys) * 1.15;
  const X = (v) => pl + (v / xmax) * (W - pl - pr), Y = (v) => pt + (1 - (v - y0) / (y1 - y0)) * (H - pt - pb);
  let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Изменение объёма опухоли по времени"><g stroke="#26364A" stroke-width="1">${[0, 0.5, 1].map((f) => `<line x1="${pl}" x2="${W - pr}" y1="${Y(y1 * f)}" y2="${Y(y1 * f)}"/>`).join('')}</g><g fill="#6C7E90" font-size="9" font-family="JetBrains Mono,monospace">${[0, 0.5, 1].map((f) => `<text x="${pl - 4}" y="${Y(y1 * f) + 3}" text-anchor="end">${Math.round(y1 * f)}</text>`).join('')}${[0, 3, 6].filter((m) => m <= xmax).map((m) => `<text x="${X(m)}" y="${H - 6}" text-anchor="middle">${m} мес</text>`).join('')}</g>`;
  const real = pts.filter((q) => !q.sim); s += `<polyline fill="none" stroke="#19C3B1" stroke-width="2" points="${real.map((q) => `${X(q.x)},${Y(q.y)}`).join(' ')}"/>`;
  for (const q of pts) {
    if (q.sim) s += `<line x1="${X(q.x)}" x2="${X(q.x)}" y1="${Y(q.hi)}" y2="${Y(q.lo)}" stroke="#F5A623" stroke-width="5" stroke-linecap="round" opacity=".35"/><line x1="${X(0)}" y1="${Y(a)}" x2="${X(q.x)}" y2="${Y(q.y)}" stroke="#F5A623" stroke-width="1.5" stroke-dasharray="4 3"/>`;
    s += `<circle cx="${X(q.x)}" cy="${Y(q.y)}" r="4.5" fill="${q.real ? '#19C3B1' : '#0B1117'}" stroke="${q.sim ? '#F5A623' : '#19C3B1'}" stroke-width="2"><title>${q.t}: ${M.nf(q.y)} см³</title></circle>`;
  }
  return s + '</svg><div class="fine">● реальное · ○ синтетика/модель (пунктир и диапазон — модельный сценарий)</div>';
}
function renderMetrics(p) {
  const m = S.metrics; let h = '';
  if (!S.segDone) { p.innerHTML = (noAnalysis() ? userNote : `<div class="hint">${ic('info')}<div>Показатели рассчитываются по маске сегментации. Запустите сегментацию — объёмы, размеры и динамика появятся здесь.</div></div><button class="btn btn-primary btn-block" data-act="goseg">${ic('play', 14)} Перейти к сегментации</button>`); return; }
  h += `<div class="hint ${S.confirmed ? 'ok' : 'warn'}">${ic(S.confirmed ? 'check' : 'warn')}<div>${S.confirmed ? '<b>Подтверждено врачом.</b> Показатели рассчитаны по подтверждённой маске.' : '<b>Проверьте маску перед использованием количественных показателей.</b> Результат создан моделью и не заменяет клиническое решение.'}</div></div>`;
  const cards = [['whole', 'Вся опухолевая область', 0, '#E8EEF4'], ['enh', 'Активная зона', 3, CL[3].color], ['nec', 'Некротическая зона', 1, CL[1].color], ['ede', 'FLAIR-зона (отёк)', 2, CL[2].color]];
  h += `<div class="metric-cards">${cards.map(([k, n, l, c]) => { const v = m.V[k]; const d = S.tp === 1 && S.baseMetrics ? M.pct(v, S.baseMetrics.V[k]) : null; const share = k === 'whole' ? '' : ` · ${M.nf(m.share[k] * 100, 0)} %`;
    return `<button class="mcard ${S.hiClass === l && l ? 'is-hi' : ''}" style="--c:${c}" data-act="hi" data-l="${l}" title="Нажмите, чтобы подсветить область в 2D и 3D"><div class="k">${n}</div><div class="v">${M.nf(v)}<small>см³${share}</small></div><div class="d ${d === null ? 'flat' : dCls(d)}">${d === null ? 'нет предыдущего' : dTxt(d) + ' к базовому'}</div></button>`; }).join('')}</div>`;
  const ed = S.maskEdited ? '<span class="src-tag ed">после правок</span>' : '<span class="src-tag">авто</span>';
  const rows = [
    ['Макс. диаметр опухоли', `${M.nf(m.diam, 0)} мм`, ed],
    ['Активная зона, срез с макс. площадью', m.bidim ? `${M.nf(m.bidim.d1, 0)} × ${M.nf(m.bidim.d2, 0)} мм` : '—', ed],
    ['Число очагов (> 0,1 см³)', m.foci, ed],
    ['Локализация', m.loc ? `${m.loc.side} · ${m.loc.lobe}` : '—', '<span class="src-tag">по MNI, ориент.</span>'],
    ['Интенсивность T1-Gd: активная / фон', m.enhRatio ? `${M.nf(m.enhRatio, 2)} ×` : '—', '<span class="src-tag">усл. ед.</span>'],
    ['Критические структуры', 'не рассчитано', '<span class="src-tag">атлас не подключен</span>'],
    ['ADC / перфузия', 'нет серий', '<span class="src-tag">недостаточно данных</span>'],
    ['Качество входных данных', '4/4 серий, MNI, без черепа', '<span class="src-tag">артефакты не оценены</span>'],
  ];
  h += `<div class="sec-title" style="margin-top:6px"><span class="panel-title">Характеристики</span></div>${rows.map(([n, v, t]) => `<div class="mrow"><span class="n">${n}</span><span class="val">${v}</span>${t}</div>`).join('')}`;
  h += `<div class="sec-title" style="margin-top:18px"><span class="panel-title">Динамика объёма, см³</span></div>${chartSvg()}`;
  if (S.follow) h += `<div style="margin-top:14px">${compareTable()}</div><button class="btn btn-block" style="margin-top:12px" data-act="cmp">${ic('compare', 15)} Сравнить исследования</button>`;
  h += `<button class="btn btn-block btn-ghost" style="margin-top:8px" data-act="goreport">${ic('doc', 15)} Создать черновик заключения</button>`;
  p.innerHTML = h;
}

/* =========================================================== СЕГМЕНТАЦИЯ */
function renderSeg(p) {
  let h = '';
  if (noAnalysis() && !S.segDone) { p.innerHTML = userNote + `<div class="sec-title"><span class="panel-title">Что доступно</span></div><div class="check"><span class="st ok">✓</span><div>Просмотр в 2D и 3D<small>MPR, окно/уровень, линейка, ROI, разрезы</small></div></div><div class="check"><span class="st bad">×</span><div>Автоматическая сегментация<small>модель не подключена для этого типа данных</small></div></div><div class="check"><span class="st bad">×</span><div>Моделирование и подбор лечения<small>требуют подтверждённой сегментации и клинических данных</small></div></div>`; return; }
  const done = S.segDone;
  h += `<div class="sec-title"><span class="panel-title">Модель</span><span class="badge badge-ai">не медицинское изделие</span></div>
    <div class="kv"><span>Название</span><b>SegResNet · BraTS-разметка</b></div><div class="kv"><span>Версия / источник</span><b>маски набора ds007045 v2.0.1</b></div><div class="kv"><span>Область применения</span><b>глиома/глиобластома, МРТ головного мозга</b></div>
    <div class="kv"><span>Входные серии</span><b>T1 · T1-Gd · T2 · FLAIR <span class="down">✓</span></b></div><div class="kv"><span>Отсутствует</span><b class="flat">DWI/ADC, перфузия — не требуются</b></div>`;
  if (!done) {
    h += `<div class="field" style="margin-top:10px"><label>Режим</label><select class="sel" ${S.segRunning ? 'disabled' : ''}><option>Стандартный</option><option>Быстрый</option><option>Исследовательский</option></select></div>
    <div class="hint">${ic('info')}<div>Ориентировочное время в демо — около 6 секунд. В демо используются <b>предрассчитанные маски</b> датасета: они проверены нейрорадиологами авторов набора, но в приложении остаются «требуют проверки врача».</div></div>`;
    if (S.segRunning) {
      h += `<div class="row sp"><b style="font-size:13px">Анализ выполняется</b><span class="range-out" id="segPct">${S.segProgress?.pct || 0} %</span></div><div class="pbar"><i id="segBar" style="width:${S.segProgress?.pct || 0}%"></i></div>
      <div class="stagelist">${app.STAGES.map((s, i) => `<div class="stage ${i < S.segProgress.stage ? 'is-done' : i === S.segProgress.stage ? 'is-run' : ''}"><i></i>${s}</div>`).join('')}</div><p class="fine">Обработка идёт в фоне — можно продолжать просмотр серий.</p>`;
    } else h += `<button class="btn btn-primary btn-block btn-lg" data-act="runseg">${ic('play', 15)} Запустить сегментацию</button>`;
  } else {
    h += `<div class="hint ${S.confirmed ? 'ok' : 'warn'}" style="margin-top:12px">${ic(S.confirmed ? 'check' : 'warn')}<div>${S.confirmed ? '<b>Подтверждено врачом.</b> Правки после подтверждения создадут новую версию.' : '<b>Требуется проверка врача.</b> Проверьте маску в срезах, при необходимости скорректируйте кистью и подтвердите результат.'}</div></div>
    <div class="sec-title" style="margin-top:14px"><span class="panel-title">Ручная коррекция</span></div>
    <div class="seg-ctl" style="width:100%;margin-bottom:10px"><button style="flex:1;justify-content:center" data-act="tool" data-t="brush" class="${S.tool === 'brush' ? 'is-active' : ''}">${ic('brush', 14)} Кисть</button><button style="flex:1;justify-content:center" data-act="tool" data-t="eraser" class="${S.tool === 'eraser' ? 'is-active' : ''}">${ic('eraser', 14)} Ластик</button><button style="flex:1;justify-content:center" data-act="tool" data-t="cc" class="${S.tool === 'cc' ? 'is-active' : ''}" title="Удалить связную область">${ic('cc', 14)} Область</button></div>
    <div class="field"><label>Размер кисти <span class="range-out" id="vBr">${S.brush.r} мм</span></label><input type="range" min="1" max="30" step="1" value="${S.brush.r}" data-in="br" aria-label="Размер кисти"></div>
    <div class="field"><label>Класс для рисования</label><div class="chips">${[3, 1, 2].map((l) => `<button class="chip ${S.brush.label === l ? 'is-active' : ''}" data-act="blabel" data-l="${l}" style="${S.brush.label === l ? '' : ''}"><i class="dot" style="--c:${CL[l].color};margin-right:6px"></i>${CL[l].short}</button>`).join('')}</div></div>
    <div class="row" style="gap:6px;margin-bottom:10px"><button class="btn btn-sm" data-act="undo" ${S.undo.length ? '' : 'disabled'}>${ic('undo', 14)} Отменить</button><button class="btn btn-sm" data-act="redo" ${S.redo.length ? '' : 'disabled'}>${ic('redo', 14)} Повторить</button>
    <button class="btn btn-sm ${S.showOrig ? 'btn-blue' : ''}" data-act="orig" ${S.maskEdited ? '' : 'disabled'} title="Показать исходную маску модели до правок">${S.showOrig ? 'Показана: до правок' : 'До / после'}</button></div>
    <div class="row" style="gap:8px;flex-wrap:wrap"><button class="btn btn-primary" style="flex:1" data-act="confirm" ${S.confirmed ? 'disabled' : ''}>${ic('check', 15)} Подтвердить результат</button><button class="btn btn-ghost" data-act="revert" ${S.maskEdited ? '' : 'disabled'}>Отменить изменения</button></div>
    <p class="fine" style="margin-top:8px">Интерполяция между срезами и пороговое выделение — в плане развития; правки сохраняются как новая версия без потери исходной маски.</p>
    <div class="sec-title" style="margin-top:18px"><span class="panel-title">Версии маски</span></div><div class="vlist">${S.versions.map((v) => `<div class="vitem"><div><b>v${v.v} · ${esc(v.who)}</b><div class="fine">${v.when.toLocaleTimeString('ru-RU')}</div></div><span class="badge ${v.status === 'подтверждено' ? 'badge-ok' : 'badge-warn'}">${v.status}</span></div>`).join('')}</div>
    <div class="sec-title" style="margin-top:18px"><span class="panel-title">Журнал изменений</span></div><div class="log">${S.log.map((l) => `<div><span>${l.t.toLocaleTimeString('ru-RU')}</span> ${esc(l.text)}</div>`).join('') || 'пусто'}</div>`;
  }
  p.innerHTML = h; $$('input[type=range]', p).forEach(rangeFill);
}

/* =========================================================== ЛЕЧЕНИЕ */
const T = () => S.treat;
const SIM_STAGES = ['Проверка параметров и области применимости', 'Применение модели ответа', 'Расчёт маски и диапазона неопределённости'];
function lvl(n, warn) { return `<div class="lvl ${warn ? 'w' : ''}">${[1, 2, 3].map((i) => `<i class="${i <= n ? 'on' : ''}"></i>`).join('')}</div>`; }
function renderTreat(p) {
  const t = T(); const disabled = noAnalysis() || !S.segDone;
  let h = '';
  if (t.view === 'home') {
    h += `<div class="sec-title"><span class="panel-title">Лечение</span><span class="badge badge-warn">поддержка решения</span></div>
    <p style="font-size:13px;color:var(--text-2);margin-bottom:6px">Оцените возможный ответ опухоли на уже выбранный план или получите несколько вариантов терапии для рассмотрения врачом и консилиумом. Пока вы ничего не запустили, результатов здесь нет.</p>`;
    if (noAnalysis()) h += userNote; else if (!S.segDone) h += `<div class="hint warn">${ic('warn')}<div>Для лечебных сценариев нужна сегментация. Сначала запустите её на вкладке «Сегментация».</div></div>`;
    h += `<div class="tr-bigbtns"><button class="tr-big" data-act="simform" ${disabled ? 'disabled' : ''}><span class="ic">${ic('trend', 20)}</span><span><b>Провести моделирование лечения</b><span>У врача уже есть предполагаемый план — оценить возможный ответ конкретной опухоли.</span></span></button>
    <button class="tr-big" data-act="selcheck" ${disabled ? 'disabled' : ''}><span class="ic">${ic('list', 20)}</span><span><b>Подбор терапии</b><span>Получить несколько обоснованных сценариев для профессионального рассмотрения.</span></span></button></div>
    <div class="hint">${ic('info')}<div>Результаты являются модельными сценариями для рассмотрения врачом и не являются назначением лечения.</div></div>`;
    if (S.simInfo) h += `<div class="row sp layer" style="--c:#fff"><div><b style="font-size:13px">Есть смоделированный сценарий</b><div class="fine">горизонт ${S.simInfo.params.horizon} мес</div></div><button class="btn btn-sm" data-act="simresult">Открыть</button></div>`;
    if (S.consilium.length) h += `<div class="sec-title" style="margin-top:12px"><span class="panel-title">Повестка консилиума</span></div>${S.consilium.map((c) => `<div class="vitem"><div><b>${esc(c.title)}</b><div class="fine">добавлено ${c.when.toLocaleTimeString('ru-RU')}</div></div></div>`).join('')}`;
  } else if (t.view === 'simform') h += simForm();
  else if (t.view === 'simrun') h += runView('Моделирование выполняется', SIM_STAGES);
  else if (t.view === 'simresult') h += simResult();
  else if (t.view === 'selcheck') h += selCheck();
  else if (t.view === 'selrun') h += runView('Подбор вариантов', ['Проверка исходных данных', 'Оценка области применимости', 'Формирование сценариев и обоснований']);
  else if (t.view === 'selresult') h += selResult();
  p.innerHTML = h; $$('input[type=range]', p).forEach(rangeFill);
}
function back(to = 'home', label = '← К выбору действия') { return `<button class="btn btn-sm btn-ghost" data-act="tview" data-v="${to}" style="margin-bottom:12px">${label}</button>`; }
function runView(title, stages) { const r = T().run || { stage: 0, pct: 0 }; return `<div class="row sp"><b style="font-size:13.5px">${title}</b><span class="range-out" id="trPct">${r.pct} %</span></div><div class="pbar"><i id="trBar" style="width:${r.pct}%"></i></div><div class="stagelist">${stages.map((s, i) => `<div class="stage ${i < r.stage ? 'is-done' : i === r.stage ? 'is-run' : ''}"><i></i>${s}</div>`).join('')}</div>`; }
function simForm() {
  const pr = T().params || { kind: 'crt', dose: 60, fractions: 30, tmz: true, bev: false, horizon: 3, start: new Date(Date.now() + 14 * 864e5).toISOString().slice(0, 10) };
  T().params = pr; const mg = S.patient.mgmt;
  return `${back()}<div class="sec-title"><span class="panel-title">Параметры моделирования</span><span class="badge badge-warn">демо-модель</span></div>
  <div class="field"><label>Тип вмешательства</label><select class="sel" data-ch="pk"><option value="crt" ${pr.kind === 'crt' ? 'selected' : ''}>Химиолучевая терапия</option><option value="rt" ${pr.kind === 'rt' ? 'selected' : ''}>Только лучевая терапия</option><option value="sys" ${pr.kind === 'sys' ? 'selected' : ''}>Только системная терапия</option><option value="surg" disabled>Хирургия (вне демо-модели)</option></select></div>
  <div class="grid2"><div class="field"><label>Суммарная доза, Гр</label><input class="inp" type="number" min="0" max="80" step="1" value="${pr.dose}" data-in="pdose"></div><div class="field"><label>Число фракций</label><input class="inp" type="number" min="1" max="40" step="1" value="${pr.fractions}" data-in="pfr"></div></div>
  <div class="row sp" style="margin-bottom:9px"><span style="font-size:13px">Темозоломид</span><button class="switch" data-act="ptmz" role="switch" aria-checked="${pr.tmz}" aria-label="Темозоломид"></button></div>
  <div class="row sp" style="margin-bottom:12px"><span style="font-size:13px">Бевацизумаб</span><button class="switch" data-act="pbev" role="switch" aria-checked="${pr.bev}" aria-label="Бевацизумаб"></button></div>
  <div class="grid2"><div class="field"><label>Дата начала</label><input class="inp" type="date" value="${pr.start}" data-ch="pstart"></div><div class="field"><label>Прогнозный горизонт</label><div class="seg-ctl" style="width:100%"><button data-act="phz" data-h="3" class="${pr.horizon === 3 ? 'is-active' : ''}" style="flex:1;justify-content:center">3 мес</button><button data-act="phz" data-h="6" class="${pr.horizon === 6 ? 'is-active' : ''}" style="flex:1;justify-content:center">6 мес</button></div></div></div>
  <div class="field"><label>Хирургический статус</label><select class="sel" disabled><option>Резекции не проводилось (дооперационное МРТ)</option></select></div>
  <div class="kv"><span>Учтено из карточки</span><b>${S.patient.age} лет · MGMT ${mg ? 'метилирован' : 'не метилирован'}</b></div>
  <div class="kv"><span>Не учтено (нет данных)</span><b class="flat">KPS, объём резекции, карта дозы RTDOSE</b></div>
  ${S.confirmed ? '' : `<div class="hint warn" style="margin-top:8px">${ic('warn')}<div>Маска ещё не подтверждена врачом — результат будет предварительным.</div></div>`}
  <button class="btn btn-primary btn-block btn-lg" style="margin-top:12px" data-act="simrun">${ic('play', 15)} Запустить моделирование</button>
  <p class="fine" style="margin-top:8px">Результат не отображается до запуска. Это демонстрационный расчёт, не клинически валидированный прогноз.</p>`;
}
function domainCheck(pr) {
  const w = [];
  if (pr.kind !== 'sys' && (pr.dose < 20 || pr.dose > 70)) w.push(`Доза ${pr.dose} Гр вне области применимости модели (20–70 Гр)`);
  if (pr.kind === 'sys' && !pr.tmz && !pr.bev) w.push('Не выбрано ни одного лечебного воздействия');
  if (pr.kind !== 'sys' && (pr.fractions < 1 || pr.fractions > 40)) w.push('Число фракций вне допустимого диапазона (1–40)');
  if (pr.kind === 'sys') pr.dose = 0;
  return w;
}
async function runSim() {
  const pr = T().params; const warn = domainCheck(pr);
  if (warn.length) { T().warn = warn; T().view = 'simresult'; T().result = null; refresh('treat'); return; }
  T().view = 'simrun'; T().run = { stage: 0, pct: 0 }; refresh('treat');
  const dur = document.body.classList.contains('no-anim') ? [50, 50, 50] : [700, 1100, 900]; const tot = dur.reduce((a, b) => a + b, 0); let el = 0;
  for (let s = 0; s < 3; s++) { T().run.stage = s; refresh('treat'); const t0 = performance.now(); await new Promise((res) => { const tick = () => { const f = Math.min(1, (performance.now() - t0) / dur[s]); T().run.pct = Math.round(((el + f * dur[s]) / tot) * 100); const b = $('#trBar'); if (b) b.style.width = T().run.pct + '%'; const q = $('#trPct'); if (q) q.textContent = T().run.pct + ' %'; f < 1 ? requestAnimationFrame(tick) : res(); }; tick(); }); el += dur[s]; }
  const e = M.effectStrength({ ...pr, mgmt: S.patient.mgmt });
  const sp = S.vols.t1.spacing; const mk = (k) => M.simulateMask(S.seg, S.depthAll, S.depthEnh, null, e * k, 11);
  const mid = mk(1), lo = mk(0.6), hi = mk(1.45);
  S.simInfo = { params: { ...pr }, e, mask: mid, V: { base: M.countVols(S.seg, sp), mid: M.countVols(mid, sp), lo: M.countVols(hi, sp), hi: M.countVols(lo, sp) } };
  // lo/hi: «оптимистичный» = большая эффективность → меньший объём
  S.sim = { mid, lo: hi, hi: lo }; S.simOn = true; app.vr().setSim(mid);
  T().warn = null; T().view = 'simresult'; app.syncVR(); app.renderAll(); refresh('layers'); refresh('metrics'); refresh('treat');
  app.log?.(`Смоделирован сценарий: ${pr.dose} Гр/${pr.fractions} фр${pr.tmz ? ' + ТМЗ' : ''}, горизонт ${pr.horizon} мес (демо-модель)`);
  toast('Модельный сценарий рассчитан — штриховка и пунктир в срезах, полупрозрачные поверхности в 3D', 'warn');
}
function simResult() {
  const t = T();
  if (t.warn) return `${back('simform', '← Изменить параметры')}<div class="hint err">${ic('warn')}<div><b>Вне области применимости.</b> Уверенный результат не формируется.<ul style="margin:6px 0 0 16px">${t.warn.map((w) => `<li>${esc(w)}</li>`).join('')}</ul></div></div>`;
  const I = S.simInfo; if (!I) return back();
  const pr = I.params; const rows = [['Вся область', 'whole'], ['Активная зона', 'enh'], ['Некроз', 'nec'], ['Отёк', 'ede']];
  const rangeBar = (k) => { const b = I.V.base[k], lo = I.V.lo[k], hi = I.V.hi[k], mid = I.V.mid[k]; const mx = Math.max(b, hi) * 1.05 || 1; const p = (v) => (v / mx) * 100; return `<div style="position:relative;height:8px;background:var(--line-2);border-radius:4px;margin-top:5px"><i style="position:absolute;left:0;width:${p(b)}%;height:100%;border-radius:4px;background:rgba(152,168,184,.55)"></i><i style="position:absolute;left:${p(lo)}%;width:${Math.max(1, p(hi) - p(lo))}%;top:-2px;height:12px;border-radius:3px;background:repeating-linear-gradient(45deg,rgba(245,166,35,.85) 0 3px,rgba(245,166,35,.25) 3px 6px);"></i><i style="position:absolute;left:${p(mid)}%;top:-3px;width:2px;height:14px;background:#fff"></i></div>`; };
  return `${back()}<div class="row sp" style="margin-bottom:10px"><span class="panel-title">Результат моделирования</span><span class="badge badge-warn">Модельный сценарий</span></div>
  <div class="hint warn">${ic('warn')}<div><b>Это не наблюдаемый результат и не прогноз.</b> Демонстрационная модель ответа (toy-response v0.1) без клинической валидации. Показан сценарий на горизонте ${pr.horizon} мес.</div></div>
  <div class="kv"><span>Параметры</span><b>${pr.kind === 'sys' ? 'системная терапия' : pr.dose + ' Гр / ' + pr.fractions + ' фр'}${pr.tmz ? ' + ТМЗ' : ''}${pr.bev ? ' + бевацизумаб' : ''}</b></div>
  ${rows.map(([n, k]) => { const d = M.pct(I.V.mid[k], I.V.base[k]); return `<div style="margin:12px 0 4px"><div class="row sp"><span style="font-size:13px">${n}</span><span class="layer-vol">${M.nf(I.V.base[k])} → <b>${M.nf(I.V.mid[k])}</b> см³ <span class="${dCls(d)}" style="font-size:11px">${M.sgn(d, 0)}%</span></span></div>${rangeBar(k)}<div class="fine" style="margin-top:3px">диапазон неопределённости: ${M.nf(I.V.lo[k])} – ${M.nf(I.V.hi[k])} см³</div></div>`; }).join('')}
  <p class="fine" style="margin-top:4px">Серая полоса — текущий объём; штриховка — диапазон; белая метка — центральная оценка.</p>
  <div class="row sp layer" style="--c:#fff;margin-top:12px"><div><b style="font-size:13px">Показать модельную маску</b><div class="fine">пунктир в срезах · полупрозрачные поверхности в 3D</div></div><button class="switch" data-act="simtoggle" role="switch" aria-checked="${S.simOn}" aria-label="Показать модельную маску"></button></div>
  <button class="btn btn-block" data-act="sim3d" style="margin-bottom:8px">${ic('cube', 15)} 3D: текущее и смоделированное состояние</button>
  <div class="sec-title" style="margin-top:14px"><span class="panel-title">Факторы модели</span></div>
  <div class="check"><span class="st ok">+</span><div>Доза лучевой терапии ${pr.dose} Гр<small>основной вклад в силу эффекта (${M.nf(I.e, 1)} мм «эрозии» границы за ${pr.horizon} мес — условная единица модели)</small></div></div>
  <div class="check"><span class="st ${pr.tmz ? 'ok' : 'miss'}">${pr.tmz ? '+' : '–'}</span><div>Темозоломид ${pr.tmz ? '' : '(не включён)'}<small>${pr.tmz ? (S.patient.mgmt ? 'MGMT метилирован — усиление эффекта' : 'MGMT не метилирован — слабое усиление') : 'без системного компонента эффект ниже'}</small></div></div>
  <div class="check"><span class="st miss">?</span><div>Похожие когорты<small>не подключено: модель не обучалась на когортах пациентов</small></div></div>
  <div class="sec-title" style="margin-top:14px"><span class="panel-title">Не хватает данных</span></div>
  <ul class="ticks warn"><li>Общее состояние (KPS/ECOG)</li><li>Данные о резекции и остаточной опухоли</li><li>Карта дозы RTDOSE и план RTPLAN</li><li>Перфузионные серии и динамика по нескольким исследованиям</li></ul>
  <div class="kv" style="margin-top:10px"><span>Версия / область</span><b>toy-response v0.1 · демо</b></div>
  <button class="btn btn-block btn-ghost" style="margin-top:10px" data-act="simreset">Сбросить сценарий</button>`;
}
function selCheck() {
  const s = T().sel; const kps = s.kps === '' ? null : +s.kps;
  const checks = [
    ['ok', 'Диагноз и степень', 'Глиобластома, WHO 4 (демо-сведения)'],
    ['ok', 'Молекулярные маркеры', `MGMT: ${S.patient.mgmt ? 'метилирован' : 'не метилирован'} · IDH: не указан (для GBM ожидается wild-type)`],
    [S.confirmed ? 'ok' : 'miss', 'МРТ и подтверждённая сегментация', S.confirmed ? 'маска подтверждена врачом' : 'маска не подтверждена — подтвердите на вкладке «Сегментация»'],
    ['ok', 'Предыдущее лечение и операции', 'не проводились (только дооперационное МРТ)'],
    [kps !== null && kps >= 0 && kps <= 100 ? 'ok' : 'miss', 'Состояние пациента (KPS)', kps !== null ? `KPS ${kps}` : 'не указано — обязательно для подбора'],
  ];
  const canRun = S.confirmed && kps !== null && kps >= 0 && kps <= 100;
  return `${back()}<div class="sec-title"><span class="panel-title">Проверка исходных данных</span></div>
  ${checks.map(([st, n, d]) => `<div class="check"><span class="st ${st}">${st === 'ok' ? '✓' : '!'}</span><div>${n}<small>${esc(d)}</small></div></div>`).join('')}
  <div class="grid2" style="margin-top:12px"><div class="field"><label>KPS (0–100)</label><input class="inp" type="number" min="0" max="100" step="10" placeholder="напр. 80" value="${esc(s.kps)}" data-in="kps"></div><div class="field"><label>Цель подбора</label><select class="sel"><option>Первичное лечение</option><option disabled>Рецидив (вне демо)</option></select></div></div>
  <div class="field"><label>Противопоказания</label><div class="chips"><label class="chip"><input type="checkbox" class="cb" data-ch="contra" data-c="hema" ${s.hema ? 'checked' : ''}> &nbsp;Тяжёлая гематотоксичность</label><label class="chip"><input type="checkbox" class="cb" data-ch="contra" data-c="lomus" ${s.lomus ? 'checked' : ''}> &nbsp;Нельзя нитрозомочевины</label></div></div>
  ${canRun ? '' : `<div class="hint warn">${ic('warn')}<div>Кнопка «Подобрать варианты» недоступна: ${!S.confirmed ? 'подтвердите сегментацию' : ''}${!S.confirmed && kps === null ? ' и ' : ''}${kps === null ? 'укажите KPS' : ''}.</div></div>`}
  <button class="btn btn-primary btn-block btn-lg" data-act="selrun" ${canRun ? '' : 'disabled'}>Подобрать варианты</button>`;
}
async function runSel() {
  T().view = 'selrun'; T().run = { stage: 0, pct: 0 }; refresh('treat'); const dur = document.body.classList.contains('no-anim') ? [40, 40, 40] : [600, 700, 800]; const tot = dur.reduce((a, b) => a + b, 0); let el = 0;
  for (let s = 0; s < 3; s++) { T().run.stage = s; refresh('treat'); const t0 = performance.now(); await new Promise((res) => { const tick = () => { const f = Math.min(1, (performance.now() - t0) / dur[s]); T().run.pct = Math.round(((el + f * dur[s]) / tot) * 100); const b = $('#trBar'); if (b) b.style.width = T().run.pct + '%'; const q = $('#trPct'); if (q) q.textContent = T().run.pct + ' %'; f < 1 ? requestAnimationFrame(tick) : res(); }; tick(); }); el += dur[s]; }
  const s = T().sel; let list = M.buildTherapy({ age: S.patient.age, mgmt: S.patient.mgmt, kps: +s.kps }); if (s.lomus) list = list.filter((x) => x.id !== 'cetg');
  T().scen = list; T().view = 'selresult'; refresh('treat'); app.log?.('Подбор терапии: сформированы сценарии для рассмотрения (демо)');
}
function selResult() {
  const list = T().scen || []; const ex = T().sel.excluded; const cmp = T().sel.cmp;
  let h = `<div class="row" style="gap:6px;margin-bottom:10px;flex-wrap:wrap"><button class="btn btn-sm btn-ghost" data-act="tview" data-v="selcheck">Изменить исходные данные</button><button class="btn btn-sm btn-ghost" data-act="newsel">Начать новый подбор</button></div>
  <div class="hint">${ic('info')}<div><b>Сценарии для рассмотрения врачом или консилиумом — не назначения.</b> Оценки качественные (3 уровня) и не объединяются в единый рейтинг. Данные пациента: ${S.patient.age} лет, MGMT ${S.patient.mgmt ? '+' : '−'}, KPS ${T().sel.kps}.</div></div>`;
  h += list.map((c, i) => `<div class="scen ${ex[c.id] ? 'is-excluded' : ''}"><div class="scen-head"><span class="scen-n">${i + 1}</span><div><b>${esc(c.title)}</b><div class="fine" style="margin-top:3px">${esc(c.params)}</div></div></div>
    <div class="scen-body"><div class="axes"><span class="a">Ожидаемый контроль опухоли</span>${lvl(c.axes.control)}<span class="a">Надёжность прогноза</span>${lvl(c.axes.reliab)}<span class="a">Полнота данных</span>${lvl(c.axes.data)}<span class="a">Соответствие критериям</span>${lvl(c.axes.applic, c.axes.applic <= 1)}</div>
    <p><b>Почему в списке:</b> ${c.reasons.map(esc).join('; ')}.</p><p><b>Учтено:</b> возраст, MGMT, KPS, диагноз. <b>Ограничения и риски:</b> ${esc(c.risks)}</p><p class="fine">Токсичность и клиническая допустимость требуют отдельных валидированных моделей — здесь не рассчитываются.</p></div>
    <details><summary>Источники (${c.src.length})</summary>${c.src.map((k) => `<a class="ref" href="${M.SOURCES[k].u}" target="_blank" rel="noopener">${esc(M.SOURCES[k].t)}</a>`).join('')}</details>
    ${ex[c.id] ? `<div class="scen-body"><span class="badge badge-mute">исключено: ${esc(ex[c.id])}</span></div>` : `<div class="scen-actions"><button class="btn btn-sm ${cmp.includes(c.id) ? 'btn-blue' : ''}" data-act="cmpadd" data-id="${c.id}">${cmp.includes(c.id) ? '✓ в сравнении' : 'Сравнить'}</button><button class="btn btn-sm" data-act="cons" data-id="${c.id}" ${S.consilium.some((x) => x.id === c.id) ? 'disabled' : ''}>${S.consilium.some((x) => x.id === c.id) ? '✓ в консилиуме' : 'Добавить в консилиум'}</button><button class="btn btn-sm btn-ghost" data-act="excl" data-id="${c.id}">Исключить с комментарием</button></div>${T().exclOpen === c.id ? `<div class="scen-body"><input class="inp" id="exclTxt" placeholder="Причина исключения (обязательно)"><button class="btn btn-sm" style="margin-top:6px" data-act="exclok" data-id="${c.id}">Исключить</button></div>` : ''}`}
    </div>`).join('');
  if (cmp.length === 2) { const [a, b] = cmp.map((id) => list.find((x) => x.id === id)); h += `<div class="sec-title" style="margin-top:14px"><span class="panel-title">Сравнение</span></div><table class="mtable"><thead><tr><th></th><th>${esc(a.title.slice(0, 22))}…</th><th>${esc(b.title.slice(0, 22))}…</th></tr></thead><tbody>${[['Контроль', 'control'], ['Надёжность', 'reliab'], ['Полнота данных', 'data'], ['Применимость', 'applic']].map(([n, k]) => `<tr><td>${n}</td><td>${'●'.repeat(a.axes[k])}${'○'.repeat(3 - a.axes[k])}</td><td>${'●'.repeat(b.axes[k])}${'○'.repeat(3 - b.axes[k])}</td></tr>`).join('')}</tbody></table>`; }
  return h + `<p class="fine" style="margin-top:10px">Нет кнопки «Применить лечение»: добавление в протокол возможно только по решению врача вне этого прототипа.</p>`;
}

/* =========================================================== ДЕЙСТВИЯ */
function act(a, b, e) {
  const l = +b.dataset.l;
  switch (a) {
    case 'goseg': app.openTab('seg'); break;
    case 'runseg': app.runSegmentation(); break;
    case 'hi': app.setHi(l); break;
    case 'style': S.layers[l].style = b.dataset.s; refresh('layers'); app.renderAll(); break;
    case 'edit': S.brush.label = l; app.setTool('brush'); app.openTab('seg'); toast(`Кисть: «${CL[l].short}». Рисуйте в срезах; правки создают новую версию маски`, ''); break;
    case 'simtoggle': S.simOn = !S.simOn; app.vr().setSim(S.simOn ? S.sim.mid : null); app.syncVR(); app.renderAll(); refresh('layers'); refresh('treat'); break;
    case 'clip': S.clip.on = !S.clip.on; app.syncVR(); refresh('layers'); break;
    case 'clipaxis': S.clip.axis = b.dataset.a; app.syncVR(); refresh('layers'); break;
    case 'clipflip': S.clip.flip = !S.clip.flip; app.syncVR(); break;
    case 'tool': app.setTool(b.dataset.t); break;
    case 'blabel': S.brush.label = l; refresh('seg'); break;
    case 'undo': app.undo(); refresh('seg'); break; case 'redo': app.redo(); refresh('seg'); break;
    case 'orig': S.showOrig = !S.showOrig; refresh('seg'); app.renderAll(); break;
    case 'confirm': app.confirmMask(); break; case 'revert': app.revertMask(); break;
    case 'cmp': app.openCompare(); break;
    case 'goreport': app.doAction('report'); break;
    case 'tview': T().view = b.dataset.v; refresh('treat'); break;
    case 'simform': T().view = 'simform'; refresh('treat'); break;
    case 'simrun': runSim(); break;
    case 'simresult': T().view = 'simresult'; refresh('treat'); break;
    case 'simreset': S.sim = null; S.simInfo = null; S.simOn = false; app.vr().setSim(null); T().view = 'home'; T().params = null; app.syncVR(); app.renderAll(); refreshAll(); break;
    case 'sim3d': S.simOn = true; app.vr().setSim(S.sim.mid); S.brainOpacity = Math.min(S.brainOpacity, 0.16); app.setLayout('vr'); app.syncVR(); refresh('layers'); break;
    case 'ptmz': T().params.tmz = !T().params.tmz; refresh('treat'); break;
    case 'pbev': T().params.bev = !T().params.bev; refresh('treat'); break;
    case 'phz': T().params.horizon = +b.dataset.h; refresh('treat'); break;
    case 'selcheck': T().view = 'selcheck'; refresh('treat'); break;
    case 'selrun': runSel(); break;
    case 'newsel': T().sel = { kps: '', excluded: {}, cmp: [] }; T().view = 'selcheck'; refresh('treat'); break;
    case 'cmpadd': { const c = T().sel.cmp; const i = c.indexOf(b.dataset.id); if (i >= 0) c.splice(i, 1); else { c.push(b.dataset.id); if (c.length > 2) c.shift(); } refresh('treat'); break; }
    case 'cons': { const s = T().scen.find((x) => x.id === b.dataset.id); S.consilium.push({ id: s.id, title: s.title, when: new Date() }); toast(`«${s.title.slice(0, 40)}…» добавлен в повестку консилиума`, 'ok'); app.log?.('В повестку консилиума добавлен сценарий: ' + s.title); refresh('treat'); break; }
    case 'excl': T().exclOpen = b.dataset.id; refresh('treat'); $('#exclTxt')?.focus(); break;
    case 'exclok': { const v = $('#exclTxt')?.value.trim(); if (!v) return toast('Укажите причину исключения', 'warn'); T().sel.excluded[b.dataset.id] = v; T().exclOpen = null; refresh('treat'); break; }
  }
}
function input(k, b) {
  const v = +b.value;
  if (k === 'op') { const l = +b.dataset.l; S.layers[l].opacity = v; S.vrOpacity[l] = l === 2 ? Math.min(0.9, v * 0.85) : Math.min(1, v * 1.35 + 0.05); rangeFill(b); app.syncVR(); app.renderAll(); }
  else if (k === 'bo') { S.brainOpacity = v; $('#vBo').textContent = Math.round(v * 100) + '%'; rangeFill(b); app.syncVR(); }
  else if (k === 'thr') { S.brainThr = v; $('#vThr').textContent = v.toFixed(2); rangeFill(b); app.syncVR(); }
  else if (k === 'clip') { S.clip.pos = v; rangeFill(b); app.syncVR(); }
  else if (k === 'br') { S.brush.r = v; $('#vBr').textContent = v + ' мм'; rangeFill(b); }
  else if (k === 'pdose') T().params.dose = v; else if (k === 'pfr') T().params.fractions = v;
  else if (k === 'kps') { T().sel.kps = b.value; clearTimeout(input._t); input._t = setTimeout(() => refresh('treat'), 600); }
}
function change(k, b) {
  if (k === 'show') { const l = +b.dataset.l; S.layers[l].show = b.checked; app.syncVR(); app.renderAll(); app.updateVRHud?.(); }
  else if (k === 'pk') { T().params.kind = b.value; if (b.value === 'sys') T().params.dose = 0; else if (!T().params.dose) T().params.dose = 60; refresh('treat'); }
  else if (k === 'pstart') T().params.start = b.value;
  else if (k === 'contra') { T().sel[b.dataset.c] = b.checked; }
}
