// «Загрузить свое МРТ»: локальное чтение NIfTI / DICOM в браузере, выбор серии, проверка качества.
// Файлы не отправляются на сервер.
import { S, app } from './state.js';
import { Vol, toArrayBuffer, parseNifti, toRAS, normalizeU8 } from './nifti.js';
import { ic, $, $$, esc, toast, openModal, closeModal } from './ui.js';

const guess = (name) => {
  const n = name.toLowerCase();
  if (/flair/.test(n)) return { key: 'flair', title: 'T2-FLAIR', seq: 'T2-FLAIR' };
  if (/t1.?(ce|gd|c\b|contrast|post)|gd|contrast|t1c/.test(n)) return { key: 't1ce', title: 'T1-Gd', seq: 'T1 с контрастом', contrast: true };
  if (/t1/.test(n)) return { key: 't1', title: 'T1', seq: 'T1' };
  if (/t2/.test(n)) return { key: 't2', title: 'T2', seq: 'T2' };
  if (/dwi|adc|diff/.test(n)) return { key: 'dwi', title: 'DWI/ADC', seq: 'DWI/ADC' };
  return null;
};
const badge = '<span class="badge badge-ai" title="Определено по заголовку файла и эвристикам; проверьте и подтвердите">определено автоматически</span>';

export function initUpload() {
  app.openUpload = () => { render0(); openModal('uploadModal'); };
  const fi = $('#fileInput'), di = $('#dirInput');
  fi.onchange = () => { const f = [...fi.files]; fi.value = ''; if (f.length) process(f); };
  di.onchange = () => { const f = [...di.files]; di.value = ''; if (f.length) process(f); };
}

function render0() {
  $('#uploadBody').innerHTML = `<div class="drop" id="dropZone"><div style="color:var(--accent)">${ic('upload', 34)}</div><h4>Перетащите файлы или папку сюда</h4><p>Поддерживаются <b>.nii / .nii.gz</b> (NIfTI-1) и <b>DICOM</b> без сжатия (.dcm, набор файлов или папка).</p>
    <div class="row"><button class="btn btn-primary" id="pickFiles">${ic('doc', 15)} Выбрать файлы</button><button class="btn" id="pickDir">${ic('folder', 15)} Выбрать папку DICOM</button><button class="btn btn-ghost" id="pickDemo">Открыть демо</button></div></div>
    <div class="hint" style="margin-top:14px">${ic('lock')}<div><b>Файлы обрабатываются локально и не отправляются на сервер.</b> Данные не сохраняются после закрытия вкладки. Браузер получает доступ только к тому, что вы явно выбрали.</div></div>
    <div class="src-list"><div class="src-item">NIfTI (.nii, .nii.gz)<span class="badge badge-ok">поддерживается</span></div><div class="src-item">DICOM (файлы/папка)<span class="badge badge-ok">поддерживается</span></div><div class="src-item">ZIP-архив<span class="badge badge-mute">позже</span></div><div class="src-item">Импорт из PACS<span class="badge badge-mute">позже</span></div></div>
    <p class="fine" style="margin-top:12px">Можно выбрать сразу несколько NIfTI-серий одного пациента (T1, T1-Gd, T2, FLAIR) — они должны иметь одинаковую матрицу. Файл с «seg» или «mask» в имени будет распознан как маска.</p>`;
  $('#pickFiles').onclick = () => $('#fileInput').click(); $('#pickDir').onclick = () => $('#dirInput').click();
  $('#pickDemo').onclick = () => { closeModal('uploadModal'); app.openDemo(app.cases[0]); };
  const dz = $('#dropZone');
  ['dragenter', 'dragover'].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add('is-over'); }));
  ['dragleave', 'drop'].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove('is-over'); }));
  dz.addEventListener('drop', async (e) => { const files = await collect(e.dataTransfer); if (files.length) process(files); });
}
async function collect(dt) {
  const out = [];
  const walk = async (entry) => { if (entry.isFile) out.push(await new Promise((r) => entry.file(r))); else if (entry.isDirectory) { const rd = entry.createReader(); let all = []; let b; do { b = await new Promise((r) => rd.readEntries(r)); all = all.concat(b); } while (b.length); for (const c of all) await walk(c); } };
  if (dt.items?.length && dt.items[0].webkitGetAsEntry) { for (const it of [...dt.items]) { const en = it.webkitGetAsEntry(); if (en) await walk(en); } } else out.push(...dt.files);
  return out;
}

/* ---------------- DICOM (без сжатия) ---------------- */
function parseDicom(buf) {
  const dv = new DataView(buf); let pos = 0; if (buf.byteLength > 132 && dv.getUint32(128, false) === 0x4449434d) pos = 132; else return null;
  const T = {}; let explicit = true; const le = true; let ts = '1.2.840.10008.1.2.1';
  const str = (o, n) => { let s = ''; for (let i = 0; i < n; i++) { const c = dv.getUint8(o + i); if (c) s += String.fromCharCode(c); } return s.trim(); };
  while (pos + 8 <= buf.byteLength) {
    const g = dv.getUint16(pos, le), e = dv.getUint16(pos + 2, le); let len, hdr, vr = '';
    if (g !== 2 && !T._tsdone) { T._tsdone = true; explicit = ts !== '1.2.840.10008.1.2'; }
    const ex = g === 2 ? true : explicit;
    if (g === 0xfffe) { len = dv.getUint32(pos + 4, le); hdr = 8; pos += hdr; continue; }
    if (ex) { vr = String.fromCharCode(dv.getUint8(pos + 4), dv.getUint8(pos + 5)); if (['OB', 'OW', 'OF', 'SQ', 'UT', 'UN', 'OD', 'OL', 'UC', 'UR'].includes(vr)) { len = dv.getUint32(pos + 8, le); hdr = 12; } else { len = dv.getUint16(pos + 6, le); hdr = 8; } } else { len = dv.getUint32(pos + 4, le); hdr = 8; }
    const tag = ((g << 16) | e) >>> 0, v = pos + hdr;
    if (tag === 0x7fe00010) { T.pix = { off: v, len }; break; }
    if (len === 0xffffffff || vr === 'SQ') { pos = v; continue; }
    if (v + len > buf.byteLength) break;
    if (T[tag] === undefined) {
      if (tag === 0x00020010) { ts = str(v, len); T.ts = ts; }
      else if (['US'].includes(vr) || (!ex && (tag === 0x00280010 || tag === 0x00280011 || tag === 0x00280100 || tag === 0x00280103))) T[tag] = dv.getUint16(v, le);
      else T[tag] = str(v, len);
    }
    pos = v + len;
  }
  return T;
}
const N = (s) => (s === undefined ? NaN : parseFloat(String(s).split('\\')[0]));
const NS = (s) => (s === undefined ? [] : String(s).split('\\').map(parseFloat));
function dicomSeries(buffers) {
  const series = {};
  for (const { buf, name } of buffers) {
    const t = parseDicom(buf); if (!t || !t.pix) continue;
    if (t.ts && !['1.2.840.10008.1.2', '1.2.840.10008.1.2.1'].includes(t.ts)) { const e = new Error(`Сжатый DICOM (transfer syntax ${t.ts}) пока не поддерживается в браузере. Экспортируйте исследование без сжатия или в NIfTI.`); e.unsupported = true; throw e; }
    const uid = t[0x0020000e] || 'unknown'; (series[uid] = series[uid] || { slices: [], desc: t[0x0008103e] || '', mod: t[0x00080060] || '', field: N(t[0x00180087]), num: t[0x00200011], date: t[0x00080020], vendor: t[0x00080070] || '', pid: t[0x00100020] }).slices.push({ t, buf });
  }
  return series;
}
function dicomToVol(ser) {
  const sl = ser.slices; const t0 = sl[0].t; const rows = t0[0x00280010], cols = t0[0x00280011]; const ps = NS(t0[0x00280030]); const io = NS(t0[0x00200037]);
  const rd = io.length === 6 ? io.slice(0, 3) : [1, 0, 0], cd = io.length === 6 ? io.slice(3, 6) : [0, 1, 0];
  const nrm = [rd[1] * cd[2] - rd[2] * cd[1], rd[2] * cd[0] - rd[0] * cd[2], rd[0] * cd[1] - rd[1] * cd[0]];
  for (const s of sl) { const ip = NS(s.t[0x00200032]); s.pos = ip.length === 3 ? ip[0] * nrm[0] + ip[1] * nrm[1] + ip[2] * nrm[2] : N(s.t[0x00200013]); s.ip = ip; }
  sl.sort((a, b) => a.pos - b.pos);
  const n = sl.length; let dz = n > 1 ? Math.abs(sl[n - 1].pos - sl[0].pos) / (n - 1) : N(t0[0x00180050]) || 1; if (!isFinite(dz) || dz <= 0) dz = 1;
  const bits = t0[0x00280100] || 16, sgn = t0[0x00280103] === 1; const data = new Float32Array(rows * cols * n);
  sl.forEach((s, k) => { const slope = N(s.t[0x00281053]) || 1, icpt = N(s.t[0x00281052]) || 0; const ar = bits === 8 ? new Uint8Array(s.buf, s.t.pix.off, rows * cols) : sgn ? new Int16Array(s.buf.slice(s.t.pix.off, s.t.pix.off + rows * cols * 2)) : new Uint16Array(s.buf.slice(s.t.pix.off, s.t.pix.off + rows * cols * 2)); for (let i = 0; i < rows * cols; i++) data[k * rows * cols + i] = ar[i] * slope + icpt; });
  // массив (col, row, slice) → индекс i + cols*(j + rows*k); аффин в LPS → RAS
  const sx = ps[1] || 1, sy = ps[0] || 1; const ip0 = sl[0].ip.length === 3 ? sl[0].ip : [0, 0, 0];
  const lps = [[rd[0] * sx, cd[0] * sy, nrm[0] * dz, ip0[0]], [rd[1] * sx, cd[1] * sy, nrm[1] * dz, ip0[1]], [rd[2] * sx, cd[2] * sy, nrm[2] * dz, ip0[2]]];
  const A = [lps[0].map((x) => -x), lps[1].map((x) => -x), lps[2], [0, 0, 0, 1]];
  return { dims: [cols, rows, n], spacing: [sx, sy, dz], data, affine: A };
}

/* ---------------- Основной процесс ---------------- */
async function process(files) {
  const body = $('#uploadBody'); const steps = [];
  const draw = () => { body.innerHTML = `<div class="up-steps">${steps.map((s) => `<div class="upstep ${s.st}"><span class="st">${s.st === 'is-done' ? '✓' : s.st === 'is-warn' ? '!' : ''}</span><div style="flex:1;min-width:0"><h5>${s.t}${s.badge ? ' ' + badge : ''}</h5>${s.sub ? `<div class="sub">${s.sub}</div>` : ''}${s.html || ''}</div></div>`).join('')}</div>${finalHtml}`; bindFinal(); };
  let finalHtml = ''; let bindFinal = () => {};
  const step = (t, o = {}) => { const s = { t, st: 'is-run', ...o }; steps.push(s); draw(); return s; };
  const done = (s, o = {}) => { Object.assign(s, { st: 'is-done' }, o); draw(); };
  try {
    const s1 = step('Чтение файлов локально', { sub: `${files.length} файл(ов) · ${(files.reduce((a, f) => a + f.size, 0) / 1e6).toFixed(1)} МБ · сеть не используется` }); await tick(); done(s1);
    const nii = files.filter((f) => /\.nii(\.gz)?$/i.test(f.name));
    const s2 = step('Определение формата');
    let series = []; // {id,title,vol:{dims,spacing,data,affine}, name, info}
    if (nii.length) {
      done(s2, { sub: `NIfTI-1 · ${nii.length} файл(ов)` }); const s3 = step('Чтение объёмов');
      for (const f of nii) {
        try { const img = parseNifti(await toArrayBuffer(f)); const r = toRAS(img); series.push({ id: f.name, name: f.name, title: guess(f.name)?.title || f.name.replace(/\.nii(\.gz)?$/i, ''), g: guess(f.name), dims: r.dims, spacing: r.spacing, raw: r.data, affine: img.affine, dim4: img.dim4, orig: { dims: img.dims, spacing: img.spacing } }); }
        catch (e) { series.push({ id: f.name, name: f.name, error: e.message }); }
      }
      done(s3);
    } else {
      const bufs = []; for (const f of files) { if (f.size < 200) continue; const b = await f.arrayBuffer(); if (b.byteLength > 132 && new DataView(b).getUint32(128, false) === 0x4449434d) bufs.push({ buf: b, name: f.name }); await tick(); }
      if (!bufs.length) throw Object.assign(new Error('Не найдено ни NIfTI, ни DICOM-файлов. Выберите .nii/.nii.gz или файлы .dcm.'), { fatal: true });
      const ser = dicomSeries(bufs); const ids = Object.keys(ser);
      done(s2, { sub: `DICOM · ${bufs.length} срезов · серий: ${ids.length}${ser[ids[0]].pid ? ' · в файлах есть идентификатор пациента — он не покидает браузер' : ''}` });
      const s3 = step('Группировка по сериям');
      for (const id of ids) { const s = ser[id]; if (s.slices.length < 8) continue; const v = dicomToVol(s); const r = toRAS(v); const nm = `${s.mod} ${s.desc}`.trim() || 'Серия ' + (s.num || ''); series.push({ id, name: nm, title: guess(s.desc)?.title || nm, g: guess(s.desc), dims: r.dims, spacing: r.spacing, raw: r.data, affine: v.affine, dicom: { field: s.field, vendor: s.vendor, date: s.date } }); }
      if (!series.length) throw Object.assign(new Error('В найденных DICOM-файлах слишком мало срезов для объёма (нужно ≥ 8 в серии).'), { fatal: true });
      done(s3);
    }
    const ok = series.filter((s) => !s.error); const bad = series.filter((s) => s.error);
    if (!ok.length) throw Object.assign(new Error(bad[0]?.error || 'Не удалось прочитать данные'), { fatal: true });
    // выбор серий
    const first = ok[0]; const isMask = (s) => /seg|mask|label|tumou?r/i.test(s.name);
    const cands = ok.filter((s) => !isMask(s)); const masks = ok.filter(isMask);
    const sel = new Set(cands.filter((s) => s.dims.join() === (cands[0] || first).dims.join()).slice(0, 4).map((s) => s.id)); if (!sel.size) sel.add(first.id);
    const s4 = step('Найденные серии', { st: 'is-done' });
    s4.html = `<div class="series-pick">${cands.map((s) => `<label><input type="${cands.length > 1 ? 'checkbox' : 'radio'}" name="ser" value="${esc(s.id)}" ${sel.has(s.id) ? 'checked' : ''}> <div><b>${esc(s.title)}</b><div class="fine">${s.dims.join('×')} · ${s.spacing.map((x) => x.toFixed(2)).join('×')} мм${s.dim4 > 1 ? ' · 4D: будет открыт первый том' : ''}${s.dims.join() !== cands[0].dims.join() ? ' · матрица отличается — будет пропущена' : ''}</div></div></label>`).join('')}${masks.map((s) => `<div class="fine">Маска: <b>${esc(s.name)}</b> — будет загружена как готовая сегментация (без анализа)</div>`).join('')}${bad.map((s) => `<div class="fine" style="color:var(--err)">${esc(s.name)}: ${esc(s.error)}</div>`).join('')}</div>`; draw();
    // распознанное
    const cur = () => cands.filter((s) => $$('input[name=ser]:checked', body).some((i) => i.value === s.id));
    const s5 = step('Автоматически распознанные сведения', { st: 'is-done', badge: true });
    const s6 = step('Проверка качества', { st: 'is-done' });
    const fill = () => {
      const chosen = (cur().length ? cur() : [cands[0] || first]); const f = chosen[0] || first; const fov = f.dims.map((d, i) => d * f.spacing[i]);
      const head = fov.every((x) => x > 120 && x < 330);
      const spx = f.spacing; const aniso0 = Math.max(...spx) / Math.min(...spx); const plane = aniso0 < 1.25 ? 'изотропный 3D-объём' : ['сагиттальная', 'корональная', 'аксиальная'][spx.indexOf(Math.max(...spx))] + ' (по толщине среза)';
      s5.html = `<div class="pill-row"><span class="chip">Область: ${head ? 'вероятно, головной мозг' : 'не определена'} <span class="fine">(по размеру поля ${fov.map((x) => Math.round(x)).join('×')} мм)</span></span><span class="chip">Тип: МРТ (предположительно)</span><span class="chip">Матрица: ${f.dims.join('×')}</span><span class="chip">Воксел: ${f.spacing.map((x) => x.toFixed(2)).join('×')} мм</span><span class="chip">Плоскость: ${plane}</span>${chosen.map((s) => `<span class="chip">${esc(s.title)}${s.g?.contrast ? ' · контраст' : ''}</span>`).join('')}${f.dicom?.field ? `<span class="chip">${f.dicom.field} Тл</span>` : ''}${f.dicom?.vendor ? `<span class="chip">${esc(f.dicom.vendor)}</span>` : ''}</div><p class="fine" style="margin-top:6px">Всё, что определено автоматически, должен подтвердить рентгенолаборант. Диагноз, маркеры и клинические данные не запрашиваются — они нужны только для анализа, который для файла не подключен.</p>`;
      const checks = [];
      const keys = chosen.map((s) => s.g?.key).filter(Boolean);
      checks.push([chosen.length >= 4 && ['t1', 't1ce', 't2', 'flair'].every((k) => keys.includes(k)) ? 'ok' : 'warn', 'Комплектность модальностей', `выбрано серий: ${chosen.length}; для модели анализа требуется T1, T1-Gd, T2, FLAIR`]);
      const vx = Math.max(...f.spacing); checks.push([vx <= 1.5 ? 'ok' : vx <= 3 ? 'warn' : 'bad', 'Разрешение', `наибольший размер вокселя ${vx.toFixed(2)} мм${vx > 1.5 ? ' — мелкие детали могут быть потеряны' : ''}`]);
      const aniso = Math.max(...f.spacing) / Math.min(...f.spacing); checks.push([aniso < 1.5 ? 'ok' : 'warn', 'Геометрия', aniso < 1.5 ? 'воксели близки к изотропным' : `анизотропия ×${aniso.toFixed(1)} — 3D-поверхности будут менее гладкими`]);
      checks.push(['ok', 'Ориентация', 'оси приведены к RAS+ по заголовку; проверьте стороны R/L визуально']);
      let mn = Infinity, mx = -Infinity, nan = 0; const d = f.raw; const step0 = Math.max(1, Math.floor(d.length / 200000)); for (let i = 0; i < d.length; i += step0) { const v = d[i]; if (!isFinite(v)) nan++; else { if (v < mn) mn = v; if (v > mx) mx = v; } }
      checks.push([mx > mn && !nan ? 'ok' : 'bad', 'Интенсивность', mx > mn ? `диапазон ${Math.round(mn)}…${Math.round(mx)}${nan ? `; нечисловых значений: ${nan}` : ''}` : 'объём пустой (постоянная интенсивность)']);
      checks.push(['warn', 'Движение и артефакты', 'автоматическая оценка не выполнялась — оценивается визуально']);
      checks.push(['bad', 'Пригодность для модели анализа', 'не подтверждена: модель обучена на MNI-регистрированных МРТ без черепа. Просмотр доступен; автоматический анализ для этого файла пока не подключен']);
      s6.html = checks.map(([st, n, d2]) => `<div class="check"><span class="st ${st === 'ok' ? 'ok' : st === 'warn' ? 'miss' : 'bad'}">${st === 'ok' ? '✓' : '!'}</span><div>${n}<small>${d2}</small></div></div>`).join('');
      draw();
    };
    fill();
    finalHtml = `<div class="row" style="justify-content:flex-end;margin-top:16px"><button class="btn btn-ghost" id="upCancel">Отмена</button><button class="btn btn-primary" id="upOpen">${ic('eye', 15)} Открыть в просмотрщике</button></div>`;
    bindFinal = () => {
      $$('input[name=ser]', body).forEach((i) => (i.onchange = fill));
      $('#upCancel', body).onclick = () => closeModal('uploadModal');
      $('#upOpen', body).onclick = async () => {
        const chosen = cur().length ? cur() : [cands[0] || first]; const base = chosen[0]; const vols = {}; const names = {}; const used = new Set();
        for (const s of chosen) { if (s.dims.join() !== base.dims.join()) continue; let k = s.g?.key && !used.has(s.g.key) ? s.g.key : 's' + used.size; used.add(k); const v = build(s); vols[k] = v; names[k] = s.title; }
        let seg = null; const mk = masks.find((m) => m.dims.join() === base.dims.join());
        if (mk) { const mu = new Uint8Array(mk.raw.length); let mxv = 0; for (let i = 0; i < mu.length; i++) { const q = Math.round(mk.raw[i]); mu[i] = q > 0 && q < 5 ? (q === 4 ? 3 : q) : 0; if (q > mxv) mxv = q; } seg = mu; }
        closeModal('uploadModal');
        await app.openUser({ vols, seg, names, meta: { fileName: chosen.map((s) => s.name).join(', '), dims: base.dims.join('×') + ' · ' + base.spacing.map((x) => x.toFixed(2)).join('×') + ' мм', field: base.dicom?.field } });
        toast('Файл открыт локально. Просмотр доступен; автоматический анализ не подключен', '');
      };
    };
    draw();
  } catch (e) {
    console.error(e);
    body.innerHTML = `<div class="hint err">${ic('warn')}<div><b>Не удалось открыть файл.</b><br>${esc(e.message)}<br><span class="fine">Данные не были сохранены и никуда не отправлялись. Что можно сделать: убедитесь, что это NIfTI-1 или несжатый DICOM; попробуйте выбрать файлы заново.</span></div></div><div class="row" style="justify-content:flex-end;margin-top:14px"><button class="btn" id="upRetry">Выбрать другие файлы</button></div>`;
    $('#upRetry').onclick = render0;
  }
}
const tick = () => new Promise((r) => setTimeout(r, 30));
function build(s) {
  let dims = s.dims, sp = s.spacing, raw = s.raw; const n = dims[0] * dims[1] * dims[2];
  let data = normalizeU8(raw, n);
  if (n > 32e6) { // прореживание ×2 для очень больших объёмов
    const [nx, ny, nz] = dims; const ox = nx >> 1, oy = ny >> 1, oz = nz >> 1; const o = new Uint8Array(ox * oy * oz);
    for (let k = 0; k < oz; k++) for (let j = 0; j < oy; j++) for (let i = 0; i < ox; i++) o[i + ox * (j + oy * k)] = data[2 * i + nx * (2 * j + ny * 2 * k)];
    data = o; dims = [ox, oy, oz]; sp = sp.map((x) => x * 2);
  }
  return new Vol(dims[0], dims[1], dims[2], data, sp, { name: s.name });
}
