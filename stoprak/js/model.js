// Расчёты и демонстрационные сценарии: метрики по маске, модель ответа на лечение (игрушечная, демо),
// варианты терапии, источники, текст заключения и ответы ИИ-помощника.
export const CLASSES = {
  1: { key: 'nec', name: 'Некроз / неусиливающееся ядро', short: 'Некроз', color: '#F5A623' },
  2: { key: 'ede', name: 'Перифокальный отёк (FLAIR-зона)', short: 'Отёк', color: '#38BDF8' },
  3: { key: 'enh', name: 'Активная (усиливающаяся) опухоль', short: 'Активная зона', color: '#FF5B61' },
};
export const nf = (x, d = 1) => Number(x).toLocaleString('ru-RU', { minimumFractionDigits: d, maximumFractionDigits: d });
export const pct = (a, b) => (b ? ((a - b) / b) * 100 : 0);
export const sgn = (x, d = 1) => (x > 0 ? '+' : x < 0 ? '−' : '') + nf(Math.abs(x), d);

/* ---------- Метрики ---------- */
function components(mask, nx, ny, nz, minVox) {
  // 6-связность, BFS по массиву индексов
  const seen = new Uint8Array(mask.length); const comps = [];
  const st = [];
  for (let s = 0; s < mask.length; s++) {
    if (!mask[s] || seen[s]) continue;
    st.length = 0; st.push(s); seen[s] = 1; let cnt = 0, cx = 0, cy = 0, cz = 0;
    while (st.length) {
      const c = st.pop(); cnt++;
      const i = c % nx, j = ((c / nx) | 0) % ny, k = (c / (nx * ny)) | 0;
      cx += i; cy += j; cz += k;
      const nb = [i > 0 ? c - 1 : -1, i < nx - 1 ? c + 1 : -1, j > 0 ? c - nx : -1, j < ny - 1 ? c + nx : -1, k > 0 ? c - nx * ny : -1, k < nz - 1 ? c + nx * ny : -1];
      for (const q of nb) if (q >= 0 && mask[q] && !seen[q]) { seen[q] = 1; st.push(q); }
    }
    if (cnt >= minVox) comps.push({ vox: cnt, c: [cx / cnt, cy / cnt, cz / cnt] });
  }
  return comps.sort((a, b) => b.vox - a.vox);
}

function farthestPair(pts, sp) {
  if (!pts.length) return { d: 0, a: null, b: null };
  const far = (p) => { let bi = 0, bd = -1; for (let i = 0; i < pts.length; i++) { const q = pts[i]; const d = ((q[0] - p[0]) * sp[0]) ** 2 + ((q[1] - p[1]) * sp[1]) ** 2 + ((q[2] - p[2]) * sp[2]) ** 2; if (d > bd) { bd = d; bi = i; } } return [bi, Math.sqrt(bd)]; };
  let [ia] = far(pts[0]); let a = pts[ia];
  let [ib, d] = far(a); let b = pts[ib];
  [ia, d] = far(b); a = pts[ia];
  return { d, a, b };
}

export function computeMetrics(seg, vol, affine, vols) {
  const { nx, ny, nz } = vol; const sp = vol.spacing; const vv = (sp[0] * sp[1] * sp[2]) / 1000; // см³ на воксель
  const cnt = [0, 0, 0, 0]; const bb = [1e9, 1e9, 1e9, -1, -1, -1];
  let sx = 0, sy = 0, sz = 0, sn = 0;
  const core = new Uint8Array(seg.length), enhM = new Uint8Array(seg.length);
  for (let k = 0, c = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++, c++) {
    const l = seg[c]; if (!l) continue; cnt[l]++;
    if (i < bb[0]) bb[0] = i; if (j < bb[1]) bb[1] = j; if (k < bb[2]) bb[2] = k; if (i > bb[3]) bb[3] = i; if (j > bb[4]) bb[4] = j; if (k > bb[5]) bb[5] = k;
    sx += i; sy += j; sz += k; sn++;
    if (l !== 2) core[c] = 1; if (l === 3) enhM[c] = 1;
  }
  const V = { nec: cnt[1] * vv, ede: cnt[2] * vv, enh: cnt[3] * vv };
  V.whole = V.nec + V.ede + V.enh; V.core = V.nec + V.enh;
  const out = { V, share: { nec: V.nec / (V.whole || 1), ede: V.ede / (V.whole || 1), enh: V.enh / (V.whole || 1) }, empty: sn === 0 };
  if (sn === 0) return { ...out, diam: 0, foci: 0, centroid: null };
  const cen = [sx / sn, sy / sn, sz / sn];
  const mni = affine ? [0, 1, 2].map((r) => affine[r][0] * cen[0] + affine[r][1] * cen[1] + affine[r][2] * cen[2] + affine[r][3]) : null;
  out.centroid = cen; out.mni = mni;
  out.extent = [(bb[3] - bb[0] + 1) * sp[0], (bb[4] - bb[1] + 1) * sp[1], (bb[5] - bb[2] + 1) * sp[2]];
  // максимальный диаметр по поверхностным вокселям всей опухоли
  const surf = [];
  for (let k = bb[2]; k <= bb[5]; k++) for (let j = bb[1]; j <= bb[4]; j++) for (let i = bb[0]; i <= bb[3]; i++) {
    const c = i + nx * (j + ny * k); if (!seg[c]) continue;
    if (i % 2 || j % 2 || k % 2) { if (!(seg[c - 1] === 0 || seg[c + 1] === 0 || seg[c - nx] === 0 || seg[c + nx] === 0 || seg[c - nx * ny] === 0 || seg[c + nx * ny] === 0)) continue; }
    else if (seg[c - 1] && seg[c + 1] && seg[c - nx] && seg[c + nx] && seg[c - nx * ny] && seg[c + nx * ny]) continue;
    surf.push([i, j, k]);
  }
  const fp = farthestPair(surf, sp);
  out.diam = fp.d; out.diamPts = [fp.a, fp.b];
  // билинейные размеры активной зоны на аксиальном срезе с максимальной площадью (по RANO — ориентир)
  let bestK = -1, bestA = 0;
  for (let k = bb[2]; k <= bb[5]; k++) { let a = 0; for (let j = bb[1]; j <= bb[4]; j++) for (let i = bb[0]; i <= bb[3]; i++) if (enhM[i + nx * (j + ny * k)]) a++; if (a > bestA) { bestA = a; bestK = k; } }
  if (bestK >= 0 && bestA > 30) {
    const pts = []; for (let j = bb[1]; j <= bb[4]; j++) for (let i = bb[0]; i <= bb[3]; i++) if (enhM[i + nx * (j + ny * bestK)]) pts.push([i, j, bestK]);
    const f = farthestPair(pts, sp);
    const dx = (f.b[0] - f.a[0]) * sp[0], dy = (f.b[1] - f.a[1]) * sp[1]; const L = Math.hypot(dx, dy) || 1; const px = -dy / L, py = dx / L;
    let mn = 1e9, mx = -1e9; for (const p of pts) { const t = (p[0] * sp[0]) * px + (p[1] * sp[1]) * py; if (t < mn) mn = t; if (t > mx) mx = t; }
    out.bidim = { slice: bestK, d1: f.d, d2: mx - mn, area: bestA * sp[0] * sp[1] / 100 };
  }
  // очаги
  const comps = components(core, nx, ny, nz, Math.round(100 / (sp[0] * sp[1] * sp[2])));
  out.foci = comps.length; out.comps = comps.map((c) => ({ ...c, cm3: c.vox * vv }));
  // интенсивности T1-Gd (усл. ед., нормировка по серии)
  if (vols?.t1ce) {
    let s = 0, n = 0, sb = 0, nb = 0; const d = vols.t1ce.data;
    for (let c = 0; c < seg.length; c++) { if (seg[c] === 3) { s += d[c]; n++; } else if (d[c] > 0 && !seg[c]) { sb += d[c]; nb++; } }
    out.enhRatio = n && nb ? (s / n) / (sb / nb) : null;
  }
  // локализация (ориентировочно, по MNI)
  if (mni) {
    const [x, y, z] = mni; const side = x > 4 ? 'правое' : x < -4 ? 'левое' : 'срединное';
    let lobe = 'лобная'; if (y < -72) lobe = 'затылочная'; else if (y < -18 && z < 12) lobe = 'височная'; else if (y < -18) lobe = 'теменная'; else if (y < 8 && z < -2) lobe = 'височная';
    out.loc = { side, lobe, text: `${side} полушарие, ${lobe} доля (ориентировочно, по MNI ${mni.map((v) => Math.round(v)).join(', ')})` };
  }
  return out;
}

/* ---------- Игрушечная модель ответа на лечение (ДЕМО) ---------- */
export function effectStrength(p) {
  const dose = p.dose || 0; const frac = Math.max(1, p.fractions || 30);
  const rt = Math.pow(dose / 60, 1.4) * (1 + 0.04 * (30 - Math.abs(frac - 30)) / 30);
  const tmz = p.tmz ? (p.mgmt ? 1.35 : 1.12) : 1.0; const bev = p.bev ? 1.08 : 1.0;
  const base = p.horizon >= 6 ? 4.6 : 2.6;
  return Math.max(0, base * (dose > 0 ? rt : 0.25) * tmz * bev);
}
export function simulateMask(seg, depthAll, depthEnh, dims, e, seed = 7) {
  const out = new Uint8Array(seg.length); const n = seg.length;
  for (let c = 0; c < n; c++) {
    const l = seg[c]; if (!l) continue;
    let h = (c * 2654435761 + seed * 40503) >>> 0; h ^= h >>> 15; h = Math.imul(h, 2246822519) >>> 0; h ^= h >>> 13;
    const noise = ((h & 255) / 255 - 0.5) * 0.9;
    const da = depthAll[c] / 4, dh = depthEnh[c] / 4;
    if (l === 3) out[c] = dh + noise > 0.35 * e ? 3 : (da + noise > 0.12 * e ? 1 : 0);
    else if (l === 1) out[c] = da + noise > 0.12 * e ? 1 : 0;
    else out[c] = da + noise > 0.75 * e ? 2 : 0;
  }
  return out;
}
export function countVols(mask, spacing) {
  const vv = (spacing[0] * spacing[1] * spacing[2]) / 1000; const c = [0, 0, 0, 0];
  for (let i = 0; i < mask.length; i++) c[mask[i]]++;
  const V = { nec: c[1] * vv, ede: c[2] * vv, enh: c[3] * vv }; V.whole = V.nec + V.ede + V.enh; return V;
}

/* ---------- Варианты терапии (сценарии для рассмотрения) ---------- */
export const SOURCES = {
  stupp2005: { t: 'Stupp R. et al. Radiotherapy plus concomitant and adjuvant temozolomide for glioblastoma. N Engl J Med 2005;352:987–996', u: 'https://doi.org/10.1056/NEJMoa043330', y: 2005 },
  perry2017: { t: 'Perry J.R. et al. Short-course radiation plus temozolomide in elderly patients with glioblastoma. N Engl J Med 2017;376:1027–1037', u: 'https://doi.org/10.1056/NEJMoa1611977', y: 2017 },
  stupp2017: { t: 'Stupp R. et al. Effect of tumor-treating fields plus maintenance temozolomide vs maintenance temozolomide alone on survival in glioblastoma. JAMA 2017;318:2306–2316', u: 'https://doi.org/10.1001/jama.2017.18718', y: 2017 },
  herrlinger2019: { t: 'Herrlinger U. et al. Lomustine-temozolomide combination therapy versus standard temozolomide therapy in patients with newly diagnosed glioblastoma with methylated MGMT promoter (CeTeG/NOA-09). Lancet 2019;393:678–688', u: 'https://doi.org/10.1016/S0140-6736(18)31791-4', y: 2019 },
  weller2021: { t: 'Weller M. et al. EANO guidelines on the diagnosis and treatment of diffuse gliomas of adulthood. Nat Rev Clin Oncol 2021;18:170–186', u: 'https://doi.org/10.1038/s41571-020-00447-z', y: 2021 },
  rano2010: { t: 'Wen P.Y. et al. Updated response assessment criteria for high-grade gliomas: RANO working group. J Clin Oncol 2010;28:1963–1972', u: 'https://doi.org/10.1200/JCO.2009.26.3541', y: 2010 },
  dataset: { t: 'Filimonova E. et al. Real-world multi-institutional glioblastoma MRI dataset (OpenNeuro ds007045, CC0). Sci Data 2026;13:1213', u: 'https://doi.org/10.18112/openneuro.ds007045.v2.0.1', y: 2026 },
};

export function buildTherapy(ctx) {
  const { age, mgmt, kps } = ctx; const elderly = age >= 65; const good = kps >= 70;
  const S = [];
  S.push({
    id: 'stupp', title: 'Хирургия по возможности + химиолучевая терапия по схеме Stupp',
    params: 'ЛТ 60 Гр / 30 фракций + темозоломид 75 мг/м² одновременно; далее адъювантный ТМЗ до 6 циклов',
    fit: elderly ? 1 : 3, reasons: [`Возраст ${age} лет`, mgmt ? 'MGMT метилирован — ожидаемо лучшая чувствительность к алкилирующей терапии' : 'MGMT не метилирован — польза ТМЗ ограничена', good ? `Общее состояние (KPS ${kps}) допускает стандартный курс` : `KPS ${kps} — требуется оценка переносимости`],
    axes: { control: elderly ? 2 : 3, reliab: 3, data: 3, applic: elderly ? 2 : 3 }, risks: 'Гематологическая токсичность ТМЗ, лучевые изменения, псевдопрогрессия в первые 3 мес.', src: ['stupp2005', 'weller2021'],
  });
  S.push({
    id: 'hypo', title: 'Гипофракционная ЛТ + темозоломид',
    params: 'ЛТ 40 Гр / 15 фракций + ТМЗ 75 мг/м²; адъювантный ТМЗ',
    fit: elderly || !good ? 3 : 1, reasons: [elderly ? 'Возраст ≥ 65 — исследованная популяция' : 'Рассматривается при сниженном статусе или необходимости сокращения курса', 'Укороченный курс — меньше визитов'],
    axes: { control: 2, reliab: elderly ? 3 : 2, data: 3, applic: elderly ? 3 : 1 }, risks: 'Оценка неопределённа вне возрастной группы исследования; вклад MGMT нужно уточнить.', src: ['perry2017', 'weller2021'],
  });
  S.push({
    id: 'ttf', title: 'Стандартная схема + поля лечения опухолей (TTFields)',
    params: 'Адъювантный ТМЗ + TTFields ≥ 18 ч/сут после завершения химиолучевой терапии',
    fit: good ? 2 : 1, reasons: ['Добавляется к адъювантной фазе', good ? 'Требуется приверженность лечению (носимое устройство)' : 'Приверженность может быть затруднена при низком KPS'],
    axes: { control: 2, reliab: 2, data: 2, applic: good ? 2 : 1 }, risks: 'Дерматологические реакции, доступность метода, влияние на качество жизни.', src: ['stupp2017'],
  });
  S.push({
    id: 'cetg', title: 'Ломустин + темозоломид (CeTeG) при метилированном MGMT',
    params: 'Ломустин + ТМЗ параллельно с ЛТ по протоколу NOA-09',
    fit: mgmt && good && !elderly ? 2 : 0, reasons: [mgmt ? 'MGMT метилирован — соответствует критерию включения исследования' : 'MGMT не метилирован — вне области применимости', 'Ограничение по возрасту и статусу в исследовании'],
    axes: { control: mgmt ? 3 : 1, reliab: 1, data: mgmt ? 2 : 1, applic: mgmt && !elderly ? 2 : 0 }, risks: 'Выраженная миелотоксичность; данные ограничены одним открытым исследованием.', src: ['herrlinger2019'],
  });
  S.push({
    id: 'palliative', title: 'Поддерживающая тактика и паллиативная ЛТ',
    params: 'Индивидуальный курс ЛТ (напр. 25 Гр / 5 фракций), симптоматическая терапия',
    fit: !good ? 3 : 0, reasons: [!good ? `KPS ${kps} — приоритет качества жизни` : 'Рассматривается, если стандартное лечение неприемлемо'],
    axes: { control: 1, reliab: 2, data: 2, applic: !good ? 3 : 0 }, risks: 'Меньший ожидаемый контроль опухоли; решение принимается совместно с пациентом.', src: ['weller2021'],
  });
  return S.sort((a, b) => b.fit - a.fit || b.axes.control - a.axes.control).slice(0, 5);
}

/* ---------- Текст заключения ---------- */
export function buildReport(st) {
  const m = st.metrics, f = st.follow; const p = st.patient; const date = new Date().toLocaleDateString('ru-RU');
  const V = m.V;
  const lines = [];
  lines.push(`ЗАКЛЮЧЕНИЕ (ЧЕРНОВИК — требует проверки и подписи врача)\nДемонстрационный случай: ${p.id}. Дата формирования: ${date}.`);
  lines.push(`1. Исследование и клинические сведения\nМРТ головного мозга. ${p.age} лет. Глиобластома (демонстрационные сведения); MGMT: ${p.mgmt ? 'метилирован' : 'не метилирован'}. Магнитное поле ${p.field || '—'} Тл, ${p.scanner || 'аппарат не указан'}. Источник данных: OpenNeuro ds007045, обезличено.`);
  lines.push(`2. Методика\nДоступны последовательности T1, T1 с контрастом (Gd), T2, T2-FLAIR; данные приведены к пространству MNI152, выполнено удаление костных структур. Диффузионные и перфузионные серии отсутствуют.`);
  lines.push(`3. Локализация и характеристики очага\n${m.loc ? m.loc.text : 'Локализация не определена'}. Число самостоятельных очагов (объём > 0,1 см³): ${m.foci}. Максимальный размер опухолевой области ${nf(m.diam, 0)} мм${m.bidim ? `; бидименсиональные размеры активной зоны на аксиальном срезе: ${nf(m.bidim.d1, 0)} × ${nf(m.bidim.d2, 0)} мм` : ''}.`);
  lines.push(`4. Количественные показатели (${st.maskEdited ? 'после ручной коррекции' : 'автоматическая сегментация'})\n• Вся опухолевая область: ${nf(V.whole)} см³\n• Активная зона: ${nf(V.enh)} см³ (${nf(m.share.enh * 100, 0)} %)\n• Некроз / неусиливающееся ядро: ${nf(V.nec)} см³ (${nf(m.share.nec * 100, 0)} %)\n• FLAIR-гиперинтенсивная зона (отёк): ${nf(V.ede)} см³ (${nf(m.share.ede * 100, 0)} %)`);
  if (f) lines.push(`5. Сравнение\n[СИНТЕТИЧЕСКАЯ ДЕМО-ДИНАМИКА, не данные пациента] Относительно исходного исследования объём активной зоны ${sgn(pct(f.V.enh, V.enh))} %, всей области ${sgn(pct(f.V.whole, V.whole))} %, некроза ${sgn(pct(f.V.nec, V.nec))} %. Возможны посттерапевтические изменения; требуется клиническая корреляция.`);
  else lines.push(`5. Сравнение\nПредыдущие исследования отсутствуют — сравнение не проводилось.`);
  lines.push(`6. Оценка динамики по критериям\nОценка по RANO ограничена: нет сведений о приёме кортикостероидов и клиническом статусе на дату исследования.`);
  lines.push(`7. Заключение\nМРТ-картина объёмного образования с усиливающейся, некротической и перифокальной отёчной зонами, соответствующая демонстрационному диагнозу. Требуется корреляция с клиническими данными и решение врача.`);
  lines.push(`8. Ограничения\nСегментация выполнена нейросетевой моделью (SegResNet, BraTS-разметка) и в датасете проверена нейрорадиологами; в приложении статус: ${st.confirmed ? 'подтверждено врачом' : 'требуется проверка врача'}. Результат не заменяет клиническое решение.`);
  return lines.join('\n\n');
}

/* ---------- ИИ-помощник ---------- */
export function aiAnswer(q, st) {
  const t = q.toLowerCase(); const m = st.metrics; const p = st.patient; const V = m?.V; const ac = [];
  const has = !!m && st.segDone;
  const src = (ks) => ks.map((k) => SOURCES[k]);
  if (/истори|суммир/.test(t)) return { text: `**Сводка по случаю ${p.id}.** ${p.age} лет, глиобластома, MGMT ${p.mgmt ? 'метилирован' : 'не метилирован'}. Доступно дооперационное МРТ (T1, T1-Gd, T2, FLAIR; ${p.field || '?'} Тл). ${has ? `По маске: вся область ${nf(V.whole)} см³, активная зона ${nf(V.enh)} см³, некроз ${nf(V.nec)} см³, отёк ${nf(V.ede)} см³.` : 'Сегментация ещё не запускалась.'}\n_Замечание:_ операция, лучевая и системная терапия в этом наборе данных отсутствуют — ${'в демо они показаны как незаполненные этапы, а не как факты.'}`, ctx: ['Карточка пациента', 'Серии'] };
  if (/сравн/.test(t) && !/план|лечен/.test(t)) return has && st.follow ? { text: `**Сравнение (демо-динамика).** Объём активной зоны: ${nf(V.enh)} → ${nf(st.follow.V.enh)} см³ (${sgn(pct(st.follow.V.enh, V.enh))} %). Вся область: ${nf(V.whole)} → ${nf(st.follow.V.whole)} см³ (${sgn(pct(st.follow.V.whole, V.whole))} %). Некроз: ${sgn(pct(st.follow.V.nec, V.nec))} %.\n⚠ Контрольная точка **синтетическая** — построена демонстрационной моделью по маске базового исследования, а не по реальному повторному МРТ.`, ctx: ['Показатели', 'Сравнение'], actions: [{ label: 'Открыть сравнение', act: 'compare' }] } : { text: 'Для сравнения сначала запустите сегментацию: нужны маски базовой и контрольной точек.', ctx: [], actions: [{ label: 'Запустить сегментацию', act: 'segment' }] };
  if (/рост|крупнейш|очаг/.test(t)) { if (!has) return { text: 'Сегментация ещё не выполнена — показывать пока нечего.', ctx: [], actions: [{ label: 'Запустить сегментацию', act: 'segment' }] }; const c = m.comps?.[0]; return { text: `Крупнейший очаг содержит ${nf(c?.cm3 ?? V.core)} см³ (без отёка); всего очагов: ${m.foci}. ${m.loc?.text ?? ''}. Показать его центр в срезах и 3D?`, ctx: ['Маска', 'Показатели'], actions: [{ label: 'Перейти к очагу', act: 'goto-largest' }] }; }
  if (/неопредел|уверен/.test(t)) return { text: 'В этой версии модель **не выдаёт карту неопределённости**, поэтому слой «неопределённая зона» не показан. Обычно границу отёка (FLAIR) и переход между активной опухолью и некрозом стоит проверить вручную — там маска наиболее чувствительна к качеству контраста и артефактам. Используйте кисть и «Проверить маску».', ctx: ['Сегментация'], actions: [{ label: 'Перейти к проверке маски', act: 'tab-seg' }] };
  if (/заключ|черновик/.test(t)) return has ? { text: 'Готовлю черновик заключения из текущих показателей. Он будет помечен как «Черновик», каждое число связано с маской.', ctx: ['Показатели', 'Маска'], actions: [{ label: 'Открыть редактор заключения', act: 'report' }] } : { text: 'Нужны результаты сегментации, чтобы подставить показатели.', ctx: [], actions: [{ label: 'Запустить сегментацию', act: 'segment' }] };
  if (/источник|руковод|литератур|найди/.test(t)) return { text: 'Ключевые источники по первичной глиобластоме (даты и версии указаны). Литературные данные относятся к популяциям исследований, а не к этому пациенту.', sources: src(['weller2021', 'stupp2005', 'rano2010', 'perry2017', 'herrlinger2019']), ctx: ['Контекст: диагноз, MGMT, возраст'] };
  if (/консилиум|сводк/.test(t)) return has ? { text: `**Сводка для консилиума (черновик).** Случай ${p.id}, ${p.age} лет, глиобластома, MGMT ${p.mgmt ? '+' : '−'}. Опухолевая область ${nf(V.whole)} см³ (активная ${nf(V.enh)}, некроз ${nf(V.nec)}, отёк ${nf(V.ede)}), ${m.foci} очаг(ов), ${m.loc?.text ?? ''}. ${st.follow ? `Демо-динамика: активная зона ${sgn(pct(st.follow.V.enh, V.enh))} %.` : ''} Вопросы: тактика (хирургия/химиолучевая терапия), необходимость уточняющих серий (перфузия). Статус маски: ${st.confirmed ? 'подтверждена врачом' : 'не подтверждена'}.`, ctx: ['Показатели', 'История', 'Лечение'] } : { text: 'Сначала выполните сегментацию — сводка строится по показателям.', actions: [{ label: 'Запустить сегментацию', act: 'segment' }] };
  if (/не хватает|данн|провер/.test(t)) return { text: `**Чего не хватает для полного анализа:**\n• Перфузия (rCBV) и DWI/ADC — серий нет;\n• KPS / ECOG — не указан в карточке;\n• Послеоперационные и контрольные МРТ — в датасете только дооперационные;\n• Данные о резекции и дозе ЛТ.\nМодель ответа на лечение работает как демонстрационный расчёт.`, ctx: ['Серии', 'Карточка пациента'] };
  if (/3d|активн.*опух/.test(t) && /откр|покаж/.test(t)) return { text: 'Включаю только активную опухоль в 3D и сворачиваю остальные слои.', actions: [{ label: 'Показать', act: '3d-enh' }], ctx: ['3D'] };
  return { text: 'Я могу: подвести сводку по пациенту, сравнить исследования, показать крупнейший очаг, подготовить сводку для консилиума и заключение, найти источники. Уточните вопрос или используйте быстрые команды ниже. Я не ставлю диагноз и не назначаю лечение.', ctx: [] };
}
