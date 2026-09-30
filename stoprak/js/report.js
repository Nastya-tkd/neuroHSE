// Редактор заключения: слева — данные и источники, справа — редактируемый черновик; версии и подпись.
import { S, app } from './state.js';
import * as M from './model.js';
import { ic, $, $$, esc, toast, openModal, confirmDialog } from './ui.js';

export function openReport() {
  if (S.mode !== 'demo') return toast('Для загруженного файла заключение недоступно: нет расчётных показателей', 'warn');
  if (!S.segDone) { toast('Для заключения нужны результаты сегментации', 'warn'); app.openTab('seg'); return; }
  if (!S.reportText) S.reportText = M.buildReport({ ...S, follow: S.follow?.metrics ? { V: S.follow.metrics.V } : null, metrics: S.baseMetrics });
  S.reportVersions = S.reportVersions || [{ n: 1, who: 'Автоматический черновик', when: new Date() }];
  render(); openModal('reportModal');
}
function render() {
  const m = S.baseMetrics; const b = $('#reportBody');
  const src = (label, val, cls) => `<div class="mrow"><span class="n">${label}</span><span class="val">${val}</span><button class="btn btn-sm btn-ghost" data-src="${cls}" title="Показать источник в срезах и 3D">Показать источник</button></div>`;
  b.innerHTML = `<div class="report-l">
    <div class="panel-title" style="margin-bottom:10px">Данные анализа</div>
    ${src('Вся область', M.nf(m.V.whole) + ' см³', 0)}${src('Активная зона', M.nf(m.V.enh) + ' см³', 3)}${src('Некроз', M.nf(m.V.nec) + ' см³', 1)}${src('Отёк (FLAIR)', M.nf(m.V.ede) + ' см³', 2)}
    <div class="kv" style="margin-top:8px"><span>Макс. диаметр</span><b>${M.nf(m.diam, 0)} мм</b></div><div class="kv"><span>Очагов</span><b>${m.foci}</b></div><div class="kv"><span>Локализация</span><b>${esc(m.loc?.side + ' · ' + m.loc?.lobe)}</b></div>
    <div class="kv"><span>Маска</span><b>${S.confirmed ? '<span class="down">подтверждена</span>' : '<span style="color:var(--warn)">не подтверждена</span>'}${S.maskEdited ? ' · правки' : ''}</b></div>
    <div class="panel-title" style="margin:18px 0 8px">Шаблон</div>
    <select class="sel" id="rpTpl"><option>Стандартный шаблон учреждения</option><option>Краткий</option><option>Личный шаблон врача</option></select>
    <p class="fine" style="margin-top:6px">Автоподставленные числа связаны с маской: «Показать источник» подсвечивает область.</p>
    <div class="panel-title" style="margin:18px 0 8px">Версии</div>
    <div class="vlist">${S.reportVersions.map((v) => `<div class="vitem"><div><b>v${v.n} · ${esc(v.who)}</b><div class="fine">${v.when.toLocaleTimeString('ru-RU')}</div></div></div>`).join('')}</div></div>
  <div class="report-r">
    ${S.reportStale ? `<div class="stale"><b>Изменение маски пересчитало показатели.</b> Числа в черновике могут быть устаревшими — обновите из данных.</div>` : ''}
    <div class="hint warn" style="margin:0">${ic('warn')}<div>Автоматически созданный текст всегда имеет статус «Черновик». Врач обязан просмотреть и подписать документ.</div></div>
    <textarea class="txt" id="rpText" aria-label="Текст заключения" spellcheck="false">${esc(S.reportText)}</textarea>
    <div class="row" style="flex-wrap:wrap;gap:8px"><button class="btn btn-sm" id="rpRebuild">Обновить из данных</button><button class="btn btn-sm" id="rpSave">Сохранить черновик</button><button class="btn btn-sm btn-ghost" id="rpCopy">Копировать</button><button class="btn btn-sm btn-ghost" id="rpExport">${ic('download', 14)} TXT</button><button class="btn btn-sm btn-ghost" onclick="window.print()">Печать</button><span style="flex:1"></span><button class="btn btn-primary" id="rpSign" ${S.reportSigned ? 'disabled' : ''}>${S.reportSigned ? 'Подписано' : 'Подписать'}</button></div>
  </div>`;
  const st = $('#rpStatus'); st.className = 'badge ' + (S.reportSigned ? 'badge-ok' : 'badge-warn'); st.textContent = S.reportSigned ? 'Подписано' : 'Черновик';
  $$('[data-src]', b).forEach((x) => (x.onclick = () => { $('#reportModal').hidden = true; const c = +x.dataset.src; app.setHi(c || 3, true); if (S.metrics?.centroid && c === 0) { /* вся область */ } }));
  const ta = $('#rpText'); ta.oninput = () => { S.reportText = ta.value; if (S.reportSigned) { S.reportSigned = false; app.updateBadge(); render(); $('#rpText').focus(); } };
  $('#rpRebuild').onclick = () => { S.reportText = M.buildReport({ ...S, follow: S.follow?.metrics ? { V: S.follow.metrics.V } : null, metrics: S.baseMetrics }); S.reportStale = false; S.reportSigned = false; S.reportVersions.push({ n: S.reportVersions.length + 1, who: 'Обновлено из данных', when: new Date() }); render(); toast('Черновик обновлён по текущим показателям', 'ok'); };
  $('#rpSave').onclick = () => { S.reportVersions.push({ n: S.reportVersions.length + 1, who: 'Правка врача (черновик)', when: new Date() }); render(); toast('Черновик сохранён как версия v' + S.reportVersions.length, 'ok'); };
  $('#rpCopy').onclick = () => { navigator.clipboard?.writeText(ta.value); toast('Текст скопирован', 'ok'); };
  $('#rpExport').onclick = () => { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([ta.value], { type: 'text/plain;charset=utf-8' })); a.download = 'zaklyuchenie.txt'; a.click(); };
  $('#rpSign').onclick = async () => {
    if (!S.confirmed) return toast('Сначала подтвердите сегментацию на вкладке «Сегментация»', 'warn');
    const ok = await confirmDialog({ title: 'Подписать заключение', html: 'Вы просмотрели документ и подписываете его как врач. Демонстрационная подпись не имеет юридической силы.', ok: 'Подписать' }); if (!ok) return;
    S.reportSigned = true; S.reportText = S.reportText.replace('(ЧЕРНОВИК — требует проверки и подписи врача)', '(ПОДПИСАНО)') + `\n\n— Подписано демо-пользователем, ${new Date().toLocaleString('ru-RU')} (демонстрационная подпись).`;
    S.reportVersions.push({ n: S.reportVersions.length + 1, who: 'Подписано врачом', when: new Date() }); app.updateBadge(); render(); toast('Заключение подписано (демо)', 'ok');
  };
}
