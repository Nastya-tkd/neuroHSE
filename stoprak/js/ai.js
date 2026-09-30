// ИИ-помощник: выдвижная панель, контекст, быстрые команды, ответы по данным открытого случая (демо, без внешних вызовов)
import { S, app } from './state.js';
import * as M from './model.js';
import { ic, $, $$, esc, md, toast } from './ui.js';

const QUICK = ['Суммируй историю пациента', 'Сравни текущую МРТ с предыдущей', 'Покажи крупнейшую зону роста', 'Почему система считает этот слой неопределённым?', 'Создай черновик заключения', 'Найди актуальные медицинские источники по этому вопросу', 'Подготовь краткую сводку для консилиума', 'Проверь, каких данных не хватает для анализа', 'Открой активную опухоль в 3D'];
let busy = false;

export function initAI() {
  $('#btnAI').onclick = () => toggle();
  $('#aiClose').onclick = () => toggle(false);
  $('#aiQuick').innerHTML = QUICK.map((q) => `<button class="chip" type="button">${esc(q)}</button>`).join('');
  $('#aiQuick').onclick = (e) => { const b = e.target.closest('.chip'); if (b) ask(b.textContent); };
  $('#aiForm').onsubmit = (e) => { e.preventDefault(); const v = $('#aiInput').value.trim(); if (v) { $('#aiInput').value = ''; ask(v); } };
  app.toggleAI = toggle; app.aiReset = reset; app.aiNote = (t) => post('a', `${esc(t)}`);
  reset();
}
function toggle(force) {
  const d = $('#aiDrawer'); const open = force ?? !d.classList.contains('is-open'); d.classList.toggle('is-open', open);
  if (open) { ctx(); setTimeout(() => $('#aiInput').focus(), 250); }
}
function ctx() {
  const has = (b) => (b ? '<span class="chip is-active">' : '<span class="chip" style="opacity:.5">');
  const demo = S.mode === 'demo';
  $('#aiCtx').innerHTML = `<div class="lbl">Контекст, доступный помощнику</div>
    ${has(demo)}Карточка пациента</span>${has(true)}Серии (${Object.keys(S.vols).length})</span>${has(S.segDone)}Маска и показатели</span>${has(!!S.follow)}Сравнение</span>${has(!!S.simInfo)}Сценарий</span>
    <div class="fine" style="width:100%">Персональные данные не покидают браузер: в демо-режиме ответы формируются локально.</div>`;
}
function reset() {
  $('#aiLog').innerHTML = ''; ctx();
  post('a', `Здравствуйте! Я помощник в контексте открытого случая${S.mode === 'demo' ? ' ' + esc(S.patient.id) : ''}. Могу суммировать данные, сравнивать исследования, выполнять действия на сайте (после вашего подтверждения) и находить источники.<div class="ctxline">Я не ставлю диагноз и не назначаю лечение.</div>`, true);
}
function post(who, html, raw) {
  const log = $('#aiLog'); const m = document.createElement('div'); m.className = 'msg ' + who; m.innerHTML = raw || who === 'u' ? html : html; log.appendChild(m); log.scrollTop = log.scrollHeight; return m;
}
async function ask(q) {
  if (busy) return; busy = true; $('#aiDrawer').classList.add('is-open'); ctx();
  post('u', esc(q)); const t = post('a', '<div class="typing"><i></i><i></i><i></i></div>');
  await new Promise((r) => setTimeout(r, document.body.classList.contains('no-anim') ? 20 : 650));
  const a = M.aiAnswer(q, S); let h = md(a.text);
  if (a.ctx?.length) h += `<div class="ctxline">Использованный контекст: ${a.ctx.join(' · ')}</div>`;
  if (a.sources) h += `<div class="refs">${a.sources.map((s) => `<a href="${s.u}" target="_blank" rel="noopener">${esc(s.t)}<small>${s.y} · открыть источник ↗</small></a>`).join('')}</div>`;
  if (a.actions) h += `<div class="actions">${a.actions.map((x) => `<button class="btn btn-sm btn-primary" data-act="${x.act}">${esc(x.label)}</button>`).join('')}</div>`;
  h += `<div class="fb"><button data-fb="up">полезно</button><button data-fb="err">ошибка</button></div>`;
  t.innerHTML = h;
  t.querySelectorAll('[data-act]').forEach((b) => (b.onclick = () => { app.doAction(b.dataset.act); b.disabled = true; }));
  t.querySelectorAll('[data-fb]').forEach((b) => (b.onclick = () => { toast(b.dataset.fb === 'up' ? 'Спасибо, отметка «полезно» сохранена' : 'Отметка об ошибке сохранена — используется для проверки ответов', 'ok'); t.querySelector('.fb').remove(); }));
  $('#aiLog').scrollTop = $('#aiLog').scrollHeight; busy = false;
}
