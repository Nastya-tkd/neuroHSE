// Иконки и небольшие UI-утилиты
const P = {
  back: 'M19 12H5M12 5l-7 7 7 7',
  ai: 'M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9zM19 16l.9 2.1L22 19l-2.1.9L19 22l-.9-2.1L16 19l2.1-.9z',
  gear: 'M12 15a3 3 0 100-6 3 3 0 000 6zM19.4 15a1.7 1.7 0 00.3 1.8l.1.1a2 2 0 11-2.8 2.8l-.1-.1a1.7 1.7 0 00-1.8-.3 1.7 1.7 0 00-1 1.5V21a2 2 0 11-4 0v-.1a1.7 1.7 0 00-1.1-1.5 1.7 1.7 0 00-1.8.3l-.1.1a2 2 0 11-2.8-2.8l.1-.1a1.7 1.7 0 00.3-1.8 1.7 1.7 0 00-1.5-1H3a2 2 0 110-4h.1a1.7 1.7 0 001.5-1.1 1.7 1.7 0 00-.3-1.8l-.1-.1a2 2 0 112.8-2.8l.1.1a1.7 1.7 0 001.8.3H9a1.7 1.7 0 001-1.5V3a2 2 0 114 0v.1a1.7 1.7 0 001 1.5 1.7 1.7 0 001.8-.3l.1-.1a2 2 0 112.8 2.8l-.1.1a1.7 1.7 0 00-.3 1.8V9a1.7 1.7 0 001.5 1H21a2 2 0 110 4h-.1a1.7 1.7 0 00-1.5 1z',
  x: 'M18 6L6 18M6 6l12 12',
  cross: 'M12 2v6M12 16v6M2 12h6M16 12h6M12 11a1 1 0 100 2 1 1 0 000-2z',
  pan: 'M5 9l-3 3 3 3M9 5l3-3 3 3M15 19l-3 3-3-3M19 9l3 3-3 3M2 12h20M12 2v20',
  zoom: 'M11 19a8 8 0 100-16 8 8 0 000 16zM21 21l-4.3-4.3M11 8v6M8 11h6',
  wl: 'M12 22a10 10 0 100-20 10 10 0 000 20zM12 2v20M12 2a10 10 0 010 20z',
  ruler: 'M3 17L17 3l4 4L7 21zM7 13l2 2M10 10l2 2M13 7l2 2',
  roi: 'M12 5c5 0 9 3 9 7s-4 7-9 7-9-3-9-7 4-7 9-7z',
  brush: 'M9.06 11.9l8.07-8.06a2.85 2.85 0 114.03 4.03l-8.06 8.08M7.07 14.94c-1.66 0-3 1.35-3 3.02 0 1.33-2.5 1.52-2 2.02 1.08 1.1 2.49 2.02 4 2.02 2.2 0 4-1.8 4-4.04a3.01 3.01 0 00-3-3.02z',
  eraser: 'M20 20H9L4 15a2 2 0 010-2.8l8-8a2 2 0 012.8 0l5.2 5.2a2 2 0 010 2.8L12 20M9 20l-4-4',
  cc: 'M12 22a10 10 0 100-20 10 10 0 000 20zM8 12h8',
  undo: 'M3 7v6h6M21 17a9 9 0 00-15-6.7L3 13',
  redo: 'M21 7v6h-6M3 17a9 9 0 0115-6.7L21 13',
  invert: 'M12 22a10 10 0 100-20 10 10 0 000 20zM12 2v20M12 2a10 10 0 010 20',
  reset: 'M3 12a9 9 0 109-9 9.75 9.75 0 00-6.7 2.7L3 8M3 3v5h5',
  camera: 'M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2zM12 17a4 4 0 100-8 4 4 0 000 8z',
  eyeoff: 'M17.9 17.9A10.1 10.1 0 0112 20c-7 0-11-8-11-8a18.5 18.5 0 015.1-5.9M9.9 4.2A9.1 9.1 0 0112 4c7 0 11 8 11 8a18.5 18.5 0 01-2.2 3.2M1 1l22 22',
  eye: 'M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8zM12 15a3 3 0 100-6 3 3 0 000 6z',
  quad: 'M3 3h8v8H3zM13 3h8v8h-8zM3 13h8v8H3zM13 13h8v8h-8z',
  single: 'M3 3h18v18H3z',
  compare: 'M3 4h8v16H3zM13 4h8v16h-8z',
  cube: 'M21 16V8l-9-5-9 5v8l9 5 9-5zM3.3 7L12 12l8.7-5M12 22V12',
  play: 'M6 4l14 8-14 8z',
  pause: 'M8 5v14M16 5v14',
  doc: 'M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8zM14 2v6h6M8 13h8M8 17h6',
  download: 'M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3',
  upload: 'M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12',
  folder: 'M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2z',
  max: 'M8 3H5a2 2 0 00-2 2v3M21 8V5a2 2 0 00-2-2h-3M3 16v3a2 2 0 002 2h3M16 21h3a2 2 0 002-2v-3',
  info: 'M12 22a10 10 0 100-20 10 10 0 000 20zM12 16v-4M12 8h.01',
  warn: 'M12 9v4M12 17h.01M10.3 3.9L2.4 18a2 2 0 001.7 3h15.8a2 2 0 001.7-3L13.7 3.9a2 2 0 00-3.4 0z',
  lock: 'M5 11h14v10H5zM8 11V7a4 4 0 118 0v4',
  check: 'M20 6L9 17l-5-5',
  chev: 'M9 18l6-6-6-6',
  trend: 'M3 17l5-6 4 4 8-10M14 5h6v6',
  list: 'M4 6h16M4 12h10M4 18h6',
  rot: 'M21 12a9 9 0 11-3-6.7L21 8M21 3v5h-5',
  layers: 'M12 2l10 5-10 5L2 7zM2 17l10 5 10-5M2 12l10 5 10-5',
  home: 'M3 11l9-8 9 8M5 10v10h5v-6h4v6h5V10',
  send: 'M22 2L11 13M22 2l-7 20-4-9-9-4z',
};
export const ic = (n, s = 17) => `<svg viewBox="0 0 24 24" width="${s}" height="${s}" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="${P[n] || ''}"/></svg>`;
export const $ = (s, r = document) => r.querySelector(s);
export const $$ = (s, r = document) => [...r.querySelectorAll(s)];
export const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
export const md = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/_(.+?)_/g, '<i>$1</i>');
export const rangeFill = (el) => { const p = ((el.value - el.min) / (el.max - el.min)) * 100; el.style.setProperty('--p', p + '%'); };

export function toast(text, kind = '', action) {
  const box = $('#toasts'); const t = document.createElement('div'); t.className = 'toast ' + kind;
  t.innerHTML = `<span>${esc(text)}</span>`;
  if (action) { const b = document.createElement('button'); b.textContent = action.label; b.onclick = () => { action.fn(); t.remove(); }; t.appendChild(b); }
  box.appendChild(t); setTimeout(() => t.remove(), action ? 7000 : 4200);
  while (box.children.length > 3) box.firstChild.remove();
}
export function openModal(id) { const m = $('#' + id); m.hidden = false; m.querySelector('button,input,textarea')?.focus?.(); return m; }
export function closeModal(id) { $('#' + id).hidden = true; }
export function confirmDialog({ title, html, ok = 'Подтвердить', cancel = 'Отмена', danger = false }) {
  return new Promise((res) => {
    $('#cfTitle').textContent = title;
    $('#cfBody').innerHTML = `<div class="cf-text">${html}</div><div class="row" style="justify-content:flex-end;margin-top:18px"><button class="btn btn-ghost" id="cfNo">${esc(cancel)}</button><button class="btn ${danger ? 'btn-danger' : 'btn-primary'}" id="cfYes">${esc(ok)}</button></div>`;
    const m = openModal('confirmModal'); const done = (v) => { closeModal('confirmModal'); res(v); };
    $('#cfNo').onclick = () => done(false); $('#cfYes').onclick = () => done(true); m._cancel = () => done(false);
  });
}
