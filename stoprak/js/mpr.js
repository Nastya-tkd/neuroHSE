// 2D-просмотрщик срезов (MPR): аксиальная / сагиттальная / корональная плоскости на canvas.
// Настоящая работа с воксельными данными: окно/уровень, наложение масок, кисть, линейка, ROI, синхронный курсор.
export const RGB = { 1: [0xF5, 0xA6, 0x23], 2: [0x38, 0xBD, 0xF8], 3: [0xFF, 0x5B, 0x61] };
const PLANES = {
  axial: { name: 'Аксиальная', ax: 2, letters: ['R', 'L', 'A', 'P'] },
  coronal: { name: 'Корональная', ax: 1, letters: ['R', 'L', 'S', 'I'] },
  sagittal: { name: 'Сагиттальная', ax: 0, letters: ['A', 'P', 'S', 'I'] },
};
const LETTERS = { R: 'R', L: 'L', A: 'A', P: 'P', S: 'S', I: 'I' };

export function planeGeom(plane, v) {
  const { nx, ny, nz } = v, [sx, sy, sz] = v.spacing;
  if (plane === 'axial') return { w: nx, h: ny, n: nz, sw: sx, sh: sy, idx: (c, r, s) => (nx - 1 - c) + nx * ((ny - 1 - r) + ny * s), ijk: (c, r, s) => [nx - 1 - c, ny - 1 - r, s], cr: (i, j, k) => [nx - 1 - i, ny - 1 - j], slice: (i, j, k) => k, setSlice: (cur, s) => (cur[2] = s) };
  if (plane === 'coronal') return { w: nx, h: nz, n: ny, sw: sx, sh: sz, idx: (c, r, s) => (nx - 1 - c) + nx * (s + ny * (nz - 1 - r)), ijk: (c, r, s) => [nx - 1 - c, s, nz - 1 - r], cr: (i, j, k) => [nx - 1 - i, nz - 1 - k], slice: (i, j, k) => j, setSlice: (cur, s) => (cur[1] = s) };
  return { w: ny, h: nz, n: nx, sw: sy, sh: sz, idx: (c, r, s) => s + nx * ((ny - 1 - c) + ny * (nz - 1 - r)), ijk: (c, r, s) => [s, ny - 1 - c, nz - 1 - r], cr: (i, j, k) => [ny - 1 - j, nz - 1 - k], slice: (i, j, k) => i, setSlice: (cur, s) => (cur[0] = s) };
}

export class SliceView {
  /** opts: {plane, mask:'main'|'follow'|'diff', title, S (shared state), app} */
  constructor(host, opts) {
    this.host = host; this.S = opts.S; this.app = opts.app; this.plane = opts.plane; this.maskMode = opts.mask || 'main';
    this.label = opts.label || '';
    this.zoom = 1; this.pan = [0, 0];
    this.canvas = document.createElement('canvas'); this.canvas.className = 'sv-canvas';
    host.appendChild(this.canvas);
    this.ctx = this.canvas.getContext('2d');
    this.buf = document.createElement('canvas'); this.bctx = this.buf.getContext('2d');
    this.ro = new ResizeObserver(() => this.resize()); this.ro.observe(host);
    this.bindEvents();
    this.measures = []; this.drag = null; this.hoverPt = null;
  }
  destroy() { this.ro.disconnect(); this.canvas.remove(); }
  resize() {
    const r = this.host.getBoundingClientRect(); const d = Math.min(window.devicePixelRatio || 1, 2);
    this.cw = Math.max(2, r.width); this.chh = Math.max(2, r.height);
    this.canvas.width = Math.round(this.cw * d); this.canvas.height = Math.round(this.chh * d); this.dpr = d;
    this.render();
  }
  get vol() { return this.S.vols[this.S.mod]; }
  geom() { return planeGeom(this.plane, this.vol); }
  layout() {
    const g = this.geom(); const mmW = g.w * g.sw, mmH = g.h * g.sh;
    const fit = Math.min((this.cw - 16) / mmW, (this.chh - 16) / mmH);
    const sc = fit * this.zoom;
    const x0 = (this.cw - mmW * sc) / 2 + this.pan[0], y0 = (this.chh - mmH * sc) / 2 + this.pan[1];
    return { g, sc, x0, y0, pxW: g.sw * sc, pxH: g.sh * sc };
  }
  toCR(ev) {
    const rect = this.canvas.getBoundingClientRect(); const L = this.layout();
    const x = ev.clientX - rect.left, y = ev.clientY - rect.top;
    return { c: Math.floor((x - L.x0) / L.pxW), r: Math.floor((y - L.y0) / L.pxH), fc: (x - L.x0) / L.pxW, fr: (y - L.y0) / L.pxH, x, y, L };
  }
  maskArr() {
    const S = this.S;
    if (this.maskMode === 'follow') return S.follow ? S.follow.mask : null;
    if (this.maskMode === 'base') return S.segDone ? S.seg : null;
    if (S.showOrig && S.segDone && S.segSrc && S.tp === 0) return S.segSrc;
    return S.segDone ? (S.tp === 1 && S.follow ? S.follow.mask : S.seg) : null;
  }

  render() {
    const S = this.S; const v = this.vol; if (!v || !this.ctx) return;
    const L = this.layout(); const { g } = L; const ctx = this.ctx;
    const slice = Math.min(g.n - 1, Math.max(0, g.slice(...S.cursor)));
    if (this.buf.width !== g.w || this.buf.height !== g.h) { this.buf.width = g.w; this.buf.height = g.h; this.img = this.bctx.createImageData(g.w, g.h); }
    const img = this.img, d = img.data, vd = v.data;
    const wl = S.wl[S.mod]; const lo = wl.lo, hi = wl.hi; const lut = new Uint8ClampedArray(256);
    for (let i = 0; i < 256; i++) { let t = (i - lo) / Math.max(1, hi - lo); t = t < 0 ? 0 : t > 1 ? 1 : t; lut[i] = (S.invert ? 1 - t : t) * 255; }
    const mask = this.maskArr(); const base = S.segDone ? S.seg : null;
    const banner = this.maskMode === 'main' && S.tp === 1 && S.follow;
    const sim = S.simOn && S.sim && this.maskMode === 'main' ? S.sim : null;
    const lay = S.layers; const hi3 = S.hiClass || 0;
    const diff = this.maskMode === 'diff' && S.follow && base;
    const w = g.w, h = g.h; const fade = S.segFade ?? 1;
    for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
      const id = g.idx(c, r, slice); const o = (r * w + c) * 4; const gv = lut[vd[id]];
      let R = gv, G = gv, B = gv;
      if (diff) {
        const a = base[id] > 0, b = S.follow.mask[id] > 0;
        if (a || b) {
          let col, al;
          if (b && !a) { col = [255, 91, 97]; al = 0.7; } else if (a && !b) { col = [34, 201, 151]; al = 0.7; } else { col = [160, 170, 182]; al = 0.28; }
          R += (col[0] - R) * al; G += (col[1] - G) * al; B += (col[2] - B) * al;
        }
      } else if (mask) {
        const l = mask[id];
        if (l && lay[l].show) {
          const ly = lay[l]; let edge = false;
          if (ly.style !== 'fill') {
            // граница класса внутри плоскости
            const cn = [c > 0 ? g.idx(c - 1, r, slice) : -1, c < w - 1 ? g.idx(c + 1, r, slice) : -1, r > 0 ? g.idx(c, r - 1, slice) : -1, r < h - 1 ? g.idx(c, r + 1, slice) : -1];
            for (const q of cn) if (q < 0 || mask[q] !== l) { edge = true; break; }
          }
          let al;
          if (ly.style === 'fill') al = ly.opacity; else if (ly.style === 'contour') al = edge ? 0.95 : 0; else al = edge ? 0.95 : ly.opacity * 0.4;
          if (hi3) al = hi3 === l ? Math.min(1, al * 1.25 + 0.08) : al * 0.35;
          al *= fade;
          const col = RGB[l];
          R += (col[0] - R) * al; G += (col[1] - G) * al; B += (col[2] - B) * al;
        }
      }
      if (sim) {
        // модельный сценарий: штриховка внутри + пунктирный контур на границе
        const mm = sim.mid[id];
        const sl = sim.lo ? sim.lo[id] : mm, sh = sim.hi ? sim.hi[id] : mm;
        if (sl !== sh && ((c + r) & 3) < 2) { R = R * 0.55 + 255 * 0.45; G = G * 0.55 + 255 * 0.45; B = B * 0.55 + 255 * 0.45; }
        if (mm && lay[mm].show) {
          const nb = [c > 0 ? sim.mid[g.idx(c - 1, r, slice)] : 0, c < w - 1 ? sim.mid[g.idx(c + 1, r, slice)] : 0, r > 0 ? sim.mid[g.idx(c, r - 1, slice)] : 0, r < h - 1 ? sim.mid[g.idx(c, r + 1, slice)] : 0];
          const edge = nb.some((q) => q !== mm);
          if (edge && ((c + r) % 4) < 3) { const col = RGB[mm]; R = R * 0.2 + (col[0] * 0.55 + 255 * 0.45) * 0.8; G = G * 0.2 + (col[1] * 0.55 + 255 * 0.45) * 0.8; B = B * 0.2 + (col[2] * 0.55 + 255 * 0.45) * 0.8; }
          else if (!edge && ((c + r) & 3) === 0) { const col = RGB[mm]; R = R * 0.75 + col[0] * 0.25; G = G * 0.75 + col[1] * 0.25; B = B * 0.75 + col[2] * 0.25; }
        }
      }
      d[o] = R; d[o + 1] = G; d[o + 2] = B; d[o + 3] = 255;
    }
    this.bctx.putImageData(img, 0, 0);
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.fillStyle = '#05080c'; ctx.fillRect(0, 0, this.cw, this.chh);
    ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(this.buf, L.x0, L.y0, w * L.pxW, h * L.pxH);

    // перекрестие
    const [cc, cr] = g.cr(...S.cursor); const cx = L.x0 + (cc + 0.5) * L.pxW, cy = L.y0 + (cr + 0.5) * L.pxH;
    if (S.crosshair) {
      ctx.strokeStyle = this.active ? 'rgba(25,195,177,.85)' : 'rgba(25,195,177,.5)'; ctx.lineWidth = 1; ctx.setLineDash([]);
      const gap = 9; ctx.beginPath();
      ctx.moveTo(L.x0, cy); ctx.lineTo(cx - gap, cy); ctx.moveTo(cx + gap, cy); ctx.lineTo(L.x0 + w * L.pxW, cy);
      ctx.moveTo(cx, L.y0); ctx.lineTo(cx, cy - gap); ctx.moveTo(cx, cy + gap); ctx.lineTo(cx, L.y0 + h * L.pxH); ctx.stroke();
    }
    // подписи ориентации
    const P = PLANES[this.plane]; ctx.font = '600 12px Inter, system-ui, sans-serif'; ctx.fillStyle = 'rgba(232,238,244,.75)'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    const mx = this.cw / 2, my = this.chh / 2;
    ctx.fillText(P.letters[0], 12, my); ctx.fillText(P.letters[1], this.cw - 12, my); ctx.fillText(P.letters[2], mx, 12); ctx.fillText(P.letters[3], mx, this.chh - 12);
    // измерения
    for (const m of this.measures) if (m.plane === this.plane && m.slice === slice) this.drawMeasure(ctx, m, L);
    if (this.drag && (this.drag.tool === 'ruler' || this.drag.tool === 'roi')) this.drawMeasure(ctx, this.drag.m, L, true);
    // курсор кисти
    if (this.hoverPt && (S.tool === 'brush' || S.tool === 'eraser')) {
      const rr = S.brush.r * L.sc; ctx.strokeStyle = S.tool === 'brush' ? 'rgba(255,255,255,.9)' : 'rgba(255,91,97,.9)'; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.arc(this.hoverPt.x, this.hoverPt.y, rr, 0, 7); ctx.stroke(); ctx.setLineDash([]);
    }
    // HUD
    ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.font = '600 11.5px Inter, system-ui, sans-serif';
    const hud = [`${this.label || P.name}`, `${S.modNames[S.mod]} · срез ${slice + 1}/${g.n}`];
    ctx.fillStyle = 'rgba(5,8,12,.55)'; ctx.fillRect(6, 6, 168, 36); ctx.fillStyle = '#E8EEF4'; ctx.fillText(hud[0], 12, 10); ctx.fillStyle = '#98A8B8'; ctx.font = '500 11px Inter, system-ui, sans-serif'; ctx.fillText(hud[1], 12, 25);
    ctx.textAlign = 'right'; ctx.fillStyle = 'rgba(232,238,244,.6)'; ctx.font = '500 10.5px "JetBrains Mono", ui-monospace, monospace';
    ctx.textBaseline = 'alphabetic'; ctx.fillText(`W ${Math.round(hi - lo)}  L ${Math.round((hi + lo) / 2)}  ×${this.zoom.toFixed(1)}`, this.cw - 8, this.chh - 7);
    if (this.hoverInfo) { ctx.fillStyle = '#E8EEF4'; ctx.fillText(this.hoverInfo, this.cw - 8, this.chh - 21); }
    if (this.maskMode === 'follow' || banner) { ctx.textAlign = 'left'; ctx.fillStyle = '#F5A623'; ctx.font = '700 10.5px Inter, system-ui, sans-serif'; ctx.fillText('СИНТЕТИЧЕСКАЯ ДЕМО-ДИНАМИКА · снимок базового МРТ', 10, this.chh - 8); }
    if (this.maskMode === 'diff') { ctx.textAlign = 'left'; ctx.fillStyle = '#98A8B8'; ctx.font = '600 10.5px Inter, system-ui, sans-serif'; ctx.fillText('красный — рост · бирюзовый — уменьшение · серый — без изменений', 10, this.chh - 8); }
    if (S.anon) { ctx.fillStyle = 'rgba(5,8,12,.85)'; ctx.fillRect(6, 6, 168, 36); ctx.fillStyle = '#98A8B8'; ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText('данные скрыты', 12, 14); }
  }
  drawMeasure(ctx, m, L, live = false) {
    const px = (c, r) => [L.x0 + c * L.pxW, L.y0 + r * L.pxH];
    ctx.lineWidth = 1.6; ctx.strokeStyle = '#FFD166'; ctx.fillStyle = '#FFD166'; ctx.setLineDash(live ? [4, 3] : []);
    const a = px(m.a[0], m.a[1]), b = px(m.b[0], m.b[1]);
    if (m.kind === 'ruler') {
      ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
      for (const p of [a, b]) { ctx.beginPath(); ctx.arc(p[0], p[1], 3, 0, 7); ctx.fill(); }
      const g = L.g; const mm = Math.hypot((m.b[0] - m.a[0]) * g.sw, (m.b[1] - m.a[1]) * g.sh);
      this.tag(ctx, (a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - 10, `${mm.toFixed(1).replace('.', ',')} мм`);
    } else {
      const cx = (a[0] + b[0]) / 2, cy = (a[1] + b[1]) / 2;
      ctx.beginPath(); ctx.ellipse(cx, cy, Math.abs(b[0] - a[0]) / 2, Math.abs(b[1] - a[1]) / 2, 0, 0, 7); ctx.stroke();
      if (m.stat) this.tag(ctx, cx, Math.min(a[1], b[1]) - 10, m.stat);
    }
    ctx.setLineDash([]);
  }
  tag(ctx, x, y, text) {
    ctx.font = '600 11.5px Inter, system-ui, sans-serif'; const w = ctx.measureText(text).width + 10;
    ctx.fillStyle = 'rgba(5,8,12,.8)'; ctx.fillRect(x - w / 2, y - 9, w, 18); ctx.fillStyle = '#FFD166'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(text, x, y);
  }

  bindEvents() {
    const cv = this.canvas; const S = this.S;
    cv.addEventListener('contextmenu', (e) => e.preventDefault());
    cv.addEventListener('pointerdown', (e) => {
      cv.setPointerCapture(e.pointerId); this.app.setActive(this);
      const p = this.toCR(e); let tool = S.tool;
      if (e.button === 2) tool = 'wl'; else if (e.button === 1 || this.app.spaceDown) tool = 'pan';
      this.drag = { tool, x: e.clientX, y: e.clientY, pan: [...this.pan], zoom: this.zoom, wl: { ...S.wl[S.mod] } };
      if (tool === 'cross') this.setCursorFrom(p);
      else if (tool === 'ruler') { this.drag.m = { kind: 'ruler', plane: this.plane, slice: p.L.g.slice(...S.cursor), a: [p.fc, p.fr], b: [p.fc, p.fr] }; }
      else if (tool === 'roi') { this.drag.m = { kind: 'roi', plane: this.plane, slice: p.L.g.slice(...S.cursor), a: [p.fc, p.fr], b: [p.fc, p.fr] }; }
      else if (tool === 'brush' || tool === 'eraser') { if (!S.segDone) { this.app.toast('Сначала запустите сегментацию — правки вносятся в маску', 'warn'); this.drag = null; return; } this.app.beginStroke(); this.paint(p, tool); }
      else if (tool === 'cc') { this.app.removeComponent(this, p); this.drag = null; }
      this.render();
    });
    cv.addEventListener('pointermove', (e) => {
      const p = this.toCR(e); this.hoverPt = p;
      this.updateHover(p);
      if (this.drag) {
        const d = this.drag, dx = e.clientX - d.x, dy = e.clientY - d.y;
        if (d.tool === 'cross') this.setCursorFrom(p);
        else if (d.tool === 'pan') { this.pan = [d.pan[0] + dx, d.pan[1] + dy]; }
        else if (d.tool === 'zoom') { this.zoom = Math.max(0.5, Math.min(12, d.zoom * Math.exp(-dy / 180))); }
        else if (d.tool === 'wl') { const wd = Math.max(4, d.wl.hi - d.wl.lo + dx * 0.9), lv = (d.wl.hi + d.wl.lo) / 2 + dy * 0.9; S.wl[S.mod] = { lo: lv - wd / 2, hi: lv + wd / 2 }; this.app.onWL(); }
        else if (d.tool === 'ruler' || d.tool === 'roi') d.m.b = [p.fc, p.fr];
        else if (d.tool === 'brush' || d.tool === 'eraser') this.paint(p, d.tool);
        this.render(); if (d.tool === 'wl') this.app.renderAll();
      } else this.render();
    });
    const end = (e) => {
      const d = this.drag; this.drag = null; if (!d) return;
      if (d.tool === 'ruler') { const m = d.m; if (Math.hypot(m.b[0] - m.a[0], m.b[1] - m.a[1]) > 0.8) { this.measures.push(m); this.app.log(`Измерение: линейка ${this.fmtLen(m)}`); } }
      if (d.tool === 'roi') { const m = d.m; m.stat = this.roiStat(m); if (m.stat) { this.measures.push(m); this.app.log('Измерение: ROI ' + m.stat); } }
      if (d.tool === 'brush' || d.tool === 'eraser') this.app.endStroke();
      this.render();
    };
    cv.addEventListener('pointerup', end); cv.addEventListener('pointercancel', end);
    cv.addEventListener('pointerleave', () => { this.hoverPt = null; this.hoverInfo = ''; this.render(); this.app.onHover(null); });
    cv.addEventListener('wheel', (e) => {
      e.preventDefault(); this.app.setActive(this);
      if (e.ctrlKey || e.metaKey) { const z = this.zoom * Math.exp(-e.deltaY / 400); this.zoom = Math.max(0.5, Math.min(12, z)); if (this.zoom < 1.02) this.pan = [0, 0]; this.render(); return; }
      const g = this.geom(); const cur = [...S.cursor]; const s = g.slice(...cur) + (e.deltaY > 0 ? 1 : -1);
      g.setSlice(cur, Math.max(0, Math.min(g.n - 1, s))); S.cursor = cur; this.app.onCursor(this);
    }, { passive: false });
    cv.addEventListener('dblclick', () => this.app.maximize(this));
  }
  fmtLen(m) { const g = this.geom(); return (Math.hypot((m.b[0] - m.a[0]) * g.sw, (m.b[1] - m.a[1]) * g.sh)).toFixed(1).replace('.', ',') + ' мм'; }
  roiStat(m) {
    const g = this.geom(); const S = this.S; const vd = this.vol.data; const slice = g.slice(...S.cursor);
    const c0 = Math.min(m.a[0], m.b[0]), c1 = Math.max(m.a[0], m.b[0]), r0 = Math.min(m.a[1], m.b[1]), r1 = Math.max(m.a[1], m.b[1]);
    const cc = (c0 + c1) / 2, rr = (r0 + r1) / 2, ac = (c1 - c0) / 2, ar = (r1 - r0) / 2; if (ac < 1 || ar < 1) return null;
    let n = 0, s = 0, s2 = 0;
    for (let r = Math.max(0, Math.floor(r0)); r <= Math.min(g.h - 1, Math.ceil(r1)); r++) for (let c = Math.max(0, Math.floor(c0)); c <= Math.min(g.w - 1, Math.ceil(c1)); c++) {
      if (((c - cc) / ac) ** 2 + ((r - rr) / ar) ** 2 > 1) continue; const v = vd[g.idx(c, r, slice)]; s += v; s2 += v * v; n++;
    }
    if (!n) return null; const mean = s / n, sd = Math.sqrt(Math.max(0, s2 / n - mean * mean));
    return `${(n * g.sw * g.sh / 100).toFixed(1).replace('.', ',')} см² · ср. ${mean.toFixed(0)} ± ${sd.toFixed(0)} у.е.`;
  }
  setCursorFrom(p) {
    const g = p.L.g; if (p.c < 0 || p.r < 0 || p.c >= g.w || p.r >= g.h) return;
    const S = this.S; const slice = g.slice(...S.cursor); const [i, j, k] = g.ijk(p.c, p.r, slice); S.cursor = [i, j, k]; this.app.onCursor(this);
  }
  updateHover(p) {
    const g = p.L.g; if (p.c < 0 || p.r < 0 || p.c >= g.w || p.r >= g.h) { this.hoverInfo = ''; this.app.onHover(null); return; }
    const S = this.S; const slice = g.slice(...S.cursor); const id = g.idx(p.c, p.r, slice); const [i, j, k] = g.ijk(p.c, p.r, slice);
    const val = this.vol.data[id]; const lab = S.segDone && S.seg ? S.seg[id] : 0;
    this.hoverInfo = `${val}${lab ? ' · ' + ['', 'некроз', 'отёк', 'активная'][lab] : ''}`;
    this.app.onHover({ i, j, k, val, lab, x: p.x, y: p.y, view: this });
  }
  paint(p, tool) {
    const S = this.S; const g = p.L.g; const slice = g.slice(...S.cursor);
    const rc = S.brush.r / g.sw, rr = S.brush.r / g.sh; const label = tool === 'brush' ? S.brush.label : 0;
    for (let r = Math.floor(p.fr - rr); r <= Math.ceil(p.fr + rr); r++) for (let c = Math.floor(p.fc - rc); c <= Math.ceil(p.fc + rc); c++) {
      if (c < 0 || r < 0 || c >= g.w || r >= g.h) continue; if (((c + 0.5 - p.fc) / rc) ** 2 + ((r + 0.5 - p.fr) / rr) ** 2 > 1) continue;
      const id = g.idx(c, r, slice); if (S.seg[id] === label) continue;
      this.app.stroke.set(id, this.app.stroke.has(id) ? this.app.stroke.get(id) : S.seg[id]); S.seg[id] = label;
    }
    this.app.renderAll(true);
  }
}
