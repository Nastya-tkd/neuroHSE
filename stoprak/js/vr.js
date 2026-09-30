// Объёмный (ray-marching) 3D-рендер МРТ + зон опухоли на WebGL2 (three.js).
// Мозг — полупрозрачная поверхность по T1, опухолевые зоны — поверхности внутри,
// клиппинг-плоскость открывает настоящий срез МРТ на разрезе.
import * as THREE from '../vendor/three.module.js';
import { OrbitControls } from '../vendor/OrbitControls.js';

export const LAYER_COLORS = {
  nec: [0xF5, 0xA6, 0x23],   // некроз — янтарный
  ede: [0x38, 0xBD, 0xF8],   // отёк — голубой
  enh: [0xFF, 0x5B, 0x61],   // активная опухоль — коралловый
};

function blur1d(src, dst, nx, ny, nz, axis) {
  const sx = 1, sy = nx, sz = nx * ny;
  const st = [sx, sy, sz][axis], n = [nx, ny, nz][axis];
  const o1 = axis === 0 ? [1, 2] : axis === 1 ? [0, 2] : [0, 1];
  const s1 = [sx, sy, sz][o1[0]], s2 = [sx, sy, sz][o1[1]];
  const n1 = [nx, ny, nz][o1[0]], n2 = [nx, ny, nz][o1[1]];
  for (let b = 0; b < n2; b++) for (let a = 0; a < n1; a++) {
    const base = a * s1 + b * s2;
    for (let t = 0; t < n; t++) {
      const c = base + t * st;
      const l = t > 0 ? src[c - st] : src[c], r = t < n - 1 ? src[c + st] : src[c];
      dst[c] = (l + 2 * src[c] + r + 2) >> 2;
    }
  }
}
export function blur3(u8, nx, ny, nz, passes = 1) {
  let a = u8, b = new Uint8Array(u8.length);
  for (let p = 0; p < passes; p++) for (let ax = 0; ax < 3; ax++) { blur1d(a, b, nx, ny, nz, ax); [a, b] = [b, a]; }
  return a === u8 ? u8.slice() : a;
}

const VERT = /* glsl */`
out vec3 vWorld;
void main(){
  vec4 w = modelMatrix * vec4(position, 1.0);
  vWorld = w.xyz;
  gl_Position = projectionMatrix * viewMatrix * w;
}`;

const FRAG = /* glsl */`
precision highp float;
precision highp sampler3D;
in vec3 vWorld;
layout(location = 0) out highp vec4 fragColor;
uniform sampler3D uVol;   // R=некроз G=отёк B=активная A=T1 (сглаженные)
uniform sampler3D uSim;   // модельный сценарий, R,G,B как выше
uniform sampler3D uMod;   // выбранная модальность для среза
uniform vec3  uBox;       // размеры бокса в мировых единицах (X=R, Y=S, Z=P)
uniform vec3  uInvDims;   // 1/nx,1/ny,1/nz
uniform float uStep;
uniform float uBrainThr, uBrainAlpha;
uniform vec3  uShow;      // видимость слоёв
uniform vec3  uOp;        // непрозрачность слоёв
uniform float uSimOn;
uniform float uClipOn;
uniform vec3  uClipN;
uniform float uClipD;
uniform vec2  uWin;       // окно/уровень для среза: lo, hi (0..1)
uniform float uHi;        // подсвеченный класс 0..3
uniform float uTime;
uniform vec3 uColNec, uColEde, uColEnh;

vec3 toTex(vec3 p){ return vec3(p.x/uBox.x + 0.5, 0.5 - p.z/uBox.z, p.y/uBox.y + 0.5); }
float hash(vec2 p){ return fract(sin(dot(p, vec2(12.9898,78.233))) * 43758.5453); }

float chan(sampler3D s, vec3 tc, int c){
  vec4 v = texture(s, tc);
  return c==0 ? v.r : c==1 ? v.g : c==2 ? v.b : v.a;
}
vec3 gradWorld(sampler3D s, vec3 tc, int c){
  vec3 e = uInvDims;
  float gx = chan(s, tc+vec3(e.x,0,0), c) - chan(s, tc-vec3(e.x,0,0), c);
  float gv = chan(s, tc+vec3(0,e.y,0), c) - chan(s, tc-vec3(0,e.y,0), c);
  float gw = chan(s, tc+vec3(0,0,e.z), c) - chan(s, tc-vec3(0,0,e.z), c);
  return vec3(gx/(e.x*uBox.x), gw/(e.z*uBox.y), -gv/(e.y*uBox.z));
}
vec3 shade(vec3 base, vec3 n, vec3 rd, float ao, float rim){
  n = normalize(n);
  if (dot(n, rd) > 0.0) n = -n;
  vec3 L1 = normalize(vec3(-0.45, 0.75, -0.55));
  vec3 L2 = normalize(-rd + vec3(0.25, 0.35, 0.0));
  float d = max(dot(n, L1), 0.0) * 0.55 + max(dot(n, L2), 0.0) * 0.55;
  vec3 h = normalize(L2 - rd);
  float sp = pow(max(dot(n, h), 0.0), 38.0) * 0.32;
  float fr = pow(1.0 - max(dot(n, -rd), 0.0), 2.6);
  return base * (0.55 + d * 1.0) * ao + vec3(sp) + vec3(0.10,0.80,0.72) * fr * rim * 1.25;
}
void addSample(inout vec4 acc, vec3 rgb, float a){
  acc.rgb += (1.0 - acc.a) * rgb * a;
  acc.a   += (1.0 - acc.a) * a;
}

void main(){
  vec3 ro = cameraPosition;
  vec3 rd = normalize(vWorld - ro);
  vec3 bmin = -0.5*uBox, bmax = 0.5*uBox;
  vec3 inv = 1.0 / rd;
  vec3 t0 = (bmin - ro) * inv, t1 = (bmax - ro) * inv;
  vec3 tmn = min(t0, t1), tmx = max(t0, t1);
  float tNear = max(max(tmn.x, tmn.y), max(tmn.z, 0.0));
  float tFar  = min(min(tmx.x, tmx.y), tmx.z);
  if (tFar <= tNear) discard;

  bool cutFace = false;
  if (uClipOn > 0.5) {
    float dn = dot(rd, uClipN);
    float f0 = dot(ro + rd*tNear, uClipN) - uClipD;
    if (f0 > 0.0) {
      if (dn < 0.0) {
        float tp = (uClipD - dot(ro, uClipN)) / dn;
        if (tp > tFar) discard;
        tNear = max(tNear, tp); cutFace = true;
      } else discard;
    } else if (dn > 0.0) {
      tFar = min(tFar, (uClipD - dot(ro, uClipN)) / dn);
    }
  }

  vec4 acc = vec4(0.0);
  float dt = uStep;
  float t = tNear + dt * hash(gl_FragCoord.xy);
  float prevB = 0.0; vec3 prevT = vec3(0.0); vec3 prevS = vec3(0.0);
  bool brainDone = false;

  if (cutFace) {
    vec3 p = ro + rd * tNear; vec3 tc = toTex(p);
    float g = texture(uMod, tc).r;
    if (g > 0.004) {
      float gy = clamp((g - uWin.x) / max(uWin.y - uWin.x, 0.02), 0.0, 1.0);
      vec3 col = vec3(gy);
      vec3 vv = texture(uVol, tc).rgb;
      float ln = smoothstep(0.42, 0.58, vv.r) * uShow.x, le = smoothstep(0.42, 0.58, vv.g) * uShow.y, lh = smoothstep(0.42, 0.58, vv.b) * uShow.z;
      col = mix(col, uColEde, le * 0.38);
      col = mix(col, uColNec, ln * 0.55);
      col = mix(col, uColEnh, lh * 0.55);
      acc = vec4(col * 0.98, 1.0);
      fragColor = vec4(acc.rgb, acc.a);
      return;
    }
  }

  for (int i = 0; i < 520; i++) {
    if (t >= tFar || acc.a > 0.985) break;
    vec3 p = ro + rd * t;
    vec3 tc = toTex(p);
    vec4 v = texture(uVol, tc);

    // --- мозг ---
    float B = v.a;
    if (!brainDone && prevB < uBrainThr && B >= uBrainThr) {
      brainDone = true;
      float f = (uBrainThr - prevB) / max(B - prevB, 1e-4);
      float th = t - dt + f * dt;
      vec3 ph = ro + rd * th; vec3 tch = toTex(ph);
      vec3 g = -gradWorld(uVol, tch, 3);
      vec3 n = normalize(g + 1e-6);
      if (dot(n, rd) > 0.0) n = -n;
      float occ = chan(uVol, toTex(ph + n * 0.011), 3);
      float ao = 1.0 - 0.55 * smoothstep(uBrainThr - 0.02, uBrainThr + 0.18, occ);
      vec3 base = mix(vec3(0.80,0.88,0.97), vec3(0.98,0.90,0.88), smoothstep(0.2, 0.9, B));
      vec3 col = shade(base, n, rd, ao, 0.9);
      float fr = pow(1.0 - max(dot(n, -rd), 0.0), 1.8);
      float a = uBrainAlpha >= 0.98 ? 1.0 : uBrainAlpha * (0.45 + 0.75 * fr);
      addSample(acc, col, clamp(a, 0.0, 1.0));
    }
    prevB = B;

    // --- опухолевые зоны ---
    vec3 cur = v.rgb;
    for (int c = 0; c < 3; c++) {
      float cv = c==0 ? cur.r : c==1 ? cur.g : cur.b;
      float pv = c==0 ? prevT.r : c==1 ? prevT.g : prevT.b;
      float sh = c==0 ? uShow.x : c==1 ? uShow.y : uShow.z;
      float op = c==0 ? uOp.x : c==1 ? uOp.y : uOp.z;
      vec3 colc = c==0 ? uColNec : c==1 ? uColEde : uColEnh;
      if (sh < 0.5) continue;
      float hiMul = uHi < 0.5 ? 1.0 : (abs(uHi - float(c+1)) < 0.5 ? 1.35 : 0.3);
      if (pv < 0.5 && cv >= 0.5) {
        float f = (0.5 - pv) / max(cv - pv, 1e-4);
        vec3 ph = ro + rd * (t - dt + f * dt);
        vec3 n = -gradWorld(uVol, toTex(ph), c);
        vec3 col = shade(colc, n + 1e-5, rd, 1.0, 0.55) * (uHi > 0.5 && hiMul > 1.0 ? 1.18 : 1.0);
        addSample(acc, col, clamp(op * hiMul, 0.0, 1.0));
      }
      // мягкий объёмный вклад — глубина
      float dens = smoothstep(0.5, 0.95, cv);
      if (dens > 0.0) addSample(acc, colc * 0.9, dens * op * 0.05 * hiMul);
    }
    prevT = cur;

    // --- модельный сценарий (штриховка) ---
    if (uSimOn > 0.5) {
      vec3 sv = texture(uSim, tc).rgb;
      for (int c = 0; c < 3; c++) {
        float cv = c==0 ? sv.r : c==1 ? sv.g : sv.b;
        float pv = c==0 ? prevS.r : c==1 ? prevS.g : prevS.b;
        float sh = c==0 ? uShow.x : c==1 ? uShow.y : uShow.z;
        vec3 colc = c==0 ? uColNec : c==1 ? uColEde : uColEnh;
        if (sh < 0.5) continue;
        if (pv < 0.5 && cv >= 0.5) {
          float f = (0.5 - pv) / max(cv - pv, 1e-4);
          vec3 ph = ro + rd * (t - dt + f * dt);
          vec3 n = -gradWorld(uSim, toTex(ph), c);
          float stripe = step(0.5, fract((gl_FragCoord.x + gl_FragCoord.y) * 0.11));
          vec3 col = mix(shade(colc, n + 1e-5, rd, 1.0, 0.3), vec3(1.0), 0.38);
          addSample(acc, col, uOp[c] * mix(0.10, 0.92, stripe));
        }
      }
      prevS = sv;
    }
    t += dt;
  }
  fragColor = vec4(acc.rgb, acc.a);
}`;

const tex3 = (data, nx, ny, nz, format, linear = true) => {
  const t = new THREE.Data3DTexture(data, nx, ny, nz);
  t.format = format; t.type = THREE.UnsignedByteType;
  t.minFilter = t.magFilter = linear ? THREE.LinearFilter : THREE.NearestFilter;
  t.wrapS = t.wrapT = t.wrapR = THREE.ClampToEdgeWrapping;
  t.unpackAlignment = 1; t.needsUpdate = true;
  return t;
};

export class VR3D {
  constructor(container, opts = {}) {
    this.el = container; this.opts = opts;
    this.renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true, powerPreference: 'high-performance', preserveDrawingBuffer: !!opts.preserve });
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, opts.maxDpr || 1.5));
    this.renderer.domElement.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;display:block;touch-action:none;';
    container.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(32, 1, 0.05, 20);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true; this.controls.dampingFactor = 0.08;
    this.controls.autoRotate = !!opts.autoRotate; this.controls.autoRotateSpeed = opts.rotateSpeed ?? 0.9;
    this.controls.minDistance = 0.9; this.controls.maxDistance = 6; this.controls.zoomSpeed = 0.7;
    this.stepMul = 1; this.quality = opts.quality || 1;
    this.controls.addEventListener('start', () => { this.stepMul = 1.9; this.controls.autoRotate = false; this.interacting = true; this.dirty = true; this.opts.onInteract?.(); });
    this.controls.addEventListener('end', () => { this.stepMul = 1; this.interacting = false; this.dirty = true; });
    this.controls.addEventListener('change', () => { this.dirty = true; });
    this.material = null; this.mesh = null;
    this.dirty = true; this.running = false; this.visible = true;
    this.uniformsInit();
    this._ro = new ResizeObserver(() => this.resize());
    this._ro.observe(container);
    this._io = new IntersectionObserver((e) => { this.visible = e[e.length - 1].isIntersecting; this.dirty = true; });
    this._io.observe(container);
    this.renderer.domElement.addEventListener('pointerdown', (e) => { this._down = [e.clientX, e.clientY, performance.now()]; });
    this.renderer.domElement.addEventListener('pointerup', (e) => {
      if (!this._down) return;
      const d = Math.hypot(e.clientX - this._down[0], e.clientY - this._down[1]);
      if (d < 4 && performance.now() - this._down[2] < 500) this.opts.onPick?.(this.pick(e.clientX, e.clientY));
    });
    this.renderer.domElement.addEventListener('pointermove', (e) => { if (this.opts.onHover && !this.interacting) this._hoverEv = e; });
    this.resize();
    this.loop = this.loop.bind(this);
    this.start();
  }

  uniformsInit() {
    const c = (a) => new THREE.Color(a[0] / 255, a[1] / 255, a[2] / 255);
    this.U = {
      uVol: { value: null }, uSim: { value: null }, uMod: { value: null },
      uBox: { value: new THREE.Vector3(1, 1, 1) }, uInvDims: { value: new THREE.Vector3(1, 1, 1) },
      uStep: { value: 0.006 }, uBrainThr: { value: 0.2 }, uBrainAlpha: { value: 0.16 },
      uShow: { value: new THREE.Vector3(1, 1, 1) }, uOp: { value: new THREE.Vector3(0.8, 0.22, 0.72) },
      uSimOn: { value: 0 }, uClipOn: { value: 0 }, uClipN: { value: new THREE.Vector3(1, 0, 0) }, uClipD: { value: 0 },
      uWin: { value: new THREE.Vector2(0.02, 0.95) }, uHi: { value: 0 }, uTime: { value: 0 },
      uColNec: { value: c(LAYER_COLORS.nec) }, uColEde: { value: c(LAYER_COLORS.ede) }, uColEnh: { value: c(LAYER_COLORS.enh) },
    };
  }

  /** data = {t1:Vol, mod:Vol, seg:Uint8Array|null}. Создаёт текстуры. */
  setData({ t1, mod, seg, spacing, focus }) {
    this.focus = focus || null;
    const { nx, ny, nz } = t1; const n = nx * ny * nz;
    this.dims = [nx, ny, nz]; this.spacing = spacing || t1.spacing;
    this.t1 = t1; this.seg = seg;
    const rgba = new Uint8Array(n * 4);
    const bt = blur3(t1.data, nx, ny, nz, 1);
    for (let i = 0; i < n; i++) rgba[i * 4 + 3] = bt[i];
    this.rgba = rgba;
    this.writeLabels(this.rgba, seg);
    const lens = [nx * this.spacing[0], ny * this.spacing[1], nz * this.spacing[2]];
    const mx = Math.max(...lens);
    this.box = new THREE.Vector3(lens[0] / mx, lens[2] / mx, lens[1] / mx); // X=R, Y=S, Z=P
    this.U.uBox.value.copy(this.box);
    this.U.uInvDims.value.set(1 / nx, 1 / ny, 1 / nz);
    this.voxW = Math.min(this.spacing[0], this.spacing[1], this.spacing[2]) / mx;
    this.U.uVol.value?.dispose();
    this.U.uVol.value = tex3(rgba, nx, ny, nz, THREE.RGBAFormat);
    this.simRgba = new Uint8Array(n * 4);
    this.U.uSim.value?.dispose();
    this.U.uSim.value = tex3(this.simRgba, nx, ny, nz, THREE.RGBAFormat);
    this.setModality(mod || t1);
    if (!this.mesh) {
      this.material = new THREE.ShaderMaterial({
        glslVersion: THREE.GLSL3, uniforms: this.U, vertexShader: VERT, fragmentShader: FRAG,
        side: THREE.BackSide, transparent: true, depthWrite: false, premultipliedAlpha: true,
      });
      this.mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), this.material);
      this.scene.add(this.mesh);
      this.cursorGroup = new THREE.Group(); this.cursorGroup.visible = false; this.scene.add(this.cursorGroup);
      const mat = new THREE.LineBasicMaterial({ color: 0x19C3B1, transparent: true, opacity: 0.85, depthTest: false });
      for (let a = 0; a < 3; a++) {
        const g = new THREE.BufferGeometry(); const v = [0, 0, 0, 0, 0, 0]; v[a] = -0.03; v[3 + a] = 0.03;
        g.setAttribute('position', new THREE.Float32BufferAttribute(v, 3));
        this.cursorGroup.add(new THREE.LineSegments(g, mat));
      }
    }
    this.mesh.scale.copy(this.box);
    this.U.uStep.value = this.voxW * 0.85 / this.quality;
    this.resetView(true);
    this.dirty = true;
  }

  writeLabels(rgba, seg) {
    const [nx, ny, nz] = this.dims; const n = nx * ny * nz;
    if (!seg) { for (let i = 0; i < n; i++) { rgba[i * 4] = rgba[i * 4 + 1] = rgba[i * 4 + 2] = 0; } return; }
    const ch = [new Uint8Array(n), new Uint8Array(n), new Uint8Array(n)];
    for (let i = 0; i < n; i++) { const l = seg[i]; if (l === 1) ch[0][i] = 255; else if (l === 2) ch[1][i] = 255; else if (l === 3) ch[2][i] = 255; }
    for (let c = 0; c < 3; c++) {
      const b = blur3(ch[c], nx, ny, nz, 2);
      for (let i = 0; i < n; i++) rgba[i * 4 + c] = b[i];
    }
  }

  /** Обновить метки (после сегментации / ручной правки) */
  setSeg(seg) {
    this.seg = seg;
    this.writeLabels(this.rgba, seg);
    this.U.uVol.value.needsUpdate = true; this.dirty = true;
  }
  /** Модельный сценарий (Uint8Array с метками 1..3) или null */
  setSim(labels) {
    if (!labels) { this.U.uSimOn.value = 0; this.dirty = true; return; }
    const [nx, ny, nz] = this.dims; const n = nx * ny * nz;
    this.writeLabels(this.simRgba, labels);
    this.U.uSim.value.needsUpdate = true; this.U.uSimOn.value = 1; this.dirty = true;
  }
  setModality(vol) {
    this.U.uMod.value?.dispose();
    this.U.uMod.value = tex3(vol.data, vol.nx, vol.ny, vol.nz, THREE.RedFormat);
    this.dirty = true;
  }
  setLayer(key, { show, opacity }) {
    const i = { nec: 0, ede: 1, enh: 2 }[key];
    const setC = (v, idx, val) => (idx === 0 ? (v.x = val) : idx === 1 ? (v.y = val) : (v.z = val));
    if (show !== undefined) setC(this.U.uShow.value, i, show ? 1 : 0);
    if (opacity !== undefined) setC(this.U.uOp.value, i, opacity);
    this.dirty = true;
  }
  setBrain({ opacity, thr }) { if (opacity !== undefined) this.U.uBrainAlpha.value = opacity; if (thr !== undefined) this.U.uBrainThr.value = thr; this.dirty = true; }
  setWindow(lo, hi) { this.U.uWin.value.set(lo, hi); this.dirty = true; }
  setHighlight(cls) { this.U.uHi.value = cls || 0; this.dirty = true; }
  /** axis: 'x'|'y'|'z' (R-L / A-P / S-I), pos 0..1 внутри бокса, flip — какую сторону убирать */
  setClip({ on, axis = 'x', pos = 0.5, flip = false }) {
    this.U.uClipOn.value = on ? 1 : 0;
    const n = new THREE.Vector3(); const b = this.box;
    // world: X=R, Y=S, Z=P ; axis x → X, y (A-P) → Z, z (S-I) → Y
    const wa = { x: 'x', y: 'z', z: 'y' }[axis];
    const sgn = flip ? -1 : 1;
    n[wa] = sgn;
    const half = b[wa] / 2;
    // pos по оси A-P считается от задней к передней: мировая Z = P, поэтому инвертируем
    const q = axis === 'y' ? 1 - pos : pos;
    const coord = -half + q * b[wa] ;
    this.U.uClipN.value.copy(n); this.U.uClipD.value = sgn * coord;
    this.dirty = true;
  }
  setCursor(i, j, k, show = true) {
    if (!this.cursorGroup) return;
    const [nx, ny, nz] = this.dims, b = this.box;
    this.cursorGroup.position.set((i + 0.5) / nx * b.x - b.x / 2, (k + 0.5) / nz * b.y - b.y / 2, b.z / 2 - (j + 0.5) / ny * b.z);
    this.cursorGroup.visible = show; this.dirty = true;
  }
  setAutoRotate(v) { this.controls.autoRotate = v; this.dirty = true; }

  resetView(instant) { this.setView('oblique'); }
  setView(name) {
    const d = this.opts.camDist || 2.55, b = this.box || new THREE.Vector3(1, 1, 1);
    let ox = -0.95, oz = -1.25;
    if (this.focus && this.dims) { const [nx, ny] = this.dims; ox = (this.focus[0] >= nx / 2 ? 1 : -1) * 0.95; oz = (this.focus[1] >= ny / 2 ? -1 : 1) * 1.25; }
    const V = {
      oblique: [ox, 0.55, oz], front: [0, 0.08, -1], back: [0, 0.08, 1],
      left: [-1, 0.08, 0], right: [1, 0.08, 0], top: [0, 1, -0.001], bottom: [0, -1, -0.001],
    }[name] || [-0.9, 0.5, -1.3];
    const v = new THREE.Vector3(...V).normalize().multiplyScalar(d);
    this.camera.position.copy(v); this.camera.up.set(0, 1, 0);
    this.controls.target.set(0, 0, 0); this.controls.update(); this.dirty = true;
  }

  resize() {
    const w = this.el.clientWidth || 2, h = this.el.clientHeight || 2;
    this.renderer.setSize(w, h, false); this.camera.aspect = w / h; this.camera.updateProjectionMatrix(); this.dirty = true;
  }

  start() { if (!this.running) { this.running = true; requestAnimationFrame(this.loop); } }
  stop() { this.running = false; }
  loop(ts) {
    if (!this.running) return;
    requestAnimationFrame(this.loop);
    if (!this.visible || !this.mesh || document.hidden) return;
    const moving = this.controls.autoRotate || this.interacting;
    const upd = this.controls.update();
    if (!(this.dirty || moving || upd)) return;
    this.U.uTime.value = ts / 1000;
    const base = this.voxW * 0.85 / this.quality;
    this.U.uStep.value = base * this.stepMul * (this.controls.autoRotate ? 1.35 : 1);
    this.renderer.render(this.scene, this.camera);
    this.dirty = false;
    this.frames = (this.frames || 0) + 1;
    if (this._hoverEv && this.opts.onHover && this.frames % 6 === 0) { const e = this._hoverEv; this._hoverEv = null; this.opts.onHover(this.pick(e.clientX, e.clientY), e); }
  }
  /** Выбор точки лучом по CPU: сначала опухолевые метки, затем поверхность мозга */
  pick(cx, cy) {
    if (!this.mesh) return null;
    const r = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((cx - r.left) / r.width) * 2 - 1, -((cy - r.top) / r.height) * 2 + 1);
    const rc = new THREE.Raycaster(); rc.setFromCamera(ndc, this.camera);
    const ro = rc.ray.origin, rd = rc.ray.direction, b = this.box;
    const bx = new THREE.Box3(new THREE.Vector3(-b.x / 2, -b.y / 2, -b.z / 2), new THREE.Vector3(b.x / 2, b.y / 2, b.z / 2));
    const hit = new THREE.Vector3();
    let start;
    if (bx.containsPoint(ro)) start = 0; else { if (!rc.ray.intersectBox(bx, hit)) return null; start = ro.distanceTo(hit); }
    const [nx, ny, nz] = this.dims; const dt = this.voxW * 0.7;
    let brainHit = null;
    for (let t = start; t < start + 4; t += dt) {
      const px = ro.x + rd.x * t, py = ro.y + rd.y * t, pz = ro.z + rd.z * t;
      if (this.U.uClipOn.value > 0.5 && px * this.U.uClipN.value.x + py * this.U.uClipN.value.y + pz * this.U.uClipN.value.z > this.U.uClipD.value) continue;
      const u = px / b.x + 0.5, w = py / b.y + 0.5, v = 0.5 - pz / b.z;
      if (u < 0 || u >= 1 || v < 0 || v >= 1 || w < 0 || w >= 1) { if (t > start + dt * 4) break; continue; }
      const i = Math.floor(u * nx), j = Math.floor(v * ny), k = Math.floor(w * nz);
      const id = i + nx * (j + ny * k);
      const lab = this.seg ? this.seg[id] : 0;
      const sh = this.U.uShow.value;
      if (lab && [0, sh.x, sh.y, sh.z][lab]) return { i, j, k, label: lab };
      if (!brainHit && this.rgba[id * 4 + 3] / 255 >= this.U.uBrainThr.value) brainHit = { i, j, k, label: 0 };
    }
    return brainHit;
  }
  screenshot() { this.renderer.render(this.scene, this.camera); return this.renderer.domElement.toDataURL('image/png'); }
  dispose() { this.stop(); this._ro.disconnect(); this._io.disconnect(); this.renderer.dispose(); this.renderer.domElement.remove(); }
}
