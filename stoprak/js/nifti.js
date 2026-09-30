// Чтение NIfTI-1 (.nii / .nii.gz) прямо в браузере + приведение к ориентации RAS+.
// Ничего не отправляется на сервер: файл читается локально через File/ArrayBuffer.

export class Vol {
  constructor(nx, ny, nz, data, spacing = [1, 1, 1], meta = {}) {
    this.nx = nx; this.ny = ny; this.nz = nz;
    this.data = data;               // Uint8Array, индекс i + nx*(j + ny*k), оси R, A, S
    this.spacing = spacing;         // мм
    this.meta = meta;
  }
  idx(i, j, k) { return i + this.nx * (j + this.ny * k); }
  get size() { return this.nx * this.ny * this.nz; }
}

async function gunzip(buf) {
  const ds = new DecompressionStream('gzip');
  const stream = new Blob([buf]).stream().pipeThrough(ds);
  return await new Response(stream).arrayBuffer();
}

export async function toArrayBuffer(src) {
  let buf = src instanceof ArrayBuffer ? src : await src.arrayBuffer();
  const b = new Uint8Array(buf, 0, 2);
  if (b[0] === 0x1f && b[1] === 0x8b) buf = await gunzip(buf);
  return buf;
}

const TYPES = {
  2: [Uint8Array, 1], 4: [Int16Array, 2], 8: [Int32Array, 4], 16: [Float32Array, 4],
  64: [Float64Array, 8], 256: [Int8Array, 1], 512: [Uint16Array, 2], 768: [Uint32Array, 4],
};

/** Разбор NIfTI-1. Возвращает {dims, spacing, data(typed, i-быстрее), affine 4x4, datatype, dim4} */
export function parseNifti(buf) {
  const dv = new DataView(buf);
  let le = true;
  let hdr = dv.getInt32(0, true);
  if (hdr !== 348) { hdr = dv.getInt32(0, false); le = false; }
  if (hdr === 540) throw new Error('NIfTI-2 пока не поддерживается. Сохраните файл как NIfTI-1 (.nii/.nii.gz).');
  if (hdr !== 348) throw new Error('Это не NIfTI-файл: заголовок не распознан.');
  const i16 = (o) => dv.getInt16(o, le), f32 = (o) => dv.getFloat32(o, le);
  const ndim = i16(40);
  const dim = []; for (let d = 1; d <= 7; d++) dim.push(i16(40 + 2 * d));
  const datatype = i16(70);
  const pix = []; for (let d = 1; d <= 3; d++) pix.push(Math.abs(f32(76 + 4 * d)) || 1);
  const voxOffset = Math.max(352, Math.round(f32(108)));
  let slope = f32(112), inter = f32(116);
  if (!slope || !isFinite(slope)) { slope = 1; inter = 0; }
  const qcode = i16(252), scode = i16(254);
  let A = [[pix[0], 0, 0, 0], [0, pix[1], 0, 0], [0, 0, pix[2], 0], [0, 0, 0, 1]];
  if (scode > 0) {
    A = [0, 1, 2].map((r) => [f32(280 + 16 * r), f32(284 + 16 * r), f32(288 + 16 * r), f32(292 + 16 * r)]).concat([[0, 0, 0, 1]]);
  } else if (qcode > 0) {
    const b = f32(256), c = f32(260), d = f32(264);
    const a = Math.sqrt(Math.max(0, 1 - (b * b + c * c + d * d)));
    const qf = f32(76) < 0 ? -1 : 1;
    const R = [
      [a * a + b * b - c * c - d * d, 2 * b * c - 2 * a * d, 2 * b * d + 2 * a * c],
      [2 * b * c + 2 * a * d, a * a + c * c - b * b - d * d, 2 * c * d - 2 * a * b],
      [2 * b * d - 2 * a * c, 2 * c * d + 2 * a * b, a * a + d * d - c * c - b * b],
    ];
    A = [0, 1, 2].map((r) => [R[r][0] * pix[0], R[r][1] * pix[1], R[r][2] * pix[2] * qf, f32(268 + 4 * r)]).concat([[0, 0, 0, 1]]);
  }
  const T = TYPES[datatype];
  if (!T) throw new Error('Неподдерживаемый тип данных NIfTI: ' + datatype);
  const [nx, ny, nz] = [dim[0], dim[1], dim[2]];
  const n3 = nx * ny * nz;
  const dim4 = ndim >= 4 ? dim[3] : 1;
  let data;
  const bytes = buf.slice(voxOffset, voxOffset + n3 * T[1]);
  if (le || T[1] === 1) data = new T[0](bytes);
  else { // big-endian → перевернуть
    const u = new Uint8Array(bytes);
    for (let o = 0; o < u.length; o += T[1]) u.subarray(o, o + T[1]).reverse();
    data = new T[0](u.buffer);
  }
  if (slope !== 1 || inter !== 0) { const f = new Float32Array(n3); for (let i = 0; i < n3; i++) f[i] = data[i] * slope + inter; data = f; }
  return { dims: [nx, ny, nz], spacing: pix, data, affine: A, datatype, dim4, ndim };
}

/** Переставить оси и отразить так, чтобы индексы i,j,k шли к Right, Anterior, Superior. */
export function toRAS(img) {
  const { dims, affine: A, spacing } = img;
  const ax = [0, 0, 0], flip = [false, false, false]; // для мировой оси w: какая ось массива, отражена ли
  const used = new Set();
  const cand = [];
  for (let a = 0; a < 3; a++) for (let w = 0; w < 3; w++) cand.push([Math.abs(A[w][a]), a, w]);
  cand.sort((p, q) => q[0] - p[0]);
  const wa = {};
  for (const [, a, w] of cand) {
    if (used.has(a) || wa[w] !== undefined) continue;
    used.add(a); wa[w] = a; ax[w] = a; flip[w] = A[w][a] < 0;
  }
  const on = [dims[ax[0]], dims[ax[1]], dims[ax[2]]];
  const osp = [spacing[ax[0]], spacing[ax[1]], spacing[ax[2]]];
  const stride = [1, dims[0], dims[0] * dims[1]];
  const out = new img.data.constructor(on[0] * on[1] * on[2]);
  let p = 0;
  for (let k = 0; k < on[2]; k++) {
    const kk = (flip[2] ? on[2] - 1 - k : k) * stride[ax[2]];
    for (let j = 0; j < on[1]; j++) {
      const jj = kk + (flip[1] ? on[1] - 1 - j : j) * stride[ax[1]];
      const sx = stride[ax[0]];
      if (!flip[0]) for (let i = 0; i < on[0]; i++) out[p++] = img.data[jj + i * sx];
      else for (let i = 0; i < on[0]; i++) out[p++] = img.data[jj + (on[0] - 1 - i) * sx];
    }
  }
  return { dims: on, spacing: osp, data: out };
}

/** Нормализация интенсивности в uint8 (1..255, 0 — фон) по перцентилям ненулевых вокселей. */
export function normalizeU8(data, n = data.length) {
  const step = Math.max(1, Math.floor(n / 400000));
  const samp = [];
  for (let i = 0; i < n; i += step) { const v = data[i]; if (v > 0 && isFinite(v)) samp.push(v); }
  if (!samp.length) return new Uint8Array(n);
  samp.sort((a, b) => a - b);
  const lo = samp[Math.floor(samp.length * 0.005)], hi = samp[Math.min(samp.length - 1, Math.floor(samp.length * 0.996))];
  const out = new Uint8Array(n), sc = 254 / (hi - lo || 1);
  for (let i = 0; i < n; i++) {
    const v = data[i];
    out[i] = v > 0 && isFinite(v) ? Math.max(1, Math.min(255, Math.round((v - lo) * sc + 1))) : 0;
  }
  return out;
}

/** Загрузка demo-файла (uint8 NIfTI, уже RAS) или пользовательского файла → Vol */
export async function loadVol(src, { normalize = false, name = '' } = {}) {
  const buf = await toArrayBuffer(src);
  const img = parseNifti(buf);
  const r = toRAS(img);
  const data = normalize || !(r.data instanceof Uint8Array) ? normalizeU8(r.data) : r.data;
  return new Vol(r.dims[0], r.dims[1], r.dims[2], data, r.spacing, { name, dim4: img.dim4, affine: img.affine, datatype: img.datatype });
}

/** Метка-объём (целые 0..4) в том же виде, без нормализации */
export async function loadLabels(src) {
  const buf = await toArrayBuffer(src);
  const img = parseNifti(buf);
  const r = toRAS(img);
  const u = new Uint8Array(r.data.length);
  for (let i = 0; i < u.length; i++) u[i] = Math.max(0, Math.min(255, Math.round(r.data[i])));
  return new Vol(r.dims[0], r.dims[1], r.dims[2], u, r.spacing, { affine: img.affine });
}
