// Общее состояние приложения и шина методов (заполняется в app.js)
export const S = {
  mode: 'demo',                 // 'demo' | 'user'
  caseId: null, meta: null, patient: null,
  vols: {}, mod: 't1ce',
  modNames: { t1: 'T1', t1ce: 'T1-Gd', t2: 'T2', flair: 'T2-FLAIR' },
  seg: null, segSrc: null, segDone: false, segFade: 1, segRunning: false,
  depthAll: null, depthEnh: null,
  follow: null, tp: 0, sim: null, simOn: false, simInfo: null,
  layers: {
    1: { show: true, opacity: 0.5, style: 'both' },
    2: { show: true, opacity: 0.32, style: 'fill' },
    3: { show: true, opacity: 0.55, style: 'both' },
  },
  vrOpacity: { 1: 0.85, 2: 0.24, 3: 0.72 },
  brainOpacity: 0.3, brainThr: 0.2, clip: { on: false, axis: 'x', pos: 0.5, flip: false },
  cursor: [0, 0, 0], wl: {}, invert: false, tool: 'cross', crosshair: true,
  brush: { r: 5, label: 3 }, undo: [], redo: [],
  layout: 'quad', hiClass: 0, anon: false, confirmed: false, maskEdited: false, reportStale: false,
  metrics: null, baseMetrics: null, log: [], versions: [],
  reportText: '', reportSigned: false, consilium: [],
  treat: { view: 'home', params: null, sel: { kps: '', excluded: {}, cmp: [] } },
  cine: false, returned: null,
};
export const app = {};
