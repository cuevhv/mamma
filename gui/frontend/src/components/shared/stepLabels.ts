/**
 * Display-layer mapping from pipeline step identifiers to user-facing
 * labels. The identifiers (`ma_cap`, `ma_masks`, ...) remain the source
 * of truth everywhere they function as keys: backend ProcessType enum,
 * runner.ALL_STEPS, task.json step keys, ML repo module names, API
 * field values. This map is consulted *only* at the rendering boundary.
 */
export const STEP_LABELS: Record<string, string> = {
  ma_cap: 'Input',
  ma_masks: 'Masks',
  ma_2d: '2D Landmarks',
  ma_3d: '3D Fitting',
  ma_vis: 'Visualization',
  export: 'Exported animation files',
};

export function stepLabel(name: string): string {
  return STEP_LABELS[name] ?? name;
}
