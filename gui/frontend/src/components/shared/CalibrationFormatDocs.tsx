import { useState } from 'react';

/**
 * Interactive documentation for the calibration formats MAMMA accepts.
 *
 * Renders a row of format chips; clicking one reveals that format's structure —
 * a faithful minimal example plus notes. The examples mirror what the loaders in
 * `capture/loaders/` actually parse (key names, distortion model, extrinsics
 * convention, units), so they double as copy-from templates.
 *
 * Shared by every calibration-input hint (New Task + Captures create forms).
 */

interface FormatDoc {
  id: string;
  label: string;      // chip text
  ext: string;        // what to point the path at
  blurb: string;      // one/two-line description (convention · distortion · units)
  example: string;    // faithful minimal snippet
  note?: string;      // extra gotcha / tip
}

const FORMATS: FormatDoc[] = [
  {
    id: 'mamma',
    label: 'MAMMA YAML',
    ext: 'a single .yaml / .yml file',
    blurb: 'Recommended. Pinhole + radtan distortion, translations in metres, one file per rig. Declare the extrinsics convention explicitly (defaults to cam2world).',
    example: `extrinsics_convention: cam2world   # or world2cam
quaternion_order: wxyz             # or xyzw (default wxyz, Hamilton)
cameras:
  cam01:
    camera_model: pinhole
    distortion_model: radtan          # or vicon_radial_2
    intrinsics: [fx, fy, cx, cy]
    distortion_coeffs: [k1, k2, p1, p2]
    resolution: [W, H]
    translation: [tx, ty, tz]         # metres
    rotation_quaternion: [w, x, y, z]`,
    note: 'Worked examples ship under configs/examples/calib/ (e.g. iphones_outdoors.yaml).',
  },
  {
    id: 'opencv_fs',
    label: 'OpenCV FileStorage',
    ext: 'one .yml, or a folder of per-camera .yml files',
    blurb: 'OpenCV %YAML:1.0 files. Per camera: K (3×3), D (Brown-Conrady), R (3×3), T (3×1). Convention world→camera: X_cam = R·X_world + T.',
    example: `%YAML:1.0
image_width: 3840
image_height: 2160
K: !!opencv-matrix          # or camera_matrix
  rows: 3
  cols: 3
  dt: d
  data: [ fx, 0, cx,  0, fy, cy,  0, 0, 1 ]
D: !!opencv-matrix          # or distortion_coefficients
  rows: 1
  cols: 5
  dt: d
  data: [ k1, k2, p1, p2, k3 ]
R: !!opencv-matrix
  rows: 3
  cols: 3
  dt: d
  data: [ r11, r12, r13,  r21, r22, r23,  r31, r32, r33 ]
T: !!opencv-matrix
  rows: 3
  cols: 1
  dt: d
  data: [ tx, ty, tz ]`,
    note: 'A folder of these (cam01.yml, cam02.yml, …) is read as one rig — each file name is its camera name.',
  },
  {
    id: 'xcp',
    label: 'Vicon .xcp',
    ext: 'a .xcp file (XML)',
    blurb: 'Vicon\'s XML calibration export — you don\'t hand-write this, Shogun/Nexus produces it. Vicon 5-param radial distortion, positions in mm (converted to m), camera pose in world (JPL [x,y,z,w] quaternion). Only VideoFile cameras are read.',
    example: `<Cameras>
  <Camera TYPE="VideoFile" NAME="IOI_01"
          SENSOR_SIZE="3840 2160" PIXEL_ASPECT_RATIO="1.0">
    <KeyFrames>
      <KeyFrame FOCAL_LENGTH="2560.0"
                PRINCIPAL_POINT="1920 1080"
                VICON_RADIAL2="0 pp_x pp_y r1 r2 r3"
                ORIENTATION="x y z w"      <!-- JPL quaternion -->
                POSITION="tx ty tz" />     <!-- millimetres -->
    </KeyFrames>
  </Camera>
</Cameras>`,
    note: 'Export it straight from your Vicon software — no manual editing needed.',
  },
  {
    id: 'opencv_json',
    label: 'OpenCV .json',
    ext: 'a .json file (nested or flat)',
    blurb: 'OpenCV-style JSON. Per camera: intrinsic_matrix (3×3), distortions (5: k1,k2,p1,p2,k3), extrinsics_matrix (3×4, world→camera [R|t]), image_size [W,H].',
    example: `{
  "cam01": {
    "intrinsic_matrix": [ [fx, 0, cx], [0, fy, cy], [0, 0, 1] ],
    "distortions": [ k1, k2, p1, p2, k3 ],
    "extrinsics_matrix": [ [r11, r12, r13, tx],
                           [r21, r22, r23, ty],
                           [r31, r32, r33, tz] ],
    "image_size": [ W, H ]
  }
}`,
    note: 'The flat form above needs no sequence key. A nested { "seq00": { "cam01": … } } wrapper is also accepted (some tools export it) — MAMMA just unwraps it and uses the first sequence.',
  },
];

export function CalibrationFormatDocs() {
  const [openId, setOpenId] = useState<string | null>(null);
  const active = FORMATS.find(f => f.id === openId) ?? null;

  return (
    <div className="space-y-2">
      <div className="text-foreground-faint uppercase tracking-[0.14em] text-[10px]">
        Supported formats — click one to see its structure:
      </div>
      <div className="flex flex-wrap gap-1.5">
        {FORMATS.map(f => {
          const on = f.id === openId;
          return (
            <button
              key={f.id}
              type="button"
              aria-pressed={on}
              onClick={() => setOpenId(on ? null : f.id)}
              className={`px-2 py-0.5 rounded text-[10.5px] font-medium border transition-colors ${
                on
                  ? 'bg-primary text-primary-foreground border-primary/40'
                  : 'bg-surface-2 text-foreground-muted border-border hover:text-foreground hover:border-border-strong'
              }`}
            >
              {f.label}
            </button>
          );
        })}
      </div>

      {active && (
        <div className="rounded-md border border-border-subtle bg-surface-1/60 p-2.5 space-y-2">
          <div className="text-foreground-faint text-[10px]">
            Point at <span className="text-foreground-muted">{active.ext}</span>.
          </div>
          <p className="text-foreground-faint">{active.blurb}</p>
          <pre className="font-mono text-foreground-subtle leading-relaxed whitespace-pre overflow-x-auto text-[10.5px]">{active.example}</pre>
          {active.note && <p className="text-foreground-faint">{active.note}</p>}
        </div>
      )}
    </div>
  );
}
