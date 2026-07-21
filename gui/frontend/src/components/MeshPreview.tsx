import { useEffect, useRef, useState, type ChangeEvent } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { X, Loader2, Play, Pause } from 'lucide-react';
import { toast } from 'sonner';

/**
 * In-browser 3D preview of a sequence's fitted SMPL-X bodies.
 *
 * The backend ships only parameters (~1-2 MB): rest-pose vertices, skin
 * weights, per-frame joint rotations and root translations. The browser does
 * the forward-kinematics + linear-blend skinning on the GPU via
 * THREE.SkinnedMesh, so the modal opens near-instantly and scrubs smoothly.
 */

interface ScenePerson {
  bodyId: number;
  color: string;
  frameCount: number;
  vShaped: string;    // b64 float32 (V,3) — rest-pose vertices, metres
  restJoints: string; // b64 float32 (55,3) — rest-pose joints, metres
  poses: string;      // b64 float32 (T,55,3) — per-frame axis-angle rotations
  transl: string;     // b64 float32 (T,3) — per-frame root translation, metres
}

interface SceneBlob {
  version: number;
  seq: string;
  fps: number;
  /** "recorded" = the capture's own rate is baked in; "fallback" = server
   *  default — a client-side fpsHint (capture.json FPS) takes precedence. */
  fpsSource?: 'recorded' | 'fallback';
  upAxis: string;             // signed axis: x|y|z|-x|-y|-z — data's UP direction
  floor: number | null;       // floor height along upAxis
  vertexCount: number;
  jointCount: number;
  parents: number[];          // kinematic parent per joint, root = -1
  faces: string;              // b64 uint32 (F,3)
  skinIndex: string;          // b64 uint16 (V,4)
  skinWeight: string;         // b64 float32 (V,4)
  frameCount: number;
  people: ScenePerson[];
}

/** base64 → Uint8Array. The machine is little-endian, matching the packed blob. */
function b64ToBytes(b64: string): Uint8Array {
  const bin = atob(b64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return u8;
}
const asF32 = (b64: string): Float32Array => { const b = b64ToBytes(b64); return new Float32Array(b.buffer, 0, b.byteLength / 4); };
const asU32 = (b64: string): Uint32Array => { const b = b64ToBytes(b64); return new Uint32Array(b.buffer, 0, b.byteLength / 4); };
const asU16 = (b64: string): Uint16Array => { const b = b64ToBytes(b64); return new Uint16Array(b.buffer, 0, b.byteLength / 2); };

/** Signed-axis string ("z", "-y", …) → unit Vector3. */
function axisVec(axis: string): THREE.Vector3 {
  const neg = axis.startsWith('-');
  const a = neg ? axis.slice(1) : axis;
  const v = new THREE.Vector3(a === 'x' ? 1 : 0, a === 'y' ? 1 : 0, a === 'z' ? 1 : 0);
  return neg ? v.negate() : v;
}

/** A posable person: skinned mesh + its bones + precomputed per-frame quaternions. */
interface Rig {
  mesh: THREE.SkinnedMesh;
  bones: THREE.Bone[];
  quats: Float32Array;   // (T,J,4) — precomputed once for playback speed
  transl: Float32Array;  // (T,3)
  rj0: [number, number, number];
  restMin: [number, number, number];  // rest-joint AABB, for camera framing
  restMax: [number, number, number];
  frameCount: number;
}

/** Imperative handle onto the running viewer, so React controls can drive it. */
interface Engine {
  seek: (frame: number) => void;
}

interface Meta { fps: number; frameCount: number; people: { bodyId: number; color: string }[]; }

const SURFACE_BG = '#101114'; // approximates --background (oklch 0.160 0.018 252)

/**
 * The viewer proper: scene fetch, three.js build, and the footer playback
 * controls — everything except the modal shell. Rendered both inline (as a
 * card on the Results page) and inside the `MeshPreview` fullscreen modal.
 * Callers that can swap `target` in place should pass a stable `key`, since the
 * fetch effect keys off `target.*` fields rather than re-running on identity.
 */
export function MeshViewer({ target, className, fpsHint }: {
  target: { tag: string; capture: string; seq: string; ma_3d_dir: string; ma_cap_dir?: string };
  className?: string;
  /** Playback rate to use when the blob has no recorded capture rate. */
  fpsHint?: number | null;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const blobRef = useRef<SceneBlob | null>(null);
  const engineRef = useRef<Engine | null>(null);
  const [phase, setPhase] = useState<'loading' | 'error' | 'ready'>('loading');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [meta, setMeta] = useState<Meta | null>(null);

  // Playback UI state. The rAF loop reads `playing`/`speed` through refs so its
  // closure never goes stale; `frame` mirrors the loop's integer frame for the
  // scrubber (updated at most fps times/sec).
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [frame, setFrame] = useState(0);
  const playingRef = useRef(playing);
  const speedRef = useRef(speed);
  useEffect(() => { playingRef.current = playing; }, [playing]);
  useEffect(() => { speedRef.current = speed; }, [speed]);

  // Fetch the scene blob once. One request + spinner: generation is ~1-2 s on
  // the first call and instant when cached.
  useEffect(() => {
    let cancelled = false;
    const q = new URLSearchParams({
      tag: target.tag, capture: target.capture, seq: target.seq, ma_3d_dir: target.ma_3d_dir,
    });
    if (target.ma_cap_dir) q.set('ma_cap_dir', target.ma_cap_dir);
    fetch(`/api/exporter/preview-scene?${q.toString()}`)
      .then(async (res) => {
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.error || `Failed to prepare the 3D scene (${res.status})`);
        return data as SceneBlob;
      })
      .then((blob) => {
        if (cancelled) return;
        blobRef.current = blob;
        setMeta({ fps: blob.fps, frameCount: blob.frameCount, people: blob.people.map(p => ({ bodyId: p.bodyId, color: p.color })) });
        setPhase('ready');
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        const msg = e instanceof Error ? e.message : String(e);
        console.error('[MeshPreview] fetch failed:', e);
        setErrorMsg(msg);
        toast.error(msg);
        setPhase('error');
      });
    return () => { cancelled = true; };
  }, [target.tag, target.capture, target.seq, target.ma_3d_dir, target.ma_cap_dir]);

  // Build and drive the three.js viewer once the scene is loaded and the canvas
  // container is mounted.
  useEffect(() => {
    if (phase !== 'ready') return;
    const container = containerRef.current;
    const blob = blobRef.current;
    if (!container || !blob) return;

    const J = blob.jointCount;
    const parents = blob.parents;
    const faces = asU32(blob.faces);
    const skinIndex = asU16(blob.skinIndex);
    const skinWeight = asF32(blob.skinWeight);

    // --- Build one posable rig per person -----------------------------------
    const rigs: Rig[] = [];
    for (const person of blob.people) {
      const vShaped = asF32(person.vShaped);
      const restJoints = asF32(person.restJoints);
      const poses = asF32(person.poses);
      const transl = asF32(person.transl);
      const T = person.frameCount;

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(vShaped, 3));
      geometry.setAttribute('skinIndex', new THREE.Uint16BufferAttribute(skinIndex, 4));
      geometry.setAttribute('skinWeight', new THREE.Float32BufferAttribute(skinWeight, 4));
      geometry.setIndex(new THREE.BufferAttribute(faces, 1));
      geometry.computeVertexNormals();

      // Bones in a rest pose: each local position is its offset from its parent
      // (root keeps its absolute rest position).
      const bones: THREE.Bone[] = [];
      for (let j = 0; j < J; j++) bones.push(new THREE.Bone());
      for (let j = 0; j < J; j++) {
        const p = parents[j];
        if (p < 0) {
          bones[j].position.set(restJoints[0], restJoints[1], restJoints[2]);
        } else {
          bones[j].position.set(
            restJoints[j * 3] - restJoints[p * 3],
            restJoints[j * 3 + 1] - restJoints[p * 3 + 1],
            restJoints[j * 3 + 2] - restJoints[p * 3 + 2],
          );
          bones[p].add(bones[j]);
        }
      }

      const material = new THREE.MeshStandardMaterial({ color: person.color, roughness: 0.75, metalness: 0.05 });
      const mesh = new THREE.SkinnedMesh(geometry, material);
      // Skinning translates far from the origin via the root bone; the rest-pose
      // bounding sphere would otherwise cull the mesh incorrectly.
      mesh.frustumCulled = false;
      mesh.add(bones[0]);
      // Capture the rest pose into the bind matrices BEFORE binding.
      mesh.updateMatrixWorld(true);
      mesh.bind(new THREE.Skeleton(bones));

      // Precompute every frame's joint quaternions once; playback just copies 4
      // floats per bone.
      const quats = new Float32Array(T * J * 4);
      const q = new THREE.Quaternion();
      const axis = new THREE.Vector3();
      for (let f = 0; f < T; f++) {
        for (let j = 0; j < J; j++) {
          const o = (f * J + j) * 3;
          const rx = poses[o], ry = poses[o + 1], rz = poses[o + 2];
          const angle = Math.hypot(rx, ry, rz);
          if (angle < 1e-8) q.set(0, 0, 0, 1);
          else q.setFromAxisAngle(axis.set(rx / angle, ry / angle, rz / angle), angle);
          const qo = (f * J + j) * 4;
          quats[qo] = q.x; quats[qo + 1] = q.y; quats[qo + 2] = q.z; quats[qo + 3] = q.w;
        }
      }

      let minx = Infinity, miny = Infinity, minz = Infinity, maxx = -Infinity, maxy = -Infinity, maxz = -Infinity;
      for (let j = 0; j < J; j++) {
        minx = Math.min(minx, restJoints[j * 3]); maxx = Math.max(maxx, restJoints[j * 3]);
        miny = Math.min(miny, restJoints[j * 3 + 1]); maxy = Math.max(maxy, restJoints[j * 3 + 1]);
        minz = Math.min(minz, restJoints[j * 3 + 2]); maxz = Math.max(maxz, restJoints[j * 3 + 2]);
      }

      rigs.push({
        mesh, bones, quats, transl, frameCount: T,
        rj0: [restJoints[0], restJoints[1], restJoints[2]],
        restMin: [minx, miny, minz], restMax: [maxx, maxy, maxz],
      });
    }

    // --- Assemble the scene -------------------------------------------------
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(SURFACE_BG);

    // World group holds all people; rotate it so the data's up-axis maps to +Y.
    const world = new THREE.Group();
    world.quaternion.copy(new THREE.Quaternion().setFromUnitVectors(axisVec(blob.upAxis), new THREE.Vector3(0, 1, 0)));
    for (const r of rigs) world.add(r.mesh);
    scene.add(world);
    world.updateMatrixWorld(true);

    scene.add(new THREE.HemisphereLight(0xffffff, 0x3a3f4a, 1.0));
    const dirLight = new THREE.DirectionalLight(0xffffff, 1.4);
    dirLight.position.set(5, 10, 7);
    scene.add(dirLight);

    // Frame-0 world-space bounding box. Skinning isn't reflected by
    // Box3.setFromObject (it reads the rest geometry), so we build it from each
    // person's rest-joint extent offset by their frame-0 translation, then rotate
    // into world space — robust for multi-person scenes spread across the floor.
    const box = new THREE.Box3();
    const corner = new THREE.Vector3();
    for (const r of rigs) {
      const tx = r.transl[0], ty = r.transl[1], tz = r.transl[2];
      for (const cx of [r.restMin[0], r.restMax[0]]) for (const cy of [r.restMin[1], r.restMax[1]]) for (const cz of [r.restMin[2], r.restMax[2]]) {
        corner.set(cx + tx, cy + ty, cz + tz).applyQuaternion(world.quaternion);
        box.expandByPoint(corner);
      }
    }
    if (box.isEmpty()) box.set(new THREE.Vector3(-1, 0, -1), new THREE.Vector3(1, 2, 1));
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const diag = Math.max(size.length(), 1e-3);

    // Ground grid: added to the SCENE (not the rotated world) at world-Y = floor,
    // so the data's floor height lands correctly after the up-axis rotation.
    const floorY = blob.floor ?? 0;
    const gridSize = Math.max(10, Math.ceil(Math.max(size.x, size.z) + 6));
    const grid = new THREE.GridHelper(gridSize, gridSize, 0x333744, 0x23262e);
    grid.position.set(center.x, floorY, center.z);
    scene.add(grid);

    const container0 = container;
    const width = container0.clientWidth || 1;
    const height = container0.clientHeight || 1;

    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 500);
    const dist = Math.max(diag * 1.6, 3);
    camera.position.copy(center).addScaledVector(new THREE.Vector3(1, 0.6, 1).normalize(), dist);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    container0.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.copy(center);
    controls.update();

    // --- Posing + playback loop --------------------------------------------
    const poseAll = (f: number) => {
      for (const r of rigs) {
        const fi = f >= r.frameCount ? r.frameCount - 1 : f; // clamp shorter tracks
        for (let j = 0; j < J; j++) {
          const qo = (fi * J + j) * 4;
          r.bones[j].quaternion.set(r.quats[qo], r.quats[qo + 1], r.quats[qo + 2], r.quats[qo + 3]);
        }
        r.bones[0].position.set(
          r.rj0[0] + r.transl[fi * 3],
          r.rj0[1] + r.transl[fi * 3 + 1],
          r.rj0[2] + r.transl[fi * 3 + 2],
        );
      }
    };

    // The recorded capture rate always wins; the host's hint covers blobs
    // where ma_cap never recorded one; 30 is the last resort.
    const fps = blob.fpsSource === 'recorded' ? blob.fps
      : (fpsHint && fpsHint > 0 ? fpsHint : blob.fps);
    const total = blob.frameCount;
    let raf = 0;
    let last = performance.now();
    let floatFrame = 0;
    let curInt = -1;

    const tick = (now: number) => {
      raf = requestAnimationFrame(tick);
      const dt = Math.min((now - last) / 1000, 0.1);
      last = now;
      if (playingRef.current && total > 1) {
        floatFrame = (floatFrame + fps * speedRef.current * dt) % total;
        if (floatFrame < 0) floatFrame += total;
      }
      const fi = Math.floor(floatFrame);
      if (fi !== curInt) {
        curInt = fi;
        poseAll(fi);
        setFrame(fi);
      }
      controls.update();
      renderer.render(scene, camera);
    };
    raf = requestAnimationFrame(tick);

    const ro = new ResizeObserver(() => {
      const w = container0.clientWidth, h = container0.clientHeight;
      if (!w || !h) return;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    });
    ro.observe(container0);

    // Scrubbing sets the accumulator and forces a re-pose on the next tick.
    engineRef.current = {
      seek: (f: number) => { floatFrame = f; curInt = -1; },
    };

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      controls.dispose();
      for (const r of rigs) {
        r.mesh.geometry.dispose();
        (r.mesh.material as THREE.Material).dispose();
        r.mesh.skeleton.dispose();
      }
      grid.geometry.dispose();
      (grid.material as THREE.Material).dispose();
      // Explicitly release the WebGL context so a quick reopen doesn't exhaust
      // the tab's context pool before GC reclaims the old canvas.
      const canvas = renderer.domElement;
      renderer.dispose();
      canvas.getContext('webgl2')?.getExtension('WEBGL_lose_context')?.loseContext();
      if (canvas.parentNode) canvas.parentNode.removeChild(canvas);
      engineRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [phase, fpsHint]);

  const onScrub = (e: ChangeEvent<HTMLInputElement>) => {
    const f = Number(e.target.value);
    setFrame(f);
    engineRef.current?.seek(f);
  };

  const total = meta?.frameCount ?? 0;

  return (
    <div className={"flex flex-col " + (className ?? '')}>
      <div className="relative flex-1 min-h-0 bg-background">
        {phase === 'loading' && (
          <div className="absolute inset-0 flex items-center justify-center text-foreground-muted text-sm gap-2 pointer-events-none">
            <Loader2 className="w-4 h-4 animate-spin" />
            Preparing 3D scene…
          </div>
        )}
        {phase === 'error' && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-center px-6">
            <div className="text-status-failed text-sm mb-2">Couldn't build the 3D preview.</div>
            {errorMsg && (
              <pre className="text-foreground-faint text-[11px] font-mono max-w-2xl whitespace-pre-wrap break-all">{errorMsg}</pre>
            )}
          </div>
        )}
        {/* The renderer's canvas mounts inside this div (only while ready). */}
        {phase === 'ready' && <div ref={containerRef} className="w-full h-full" />}
      </div>

      {phase === 'ready' && meta && (
        <div className="flex items-center gap-3 px-3 py-2 bg-surface-2 border-t border-border text-xs">
          <button
            onClick={() => setPlaying(p => !p)}
            className="inline-flex items-center justify-center w-7 h-7 rounded-md bg-surface-3 hover:bg-surface-4 border border-border text-foreground-muted hover:text-foreground transition-colors flex-shrink-0"
            aria-label={playing ? 'Pause' : 'Play'}
            title={playing ? 'Pause' : 'Play'}
          >
            {playing ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
          </button>
          <input
            type="range"
            min={0}
            max={Math.max(total - 1, 0)}
            value={frame}
            onChange={onScrub}
            className="flex-1 accent-primary cursor-pointer"
            aria-label="Frame"
          />
          <span className="font-mono text-foreground-muted tabular-nums flex-shrink-0">
            {frame + 1} / {total}
          </span>
          <label className="flex items-center gap-1.5 text-foreground-muted flex-shrink-0">
            Speed
            <select
              value={speed}
              onChange={e => setSpeed(Number(e.target.value))}
              className="bg-surface-3 border border-border rounded px-1.5 py-0.5 text-foreground"
            >
              <option value={0.25}>0.25×</option>
              <option value={0.5}>0.5×</option>
              <option value={1}>1×</option>
              <option value={2}>2×</option>
            </select>
          </label>
          <div className="flex items-center gap-2 flex-shrink-0 pl-1" title={`${meta.people.length} ${meta.people.length === 1 ? 'body' : 'bodies'}`}>
            <span className="text-foreground-faint">{meta.people.length} {meta.people.length === 1 ? 'body' : 'bodies'}</span>
            <div className="flex items-center gap-1">
              {meta.people.map(p => (
                <span key={p.bodyId} className="w-2.5 h-2.5 rounded-full ring-1 ring-inset ring-white/10" style={{ backgroundColor: p.color }} />
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Fullscreen modal wrapper around `MeshViewer`: overlay + card + header, with
 * Escape and backdrop-click close. The viewer's fetch/build/controls all live
 * in `MeshViewer`; this shell just frames it.
 */
export function MeshPreview({ target, onClose, fpsHint }: {
  target: { tag: string; capture: string; seq: string; ma_3d_dir: string; ma_cap_dir?: string };
  onClose: () => void;
  fpsHint?: number | null;
}) {
  // Escape closes the modal.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 bg-black/85 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="w-[95vw] h-[90vh] rounded-xl bg-surface-1 border border-border flex flex-col shadow-2xl shadow-black/60 ring-1 ring-inset ring-white/[0.03] overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="flex items-center justify-between gap-3 p-3 border-b border-border-subtle bg-surface-2/40">
          <div className="min-w-0 flex items-center gap-2">
            <span className="text-foreground text-sm font-mono font-medium truncate" title={target.seq}>{target.seq}</span>
            <span className="text-foreground-faint text-sm">·</span>
            <span className="text-foreground-muted text-sm">3D preview</span>
          </div>
          <button
            onClick={onClose}
            className="text-foreground-subtle hover:text-foreground p-1.5 -m-1.5 rounded-md hover:bg-surface-3 transition-colors"
            aria-label="Close"
          >
            <X className="w-5 h-5" />
          </button>
        </header>

        <MeshViewer target={target} className="flex-1 min-h-0" fpsHint={fpsHint} />
      </div>
    </div>
  );
}
