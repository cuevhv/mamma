import { useCallback, useEffect, useRef, useState } from 'react';
import {
  AlertCircle,
  AlertTriangle,
  Check,
  ChevronDown,
  Download,
  FileJson,
  Folder,
  FolderOpen,
  Info,
  Loader2,
  Terminal,
  X,
} from 'lucide-react';
import { toast } from 'sonner';
import { MultiSelectDropdown } from './MultiSelectDropdown';
import { CalibrationStatus } from './CalibrationStatus';
import { UpAxisValue } from './UpAxisToggle';
import { PresetDigestCard, PresetDigest, PresetOverrides } from './PresetDigest';
import type { StepSettingsSchema } from '../lib/flagModel';
import { CommandPreviewModal, type CommandPreview } from './CommandPreviewModal';
import { CaptureSelect } from './CaptureSelect';
import { PresetPicker } from './PresetPicker';
import { stepLabel } from './shared/stepLabels';
import { formatTaskId } from './shared/formatTaskId';
import { CreateField, FieldRow, FootageHint, CalibrationHint } from './shared/CreateCaptureFields';

const DEFAULT_CAMERAS = Array.from({ length: 33 }, (_, i) => `IOI_${String(i + 1).padStart(2, '0')}`);

interface PresetSummary {
  name: string;
  displayName: string;
  description: string;
  path: string;
  /** "user" (writable) or "example" (read-only, shipped under configs/examples/presets/). */
  source?: 'user' | 'example';
}

/** Subset of /api/captures we use to populate the picker dropdown. */
interface CaptureSummary {
  id: string;
  captureName: string;
  jsonPath: string;
  seqNames: string[];
  cams: string[];
  thumbnailPath: string | null;
}

/** A previously-used Output ID for the picked capture. Powers the
 *  "extend a previous run" UX in the Output ID field. */
interface RunGroupSummary {
  outputId: string;
  submissions: number;
  lastSubmittedAt: string;
  outputDir: string;
  datasetName: string;
  /** {seqName: stepNamesAlreadyDone} — drives the smart-default for
   *  Run steps when the user picks this group. */
  stepsDone: Record<string, string[]>;
}

/** Mirrors the /api/captures/preflight response. Both sides are
 *  independent so the form can show two badges that flip green
 *  independently. */
interface PreflightResponse {
  footage: {
    ok: boolean;
    error: string | null;
    sequences: number;
    cameras: number;
    layout: 'videos' | 'images' | 'mixed' | null;
    firstSequence: string | null;
    sequenceNames: string[];
    cameraNames: string[];
  };
  calibration: {
    ok: boolean;
    error: string | null;
    cameraCount: number;
    distortionModels: string[];
    cameraNames: string[];
  };
}

type StepDoneStatus = 'all' | 'partial' | 'none';

/** "Across the chosen sequences, how done is this step in this run group?"
 *  Powers both the auto-default-uncheck and the inline status pill. */
function stepDoneStatus(
  group: RunGroupSummary | null,
  sequences: string[],
  stepName: string,
): { status: StepDoneStatus; doneCount: number; total: number } {
  if (!group || sequences.length === 0) return { status: 'none', doneCount: 0, total: sequences.length };
  let done = 0;
  for (const seq of sequences) {
    if ((group.stepsDone[seq] ?? []).includes(stepName)) done++;
  }
  if (done === 0) return { status: 'none', doneCount: 0, total: sequences.length };
  if (done === sequences.length) return { status: 'all', doneCount: done, total: sequences.length };
  return { status: 'partial', doneCount: done, total: sequences.length };
}

const RTF = typeof Intl !== 'undefined' && (Intl as any).RelativeTimeFormat
  ? new (Intl as any).RelativeTimeFormat('en', { numeric: 'auto' })
  : null;
function formatRelativeTime(iso: string): string {
  const t = Date.parse(iso);
  if (Number.isNaN(t) || !RTF) return iso;
  const diffMs = t - Date.now();
  const absMin = Math.abs(diffMs) / 60_000;
  if (absMin < 1) return 'just now';
  if (absMin < 60) return RTF.format(Math.round(diffMs / 60_000), 'minute');
  if (absMin < 60 * 24) return RTF.format(Math.round(diffMs / 3_600_000), 'hour');
  if (absMin < 60 * 24 * 30) return RTF.format(Math.round(diffMs / 86_400_000), 'day');
  return new Date(t).toLocaleDateString();
}

/** Quick basename of a posix-ish path. Used to default the output name
 *  from the footage root the user typed. */
function basenameOf(p: string): string {
  if (!p) return '';
  return p.replace(/\\/g, '/').replace(/\/+$/, '').split('/').pop() ?? '';
}

const CAPTURE_MODE_LS_KEY = 'mammaNewTaskCaptureMode';
const PREFLIGHT_DEBOUNCE_MS = 400;

/** Payload handed to the form when the user clicks "Re-run…" on a result.
 *  Pre-fills the form to continue that exact run (same output-id/dir/config)
 *  so the user only chooses which steps to force-re-run and edits their config. */
export interface ReRunInit {
  /** The run's task id — its frozen config is loaded as the config baseline. */
  taskId: string;
  /** capture.json path (relative to the mount, or absolute) — locked. */
  captureJsonPath: string;
  /** The run's output-id + out_dir — the re-run reads its reused steps from here. */
  outputId: string;
  outputDir: string;
  /** Sequences the run covered; pre-selected. */
  seqNames: string[];
}

/** Steps that depend (transitively) on `stepName`, given the digest's
 *  per-step `dependencies`. Used to auto-include downstream steps when an
 *  upstream one is marked for re-run (e.g. ma_3d → ma_vis). */
function downstreamSteps(stepName: string, steps: { name: string; dependencies: string[] }[]): string[] {
  const out = new Set<string>();
  const stack = [stepName];
  while (stack.length) {
    const cur = stack.pop()!;
    for (const s of steps) {
      if (s.dependencies.includes(cur) && !out.has(s.name)) {
        out.add(s.name);
        stack.push(s.name);
      }
    }
  }
  return [...out];
}

interface Props {
  /** Called with the new task's id on successful submission. */
  onSubmitted?: (taskId: number) => void;
  /** When set, the form opens in "Re-run" mode pre-filled to continue this run. */
  reRunInit?: ReRunInit | null;
}

/**
 * Three-step task-submission form:
 *   Step 1 — Capture: foreground "Create new" (footage + calibration with
 *            live preflight); "Pick existing" toggle as the secondary mode.
 *   Step 2 — Pipeline Configuration Preset: existing preset dropdown +
 *            PresetDigestCard, just relabelled here.
 *   Step 3 — Run details: sequences, cameras, output dir/id, run steps,
 *            sequence order, submit.
 *
 * Step 1 is gated by completed[1] (capture chosen + sequences loaded).
 * Step 2 by completed[2] (preset + digest loaded). Step 3 is the
 * terminal step with the existing canSubmit check.
 *
 * No behaviour changes inside Step 2 or Step 3 — they're the same
 * controls in a different chrome. Step 1 replaces the old flat
 * dropdown + "Use a custom path…" affordance with the new Create /
 * Pick toggle.
 */
export function NewTaskForm({ onSubmitted, reRunInit }: Props) {
  const isReRun = !!reRunInit;
  // --- step accordion ---
  const [activeStep, setActiveStep] = useState<1 | 2 | 3>(1);
  const [completed1, setCompleted1] = useState(false);
  const [completed2, setCompleted2] = useState(false);
  // Re-run: the steps to redo. They run under a FRESH output-id; the unchecked
  // (earlier) steps are read from the source run via path overrides — nothing
  // is copied or overwritten (MAMMA steps skip when outputs already exist, so
  // re-running in place can't force a redo; a new run is the reliable way).
  const [forceSteps, setForceSteps] = useState<Set<string>>(new Set());

  // --- capture (Step 1) ---
  const [captureJsonPath, setCaptureJsonPath] = useState('');
  const [captures, setCaptures] = useState<CaptureSummary[]>([]);
  const [capturePathError, setCapturePathError] = useState('');
  /** Step 1 has two modes. Create = on-the-spot capture-json creation
   *  (primary). Pick = the existing-captures dropdown (secondary). The
   *  old "custom path" free-text input is gone — hand-edited capture
   *  files belong under configs/examples/captures/ where they appear
   *  in the dropdown as 'example' rows. */
  const [captureMode, setCaptureMode] = useState<'create' | 'pick'>(() => {
    try {
      const stored = localStorage.getItem(CAPTURE_MODE_LS_KEY);
      return stored === 'pick' ? 'pick' : 'create';
    } catch {
      return 'create';
    }
  });
  // Create-mode inputs.
  const [createIoiRoot, setCreateIoiRoot] = useState('');
  const [createCalib, setCreateCalib] = useState('');
  const [createOutputName, setCreateOutputName] = useState('');
  const [createCamFps, setCreateCamFps] = useState('30');   // capture.json cam_fps
  // World up-axis chosen in the form ('auto' = let the backend auto-detect).
  // Persisted into the new capture.json by /api/captures/generate-json.
  const [createUpAxis, setCreateUpAxis] = useState<UpAxisValue>('auto');
  // Per-input "what goes here?" disclosures — toggled by a small (?) icon
  // next to each field's label, so the user can read the structure for
  // the field they're hovering rather than wading through a combined hint.
  const [footageHintOpen, setFootageHintOpen] = useState(false);
  const [calibHintOpen, setCalibHintOpen] = useState(false);
  // When the user types into the Capture name field, mark it touched so
  // we stop auto-syncing it from the footage basename. Reset to untouched
  // on a clear, so the auto-sync kicks back in.
  const [outputNameTouched, setOutputNameTouched] = useState(false);
  const [preflight, setPreflight] = useState<PreflightResponse | null>(null);
  const [preflightInFlight, setPreflightInFlight] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  // When the backend returns 409 for an existing capture name, we
  // remember the conflicting name so the panel can render an
  // "Overwrite?" confirmation. Cleared once the user picks Overwrite
  // or Cancel.
  const [pendingOverwrite, setPendingOverwrite] = useState<{ name: string } | null>(null);

  // --- preset (Step 2) ---
  const [presets, setPresets] = useState<PresetSummary[]>([]);
  const [stepSettings, setStepSettings] = useState<StepSettingsSchema>({});
  const [presetName, setPresetName] = useState<string>('');
  const [digest, setDigest] = useState<PresetDigest | null>(null);
  const [stepEnabled, setStepEnabled] = useState<Record<string, boolean>>({});
  // Inline edits to the preset for this submission only. Cleared when the
  // preset selection changes so users start from a clean slate.
  const [overrides, setOverrides] = useState<PresetOverrides>({});

  // --- run details (Step 3) ---
  const [availableSeqNames, setAvailableSeqNames] = useState<string[]>([]);
  const [selectedSeqNames, setSelectedSeqNames] = useState<string[]>([]);
  // Empty by default — the cameras list is sourced from the selected
  // capture. DEFAULT_CAMERAS is still used as a legacy fallback inside
  // fetchSequences() when a capture.json doesn't list its cameras.
  const [availableCameras, setAvailableCameras] = useState<string[]>([]);
  const [selectedCameras, setSelectedCameras] = useState<string[]>([]);
  const [outputDir, setOutputDir] = useState('');
  const [outputId, setOutputId] = useState('');
  // Execution (engine + where-to-run) now lives in the preset's global block
  // (PresetDigestCard). It flows to the backend via `overrides.global`, so the
  // form no longer holds engine/cluster state here.
  /** Existing run groups for the picked capture — populated when capture
   *  changes. The form uses this to suggest "continue a previous run"
   *  affordances and surface what's already done. */
  const [runGroups, setRunGroups] = useState<RunGroupSummary[]>([]);

  // Per-task sequence dispatch order. Form default is sequence-major
  // — finish each sequence end-to-end before the next, so the user
  // sees a complete result for sequence 1 before sequence 2 starts.
  const [sequenceMajor, setSequenceMajor] = useState(true);

  const [submitting, setSubmitting] = useState(false);

  // Persist the Create/Pick mode so power users who always Pick don't
  // get re-prompted to Create. Wrapped in try/catch so storage failures
  // in private-window contexts don't break the form.
  useEffect(() => {
    try { localStorage.setItem(CAPTURE_MODE_LS_KEY, captureMode); }
    catch { /* swallow */ }
  }, [captureMode]);

  // --- effects ----------------------------------------------------------

  useEffect(() => {
    // Populate the picker from the same DB-backed list the Captures tab
    // uses, not the disk listing — captures stored anywhere on disk
    // appear in this dropdown the moment they're created.
    fetch('/api/captures').then(r => r.ok ? r.json() : []).then(setCaptures).catch(() => {});
    fetch('/api/task-presets').then(r => r.ok ? r.json() : []).then((list: PresetSummary[]) => {
      setPresets(list);
      // Don't auto-pick a preset in Re-run mode — the config comes from the
      // run's frozen config, loaded separately below.
      if (!isReRun && list.length > 0 && !presetName) setPresetName(list[0].name);
    }).catch(() => {});
    // Curated per-step "common settings" schema for the friendly widgets.
    // Static catalogue, so fetch once; failures degrade to raw flags only.
    fetch('/api/steps/settings').then(r => r.ok ? r.json() : {}).then(setStepSettings).catch(() => {});
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (isReRun) return;  // digest comes from the run's frozen config, not a preset
    if (!presetName) { setDigest(null); setOverrides({}); setCompleted2(false); return; }
    let cancelled = false;
    fetch(`/api/task-presets/${encodeURIComponent(presetName)}/digest`)
      .then(r => r.ok ? r.json() : null)
      .then((d: PresetDigest | null) => {
        if (cancelled) return;
        if (!d) {
          setDigest(null);
          setStepEnabled({});
          setOverrides({});
          setCompleted2(false);
          toast.error(`Could not load preset '${presetName}'`);
          return;
        }
        setDigest(d);
        const init: Record<string, boolean> = {};
        for (const s of d.steps) init[s.name] = s.enabled;
        setStepEnabled(init);
        setOverrides({});
        setCompleted2(true);
      })
      .catch(() => { if (!cancelled) { setDigest(null); setCompleted2(false); } });
    return () => { cancelled = true; };
  }, [presetName]);

  useEffect(() => {
    setSelectedCameras(prev => prev.filter(c => availableCameras.includes(c)));
  }, [availableCameras]);

  // Refresh the run-groups list when the picked capture changes.
  // Output ID is namespace-scoped to the capture (outputs land at
  // <out_dir>/<step>/<output_id>/<dataset>/<seq>/) so reusing an ID
  // across captures would write into the wrong dataset namespace —
  // we clear it on capture switch.
  useEffect(() => {
    if (!captureJsonPath) {
      setRunGroups([]);
      setOutputId('');
      return;
    }
    let cancelled = false;
    fetch(`/api/captures/run-groups?path=${encodeURIComponent(captureJsonPath)}`)
      .then(r => r.ok ? r.json() : { runGroups: [] })
      .then((d: { runGroups: RunGroupSummary[] }) => {
        if (cancelled) return;
        setRunGroups(d.runGroups || []);
      })
      .catch(() => { if (!cancelled) setRunGroups([]); });
    setOutputId('');
    return () => { cancelled = true; };
  }, [captureJsonPath]);

  // Re-run mode: pre-fill the form from the given run. Runs once on mount.
  // Sequences + cameras are FIXED to what the run used (not editable) to avoid
  // any mismatch with the reused steps. The output-id is left blank on purpose —
  // a re-run always writes to a fresh run id and reads the reused steps from the
  // source via paths. The config baseline is the run's frozen config.
  useEffect(() => {
    if (!reRunInit) return;
    let cancelled = false;
    setCaptureMode('pick');
    setCaptureJsonPath(reRunInit.captureJsonPath);
    setOutputDir(reRunInit.outputDir);
    setSelectedSeqNames(reRunInit.seqNames);   // fixed
    setCompleted1(true);
    setActiveStep(3);
    fetch(`/api/tasks/${encodeURIComponent(reRunInit.taskId)}/rerun-config`)
      .then(r => r.ok ? r.json() : null)
      .then((d: PresetDigest | null) => {
        if (cancelled || !d) {
          if (!cancelled) toast.error('Could not load the run configuration to re-run.');
          return;
        }
        setDigest(d);
        const init: Record<string, boolean> = {};
        for (const s of d.steps) init[s.name] = s.enabled;
        setStepEnabled(init);   // the run's pipeline; the re-run picks a subset
        setSelectedCameras(d.global.camNames || []);   // fixed to the run's cameras
        // Reconstruct the 'cluster' pseudo-engine. Submission resolves
        // engine 'cluster' into the profile's real engine (apptainer + sif)
        // before freezing the config, so a cluster run's config stores
        // engine='apptainer' — but cluster_profile survives. Without this,
        // the Engine dropdown pre-selects 'apptainer' and, worse,
        // re-submitting routes the run to THIS machine (cluster routing
        // only triggers on the literal engine 'cluster').
        setOverrides(d.global.clusterProfile
          ? { global: { engine: 'cluster', cluster_profile: d.global.clusterProfile } }
          : {});
        setCompleted2(true);
      })
      .catch(() => { if (!cancelled) toast.error('Could not reach the backend to load the run config.'); });
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Auto-fill the Capture name from the footage basename until the
  // user manually edits the field. Once they touch it, we stop tracking
  // — they own the value from then on.
  useEffect(() => {
    if (outputNameTouched) return;
    setCreateOutputName(basenameOf(createIoiRoot));
  }, [createIoiRoot, outputNameTouched]);

  // Debounced preflight: as the user types into Create-mode's footage /
  // calibration inputs, hit /api/captures/preflight to surface live
  // validation badges. AbortController so a stale slow response can't
  // overwrite a newer one.
  useEffect(() => {
    if (captureMode !== 'create') return;
    if (!createIoiRoot && !createCalib) { setPreflight(null); return; }

    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      setPreflightInFlight(true);
      try {
        const res = await fetch('/api/captures/preflight', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ioiRoot: createIoiRoot, calib: createCalib }),
          signal: controller.signal,
        });
        if (res.ok) {
          const data: PreflightResponse = await res.json();
          if (!controller.signal.aborted) setPreflight(data);
        }
      } catch (err: any) {
        if (err?.name !== 'AbortError') {
          // Network blip — keep the previous badge so the UI doesn't
          // flash empty on a transient.
        }
      } finally {
        if (!controller.signal.aborted) setPreflightInFlight(false);
      }
    }, PREFLIGHT_DEBOUNCE_MS);

    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [captureMode, createIoiRoot, createCalib]);

  // When preflight comes back green on BOTH sides while in Create mode,
  // promote the result into Step 3 immediately — the user can start
  // configuring the run without explicitly saving the capture. The
  // actual capture.json is written on the way to /api/tasks when they
  // click Run. If preflight goes red (or empties out), we clear the
  // dropdowns and bounce Step 1 back to "not complete".
  //
  // We deliberately skip this in Pick mode — there fetchSequences()
  // owns populating Step 3 from the picked capture file.
  useEffect(() => {
    if (captureMode !== 'create') return;
    if (preflight && preflight.footage.ok && preflight.calibration.ok) {
      setAvailableSeqNames(preflight.footage.sequenceNames);
      setAvailableCameras(preflight.footage.cameraNames);
      setSelectedSeqNames(prev => prev.filter(s => preflight.footage.sequenceNames.includes(s)));
      setSelectedCameras(prev => prev.filter(c => preflight.footage.cameraNames.includes(c)));
      setCapturePathError('');
      setCompleted1(true);
      setActiveStep(prev => prev === 1 ? 2 : prev);
    } else {
      // Clear when preflight is incomplete/red AND we haven't already
      // saved a capture (captureJsonPath stays empty in Create mode
      // until Run). This is the "you edited the path, dropdowns reset"
      // behaviour.
      setAvailableSeqNames([]);
      setAvailableCameras([]);
      setSelectedSeqNames([]);
      setSelectedCameras([]);
      setCompleted1(false);
    }
  }, [captureMode, preflight]);

  // Clear any stale "Overwrite?" confirm when the user edits the
  // Create-mode inputs after we showed one — they're clearly trying to
  // change something rather than confirm.
  useEffect(() => {
    setPendingOverwrite(null);
  }, [createIoiRoot, createCalib, createOutputName]);

  // Mode switch — keep entered paths so the user can flip back, but
  // clear the shared dropdowns + selections so they reflect only the
  // currently active mode's data. Skipped in Re-run mode, where the capture
  // + sequences are pre-filled and the mode toggle isn't shown (the reset
  // would otherwise wipe the pre-filled sequence list).
  useEffect(() => {
    if (isReRun) return;
    setCaptureJsonPath('');
    setAvailableSeqNames([]);
    setAvailableCameras([]);
    setSelectedSeqNames([]);
    setSelectedCameras([]);
    setCompleted1(false);
    setCapturePathError('');
    setPendingOverwrite(null);
  }, [captureMode, isReRun]);

  // --- callbacks --------------------------------------------------------

  const fetchSequences = useCallback(async (path: string) => {
    if (!path) return;
    try {
      const res = await fetch('/api/captures/parse-sequences', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ captureJsonPath: path }),
      });
      if (res.ok) {
        const d = await res.json();
        setAvailableSeqNames(d.sequences || []);
        setAvailableCameras(Array.isArray(d.cameras) && d.cameras.length > 0 ? d.cameras : DEFAULT_CAMERAS);
        setSelectedSeqNames([]);
        setCapturePathError(d.sequences?.length ? '' : 'No sequences found in capture file.');
        if (d.sequences?.length) {
          setCompleted1(true);
          // Auto-advance to Step 2 once a capture is locked in.
          setActiveStep(prev => prev === 1 ? 2 : prev);
        } else {
          setCompleted1(false);
        }
      } else {
        const err = await res.json().catch(() => ({}));
        setCapturePathError(`Could not load sequences: ${err.error || 'File not found'}`);
        setAvailableCameras([]);
        setAvailableSeqNames([]);
        setSelectedSeqNames([]);
        setCompleted1(false);
      }
    } catch {
      setCapturePathError('Failed to connect to server.');
      setCompleted1(false);
    }
  }, []);

  const adoptCapture = useCallback(async (path: string) => {
    setCaptureJsonPath(path);
    setCapturePathError('');
    await fetchSequences(path);
  }, [fetchSequences]);

  /** Resolve the capture.json path to use for a Run.
   *
   * Pick mode: returns the already-set captureJsonPath, or null if
   * nothing's picked.
   * Create mode: writes capture.json now (calls /generate-json with
   * the current paths + name). On a 409 name collision, sets
   * pendingOverwrite and returns null so the caller can wait for the
   * user's confirmation; pass overwrite=true to retry through. On
   * success, sets captureJsonPath and returns the absolute path.
   */
  const saveCaptureIfNeeded = useCallback(async (overwrite: boolean = false): Promise<string | null> => {
    if (captureMode === 'pick') return captureJsonPath || null;
    // Already saved earlier in this session (e.g. user clicked Run, got
    // a task started, came back to the form). The path is still valid.
    if (captureJsonPath) return captureJsonPath;

    setCreateError(null);
    try {
      const res = await fetch('/api/captures/generate-json', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ioiRoot: createIoiRoot,
          calib: createCalib,
          outputName: createOutputName || undefined,
          camFps: createCamFps,
          upAxis: createUpAxis,
          ...(overwrite ? { overwrite: true } : {}),
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({} as { error?: string; code?: string; existingName?: string }));
        if (res.status === 409 && err?.code === 'name_in_use') {
          setPendingOverwrite({
            name: err.existingName || createOutputName || basenameOf(createIoiRoot) || 'this capture',
          });
          return null;
        }
        toast.error(`Failed to save capture: ${err?.error || `HTTP ${res.status}`}`);
        return null;
      }
      const created = await res.json() as { outputName: string; path: string; sequenceCount: number };
      setPendingOverwrite(null);
      // Reconcile the captures list so we pick up the absolute
      // jsonPath that /api/captures returns from the DB row.
      let chosenPath = created.path;  // relative-to-MOUNT_POINT fallback
      try {
        const listRes = await fetch('/api/captures');
        if (listRes.ok) {
          const list: CaptureSummary[] = await listRes.json();
          setCaptures(list);
          const match = list.find(c => c.captureName === created.outputName);
          if (match) chosenPath = match.jsonPath;
        }
      } catch { /* fall through with the relative path */ }
      setCaptureJsonPath(chosenPath);
      toast.success(
        `${overwrite ? 'Overwrote' : 'Saved'} capture '${created.outputName}' · ${created.sequenceCount} seq${created.sequenceCount === 1 ? '' : 's'}`
      );
      return chosenPath;
    } catch {
      toast.error('Failed to reach the backend while saving the capture.');
      return null;
    }
  }, [captureMode, captureJsonPath, createIoiRoot, createCalib, createOutputName, createCamFps, createUpAxis]);

  // When the user changes any Create-mode input after a prior Run
  // saved a capture this session, invalidate captureJsonPath so the
  // next Run re-saves with the new values.
  useEffect(() => {
    if (captureMode === 'create' && captureJsonPath) {
      setCaptureJsonPath('');
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [createIoiRoot, createCalib, createOutputName, createCamFps]);

  // Frame range is only consumed by ma_cap (it bakes --start/--end into the
  // per-camera NPZs downstream steps inherit). Whenever ma_cap is deselected,
  // a pending range edit would silently do nothing — drop it so the submitted
  // config reflects reality. Used by both the normal Run-steps toggles and the
  // Re-run steps picker.
  const dropFrameRangeOverride = useCallback(() => {
    setOverrides(prev => {
      if (prev.global?.start_frame === undefined && prev.global?.end_frame === undefined) return prev;
      const { start_frame: _s, end_frame: _e, ...g } = prev.global ?? {};
      return { ...prev, global: g };
    });
  }, []);

  const submit = async (overwriteOnConflict: boolean = false) => {
    if (submitting) return;
    if (!digest) { toast.error('Pick a preset first'); return; }
    if (!completed1) { toast.error('Pick or define a capture in Step 1 first'); return; }
    if (selectedSeqNames.length === 0) { toast.error('Select at least one sequence'); return; }
    const processes = digest.steps.filter(s => stepEnabled[s.name]).map(s => s.name);
    if (processes.length === 0) { toast.error('All steps are disabled — pick at least one'); return; }
    // Re-run: at least one step must be picked to run.
    const reRunArr = isReRun ? [...forceSteps].filter(s => processes.includes(s)) : [];
    if (isReRun && reRunArr.length === 0) { toast.error('Pick at least one step to re-run'); return; }

    // Re-run always writes to a FRESH output-id, running ONLY the picked steps
    // and reading the reused (unchecked) steps from the source run via
    // --<step>_dir path overrides — nothing is overwritten (MAMMA steps skip
    // when outputs already exist, so re-running in place can't force a redo).
    const processesToSend = isReRun ? reRunArr : processes;
    const outputIdToSend = isReRun ? '' : outputId;
    const reuseToSend = isReRun && reRunInit
      ? {
          // taskId lets the backend read the source run's frozen config, so
          // chained re-runs resolve reused inputs to where they actually live.
          taskId: reRunInit.taskId,
          outputId: reRunInit.outputId,
          outputDir: reRunInit.outputDir,
          dataset: digest.global.datasetName || '',
        }
      : undefined;

    setSubmitting(true);
    try {
      // Step 0: write the capture.json if we're in Create mode and
      // haven't already. On 409 the helper sets pendingOverwrite and
      // returns null; the user clicks the inline Overwrite button which
      // re-fires submit(true).
      const effectivePath = await saveCaptureIfNeeded(overwriteOnConflict);
      if (!effectivePath) return;

      const res = await fetch('/api/tasks', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          captureJsonPath: effectivePath,
          taskJsonPath: digest.path,
          seqNames: selectedSeqNames,
          cameras: selectedCameras,
          outputDir,
          outputId: outputIdToSend,
          processes: processesToSend,
          reuseInputsFrom: reuseToSend,
          taskOverrides: overrides,
          sequenceMajor,
        }),
      });
      if (res.ok) {
        const d = await res.json();
        toast.success(`Task #${d.taskId} started`);
        onSubmitted?.(d.taskId);
      } else {
        const err = await res.json().catch(() => ({}));
        toast.error(`Failed: ${err.error || res.statusText}`);
      }
    } catch (e) {
      toast.error('Error starting task. See console for details.');
      console.error(e);
    } finally {
      setSubmitting(false);
    }
  };

  // Command preview — resolve the exact per-step shell command from the
  // backend (same materialize path as a real run; no task is created).
  // Triggered by each step's "View command" button.
  const [cmdPreview, setCmdPreview] = useState<{
    open: boolean; loading: boolean; error: string | null;
    data: CommandPreview | null; focusStep: string | null;
  }>({ open: false, loading: false, error: null, data: null, focusStep: null });

  const openCommandPreview = async (focusStep: string | null = null) => {
    if (!digest) { toast.error('Pick a preset first'); return; }
    if (selectedSeqNames.length === 0) { toast.error('Select at least one sequence'); return; }
    if (!captureJsonPath) { toast.error('Pick or save a capture first to preview commands'); return; }
    const processes = digest.steps.filter(s => stepEnabled[s.name]).map(s => s.name);
    setCmdPreview({ open: true, loading: true, error: null, data: null, focusStep });
    try {
      const res = await fetch('/api/tasks/preview-commands', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          captureJsonPath, taskJsonPath: digest.path,
          seqNames: selectedSeqNames, cameras: selectedCameras,
          outputDir, outputId, processes,
          taskOverrides: overrides, sequenceMajor,
        }),
      });
      const d = await res.json().catch(() => ({}));
      if (res.ok) setCmdPreview(p => ({ ...p, loading: false, data: d }));
      else setCmdPreview(p => ({ ...p, loading: false, error: d.error || `HTTP ${res.status}` }));
    } catch {
      setCmdPreview(p => ({ ...p, loading: false, error: 'Could not reach the server.' }));
    }
  };

  // --- derived ----------------------------------------------------------

  // Submit-ready when Step 1 is locked in (Pick-mode capture chosen or
  // Create-mode preflight green), a preset is loaded, and at least one
  // sequence is selected. captureJsonPath isn't required directly —
  // submit() saves the capture.json transparently on Run when needed.
  const canSubmit = !!digest && completed1 && selectedSeqNames.length > 0 && !submitting;

  const knownCaptureMatch = captures.find(c => c.jsonPath === captureJsonPath) ?? null;
  const captureLabel = knownCaptureMatch?.captureName
    || basenameOf(captureJsonPath).replace(/\.json$/i, '')
    || null;

  // In Re-run mode the sequence/camera sets are fixed to what the source run
  // used and the pickers aren't rendered, so the available* lists stay empty —
  // summarize from the selected (fixed) sets there instead.
  const summarySeqCount = isReRun ? selectedSeqNames.length : availableSeqNames.length;
  const summaryCamCount = isReRun ? selectedCameras.length : availableCameras.length;
  const step1Summary = completed1 && captureLabel
    ? `${captureLabel} · ${summarySeqCount} sequence${summarySeqCount === 1 ? '' : 's'} · ${summaryCamCount} camera${summaryCamCount === 1 ? '' : 's'}`
    : null;
  const step2Summary = completed2 && digest
    ? digest.displayName || presetName
    : null;

  // --- UI ---------------------------------------------------------------

  return (
    <div className="w-full max-w-5xl mx-auto px-6 py-8 space-y-4">
      <div>
        <h2 className="text-3xl text-foreground tracking-tight font-medium mb-1">{isReRun ? 'Re-run' : 'New task'}</h2>
        <p className="text-foreground-muted text-sm">
          {isReRun
            ? <>Re-running from task <span className="font-mono text-foreground">{formatTaskId(reRunInit?.outputId)}</span> into a new run: pick the steps to redo and edit their config.<br />Redone steps take their inputs from that task's existing outputs; all new results are written to the new run, and the original task is never modified.</>
            : 'Pick a capture, choose a pipeline configuration preset, and submit.'}
        </p>
      </div>

      {/* STEP 1 — Capture */}
      <StepCard
        n={1}
        title="Capture"
        active={activeStep === 1}
        completed={completed1}
        summary={step1Summary}
        onToggle={() => setActiveStep(s => s === 1 ? 1 : 1)}
        onHeaderClick={() => setActiveStep(1)}
      >
        {isReRun ? (
          <div className="flex items-start gap-2.5 rounded-md border border-border-subtle bg-surface-2/40 p-3 text-xs text-foreground-muted">
            <Info className="w-4 h-4 text-primary flex-shrink-0 mt-0.5" />
            <div className="leading-relaxed">
              Re-running task <span className="font-mono text-foreground">{formatTaskId(reRunInit?.outputId)}</span> into a new run —
              pick the steps to redo in <span className="text-foreground">Step 3</span>; the rest are reused from here.
            </div>
          </div>
        ) : (
        <div className="space-y-4">
          {/* Mode toggle */}
          <div className="inline-flex rounded-md border border-border bg-surface-2 p-0.5 text-[12px]">
            <button
              type="button"
              onClick={() => setCaptureMode('create')}
              className={
                'px-3 py-1.5 rounded transition-colors ' +
                (captureMode === 'create'
                  ? 'bg-primary-muted-strong text-primary'
                  : 'text-foreground-muted hover:text-foreground')
              }
            >
              Define new capture
            </button>
            <button
              type="button"
              onClick={() => setCaptureMode('pick')}
              className={
                'px-3 py-1.5 rounded transition-colors ' +
                (captureMode === 'pick'
                  ? 'bg-primary-muted-strong text-primary'
                  : 'text-foreground-muted hover:text-foreground')
              }
            >
              Pick existing
            </button>
          </div>

          {captureMode === 'create' ? (
            <CreateCapturePanel
              ioiRoot={createIoiRoot}
              setIoiRoot={setCreateIoiRoot}
              calib={createCalib}
              setCalib={setCreateCalib}
              outputName={createOutputName}
              onOutputNameChange={(v) => {
                setCreateOutputName(v);
                setOutputNameTouched(true);
              }}
              defaultOutputName={basenameOf(createIoiRoot)}
              camFps={createCamFps}
              onCamFpsChange={setCreateCamFps}
              footageHintOpen={footageHintOpen}
              setFootageHintOpen={setFootageHintOpen}
              calibHintOpen={calibHintOpen}
              setCalibHintOpen={setCalibHintOpen}
              preflight={preflight}
              preflightInFlight={preflightInFlight}
              createError={createError}
              upAxis={createUpAxis}
              onUpAxisChange={setCreateUpAxis}
            />
          ) : (
            <PickCapturePanel
              captures={captures}
              currentPath={captureJsonPath}
              onPick={async (next) => {
                if (!next) return;
                await adoptCapture(next);
              }}
              pathError={capturePathError}
            />
          )}
        </div>
        )}
      </StepCard>

      {/* STEP 2 — Pipeline Configuration Preset */}
      <StepCard
        n={2}
        title="Pipeline Configuration Preset"
        active={activeStep === 2}
        completed={completed2}
        summary={step2Summary}
        onHeaderClick={() => setActiveStep(2)}
      >
        <div className="space-y-3">
          {isReRun ? (
            <p className="text-foreground-faint text-[11px] leading-snug">
              This is the exact configuration the run used. Edit the settings for the step(s) you're re-running below —
              changes only affect the steps you re-run in Step 3.
            </p>
          ) : (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-foreground-subtle font-medium mb-1">Preset</div>
              <p className="text-foreground-faint text-[11px] mb-2 leading-snug">
                A preset is a reusable pipeline configuration: the engine, per-step flags and settings a run uses.
                Pick one to start from; expand it below to review or edit, and optionally "Save as preset…".
              </p>
              <PresetPicker presets={presets} value={presetName} onPick={setPresetName} />
            </div>
          )}
          <PresetDigestCard
            digest={digest}
            overrides={overrides}
            stepSettings={stepSettings}
            onOverridesChange={setOverrides}
            frameRangeLockedNote={(() => {
              // Only ma_cap consumes the frame range (baked into its NPZs), so
              // lock the editor whenever ma_cap won't run — in either mode.
              // digest is null while the preset/rerun-config fetch is in flight.
              if (!digest) return undefined;
              const hasCap = digest.steps.some(s => s.name === 'ma_cap');
              if (!hasCap) return undefined;
              if (isReRun && !forceSteps.has('ma_cap')) {
                return "Frame range is inherited from the source run's ma_cap — include ma_cap in the steps to re-run (Step 3) to change it.";
              }
              if (!isReRun && !stepEnabled['ma_cap']) {
                return "ma_cap is unchecked in Run steps (Step 3) — the frame range comes from the existing ma_cap outputs. Check ma_cap to change it.";
              }
              return undefined;
            })()}
            onPresetSaved={(newName) => {
              fetch('/api/task-presets')
                .then(r => r.ok ? r.json() : [])
                .then((list: PresetSummary[]) => {
                  setPresets(list);
                  setPresetName(newName);
                  setOverrides({});
                })
                .catch(() => {});
            }}
            onPresetDeleted={() => {
              fetch('/api/task-presets')
                .then(r => r.ok ? r.json() : [])
                .then((list: PresetSummary[]) => {
                  setPresets(list);
                  setPresetName(list[0]?.name ?? '');
                  setOverrides({});
                })
                .catch(() => {});
            }}
          />
        </div>
      </StepCard>

      {/* STEP 3 — Run details */}
      <StepCard
        n={3}
        title="Run details"
        active={activeStep === 3}
        completed={false}
        summary={null}
        onHeaderClick={() => setActiveStep(3)}
      >
        <div className="space-y-5">
          {/* Empty-state hint: when no capture is picked yet, the
              sequence and camera lists will be empty. Make that visible
              up-front (rather than rely on the dropdowns silently
              showing "No sequences selected"), and offer a one-click
              jump back to Step 1. */}
          {!captureJsonPath && (
            <div className="flex items-start gap-2 p-3 rounded-md border border-border-subtle bg-surface-2/40 text-[12px] text-foreground-muted">
              <Info className="w-3.5 h-3.5 mt-0.5 flex-shrink-0 text-primary" aria-hidden />
              <div className="leading-relaxed">
                Sequences and cameras come from the capture you pick in{' '}
                <button
                  type="button"
                  onClick={() => setActiveStep(1)}
                  className="text-primary hover:underline font-medium"
                >
                  Step 1
                </button>
                . Once a capture is loaded, both lists below will populate.
              </div>
            </div>
          )}

          {isReRun ? (
            /* Re-run: sequences + cameras are fixed to what the run used. */
            <>
              <FieldRow label="Seq Names">
                <div className="flex flex-wrap gap-1.5 pt-1">
                  {selectedSeqNames.length
                    ? selectedSeqNames.map(s => (
                        <span key={s} className="inline-flex items-center bg-surface-2 border border-border text-foreground-muted px-2 py-0.5 rounded-md text-xs font-mono">{s}</span>
                      ))
                    : <span className="text-foreground-faint text-xs">—</span>}
                </div>
              </FieldRow>
              <FieldRow label="Cameras">
                <div className="flex flex-wrap gap-1.5 pt-1">
                  {selectedCameras.length
                    ? selectedCameras.map(c => (
                        <span key={c} className="inline-flex items-center bg-surface-2 border border-border text-foreground-muted px-2 py-0.5 rounded-md text-xs font-mono">{c}</span>
                      ))
                    : <span className="text-foreground-faint text-xs">inherited from the run</span>}
                </div>
              </FieldRow>
              <p className="pl-36 text-foreground-faint text-[11px] leading-snug">
                Sequences and cameras are fixed to what task {formatTaskId(reRunInit?.outputId)} used.
              </p>
            </>
          ) : (
            <>
              {/* Sequences */}
              <MultiSelectDropdown
                label="Seq Names"
                options={availableSeqNames}
                selected={selectedSeqNames}
                onToggle={(n) => setSelectedSeqNames(prev => prev.includes(n) ? prev.filter(x => x !== n) : [...prev, n])}
                onSelectAll={() => setSelectedSeqNames(selectedSeqNames.length === availableSeqNames.length ? [] : [...availableSeqNames])}
                onRemove={(n) => setSelectedSeqNames(prev => prev.filter(x => x !== n))}
                placeholder="No sequences selected"
                searchable
              />

              {/* Cameras */}
              <MultiSelectDropdown
                label="Cameras"
                options={availableCameras}
                selected={selectedCameras}
                onToggle={(c) => setSelectedCameras(prev => prev.includes(c) ? prev.filter(x => x !== c) : [...prev, c])}
                onSelectAll={() => setSelectedCameras(selectedCameras.length === availableCameras.length ? [] : [...availableCameras])}
                onRemove={(c) => setSelectedCameras(prev => prev.filter(x => x !== c))}
                columns={3}
                placeholder="No cameras selected"
              />
              {selectedCameras.length > 0 && selectedCameras.length < 3 && (
                <div className="flex items-start gap-4">
                  <div className="w-32 flex-shrink-0" />
                  <div className="flex items-start gap-1.5 text-status-failed text-xs">
                    <AlertCircle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                    <span>
                      Multi-view 3D reconstruction needs <strong>at least 3 cameras</strong> — with fewer views, each landmark is rarely visible in enough unoccluded cameras for stable triangulation and the <span className="font-mono">ma_3d</span> step will typically fail.
                    </span>
                  </div>
                </div>
              )}
              {selectedCameras.length === 3 && (
                <div className="flex items-start gap-4">
                  <div className="w-32 flex-shrink-0" />
                  <div className="flex items-start gap-1.5 text-yellow-400 text-xs">
                    <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                    <span>
                      3 cameras — the pipeline will run, but quality degrades quickly with self-occlusion. <strong>4 or more cameras</strong> recommended for robust 3D reconstruction.
                    </span>
                  </div>
                </div>
              )}
            </>
          )}

          {/* Camera-rig 3D preview — reuses the calibration preview (the
              "Preview camera rig" button + Rerun viewer) so the user can eyeball
              this capture's cameras in the 3D scene before running. */}
          {captureJsonPath && (
            <div className="flex items-start gap-4">
              <div className="w-32 flex-shrink-0" />
              <div className="flex-1 min-w-0">
                <CalibrationStatus captureJsonPath={captureJsonPath} />
              </div>
            </div>
          )}

          {/* Output dir + id — hidden in Re-run: it always writes to a fresh
              run id (the "Previous tasks" picker/output-id aren't the user's to
              change here — the run being re-run is fixed). */}
          {!isReRun && (<>
          <FieldRow label="Output dir">
            <div className="flex-1 relative">
              <input
                type="text"
                value={outputDir}
                onChange={(e) => setOutputDir(e.target.value)}
                placeholder="optional — defaults to ./var/output/"
                className="w-full bg-surface-2 border border-border rounded-md px-3 py-2 pr-9 text-foreground text-sm focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition-colors placeholder:text-foreground-faint"
              />
              <Folder className="absolute right-2.5 top-1/2 -translate-y-1/2 w-4 h-4 text-foreground-subtle pointer-events-none" />
            </div>
          </FieldRow>

          <FieldRow label="Output ID">
            <div className="space-y-2">
              <input
                type="text"
                value={outputId}
                onChange={(e) => setOutputId(e.target.value)}
                placeholder="optional — defaults to a fresh task id"
                className="w-full bg-surface-2 border border-border rounded-md px-3 py-2 text-foreground text-sm focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition-colors placeholder:text-foreground-faint"
              />
              <div className="text-foreground-faint text-[12px] leading-relaxed">
                To continue from previous runs, put their task-id here.
              </div>
              {runGroups.length > 0 && (
                <div className="space-y-1.5">
                  <div className="text-foreground-subtle text-[10px] uppercase tracking-wider font-medium">Previous tasks</div>
                  <div className="flex flex-wrap gap-1.5">
                    {runGroups.slice(0, 8).map(g => {
                      const totalCells = Object.keys(g.stepsDone).length * 5; // approx — just a hint
                      const doneCells = Object.values(g.stepsDone).reduce((sum, arr) => sum + arr.length, 0);
                      const isSelected = outputId === g.outputId;
                      return (
                        <button
                          key={g.outputId}
                          type="button"
                          onClick={() => {
                            setOutputId(g.outputId);
                            if (digest) {
                              setStepEnabled(prev => {
                                const next = { ...prev };
                                for (const step of digest.steps) {
                                  const { status } = stepDoneStatus(g, selectedSeqNames, step.name);
                                  if (status === 'all') next[step.name] = false;
                                  else if (!prev[step.name] && step.enabled) next[step.name] = true;
                                }
                                return next;
                              });
                              // ma_cap auto-unchecked (already done in this group) →
                              // a pending frame-range edit would be inert; drop it.
                              if (stepDoneStatus(g, selectedSeqNames, 'ma_cap').status === 'all') {
                                dropFrameRangeOverride();
                              }
                            }
                          }}
                          title={`Last submitted ${formatRelativeTime(g.lastSubmittedAt)} · ${g.submissions} submission${g.submissions === 1 ? '' : 's'}`}
                          className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs border transition-colors ${
                            isSelected
                              ? 'bg-primary-muted-strong border-primary/55 text-primary ring-1 ring-inset ring-white/10'
                              : 'bg-surface-2 border-border text-foreground-muted hover:border-border-strong hover:bg-surface-3 hover:text-foreground'
                          }`}
                        >
                          <span className="font-mono">{g.outputId}</span>
                          <span className="opacity-60">·</span>
                          <span className="tabular-nums">{doneCells}/{totalCells || '—'} done</span>
                          <span className="opacity-60">·</span>
                          <span className="text-foreground-faint">{formatRelativeTime(g.lastSubmittedAt)}</span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          </FieldRow>
          </>)}

          {/* Run steps (normal) / Steps to re-run (Re-run mode). Each step
              shows a small status pill with what the source run already
              produced. In Re-run mode the checkbox picks a step to run under
              the new run id; ticking one auto-includes its downstream
              dependents (they'd otherwise be built from stale inputs). */}
          {digest && digest.steps.length > 0 && (
            <FieldRow label={isReRun ? 'Steps to re-run' : 'Run steps'}>
              <div className="space-y-1.5">
                <div className="flex flex-wrap gap-1.5">
                  {(() => {
                    const groupId = isReRun ? reRunInit?.outputId : outputId;
                    const activeGroup = runGroups.find(g => g.outputId === groupId) ?? null;
                    const shown = isReRun ? digest.steps.filter(s => stepEnabled[s.name]) : digest.steps;
                    return shown.map(step => {
                      const { status, doneCount, total } = stepDoneStatus(activeGroup, selectedSeqNames, step.name);
                      const on = isReRun ? forceSteps.has(step.name) : !!stepEnabled[step.name];
                      const toggle = (checked: boolean) => {
                        if (step.name === 'ma_cap' && !checked) dropFrameRangeOverride();
                        if (!isReRun) { setStepEnabled(prev => ({ ...prev, [step.name]: checked })); return; }
                        setForceSteps(prev => {
                          const next = new Set(prev);
                          if (checked) {
                            next.add(step.name);
                            // Pull in downstream dependents that are part of this run.
                            for (const d of downstreamSteps(step.name, digest.steps)) {
                              if (stepEnabled[d]) next.add(d);
                            }
                          } else {
                            next.delete(step.name);
                          }
                          return next;
                        });
                      };
                      return (
                        <label
                          key={step.name}
                          className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-md border text-xs cursor-pointer transition-colors ${
                            on
                              ? 'bg-primary-muted border-primary/40 text-primary'
                              : 'bg-surface-2 border-border text-foreground-muted hover:border-border-strong hover:text-foreground'
                          }`}
                        >
                          <input
                            type="checkbox"
                            checked={on}
                            onChange={(e) => toggle(e.target.checked)}
                            className="w-3 h-3 accent-primary"
                          />
                          <span>{stepLabel(step.name)}</span>
                          <span className="text-[10px] opacity-60 font-mono">{step.name}</span>
                          {status === 'all' && (
                            <span
                              className="ml-1 inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] uppercase tracking-wider bg-status-completed-bg border border-status-completed/35 text-status-completed"
                              title={isReRun
                                ? `Currently done for all ${total} chosen sequences — check to re-run it.`
                                : `Already done for all ${total} chosen sequences — runner will skip via DONE sentinel.`}
                            >
                              ✓ done
                            </span>
                          )}
                          {status === 'partial' && (
                            <span
                              className="ml-1 inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] uppercase tracking-wider bg-status-pending-bg border border-status-pending/35 text-status-pending"
                              title={`Done for ${doneCount}/${total} chosen sequences.`}
                            >
                              ↻ partial {doneCount}/{total}
                            </span>
                          )}
                        </label>
                      );
                    });
                  })()}
                </div>
                {isReRun && (
                  <p className="text-foreground-faint text-[11px] leading-snug">
                    Checked steps run under a new run id with the edited config; the unchecked earlier steps are read
                    from this run as input (nothing is copied or overwritten). Steps that depend on a checked one are
                    pulled in automatically.
                  </p>
                )}
              </div>
            </FieldRow>
          )}

          {/* Sequence dispatch order — only meaningful when the task covers
              more than one sequence (with a single sequence both orders are
              identical), so hide it otherwise. The sequenceMajor state keeps
              its value either way. */}
          {selectedSeqNames.length > 1 && (
          <FieldRow label="Sequence order">
            <div className="flex flex-wrap gap-2">
              <label
                className={`inline-flex items-start gap-2 px-3 py-2 rounded-md border text-sm cursor-pointer transition-colors max-w-xs ${
                  !sequenceMajor
                    ? 'bg-primary-muted border-primary/40 text-primary'
                    : 'bg-surface-2 border-border text-foreground-muted hover:border-border-strong hover:text-foreground'
                }`}
              >
                <input
                  type="radio"
                  name="sequence-order"
                  checked={!sequenceMajor}
                  onChange={() => setSequenceMajor(false)}
                  className="mt-0.5 w-3.5 h-3.5 accent-primary"
                />
                <div className="flex-1 min-w-0">
                  <div className="font-medium">Step-major</div>
                  <div className="text-xs opacity-75 mt-0.5 leading-snug">
                    Finish each step across all sequences before continuing to the next step.
                  </div>
                </div>
              </label>
              <label
                className={`inline-flex items-start gap-2 px-3 py-2 rounded-md border text-sm cursor-pointer transition-colors max-w-xs ${
                  sequenceMajor
                    ? 'bg-primary-muted border-primary/40 text-primary'
                    : 'bg-surface-2 border-border text-foreground-muted hover:border-border-strong hover:text-foreground'
                }`}
              >
                <input
                  type="radio"
                  name="sequence-order"
                  checked={sequenceMajor}
                  onChange={() => setSequenceMajor(true)}
                  className="mt-0.5 w-3.5 h-3.5 accent-primary"
                />
                <div className="flex-1 min-w-0">
                  <div className="font-medium">Sequence-major <span className="opacity-60 font-normal text-xs">(default)</span></div>
                  <div className="text-xs opacity-75 mt-0.5 leading-snug">
                    Finish each sequence end-to-end before the next sequence.
                  </div>
                </div>
              </label>
            </div>
          </FieldRow>
          )}

          {/* Overwrite confirm — appears when the first Run attempt
              hit a name collision while saving the Create-mode capture.
              The user clicks Overwrite to retry the save with
              overwrite=true and continue to /api/tasks, or Cancel and
              edit the Capture name in Step 1. */}
          {pendingOverwrite && (
            <div className="rounded-md border border-status-pending/40 bg-status-pending-bg/40 p-3 text-[12px] text-foreground leading-relaxed">
              <div className="flex items-start gap-2">
                <AlertTriangle className="w-4 h-4 text-status-pending flex-shrink-0 mt-0.5" aria-hidden />
                <div className="flex-1 min-w-0 space-y-2">
                  <div>
                    A capture named{' '}
                    <span className="font-mono text-foreground">{pendingOverwrite.name}</span>{' '}
                    already exists. Overwriting replaces its{' '}
                    <span className="font-mono">capture.json</span> on disk
                    and updates the row in the local database; sequences
                    that were used by past tasks are preserved as history.
                    {' '}Rename it in <button
                      type="button"
                      onClick={() => setActiveStep(1)}
                      className="text-primary hover:underline"
                    >Step 1</button> if you'd rather keep both.
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <button
                      type="button"
                      onClick={() => submit(true)}
                      disabled={submitting}
                      className="inline-flex items-center gap-1.5 px-2.5 py-1 text-[11px] rounded bg-status-pending/80 hover:bg-status-pending text-background font-medium disabled:opacity-60 disabled:cursor-not-allowed transition-colors"
                    >
                      {submitting ? <Loader2 className="w-3 h-3 animate-spin" /> : <Download className="w-3 h-3" />}
                      {submitting ? 'Overwriting…' : 'Overwrite & Run'}
                    </button>
                    <button
                      type="button"
                      onClick={() => setPendingOverwrite(null)}
                      disabled={submitting}
                      className="px-2.5 py-1 text-[11px] rounded border border-border text-foreground-muted hover:text-foreground transition-colors disabled:opacity-60"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Submit */}
          <div className="flex items-center justify-end gap-2 pt-1">
            <button
              type="button"
              onClick={() => openCommandPreview()}
              disabled={!digest}
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-md text-sm font-medium border border-border text-foreground-muted hover:text-foreground hover:border-border-strong transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
              title="Preview the exact commands all enabled steps will run"
            >
              <Terminal className="w-4 h-4" />
              View commands
            </button>
            <button
              onClick={() => submit(false)}
              disabled={!canSubmit}
              className={`inline-flex items-center gap-2 px-6 py-2.5 bg-primary hover:opacity-90 active:opacity-100 text-primary-foreground rounded-md text-sm font-medium transition-all shadow-sm shadow-black/30 disabled:opacity-40 disabled:cursor-not-allowed ${submitting ? 'cursor-wait' : ''}`}
            >
              {submitting ? 'Starting…' : 'Run'}
            </button>
          </div>
        </div>
      </StepCard>

      <CommandPreviewModal
        open={cmdPreview.open}
        loading={cmdPreview.loading}
        error={cmdPreview.error}
        data={cmdPreview.data}
        focusStep={cmdPreview.focusStep}
        onClose={() => setCmdPreview(p => ({ ...p, open: false }))}
      />
    </div>
  );
}

// ─── StepCard chrome ───────────────────────────────────────────────────

function StepCard({
  n, title, active, completed, summary, gated, gatedHint, onHeaderClick, children, onToggle,
}: {
  n: 1 | 2 | 3;
  title: string;
  active: boolean;
  completed: boolean;
  summary: string | null;
  gated?: boolean;
  gatedHint?: string;
  onHeaderClick?: () => void;
  onToggle?: () => void;
  children: React.ReactNode;
}) {
  // Suppress unused-warning for onToggle which was an earlier param shape.
  void onToggle;
  const interactive = !gated;
  return (
    <div className="bg-surface-1 border border-border-subtle rounded-xl shadow-sm shadow-black/30 ring-1 ring-inset ring-white/[0.02] overflow-hidden">
      <button
        type="button"
        onClick={() => { if (interactive && onHeaderClick) onHeaderClick(); }}
        aria-expanded={active}
        disabled={!interactive}
        className={
          'w-full px-5 py-3 flex items-center justify-between gap-4 text-left transition-colors ' +
          (interactive ? 'hover:bg-surface-2/40 cursor-pointer' : 'cursor-not-allowed opacity-70') +
          (active ? ' border-b border-border-subtle' : '')
        }
      >
        <div className="flex items-baseline gap-3 min-w-0">
          <ChevronDown
            className={
              'w-3.5 h-3.5 text-foreground-faint transition-transform flex-shrink-0 ' +
              (active ? '' : '-rotate-90')
            }
            aria-hidden
          />
          <span className="text-foreground-faint text-[10.5px] uppercase tracking-[0.18em] font-medium">
            Step {n}
          </span>
          <h3 className="text-foreground text-[13px] font-medium truncate">{title}</h3>
        </div>
        <div className="flex items-center gap-3 flex-shrink-0">
          {!active && summary && (
            <span className="text-foreground-muted text-[11.5px] truncate max-w-[420px]">{summary}</span>
          )}
          {gated && gatedHint && !active && (
            <span className="text-foreground-faint text-[11px] italic">{gatedHint}</span>
          )}
          {completed && (
            <span className="inline-flex items-center justify-center w-4 h-4 rounded-full bg-status-completed-bg ring-1 ring-inset ring-status-completed/40">
              <Check className="w-2.5 h-2.5 text-status-completed" />
            </span>
          )}
        </div>
      </button>
      {active && <div className="px-5 py-4">{children}</div>}
    </div>
  );
}

// ─── Step 1: Create panel ──────────────────────────────────────────────

function CreateCapturePanel({
  ioiRoot, setIoiRoot, calib, setCalib,
  outputName, onOutputNameChange, defaultOutputName,
  camFps, onCamFpsChange,
  footageHintOpen, setFootageHintOpen,
  calibHintOpen, setCalibHintOpen,
  preflight, preflightInFlight, createError,
  upAxis, onUpAxisChange,
}: {
  ioiRoot: string; setIoiRoot: (v: string) => void;
  calib: string; setCalib: (v: string) => void;
  outputName: string;
  onOutputNameChange: (v: string) => void;
  defaultOutputName: string;
  camFps: string; onCamFpsChange: (v: string) => void;
  footageHintOpen: boolean; setFootageHintOpen: (v: boolean) => void;
  calibHintOpen: boolean; setCalibHintOpen: (v: boolean) => void;
  preflight: PreflightResponse | null;
  preflightInFlight: boolean;
  createError: string | null;
  upAxis: UpAxisValue;
  onUpAxisChange: (v: UpAxisValue) => void;
}) {
  // Only render a badge once at least one preflight response has landed,
  // OR once the user has typed something + the debounce has elapsed.
  // Otherwise empty values flash a red "required" badge on first paint.
  const showFootageBadge = !!ioiRoot;

  return (
    <div className="space-y-3">
      {/* Footage root */}
      <CreateField
        label="Footage root"
        hintOpen={footageHintOpen}
        onHintToggle={() => setFootageHintOpen(!footageHintOpen)}
        hint={<FootageHint />}
      >
        <div className="space-y-1">
          <div className="relative">
            <FolderOpen className="absolute left-2.5 top-1/2 -translate-y-1/2 w-4 h-4 text-foreground-subtle pointer-events-none" aria-hidden />
            <input
              type="text"
              value={ioiRoot}
              onChange={(e) => setIoiRoot(e.target.value)}
              placeholder="/absolute/path/to/footage/root"
              className="w-full bg-surface-2 border border-border rounded-md pl-9 pr-3 py-2 text-foreground text-sm font-mono focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition-colors placeholder:text-foreground-faint"
            />
          </div>
          {showFootageBadge && (
            <PreflightBadge
              inFlight={preflightInFlight}
              ok={preflight?.footage.ok ?? false}
              okLabel={preflight?.footage.ok
                ? `${preflight.footage.sequences} sequence${preflight.footage.sequences === 1 ? '' : 's'} · ${preflight.footage.layout} layout`
                : ''}
              errorLabel={preflight?.footage.error ?? null}
            />
          )}
        </div>
      </CreateField>

      {/* Calibration */}
      <CreateField
        label="Calibration"
        hintOpen={calibHintOpen}
        onHintToggle={() => setCalibHintOpen(!calibHintOpen)}
        hint={<CalibrationHint />}
      >
        <div className="space-y-1">
          <div className="relative">
            <FileJson className="absolute left-2.5 top-1/2 -translate-y-1/2 w-4 h-4 text-foreground-subtle pointer-events-none" aria-hidden />
            <input
              type="text"
              value={calib}
              onChange={(e) => setCalib(e.target.value)}
              placeholder="/path/to/calibration.yaml"
              className="w-full bg-surface-2 border border-border rounded-md pl-9 pr-3 py-2 text-foreground text-sm font-mono focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition-colors placeholder:text-foreground-faint"
            />
          </div>
          <CalibrationStatus calibPath={calib} upAxis={upAxis} onUpAxisChange={onUpAxisChange} />
        </div>
      </CreateField>

      {/* Camera FPS + Capture name on one compact row. FPS -> capture.json's
          cam_fps (default 30); name -> the saved capture.json basename. */}
      <div className="flex items-center gap-3">
        <label className="text-foreground-muted text-xs uppercase tracking-wider font-medium w-32 text-right flex-shrink-0">Camera FPS</label>
        <input
          type="number"
          min={1}
          step={1}
          value={camFps}
          onChange={(e) => onCamFpsChange(e.target.value)}
          placeholder="30"
          className="w-20 bg-surface-2 border border-border rounded-md px-3 py-2 text-foreground text-sm font-mono focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition-colors placeholder:text-foreground-faint"
        />
        <label className="text-foreground-muted text-xs uppercase tracking-wider font-medium flex-shrink-0">Capture name</label>
        <input
          type="text"
          value={outputName}
          onChange={(e) => onOutputNameChange(e.target.value)}
          placeholder={defaultOutputName || 'capture-name'}
          className="flex-1 min-w-0 bg-surface-2 border border-border rounded-md px-3 py-2 text-foreground text-sm font-mono focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition-colors placeholder:text-foreground-faint"
        />
      </div>
      <div className="pl-[8.75rem] -mt-1 text-foreground-faint text-[10.5px]">
        Camera FPS defaults to 30. Capture saved as{' '}
        <span className="font-mono">{(outputName.trim() || defaultOutputName || 'capture-name')}.json</span>{' '}
        under <span className="font-mono">gui/var/interface/capture_jsons/</span> — both editable later.
      </div>

      {/* No "Save" button: once the preflight badges are both green
          the capture is considered ready, and the file is actually
          written on the way to /api/tasks when the user clicks Run.
          Any backend save error surfaces here. */}
      <div className="text-[10.5px] text-foreground-faint leading-relaxed">
        The capture.json file itself is saved when you click{' '}
        <span className="font-medium text-foreground">Run</span> in Step 3.
      </div>
      {createError && (
        <div className="flex items-start gap-1.5 text-status-failed text-[11px]">
          <AlertCircle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
          <span>{createError}</span>
        </div>
      )}
    </div>
  );
}

// ─── Step 1: Pick panel ────────────────────────────────────────────────

function PickCapturePanel({
  captures, currentPath, onPick, pathError,
}: {
  captures: CaptureSummary[];
  currentPath: string;
  onPick: (path: string) => void;
  pathError: string;
}) {
  return (
    <div className="space-y-2">
      <CaptureSelect captures={captures} value={currentPath} onPick={onPick} />
      {pathError && (
        <div className="flex items-center gap-1.5 text-status-failed text-xs">
          <AlertCircle className="w-3.5 h-3.5 flex-shrink-0" />
          {pathError}
        </div>
      )}
    </div>
  );
}

// ─── Validation badge ──────────────────────────────────────────────────

function PreflightBadge({
  inFlight, ok, okLabel, errorLabel,
}: {
  inFlight: boolean;
  ok: boolean;
  okLabel: string;
  errorLabel: string | null;
}) {
  if (inFlight) {
    return (
      <div className="inline-flex items-center gap-1.5 text-[11px] text-foreground-muted">
        <Loader2 className="w-3 h-3 animate-spin" aria-hidden />
        Checking…
      </div>
    );
  }
  if (ok) {
    return (
      <div className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-status-completed-bg ring-1 ring-inset ring-status-completed/30 text-status-completed text-[11px]">
        <Check className="w-3 h-3" aria-hidden />
        {okLabel}
      </div>
    );
  }
  if (errorLabel) {
    return (
      <div className="inline-flex items-start gap-1.5 px-2 py-0.5 rounded-md bg-status-failed-bg ring-1 ring-inset ring-status-failed/30 text-status-failed text-[11px] max-w-full">
        <X className="w-3 h-3 flex-shrink-0 mt-0.5" aria-hidden />
        <span className="break-words">{errorLabel}</span>
      </div>
    );
  }
  return null;
}

