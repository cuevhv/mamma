import { useState, useEffect, useMemo, type ReactNode } from 'react';
import { ArrowLeft, X, FileJson, ChevronLeft, ChevronRight, ChevronDown, Film, Maximize2, Video, RotateCw } from 'lucide-react';
import { toast } from 'sonner';
import { ALL_STEPS, buildProcessRows, rowRollupStatus, RowStatus } from './ProcessTable';
import { useTaskPolling } from './shared/useTaskPolling';
import { FileViewerModal } from './shared/FileViewerModal';
import { Skeleton } from './shared/Skeleton';
import { Thumbnail } from './shared/Thumbnail';
import { RerunWebViewer } from './RerunWebViewer';
import { UpAxisValue, upAxisLabel } from './UpAxisToggle';
import { HtmlViewer } from './HtmlViewer';
import { NpzViewer } from './NpzViewer';
import { StepOutputs } from './StepOutputs';
import { ResultExport } from './ResultExport';
import { formatTaskId } from './shared/formatTaskId';
import { formatRelativeTime } from './shared/relativeTime';

/** Shape returned by /api/tasks/history for one run's processes. */
interface HistoryProcess {
  processId: string;
  processType: string;
  status: string;
  pid?: string | null;
  outFile?: string | null;
  errFile?: string | null;
}

interface HistorySequence {
  seqName: string;
  processes: HistoryProcess[];
}

interface HistoryTask {
  taskId: string;
  captureName: string;
  /** Source preset path recorded at submit time. Null/undefined for
   *  legacy or CLI-imported rows. */
  presetPath?: string | null;
  username: string;
  createdAt: string;
  sequences: HistorySequence[];
}

interface ActiveTask {
  taskId: string;
  captureName: string;
  username: string;
  createdAt: string;
  runnerPid?: string | null;
  processes: Array<HistoryProcess & { sequenceName: string }>;
}

interface SequenceInfo {
  name: string;
  path?: string | null;
  ioiPath?: string | null;
  subjects: string[];
  numSubjects: number;
  cameras: string[];
  previewImage?: string | null;
  cameraPreviews?: Record<string, string | null>;
}

interface Task {
  taskId: string;
  seqNames: string[];
  createdAt: string;
}

interface CaptureDetailProps {
  captureName: string;
  onBack: () => void;
  /** Jump to the Exporter tab (for tool setup / custom-path export). */
  onGoToExporter?: () => void;
  /** Optional deep-link state (set when arriving from a Tasks-tab cell click).
   *  Pre-selects the Outputs explorer's task/sequence/process so the user
   *  lands directly on the cell's output directory. */
  initial?: {
    taskId?: string;
    sequence?: string;
    process?: string;
  };
}

interface APITask {
  id: string;
  status: string;
  startedAt: string;
  user: string;
  processes: string[];
  sequenceNames: string[];
  outputPath?: string | null;
  outputId?: string | null;
  datasetName?: string | null;
}

interface CaptureData {
  captureName: string;
  tasks: APITask[];
  sequences: SequenceInfo[];
  // Essential capture metadata for the info card (best-effort from the
  // capture JSON; may be absent for examples or missing/unparseable JSON).
  cams?: string[];
  camFps?: number | null;
  calib?: string | null;
  dataPath?: string | null;
  captureJsonPath?: string | null;
  ioiRoot?: string | null;
  thumbnailPath?: string | null;
}

// Glyphs prefixed to task-option labels in the Task dropdown so each run's
// rolled-up status reads at a glance even though <option> can't be styled.
const STATUS_GLYPH: Record<RowStatus, string> = {
  Running: '●',
  Failed: '✗',
  Mixed: '~',
  Pending: '…',
  Completed: '✓',
};

// Worst-case (most attention-grabbing) status across a task's per-sequence rows.
function aggregateTaskStatus(perRow: RowStatus[]): RowStatus {
  const priority: RowStatus[] = ['Running', 'Failed', 'Mixed', 'Pending', 'Completed'];
  const set = new Set(perRow);
  for (const s of priority) if (set.has(s)) return s;
  return 'Pending';
}

function truncate(s: string, n: number): string {
  return s.length <= n ? s : s.slice(0, n - 1) + '…';
}

/**
 * Plays an MP4 via the existing `/api/files/stream` route. Browsers
 * decode a narrow set of codecs natively (H.264 baseline/main + a few
 * others); MAMMA's `videos_crf24` / `videos_crf16` / `videos_light`
 * dirs sometimes ship variants the browser can't open (e.g. HEVC
 * tone-mapped iPhone captures). Catch the `<video onError>` and
 * surface a friendly fallback message so users know the file isn't
 * corrupted — they just need a native player.
 */
function VideoPlayer({ relPath }: { relPath: string }) {
  const [status, setStatus] = useState<'loading' | 'ok' | 'failed'>('loading');
  const [launching, setLaunching] = useState(false);
  // Re-mounting via key={relPath} would also work, but resetting state
  // explicitly keeps the modal mounted (preserves the close-button slot).
  useEffect(() => { setStatus('loading'); }, [relPath]);
  const fileName = relPath.split('/').pop() || '';
  const openNative = async () => {
    if (launching) return;
    setLaunching(true);
    try {
      const res = await fetch('/api/files/open-native', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: relPath }),
      });
      const data = await res.json();
      if (res.ok) {
        toast.success(`Opening ${fileName} in ${data.binary ? data.binary.split('/').pop() : 'native viewer'}…`);
      } else {
        toast.error(data.error || `Failed to launch (${res.status})`);
      }
    } catch (e) {
      console.error(e);
      toast.error('Failed to reach the backend.');
    } finally {
      setLaunching(false);
    }
  };
  return (
    <>
      <video
        key={relPath}
        src={`/api/files/stream?path=${encodeURIComponent(relPath)}`}
        controls
        autoPlay
        onLoadedData={() => setStatus('ok')}
        onError={() => setStatus('failed')}
        style={{ display: status === 'failed' ? 'none' : 'block' }}
        className="w-full max-h-[82vh] rounded-lg object-contain"
      />
      {status === 'failed' && (
        <div className="rounded-lg bg-surface-2 border border-border-subtle p-6 text-center max-w-2xl mx-auto">
          <div className="text-foreground text-sm font-medium mb-2">
            Your browser can't play this video format.
          </div>
          <div className="text-foreground-muted text-xs leading-relaxed">
            The file isn't corrupted — many browsers don't support codecs
            like HEVC, ProRes, or some H.264 profiles inline. Open it
            with a native player (VLC, mpv, QuickTime, ffplay) instead.
          </div>
          <div className="mt-4 flex items-center justify-center gap-2 text-xs text-foreground-subtle font-mono break-all">
            {fileName}
          </div>
          <button
            type="button"
            onClick={openNative}
            disabled={launching}
            className="inline-flex items-center gap-1.5 mt-3 px-3 py-1.5 rounded-md bg-primary-muted text-primary text-xs font-medium hover:bg-primary-muted-strong transition-colors disabled:opacity-60"
          >
            {launching ? 'Launching…' : 'Open in native player'}
          </button>
        </div>
      )}
    </>
  );
}

export function CaptureDetail({ captureName, onBack, initial, onGoToExporter }: CaptureDetailProps) {
  // Initial values come from the deep-link prop (Tasks → Results jump). The
  // existing auto-default effects only fire when these are empty, so a
  // pre-set value isn't clobbered.
  const [selectedTaskId, setSelectedTaskId] = useState<string>(initial?.taskId ?? '');
  const [selectedSequence, setSelectedSequence] = useState(initial?.sequence ?? '');
  const [captureData, setCaptureData] = useState<CaptureData | null>(null);
  const [loading, setLoading] = useState(true);
  const [playingVideoRelPath, setPlayingVideoRelPath] = useState<string | null>(null);
  const [playingImageRelPath, setPlayingImageRelPath] = useState<string | null>(null);
  /** Ordered image relPaths in the folder the lightbox was opened from,
   *  used for prev/next navigation. */
  const [imageSiblings, setImageSiblings] = useState<string[]>([]);
  /** ma_vis preview.mp4 relPath for the current task/sequence, or null if it
   *  doesn't exist — fills the spare cell in the step grid when present. */
  const [previewRelPath, setPreviewRelPath] = useState<string | null>(null);
  // Run config viewer state — shared across all the "view config" pill actions.
  const [taskConfigViewer, setTaskConfigViewer] = useState<{ name: string; path: string } | null>(null);
  /** When set, the embedded Rerun web viewer is open for this .rrd. */
  const [rrdWebViewer, setRrdWebViewer] = useState<{ path: string; name: string } | null>(null);
  const [calibPreviewBusy, setCalibPreviewBusy] = useState(false);
  const [calibUpAxis, setCalibUpAxis] = useState<UpAxisValue>('auto');
  /** When set, the embedded HTML viewer is open for this .html / .htm. */
  const [htmlViewer, setHtmlViewer] = useState<{ path: string; name: string } | null>(null);
  /** When set, the .npz inspector is open for this archive. */
  const [npzViewer, setNpzViewer] = useState<{ path: string; name: string } | null>(null);

  // Lightbox prev/next within the folder the image was opened from. Index is
  // derived from the live siblings list so it survives re-renders; navigation
  // wraps around the ends.
  const imageIndex = playingImageRelPath ? imageSiblings.indexOf(playingImageRelPath) : -1;
  const stepImage = (delta: number) => {
    if (imageIndex < 0 || imageSiblings.length < 2) return;
    const next = (imageIndex + delta + imageSiblings.length) % imageSiblings.length;
    setPlayingImageRelPath(imageSiblings[next]);
  };

  // Arrow keys page through the gallery while the image lightbox is open.
  useEffect(() => {
    if (!playingImageRelPath) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') { e.preventDefault(); stepImage(-1); }
      else if (e.key === 'ArrowRight') { e.preventDefault(); stepImage(1); }
      else if (e.key === 'Escape') { setPlayingImageRelPath(null); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playingImageRelPath, imageSiblings]);

  /** Build + open a camera-rig .rrd from this capture's calibration so the user
   *  can sanity-check the rig/convention. Resolves the calib via the capture
   *  json on the backend (handles relative '../calib/...' paths). */
  const previewCalibRig = async (axis: UpAxisValue = calibUpAxis) => {
    if (!captureData?.captureJsonPath || calibPreviewBusy) return;
    setCalibPreviewBusy(true);
    try {
      const res = await fetch('/api/calib/preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ captureJsonPath: captureData.captureJsonPath, upAxis: axis }),
      });
      const data = await res.json();
      if (!res.ok) {
        toast.error(data.error || `Could not build camera-rig preview (${res.status})`);
        return;
      }
      const name = data.upAxis ? `camera rig · ${upAxisLabel(data.upAxis)} up` : 'camera rig';
      setRrdWebViewer({ path: data.rrdPath, name });
    } catch (e) {
      console.error(e);
      toast.error('Failed to reach the backend for the camera-rig preview.');
    } finally {
      setCalibPreviewBusy(false);
    }
  };

  const openTaskConfig = async (taskId: string) => {
    try {
      const res = await fetch(`/api/tasks/${encodeURIComponent(taskId)}/config-path`);
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        toast.error(err.error || `Failed to locate run config (${res.status})`);
        return;
      }
      const data = await res.json();
      setTaskConfigViewer({ name: `run_${taskId}.json`, path: data.path });
    } catch (e) {
      console.error(e);
      toast.error('Failed to load run config. See console.');
    }
  };

  /** Open the source preset a run was created from (same endpoint as the run
   *  config; it also returns `presetPath`). Surfaced on the entry step. */
  const openTaskPreset = async (taskId: string) => {
    try {
      const res = await fetch(`/api/tasks/${encodeURIComponent(taskId)}/config-path`);
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        toast.error(err.error || `Failed to locate preset (${res.status})`);
        return;
      }
      const data = await res.json();
      if (!data.presetPath) {
        toast.error('No source preset was recorded for this run.');
        return;
      }
      setTaskConfigViewer({ name: data.presetPath.split('/').pop() || 'preset.yaml', path: data.presetPath });
    } catch (e) {
      console.error(e);
      toast.error('Failed to load preset. See console.');
    }
  };

  /** Launch the native Rerun viewer for a .rrd file via the backend.
   *  We use the native viewer (not the web embed) because GB-scale .rrd
   *  files routinely exceed browser memory limits. The Rerun process
   *  pops up on the same machine the Flask backend runs on. */
  const openRrd = async (path: string, fresh = false) => {
    try {
      const res = await fetch('/api/rrd/open', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path, reset_layout: fresh }),
      });
      const data = await res.json();
      if (res.ok) {
        toast.success(`Opening ${path.split('/').pop()} in Rerun${fresh && data.layout_reset ? ' (fresh layout)' : ''}…`, {
          description: "If you don't see it, check behind this window — it may already be open.",
          duration: 6000,
        });
      } else {
        toast.error(data.error || `Failed to launch (${res.status})`);
      }
    } catch (e) {
      console.error(e);
      toast.error('Failed to reach the backend.');
    }
  };

  // Live polling of /api/processes/active so cells update while a run is in flight.
  const { data: activeData } = useTaskPolling<ActiveTask[]>('/api/processes/active', { intervalMs: 2000 });
  // Full history for this capture; refreshed on focus / explicit reload elsewhere.
  const { data: historyData, refresh: refreshHistory } = useTaskPolling<HistoryTask[]>('/api/tasks/history', { intervalMs: 0 });

  useEffect(() => {
    setLoading(true);
    fetch(`/api/captures/${captureName}`)
      .then(res => {
        if (!res.ok) throw new Error('Failed to fetch capture details');
        return res.json();
      })
      .then((data: CaptureData) => {
        setCaptureData(data);
        // Don't clobber a deep-linked pre-selection from the Tasks-tab jump.
        // The current state could already hold an initial.* value supplied
        // via props, in which case we leave it alone.
        setSelectedSequence(prev => prev || (data.sequences?.[0]?.name ?? ''));
        if (data.tasks?.length) {
          setSelectedTaskId(prev => prev || data.tasks[0].id);
        }
      })
      .catch(err => console.error(err))
      .finally(() => setLoading(false));
  }, [captureName]);

  const selectedTask = useMemo(
    () => captureData?.tasks.find(t => t.id === selectedTaskId) ?? null,
    [captureData, selectedTaskId]
  );

  // Refresh history once per matrix-relevant change so it reflects newly-finished runs.
  useEffect(() => { refreshHistory(); }, [captureName, refreshHistory]);

  // Per-capture run list, derived from history + any live tasks for the same capture.
  const runsForCapture = useMemo<HistoryTask[]>(() => {
    const fromHistory = (historyData ?? []).filter(t => t.captureName === captureName);
    if (!activeData) return fromHistory;
    // Replace history rows with live ones when a task is currently active.
    const liveByTaskId = new Map<string, ActiveTask>();
    for (const t of activeData) {
      if (t.captureName === captureName) liveByTaskId.set(t.taskId, t);
    }
    return fromHistory.map(h => {
      const live = liveByTaskId.get(h.taskId);
      if (!live) return h;
      // Pivot live's flat processes into history's nested shape.
      const seqMap: Record<string, HistoryProcess[]> = {};
      for (const p of live.processes) {
        (seqMap[p.sequenceName] ||= []).push({
          processId: p.processId,
          processType: p.processType,
          status: p.status,
          pid: p.pid,
          outFile: p.outFile,
          errFile: p.errFile,
        });
      }
      return {
        ...h,
        sequences: Object.entries(seqMap).map(([seqName, processes]) => ({ seqName, processes })),
      };
    });
  }, [historyData, activeData, captureName]);

  // Flatten all (task, seq, process) tuples across this capture's runs
  // into the per-(task, seq) row shape the table consumes.
  const tableData = useMemo(() => {
    const flat: Array<{
      taskId: string; seqName: string; presetPath?: string | null; createdAt?: string;
      processType: string; processId: string; status: string;
      pid?: string | null; outFile?: string | null; errFile?: string | null;
    }> = [];
    for (const run of runsForCapture) {
      for (const seq of run.sequences) {
        for (const p of seq.processes) {
          flat.push({
            taskId: run.taskId,
            seqName: seq.seqName,
            presetPath: run.presetPath ?? null,
            createdAt: run.createdAt,
            processType: p.processType,
            processId: p.processId,
            status: p.status,
            pid: p.pid,
            outFile: p.outFile,
            errFile: p.errFile,
          });
        }
      }
    }
    const { rows, stepsInUse } = buildProcessRows(flat);
    const orderedSteps = ALL_STEPS.filter(s => stepsInUse.has(s));
    return { rows, steps: orderedSteps };
  }, [runsForCapture]);

  // The file explorer still uses selectedTaskId for output-path construction;
  // default it to the most recent task once we have one.
  useEffect(() => {
    if (!selectedTaskId && runsForCapture.length > 0) {
      setSelectedTaskId(runsForCapture[0].taskId);
    }
  }, [runsForCapture, selectedTaskId]);

  // Task options for the Outputs explorer's Task dropdown — every run for
  // this capture, with a rolled-up status and a relative-time label so the
  // user can identify which run they're browsing without leaving the card.
  const taskOptions = useMemo(() => {
    return runsForCapture.map(run => {
      const flat = run.sequences.flatMap(seq =>
        seq.processes.map(p => ({
          taskId: run.taskId,
          seqName: seq.seqName,
          createdAt: run.createdAt,
          processType: p.processType,
          processId: p.processId,
          status: p.status,
          pid: p.pid,
          outFile: p.outFile,
          errFile: p.errFile,
        }))
      );
      const { rows } = buildProcessRows(flat);
      const rolled = aggregateTaskStatus(rows.map(r => rowRollupStatus(r.cells)));
      const apiTask = captureData?.tasks.find(t => t.id === run.taskId);
      const outputId = apiTask?.outputId || run.taskId;
      return {
        taskId: run.taskId,
        outputId,
        status: rolled,
        createdAt: run.createdAt,
        relativeTime: formatRelativeTime(run.createdAt),
      };
    });
  }, [runsForCapture, captureData]);

  const availableSequenceNames = useMemo(
    () => selectedTask?.sequenceNames ?? [],
    [selectedTask]
  );

  const availableProcesses = useMemo(() => {
    // Sort by the canonical pipeline order (ma_cap → ma_vis) so the
    // stacked sections match the matrix's column order. Unknown steps
    // fall to the end, sorted alphabetically among themselves.
    const set = new Set(selectedTask?.processes ?? []);
    const known = ALL_STEPS.filter(s => set.has(s));
    const unknown = [...set].filter(s => !ALL_STEPS.includes(s)).sort();
    return [...known, ...unknown];
  }, [selectedTask]);

  const sequenceOptions = useMemo(() => {
    if (!captureData) return [];
    if (!availableSequenceNames.length) return captureData.sequences;
    return captureData.sequences.filter(s => availableSequenceNames.includes(s.name));
  }, [captureData, availableSequenceNames]);

  // Per-step stdout/stderr log paths for the selected task + sequence, so each
  // step card can offer `out`/`err` buttons. Logs live in jobs_log_dir (not the
  // output dir), so they come from the process rows rather than /api/files/list.
  const logsByStep = useMemo(() => {
    const m = new Map<string, { outFile?: string | null; errFile?: string | null }>();
    const run = runsForCapture.find(r => r.taskId === selectedTaskId);
    const seq = run?.sequences.find(s => s.seqName === selectedSequence);
    for (const p of seq?.processes ?? []) {
      m.set(p.processType, { outFile: p.outFile, errFile: p.errFile });
    }
    return m;
  }, [runsForCapture, selectedTaskId, selectedSequence]);

  useEffect(() => {
    if (!sequenceOptions.length) {
      setSelectedSequence('');
      return;
    }
    if (!sequenceOptions.find(s => s.name === selectedSequence)) {
      setSelectedSequence(sequenceOptions[0].name);
    }
  }, [sequenceOptions]);

  // The Outputs explorer used to render one step at a time, picked from a
  // dropdown. We now stack one section per step instead — `baseRelPathFor`
  // computes the per-step root path that each <StepOutputs> uses to start
  // browsing. Mirrors mamma_apptainer's output convention:
  //   <output_path>/<step>/<output_id>/<dataset_name>/<seq>/
  // Falls back to the legacy MOUNT_POINT-relative shape when older DB
  // rows are missing the new fields.
  const baseRelPathFor = (step: string): string => {
    if (!selectedTask || !selectedSequence || !captureData) return '';
    const outPath = selectedTask.outputPath;
    const outId = selectedTask.outputId || selectedTaskId;
    const dataset = selectedTask.datasetName || captureData.captureName;
    if (outPath) return `${outPath}/${step}/${outId}/${dataset}/${selectedSequence}`;
    return `output/${step}/${outId}/${dataset}/${selectedSequence}`;
  };

  // Probe for the ma_vis preview.mp4 of the current task/sequence so the step
  // grid can show it inline (fills the otherwise-empty trailing cell). Cleared
  // and re-checked whenever the selection changes.
  useEffect(() => {
    setPreviewRelPath(null);
    if (!availableProcesses.includes('ma_vis')) return;
    const base = baseRelPathFor('ma_vis');
    if (!base) return;
    const controller = new AbortController();
    fetch(`/api/files/list?path=${encodeURIComponent(base)}`, { signal: controller.signal })
      .then(r => (r.ok ? r.json() : null))
      .then(d => {
        if (d && Array.isArray(d.files) && d.files.some((f: { name: string }) => f.name === 'preview.mp4')) {
          setPreviewRelPath(`${base}/preview.mp4`);
        }
      })
      .catch(() => {});
    return () => controller.abort();
    // baseRelPathFor is a render-local closure over these same deps.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedTaskId, selectedSequence, captureData, availableProcesses]);

  const tasks: Task[] = captureData
    ? captureData.tasks.map(t => ({
        taskId: t.id,
        seqNames: t.sequenceNames ?? [],
        createdAt: t.startedAt
      }))
    : [];

  const numberOfSequences = captureData?.sequences?.length ?? 0;

  // Derived values for the capture info card (see the grid below).
  const infoCams = captureData?.cams ?? [];

  if (loading) {
    // Layout-shaped skeleton: title bar → runs-summary card → outputs
    // explorer card. Mirrors the post-load layout closely so the page
    // doesn't shift when content arrives — and a shimmer beats a stalled
    // spinner for "this is doing something."
    return (
      <div className="px-6 py-8">
        <div className="max-w-[1600px] mx-auto">
          <div className="mb-8">
            <Skeleton className="h-3 w-24 mb-4" />
            <Skeleton className="h-8 w-64 mb-2" />
            <Skeleton className="h-3 w-80" />
          </div>
          <div className="bg-surface-1 border border-border-subtle rounded-xl p-5 mb-6">
            <Skeleton className="h-4 w-48 mb-2" />
            <Skeleton className="h-3 w-72 mb-4" />
            <div className="flex gap-2 flex-wrap">
              {Array.from({ length: 4 }).map((_, i) => <Skeleton key={i} className="h-7 w-32" />)}
            </div>
          </div>
          <div className="bg-surface-1 border border-border-subtle rounded-xl p-6">
            <Skeleton className="h-5 w-44 mb-2" />
            <Skeleton className="h-3 w-96 mb-4" />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
              <Skeleton className="h-10" />
              <Skeleton className="h-10" />
            </div>
            <div className="flex flex-col gap-3">
              {Array.from({ length: 3 }).map((_, i) => <Skeleton key={i} className="h-10" />)}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="px-6 py-8">
      <div className="max-w-[1600px] mx-auto">
        <div className="mb-8">
          <button
            onClick={onBack}
            className="inline-flex items-center gap-1.5 text-foreground-muted hover:text-foreground text-sm transition-colors mb-4"
          >
            <ArrowLeft className="w-4 h-4" />
            All captures
          </button>
          <h2 className="text-3xl text-foreground tracking-tight font-medium mb-1">{captureName}</h2>
          <p className="text-foreground-muted text-sm">Pipeline runs, status, and outputs for this capture.</p>
        </div>

        {/* Top region: capture info card (left) + Export animation (right).
            Two columns on wide screens, stacked on narrow. The info card holds
            the thumbnail, at-a-glance stats, camera chips, and the run picker.
            Best-effort: each stat renders only when its data is present, so
            example captures / missing JSON degrade gracefully. */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6 items-stretch">
          {captureData && (
            <div className="h-full bg-surface-1 border border-border-subtle rounded-xl p-5 shadow-sm shadow-black/30 ring-1 ring-inset ring-white/[0.02]">
              <div className="flex gap-4">
                <Thumbnail
                  path={captureData.thumbnailPath}
                  alt={captureName}
                  loading="eager"
                  className="w-44 aspect-video shrink-0"
                />
                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between gap-3">
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-4 gap-y-3 flex-1 min-w-0">
                      <StatItem label="Cameras" value={infoCams.length ? String(infoCams.length) : '—'} />
                      {captureData.camFps != null && (
                        <StatItem label="FPS" value={`${captureData.camFps}`} />
                      )}
                      <StatItem label="Sequences" value={String(numberOfSequences)} />
                      <StatItem label="Runs" value={String(taskOptions.length)} />
                      {captureData.calib && (
                        <StatItem label="Calibration" value={captureData.calib} title={captureData.calib} mono />
                      )}
                    </div>
                    {captureData.captureJsonPath && (
                      <div className="shrink-0 flex flex-col items-stretch gap-1">
                        <button
                          onClick={() => window.location.reload()}
                          title="Reload this page"
                          className="inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-xs text-foreground-muted hover:text-foreground bg-surface-2 hover:bg-surface-3 ring-1 ring-inset ring-border transition-colors whitespace-nowrap"
                        >
                          <RotateCw className="w-3.5 h-3.5" /> Reload Page
                        </button>
                        <button
                          onClick={() => setTaskConfigViewer({ name: 'capture.json', path: captureData.captureJsonPath! })}
                          title="View capture config (capture.json)"
                          className="inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-xs text-foreground-muted hover:text-foreground bg-surface-2 hover:bg-surface-3 ring-1 ring-inset ring-border transition-colors whitespace-nowrap"
                        >
                          <FileJson className="w-3.5 h-3.5" /> View Capture config
                        </button>
                        {/* Camera rig + up-axis as ONE joined control, so it reads as
                            a unit: the dropdown sets the axis the button previews with. */}
                        <div className="flex items-stretch rounded-md border border-border overflow-hidden">
                          <button
                            onClick={() => previewCalibRig()}
                            disabled={calibPreviewBusy}
                            title="Open the camera rig in 3D, oriented by the up-axis on the right"
                            className="flex-1 inline-flex items-center gap-1.5 px-2 py-1 text-xs text-foreground-muted hover:text-foreground bg-surface-2 hover:bg-surface-3 transition-colors whitespace-nowrap disabled:opacity-50 disabled:cursor-not-allowed"
                          >
                            <Video className="w-3.5 h-3.5" /> {calibPreviewBusy ? 'Building…' : 'Camera rig'}
                          </button>
                          <div className="relative inline-flex items-stretch border-l border-border">
                            <select
                              value={calibUpAxis}
                              // Re-render an open rig preview with the new up-axis.
                              onChange={(e) => {
                                const a = e.target.value as UpAxisValue;
                                setCalibUpAxis(a);
                                if (rrdWebViewer?.name.startsWith('camera rig')) previewCalibRig(a);
                              }}
                              title="World up-axis used to orient the camera-rig preview"
                              aria-label="Up-axis for the camera-rig preview"
                              className="appearance-none bg-surface-2 hover:bg-surface-3 text-foreground-muted text-xs pl-1.5 pr-5 focus:outline-none cursor-pointer transition-colors"
                            >
                              <option value="auto">up: auto</option>
                              <option value="x">up: +X</option>
                              <option value="-x">up: −X</option>
                              <option value="y">up: +Y</option>
                              <option value="-y">up: −Y</option>
                              <option value="z">up: +Z</option>
                              <option value="-z">up: −Z</option>
                            </select>
                            <ChevronDown className="pointer-events-none absolute right-1 top-1/2 -translate-y-1/2 w-3 h-3 text-foreground-faint" />
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                  {captureData.dataPath && (
                    <div className="mt-3">
                      <StatItem label="Source data" value={captureData.dataPath} title={captureData.dataPath} mono />
                    </div>
                  )}
                </div>
              </div>

              {infoCams.length > 0 && (
                <div className="mt-4 flex flex-wrap items-center gap-1">
                  {infoCams.slice(0, 8).map(c => (
                    <span
                      key={c}
                      className="bg-surface-2 text-foreground-muted rounded px-1.5 py-0.5 text-[11px] font-mono ring-1 ring-inset ring-border"
                    >
                      {c}
                    </span>
                  ))}
                  {infoCams.length > 8 && (
                    <span className="text-foreground-faint text-[11px] px-1" title={infoCams.join(', ')}>
                      +{infoCams.length - 8} more
                    </span>
                  )}
                </div>
              )}

              {/* Run picker — newest first; horizontally scrollable so older
                  runs stay reachable without growing the card. Selecting a pill
                  drives the Outputs explorer below; the icon opens its config. */}
              {taskOptions.length > 0 && (
                <div className="mt-4 pt-4 border-t border-border-subtle">
                  <div className="text-foreground-muted text-[11px] uppercase tracking-wider font-medium mb-2">
                    Runs ({taskOptions.length})
                  </div>
                  <div className="flex gap-1.5 overflow-x-auto pb-1">
                    {taskOptions.map(t => {
                      const isSelected = selectedTaskId === t.taskId;
                      return (
                        <div
                          key={t.taskId}
                          className={`inline-flex items-stretch shrink-0 rounded-md border text-xs overflow-hidden transition-colors ${
                            isSelected
                              ? 'bg-primary-muted-strong border-primary/45 ring-1 ring-inset ring-white/10'
                              : 'bg-surface-2 border-border hover:border-border-strong'
                          }`}
                        >
                          <button
                            onClick={() => setSelectedTaskId(t.taskId)}
                            className={`inline-flex items-center gap-2 px-2.5 py-1.5 ${isSelected ? 'text-primary' : 'text-foreground-muted hover:bg-surface-3 hover:text-foreground'} transition-colors`}
                            title={`Browse outputs of task ${formatTaskId(t.taskId)}`}
                          >
                            <span className="font-mono text-primary">{formatTaskId(t.taskId)}</span>
                            <span className="opacity-60">·</span>
                            <span className="whitespace-nowrap">{STATUS_GLYPH[t.status]} {t.status}</span>
                            {t.relativeTime && <><span className="opacity-60">·</span><span className="text-foreground-faint whitespace-nowrap">{t.relativeTime}</span></>}
                          </button>
                          <button
                            onClick={() => openTaskConfig(t.taskId)}
                            className={`inline-flex items-center px-2 border-l ${isSelected ? 'border-primary/30 text-primary hover:bg-primary-muted' : 'border-border text-foreground-subtle hover:bg-surface-3 hover:text-foreground'} transition-colors`}
                            title={`View task ${formatTaskId(t.taskId)} config (task_${t.taskId}.json)`}
                          >
                            <FileJson className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}

          <ResultExport captureName={captureName} initialSeq={selectedSequence} onGoToExporter={onGoToExporter} />
        </div>

        <div className="grid grid-cols-1 gap-6">
          <div className="lg:col-span-1">
            <div className="bg-surface-1 border border-border-subtle rounded-xl p-6 shadow-sm shadow-black/30 ring-1 ring-inset ring-white/[0.02]">
              <h3 className="text-foreground text-lg font-medium tracking-tight mb-1">Outputs explorer</h3>
              <p className="text-foreground-muted text-xs mb-4">
                One section per pipeline step — pick a task and a sequence, every step's outputs are listed below.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="text-foreground-muted text-[11px] uppercase tracking-wider font-medium mb-1.5 block">Task</label>
                  <select
                    value={selectedTaskId}
                    onChange={e => setSelectedTaskId(e.target.value)}
                    disabled={taskOptions.length === 0}
                    className="w-full bg-surface-2 border border-border rounded-md px-3 py-2 text-foreground text-sm focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition-colors disabled:opacity-50"
                  >
                    {taskOptions.length === 0 && <option value="">No runs yet</option>}
                    {taskOptions.map(t => (
                      <option key={t.taskId} value={t.taskId} className="bg-surface-2">
                        {formatTaskId(t.taskId)} — {truncate(t.outputId, 24)} {STATUS_GLYPH[t.status]} {t.status}{t.relativeTime ? ` · ${t.relativeTime}` : ''}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="text-foreground-muted text-[11px] uppercase tracking-wider font-medium mb-1.5 block">Sequence</label>
                  <select
                    value={selectedSequence}
                    onChange={e => setSelectedSequence(e.target.value)}
                    className="w-full bg-surface-2 border border-border rounded-md px-3 py-2 text-foreground text-sm focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition-colors"
                  >
                    {sequenceOptions.length === 0 && <option value="">No sequences found</option>}
                    {sequenceOptions.map(sequence => (
                      <option key={sequence.name} value={sequence.name} className="bg-surface-2">
                        {sequence.name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {availableProcesses.length === 0 ? (
                <div className="text-foreground-subtle text-sm text-center py-8">
                  This task has no recorded steps.
                </div>
              ) : (
                <div className="grid grid-cols-1 xl:grid-cols-2 gap-3 items-start">
                  {availableProcesses.map(step => (
                    <StepOutputs
                      // Re-mount when the underlying selection changes so each
                      // step starts collapsed/refreshed cleanly. Without the
                      // key, the <StepOutputs> would keep its old relPath
                      // pointing at the previous task/sequence.
                      key={`${selectedTaskId}::${selectedSequence}::${step}`}
                      step={step}
                      baseRelPath={baseRelPathFor(step)}
                      defaultOpen={initial?.process ? step === initial.process : true}
                      scrollIntoViewOnMount={initial?.process === step}
                      onPlayVideo={setPlayingVideoRelPath}
                      onPlayImage={(relPath, siblings) => {
                        setImageSiblings(siblings);
                        setPlayingImageRelPath(relPath);
                      }}
                      onOpenRrdBrowser={(path, name) => setRrdWebViewer({ path, name })}
                      onOpenRrdNative={openRrd}
                      onOpenHtml={(path, name) => setHtmlViewer({ path, name })}
                      onOpenNpz={(path, name) => setNpzViewer({ path, name })}
                      // Reuses the shared FileViewerModal (already wired
                      // for task-config viewing) — it renders type-aware
                      // bodies for json/csv/yaml/plain text.
                      onOpenText={(path, name) => setTaskConfigViewer({ path, name })}
                      outLog={logsByStep.get(step)?.outFile}
                      errLog={logsByStep.get(step)?.errFile}
                      headerExtras={step === 'ma_cap' && selectedTaskId ? (
                        <>
                          <button
                            type="button"
                            onClick={() => openTaskPreset(selectedTaskId)}
                            title="View the source preset this run was created from"
                            className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-mono text-foreground-muted hover:text-foreground hover:bg-surface-3 ring-1 ring-inset ring-border transition-colors"
                          >
                            <FileJson className="w-3 h-3" /> preset
                          </button>
                          <button
                            type="button"
                            onClick={() => openTaskConfig(selectedTaskId)}
                            title={`View the run config (run_${selectedTaskId}.json)`}
                            className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-mono text-foreground-muted hover:text-foreground hover:bg-surface-3 ring-1 ring-inset ring-border transition-colors"
                          >
                            <FileJson className="w-3 h-3" /> task
                          </button>
                        </>
                      ) : undefined}
                    />
                  ))}

                  {/* ma_vis preview — fills the spare grid cell when a
                      preview.mp4 exists for this task/sequence. Matches the step
                      cards' look + fixed height for a tidy grid. */}
                  {previewRelPath && (
                    <div className="bg-background border border-border-subtle rounded-lg overflow-hidden flex flex-col">
                      <div className="w-full flex items-center gap-2 px-3 py-2 bg-surface-1 border-b border-border-subtle">
                        <Film className="w-4 h-4 text-foreground-muted" />
                        <span className="text-foreground text-sm font-medium">Preview</span>
                        <span className="text-foreground-faint text-xs font-mono">(preview.mp4)</span>
                        <button
                          onClick={() => setPlayingVideoRelPath(previewRelPath)}
                          className="ml-auto p-1 rounded-md text-foreground-muted hover:text-foreground hover:bg-surface-3 transition-colors"
                          title="Open fullscreen"
                        >
                          <Maximize2 className="w-3.5 h-3.5" />
                        </button>
                      </div>
                      <div className="h-[300px] bg-background flex items-center justify-center p-2">
                        <video
                          key={previewRelPath}
                          src={`/api/files/stream?path=${encodeURIComponent(previewRelPath)}`}
                          controls
                          className="max-h-full max-w-full rounded"
                        />
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>

        {(playingVideoRelPath || playingImageRelPath) && (
          <div
            className="fixed inset-0 bg-black/85 backdrop-blur-sm flex items-center justify-center z-50 p-4"
            onClick={() => {
              setPlayingVideoRelPath(null);
              setPlayingImageRelPath(null);
            }}
          >
            <div
              className="relative w-[90vw] max-w-[1400px] max-h-[90vh]"
              onClick={e => e.stopPropagation()}
            >
              <button
                onClick={() => {
                  setPlayingVideoRelPath(null);
                  setPlayingImageRelPath(null);
                }}
                className="absolute top-2 right-2 text-foreground hover:text-foreground transition-colors z-10 bg-surface-2 hover:bg-surface-3 rounded-md p-1.5 ring-1 ring-border"
              >
                <X className="w-5 h-5" />
              </button>
              {playingVideoRelPath && (
                <VideoPlayer relPath={playingVideoRelPath} />
              )}
              {playingImageRelPath && (
                <div className="flex max-h-[82vh] items-center justify-center overflow-hidden rounded-lg bg-black/30 p-2">
                  <img
                    src={`/api/files/image?path=${encodeURIComponent(playingImageRelPath)}`}
                    alt={playingImageRelPath.split('/').pop()}
                    className="block max-h-full max-w-full w-auto h-auto object-contain"
                  />
                </div>
              )}
              {playingImageRelPath && imageSiblings.length > 1 && (
                <>
                  <button
                    onClick={() => stepImage(-1)}
                    className="absolute left-2 top-1/2 -translate-y-1/2 z-10 bg-surface-2/90 hover:bg-surface-3 text-foreground rounded-full p-2 ring-1 ring-border transition-colors"
                    aria-label="Previous image"
                    title="Previous (←)"
                  >
                    <ChevronLeft className="w-5 h-5" />
                  </button>
                  <button
                    onClick={() => stepImage(1)}
                    className="absolute right-2 top-1/2 -translate-y-1/2 z-10 bg-surface-2/90 hover:bg-surface-3 text-foreground rounded-full p-2 ring-1 ring-border transition-colors"
                    aria-label="Next image"
                    title="Next (→)"
                  >
                    <ChevronRight className="w-5 h-5" />
                  </button>
                </>
              )}
              <p className="text-foreground-subtle text-xs font-mono mt-2 text-center">
                {(playingVideoRelPath || playingImageRelPath || '').split('/').pop()}
                {playingImageRelPath && imageSiblings.length > 1 && (
                  <span className="text-foreground-faint">
                    {'  '}· {imageIndex + 1} / {imageSiblings.length}
                  </span>
                )}
              </p>
            </div>
          </div>
        )}

        <FileViewerModal file={taskConfigViewer} onClose={() => setTaskConfigViewer(null)} />

        {rrdWebViewer && (
          <RerunWebViewer
            key={rrdWebViewer.path}
            rrdPath={rrdWebViewer.path}
            fileName={rrdWebViewer.name}
            onClose={() => setRrdWebViewer(null)}
            onOpenNative={(fresh) => {
              const path = rrdWebViewer.path;
              setRrdWebViewer(null);
              openRrd(path, fresh);
            }}
          />
        )}

        {htmlViewer && (
          <HtmlViewer
            htmlPath={htmlViewer.path}
            fileName={htmlViewer.name}
            onClose={() => setHtmlViewer(null)}
          />
        )}

        {npzViewer && (
          <NpzViewer
            npzPath={npzViewer.path}
            fileName={npzViewer.name}
            onClose={() => setNpzViewer(null)}
          />
        )}
      </div>
    </div>
  );
}

/** One labelled stat in the capture info card. `value` may be any node (e.g. the
 *  coloured "Latest run" status). `mono` renders paths/filenames in a compact
 *  monospace; `title` exposes the untruncated value on hover. */
function StatItem({ label, value, title, mono, className }: {
  label: string;
  value: ReactNode;
  title?: string;
  mono?: boolean;
  className?: string;
}) {
  return (
    <div className={`min-w-0 ${className ?? ''}`}>
      <div className="text-foreground-muted text-[11px] uppercase tracking-wider font-medium mb-1">{label}</div>
      <div className={`text-foreground truncate ${mono ? 'font-mono text-xs' : 'text-sm'}`} title={title}>
        {value}
      </div>
    </div>
  );
}
