import { useState, useEffect, useMemo } from 'react';
import { ArrowLeft, X, FileJson } from 'lucide-react';
import { toast } from 'sonner';
import { ALL_STEPS, buildProcessRows, rowRollupStatus, RowStatus } from './ProcessTable';
import { useTaskPolling } from './shared/useTaskPolling';
import { FileViewerModal } from './shared/FileViewerModal';
import { Skeleton } from './shared/Skeleton';
import { RerunWebViewer } from './RerunWebViewer';
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
  // Run config viewer state — shared across all the "view config" pill actions.
  const [taskConfigViewer, setTaskConfigViewer] = useState<{ name: string; path: string } | null>(null);
  /** When set, the embedded Rerun web viewer is open for this .rrd. */
  const [rrdWebViewer, setRrdWebViewer] = useState<{ path: string; name: string } | null>(null);
  /** When set, the embedded HTML viewer is open for this .html / .htm. */
  const [htmlViewer, setHtmlViewer] = useState<{ path: string; name: string } | null>(null);
  /** When set, the .npz inspector is open for this archive. */
  const [npzViewer, setNpzViewer] = useState<{ path: string; name: string } | null>(null);

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
        toast.success(`Opening ${path.split('/').pop()} in Rerun${fresh && data.layout_reset ? ' (fresh layout)' : ''}…`);
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

  const tasks: Task[] = captureData
    ? captureData.tasks.map(t => ({
        taskId: t.id,
        seqNames: t.sequenceNames ?? [],
        createdAt: t.startedAt
      }))
    : [];

  const numberOfSequences = captureData?.sequences?.length ?? 0;

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

        {/* Recent runs summary — gives the user a "what produced these
            outputs" answer without dragging the live matrix in here. The
            matrix lives only in the Tasks tab now. */}
        {taskOptions.length > 0 && (
          <section className="mb-6">
            <div className="bg-surface-1 border border-border-subtle rounded-xl p-5 shadow-sm shadow-black/30 ring-1 ring-inset ring-white/[0.02]">
              <div className="flex items-end justify-between gap-4 mb-4 flex-wrap">
                <div>
                  <h3 className="text-foreground text-lg font-medium tracking-tight">Runs for this capture</h3>
                  <p className="text-foreground-muted text-sm mt-0.5">
                    {taskOptions.length} run{taskOptions.length === 1 ? '' : 's'} · {numberOfSequences} sequence{numberOfSequences === 1 ? '' : 's'} · live monitoring is in the Tasks tab.
                  </p>
                </div>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {taskOptions.map(t => {
                  const isSelected = selectedTaskId === t.taskId;
                  return (
                    <div
                      key={t.taskId}
                      className={`inline-flex items-stretch rounded-md border text-xs overflow-hidden transition-colors ${
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
                        <span>{STATUS_GLYPH[t.status]} {t.status}</span>
                        {t.relativeTime && <><span className="opacity-60">·</span><span className="text-foreground-faint">{t.relativeTime}</span></>}
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
          </section>
        )}

        <ResultExport captureName={captureName} initialSeq={selectedSequence} onGoToExporter={onGoToExporter} />

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
                <div className="flex flex-col gap-3">
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
                      onPlayImage={setPlayingImageRelPath}
                      onOpenRrdBrowser={(path, name) => setRrdWebViewer({ path, name })}
                      onOpenRrdNative={openRrd}
                      onOpenHtml={(path, name) => setHtmlViewer({ path, name })}
                      onOpenNpz={(path, name) => setNpzViewer({ path, name })}
                      // Reuses the shared FileViewerModal (already wired
                      // for task-config viewing) — it renders type-aware
                      // bodies for json/csv/yaml/plain text.
                      onOpenText={(path, name) => setTaskConfigViewer({ path, name })}
                    />
                  ))}
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
                <div className="max-h-[82vh] overflow-auto rounded-lg bg-black/30 p-2">
                  <img
                    src={`/api/files/image?path=${encodeURIComponent(playingImageRelPath)}`}
                    alt={playingImageRelPath.split('/').pop()}
                    className="block h-auto w-auto max-w-none"
                  />
                </div>
              )}
              <p className="text-foreground-subtle text-xs font-mono mt-2 text-center">
                {(playingVideoRelPath || playingImageRelPath || '').split('/').pop()}
              </p>
            </div>
          </div>
        )}

        <FileViewerModal file={taskConfigViewer} onClose={() => setTaskConfigViewer(null)} />

        {rrdWebViewer && (
          <RerunWebViewer
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
