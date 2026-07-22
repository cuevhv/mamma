import { useEffect, useState, useCallback } from 'react';
import { Download, Check, Loader2, AlertTriangle, ChevronDown, ChevronRight, ChevronLeft, Search, Box, FolderOpen, X, FileJson } from 'lucide-react';
import { ExportPanel, ExportJobStatus, jget, jpost, type Readiness, type Job } from './ExportPanel';
import { SequenceSelect } from './SequenceSelect';
import { MeshViewer } from './MeshPreview';
import { StepOutputs } from './StepOutputs';
import { ALL_STEPS } from './ProcessTable';
import { RerunWebViewer, openRrdNative } from './RerunWebViewer';
import { NpzViewer } from './NpzViewer';
import { HtmlViewer } from './HtmlViewer';
import { FileViewerModal } from './shared/FileViewerModal';
import { VideoPlayer } from './CaptureDetail';

/** SMPL-X Exporter tab — a full-screen workbench: on the LEFT the setup
 *  (Blender + add-on downloads), the sequence source (detected results OR a
 *  custom path) as an always-open list, and the shared export panel; on the
 *  RIGHT an embedded 3D viewer that previews the clicked sequence and a log
 *  card for export progress. The export core lives in ExportPanel so it's
 *  reused on the results page (which keeps the dropdown + modal shape). */

interface Seq { tag: string; capture: string; seq: string; people: number; ma_3d_dir: string; ma_cap_dir?: string; already_exported: boolean; mtime?: number; }

/** The task behind a disk-discovered item, resolved by the backend so the
 *  Outputs view can offer the run-config + per-step log buttons. */
interface TaskCtx { taskId?: string; taskJsonPath?: string | null; logs?: Record<string, { out?: string | null; err?: string | null }>; }

const seqKey = (s: Seq) => `${s.tag}/${s.capture}/${s.seq}::${s.ma_3d_dir}`;

export function Exporter() {
  const [ready, setReady] = useState<Readiness | null>(null);
  const [seqs, setSeqs] = useState<Seq[]>([]);
  const [source, setSource] = useState<'detected' | 'custom'>('detected');
  const [sels, setSels] = useState<string[]>([]);
  /** The row currently shown in the right-hand 3D viewer (click to preview). */
  const [active, setActive] = useState<Seq | null>(null);
  /** Which view the right-hand card shows for the active item. Persists
   *  across `active` changes (deliberately not reset when the selection moves). */
  const [rightView, setRightView] = useState<'viewer' | 'outputs'>('viewer');
  // Outputs-explorer modal / lightbox state — mirrors the capture Results page.
  const [rrdWebViewer, setRrdWebViewer] = useState<{ path: string; name: string } | null>(null);
  const [htmlViewer, setHtmlViewer] = useState<{ path: string; name: string } | null>(null);
  const [npzViewer, setNpzViewer] = useState<{ path: string; name: string } | null>(null);
  const [textViewer, setTextViewer] = useState<{ path: string; name: string } | null>(null);
  const [playingVideoRelPath, setPlayingVideoRelPath] = useState<string | null>(null);
  const [playingImageRelPath, setPlayingImageRelPath] = useState<string | null>(null);
  const [imageSiblings, setImageSiblings] = useState<string[]>([]);
  /** Mirror of ExportPanel's internal job, surfaced in the log card. */
  const [exportJob, setExportJob] = useState<Job | null>(null);
  /** Task resolved from the active item's output id (null = not in the DB). */
  const [taskCtx, setTaskCtx] = useState<TaskCtx | null>(null);
  // custom path
  const [customPath, setCustomPath] = useState('');
  const [customSeqs, setCustomSeqs] = useState<Seq[]>([]);
  const [customSels, setCustomSels] = useState<string[]>([]);
  const [scan, setScan] = useState<{ state: 'idle' | 'scanning' | 'done' | 'error'; msg?: string }>({ state: 'idle' });
  // tools
  const [dlJob, setDlJob] = useState<Job | null>(null);
  const [showAddonForm, setShowAddonForm] = useState(false);
  const [creds, setCreds] = useState({ username: '', password: '' });
  const [setupOpen, setSetupOpen] = useState(true);

  const refreshReady = useCallback(async () => setReady(await jget<Readiness>('/api/exporter/readiness')), []);
  const refreshSeqs = useCallback(async () => setSeqs((await jget<{ sequences: Seq[] }>('/api/exporter/sequences')).sequences), []);
  useEffect(() => { refreshReady(); refreshSeqs(); }, [refreshReady, refreshSeqs]);

  const toolsReady = !!ready?.blender.present && !!ready?.addon.present;
  useEffect(() => { if (toolsReady) setSetupOpen(false); }, [toolsReady]);

  // Poll a download job until done, then refresh readiness.
  useEffect(() => {
    if (!dlJob || dlJob.state !== 'running') return;
    const t = setInterval(async () => {
      const j = await jget<Job>(`/api/exporter/job/${dlJob.id}`);
      setDlJob(j);
      if (j.state !== 'running') refreshReady();
    }, 1500);
    return () => clearInterval(t);
  }, [dlJob, refreshReady]);

  // Resolve the active item back to its task (best-effort: CLI runs and
  // custom-path items aren't in the DB and simply get no task/log buttons).
  useEffect(() => {
    setTaskCtx(null);
    if (!active) return;
    const c = new AbortController();
    const q = new URLSearchParams({ tag: active.tag, capture: active.capture, seq: active.seq });
    fetch(`/api/exporter/task-context?${q}`, { signal: c.signal })
      .then(r => (r.ok ? r.json() : null))
      .then(d => { if (d && d.taskId) setTaskCtx(d); })
      .catch(() => {});
    return () => c.abort();
  }, [active]);

  // Image lightbox prev/next within the folder it was opened from; index is
  // derived from the live siblings list so it survives re-renders and wraps.
  const imageIndex = playingImageRelPath ? imageSiblings.indexOf(playingImageRelPath) : -1;
  const stepImage = (delta: number) => {
    if (imageIndex < 0 || imageSiblings.length < 2) return;
    const next = (imageIndex + delta + imageSiblings.length) % imageSiblings.length;
    setPlayingImageRelPath(imageSiblings[next]);
  };

  // One keydown effect for the video/image overlays: Escape closes the topmost
  // one; arrows page the gallery while the image lightbox is open. The other
  // viewers (rrd/html/npz/text) own their own Escape handling.
  useEffect(() => {
    if (!playingVideoRelPath && !playingImageRelPath) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { setPlayingVideoRelPath(null); setPlayingImageRelPath(null); }
      else if (playingImageRelPath && e.key === 'ArrowLeft') { e.preventDefault(); stepImage(-1); }
      else if (playingImageRelPath && e.key === 'ArrowRight') { e.preventDefault(); stepImage(1); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playingVideoRelPath, playingImageRelPath, imageSiblings]);

  const startBlender = async () => {
    const r = await jpost<{ job_id: string }>('/api/exporter/download-blender');
    setDlJob({ id: r.job_id, state: 'running', log_tail: [], outputs: [], error: null, kind: 'blender' });
  };
  const startAddon = async () => {
    const r = await jpost<{ job_id: string }>('/api/exporter/download-addon', creds);
    setShowAddonForm(false); setCreds({ username: '', password: '' });
    setDlJob({ id: r.job_id, state: 'running', log_tail: [], outputs: [], error: null, kind: 'addon' });
  };

  const runScan = async () => {
    if (!customPath.trim()) return;
    setScan({ state: 'scanning' }); setCustomSeqs([]); setCustomSels([]);
    try {
      const r = await jget<{ sequences: Seq[]; error?: string }>(`/api/exporter/scan?path=${encodeURIComponent(customPath.trim())}`);
      if (r.error) { setScan({ state: 'error', msg: r.error }); return; }
      setCustomSeqs(r.sequences);
      if (r.sequences.length === 1) setCustomSels([seqKey(r.sequences[0])]);
      setScan({ state: 'done', msg: `${r.sequences.length} sequence(s) found` });
    } catch (e) { setScan({ state: 'error', msg: String(e) }); }
  };

  const list = source === 'detected' ? seqs : customSeqs;
  const selKeys = source === 'detected' ? sels : customSels;
  const targets = list.filter(s => selKeys.includes(seqKey(s)));

  const card = 'bg-surface-1 border border-border-subtle rounded-xl p-5 shadow-sm shadow-black/30';
  const ToolRow = ({ name, present, onDl }: { name: string; present: boolean; onDl: () => void }) => (
    <div className="flex items-center justify-between py-1.5">
      <span className="text-sm text-foreground">{name}</span>
      {present ? (
        <span className="inline-flex items-center gap-1.5 text-status-completed text-xs"><Check className="w-4 h-4" /> Ready</span>
      ) : (
        <button onClick={onDl} className="inline-flex items-center gap-1.5 px-2.5 py-1 bg-primary text-primary-foreground rounded-md text-xs font-medium">
          <Download className="w-3.5 h-3.5" /> Download
        </button>
      )}
    </div>
  );
  const tab = (id: 'detected' | 'custom', label: string) => (
    <button onClick={() => setSource(id)}
      className={`px-3 py-1 rounded-md text-xs transition-colors ${source === id ? 'bg-surface-2 text-foreground ring-1 ring-inset ring-border' : 'text-foreground-muted hover:text-foreground'}`}>
      {label}
    </button>
  );
  const emptyPanel = (msg: string) => (
    <div className="h-full min-h-0 flex items-center justify-center bg-surface-2 border border-border rounded-md px-4 text-center text-foreground-faint text-sm">{msg}</div>
  );
  // Right-card view switch — same idiom as the source `tab()` above.
  const viewTab = (id: 'viewer' | 'outputs', label: string, Icon: typeof Box) => (
    <button onClick={() => setRightView(id)}
      className={`px-2.5 py-1 rounded-md text-xs inline-flex items-center gap-1.5 transition-colors ${rightView === id ? 'bg-surface-2 text-foreground ring-1 ring-inset ring-border' : 'text-foreground-muted hover:text-foreground'}`}>
      <Icon className="w-3.5 h-3.5" /> {label}
    </button>
  );

  return (
    // Full-width workbench for this tab (App imposes no width wrapper — each page
    // owns its container; the other tabs stay centered/max-width as before).
    <div className="w-full px-4 lg:px-6 py-4">
      <div className="mb-3">
        <h1 className="text-foreground text-xl font-semibold">Exporter</h1>
        <p className="text-foreground-muted text-sm mt-1">Export the SMPL-X animation to standard formats for render/game engines and 3D tools. Click a sequence to preview it, tick the ones to export.</p>
      </div>

      {/* On xl the grid fills the viewport (100vh − nav 3.5rem − header/padding);
          below xl it collapses to a single natural-height column. */}
      <div className="grid grid-cols-1 xl:grid-cols-[minmax(380px,2fr)_3fr] gap-4 xl:h-[calc(100vh-10rem)] min-h-[540px]">
        {/* ── LEFT: setup · source · list · export ─────────────────────────── */}
        <div className="flex flex-col gap-3 min-h-0">
          {/* ① Export tools */}
          <div className={`${card} flex-shrink-0`}>
            <button onClick={() => setSetupOpen(o => !o)} className="flex items-center gap-2 w-full text-left">
              {setupOpen ? <ChevronDown className="w-4 h-4 text-foreground-subtle" /> : <ChevronRight className="w-4 h-4 text-foreground-subtle" />}
              <span className="text-sm font-medium text-foreground">Export tools</span>
              {toolsReady
                ? <span className="ml-auto inline-flex items-center gap-1.5 text-status-completed text-xs"><Check className="w-4 h-4" /> Blender + add-on ready</span>
                : <span className="ml-auto text-status-pending text-xs">setup needed for FBX/ABC/BVH/USD</span>}
            </button>
            {setupOpen && (
              <div className="mt-3 pl-6 space-y-1">
                <ToolRow name="Portable Blender 4.5 LTS" present={!!ready?.blender.present} onDl={startBlender} />
                <div>
                  <ToolRow name="SMPL-X add-on (gated)" present={!!ready?.addon.present} onDl={() => setShowAddonForm(s => !s)} />
                  {showAddonForm && !ready?.addon.present && (
                    <div className="ml-0 mt-1 rounded-md border border-border-subtle bg-surface-2/60 p-3 space-y-2">
                      <p className="text-foreground-subtle text-xs">Sign in with your SMPL-X account (smpl-x.is.tue.mpg.de).</p>
                      <input className="w-full bg-surface-2 border border-border rounded-md px-2 py-1 text-xs text-foreground" placeholder="Username" value={creds.username} onChange={e => setCreds(c => ({ ...c, username: e.target.value }))} />
                      <input type="password" className="w-full bg-surface-2 border border-border rounded-md px-2 py-1 text-xs text-foreground" placeholder="Password" value={creds.password} onChange={e => setCreds(c => ({ ...c, password: e.target.value }))} />
                      <button onClick={startAddon} disabled={!creds.username || !creds.password} className="inline-flex items-center gap-1.5 px-2.5 py-1 bg-primary text-primary-foreground rounded-md text-xs font-medium disabled:opacity-40">
                        <Download className="w-3.5 h-3.5" /> Download add-on
                      </button>
                    </div>
                  )}
                </div>
                <p className="text-foreground-faint text-xs pt-1">The <code className="font-mono">npz</code> format needs none of this; FBX/ABC/BVH/USD need both.</p>
                {dlJob && (
                  <div className="text-xs pt-1">
                    {dlJob.state === 'running' && <span className="inline-flex items-center gap-1.5 text-status-running"><Loader2 className="w-3.5 h-3.5 animate-spin" /> downloading {dlJob.kind}…</span>}
                    {dlJob.state === 'error' && <span className="inline-flex items-center gap-1.5 text-status-failed"><AlertTriangle className="w-3.5 h-3.5" /> {dlJob.error}</span>}
                    {dlJob.state === 'ready' && <span className="inline-flex items-center gap-1.5 text-status-completed"><Check className="w-3.5 h-3.5" /> {dlJob.kind} ready</span>}
                    {dlJob.state !== 'ready' && dlJob.log_tail.length > 0 && <span className="text-foreground-faint ml-2 font-mono">{dlJob.log_tail[dlJob.log_tail.length - 1].slice(0, 80)}</span>}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* ② What to export — source tabs (+ custom path input) */}
          <div className={`${card} flex-shrink-0`}>
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-foreground">Select sequences to export</span>
              <div className="flex gap-1">{tab('detected', 'Detected results')}{tab('custom', 'Custom path')}</div>
            </div>
            {source === 'custom' && (
              <div className="mt-2 space-y-2">
                <div className="flex gap-2">
                  <input value={customPath} onChange={e => setCustomPath(e.target.value)} onKeyDown={e => e.key === 'Enter' && runScan()}
                    placeholder="/path/to/ma_3d (or a sequence folder with smplx_params_*.npz)"
                    className="flex-1 bg-surface-2 border border-border rounded-md px-2 py-2 text-sm text-foreground font-mono" />
                  <button onClick={runScan} disabled={!customPath.trim() || scan.state === 'scanning'}
                    className="inline-flex items-center gap-1.5 px-3 py-2 bg-surface-2 border border-border rounded-md text-sm text-foreground disabled:opacity-40">
                    {scan.state === 'scanning' ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />} Find
                  </button>
                </div>
                {scan.state === 'error' && <p className="text-status-failed text-xs"><AlertTriangle className="w-3.5 h-3.5 inline mr-1" />{scan.msg}</p>}
                {scan.state === 'done' && customSeqs.length === 0 && <p className="text-foreground-muted text-xs">No <code className="font-mono">smplx_params_*.npz</code> found under that path.</p>}
              </div>
            )}
            {source === 'detected' && seqs.length === 0 && (
              <p className="mt-2 text-foreground-muted text-sm">No completed <code className="font-mono">ma_3d</code> results found. Run the pipeline, or use <button onClick={() => setSource('custom')} className="text-primary hover:underline">Custom path</button>.</p>
            )}
          </div>

          {/* ③ Sequence list — always-open panel that fills the remaining height.
              The xl floor keeps the list usable when the setup card is expanded
              AND the formats card shows the missing-tools notice (both tall). */}
          <div className="flex-1 min-h-[16rem] xl:min-h-[8rem]">
            {source === 'detected' ? (
              seqs.length > 0
                ? <SequenceSelect variant="panel" seqs={seqs} values={sels} onChange={setSels} getKey={seqKey} showCapture
                    activeKey={active ? seqKey(active) : null} onActivate={setActive} />
                : emptyPanel('Detected ma_3d results will be listed here.')
            ) : (
              customSeqs.length > 0
                ? <SequenceSelect variant="panel" seqs={customSeqs} values={customSels} onChange={setCustomSels} getKey={seqKey} showCapture
                    activeKey={active ? seqKey(active) : null} onActivate={setActive} />
                : emptyPanel(scan.state === 'scanning' ? 'Scanning…' : 'Find a path to list its sequences here.')
            )}
          </div>

          {/* ④ Formats & options + run (shared) */}
          <div className={`${card} flex-shrink-0`}>
            <div className="text-sm font-medium text-foreground mb-3">Formats & options</div>
            <ExportPanel targets={targets} readiness={ready} onNeedTools={() => setSetupOpen(true)} hideJobStatus onJobChange={setExportJob} />
          </div>
        </div>

        {/* ── RIGHT: 3D viewer / outputs · export log ──────────────────────── */}
        <div className="flex flex-col gap-4 min-h-0">
          {/* Preview card — 3D viewer OR the per-step Outputs explorer for the
              active (clicked) row; the view switch persists across selections. */}
          <div className="flex-1 min-h-[20rem] xl:min-h-0 flex flex-col bg-background border border-border-subtle rounded-lg overflow-hidden">
            <div className="flex items-center gap-2 px-3 py-2 bg-surface-1 border-b border-border-subtle flex-shrink-0">
              {rightView === 'viewer'
                ? <Box className="w-4 h-4 text-foreground-muted" />
                : <FolderOpen className="w-4 h-4 text-foreground-muted" />}
              <span className="text-foreground text-sm font-medium">{rightView === 'viewer' ? '3D Viewer' : 'Outputs'}</span>
              {active && <span className="min-w-0 text-foreground-faint text-xs font-mono truncate">{active.capture} / {active.seq}</span>}
              <div className="ml-auto flex gap-1 bg-surface-2/50 rounded-lg p-0.5 flex-shrink-0">
                {viewTab('viewer', '3D Viewer', Box)}
                {viewTab('outputs', 'Outputs', FolderOpen)}
              </div>
            </div>
            <div className="flex-1 min-h-0">
              {rightView === 'viewer' ? (
                active ? (
                  <MeshViewer key={seqKey(active)} target={active} className="h-full" />
                ) : (
                  <div className="h-full flex flex-col items-center justify-center gap-2 text-foreground-faint">
                    <Box className="w-8 h-8 opacity-40" />
                    <span className="text-sm">Click a sequence to preview it here</span>
                  </div>
                )
              ) : (
                active ? (
                  // Re-mount the whole stack on selection change (key) so each
                  // step's in-section navigation resets to its new root.
                  <div key={seqKey(active)} className="h-full overflow-y-auto p-3 space-y-3">
                    {[...ALL_STEPS, 'export'].map(step => (
                      <StepOutputs
                        key={step}
                        step={step}
                        defaultOpen={false}
                        // Same substitution the backend uses for ma_cap: swap the
                        // `/ma_3d/` segment of this row's dir for the target step,
                        // then append the sequence name. For 'export' this lands on
                        // the exporter's default output dir.
                        baseRelPath={`${active.ma_3d_dir.replace('/ma_3d/', `/${step}/`)}/${active.seq}`}
                        onPlayVideo={setPlayingVideoRelPath}
                        onPlayImage={(relPath, siblings) => { setImageSiblings(siblings); setPlayingImageRelPath(relPath); }}
                        onOpenRrdBrowser={(path, name) => setRrdWebViewer({ path, name })}
                        onOpenRrdNative={(relPath, fresh) => openRrdNative(relPath, fresh)}
                        onOpenHtml={(path, name) => setHtmlViewer({ path, name })}
                        onOpenNpz={(path, name) => setNpzViewer({ path, name })}
                        onOpenText={(path, name) => setTextViewer({ path, name })}
                        outLog={taskCtx?.logs?.[step]?.out ?? undefined}
                        errLog={taskCtx?.logs?.[step]?.err ?? undefined}
                        headerExtras={taskCtx?.taskJsonPath ? (
                          <button
                            type="button"
                            onClick={() => setTextViewer({ name: `run_${taskCtx.taskId}.json`, path: taskCtx.taskJsonPath! })}
                            title={`View the run config (run_${taskCtx.taskId}.json)`}
                            className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-mono text-foreground-muted hover:text-foreground hover:bg-surface-3 ring-1 ring-inset ring-border transition-colors"
                          >
                            <FileJson className="w-3 h-3" /> task
                          </button>
                        ) : undefined}
                      />
                    ))}
                  </div>
                ) : (
                  <div className="h-full flex flex-col items-center justify-center gap-2 text-foreground-faint">
                    <FolderOpen className="w-8 h-8 opacity-40" />
                    <span className="text-sm">Click a sequence to browse its outputs</span>
                  </div>
                )
              )}
            </div>
          </div>

          {/* Export log — mirrors ExportPanel's job status */}
          <div className="flex-shrink-0 max-h-[35%] min-h-[9rem] overflow-y-auto bg-background border border-border-subtle rounded-lg">
            <div className="sticky top-0 flex items-center gap-2 px-3 py-2 bg-surface-1 border-b border-border-subtle">
              <span className="text-foreground text-sm font-medium">Export log</span>
            </div>
            <div className="p-3">
              {exportJob
                ? <ExportJobStatus job={exportJob} />
                : <p className="text-foreground-faint text-sm">Run an export to see progress here.</p>}
            </div>
          </div>
        </div>
      </div>

      {/* Video / image lightbox for the Outputs explorer (backdrop + Escape
          close; arrows page the gallery). Mirrors the capture Results page. */}
      {(playingVideoRelPath || playingImageRelPath) && (
        <div
          className="fixed inset-0 bg-black/85 backdrop-blur-sm flex items-center justify-center z-50 p-4"
          onClick={() => { setPlayingVideoRelPath(null); setPlayingImageRelPath(null); }}
        >
          <div className="relative w-[90vw] max-w-[1400px] max-h-[90vh]" onClick={e => e.stopPropagation()}>
            <button
              onClick={() => { setPlayingVideoRelPath(null); setPlayingImageRelPath(null); }}
              className="absolute top-2 right-2 text-foreground hover:text-foreground transition-colors z-10 bg-surface-2 hover:bg-surface-3 rounded-md p-1.5 ring-1 ring-border"
            >
              <X className="w-5 h-5" />
            </button>
            {playingVideoRelPath && <VideoPlayer relPath={playingVideoRelPath} />}
            {playingImageRelPath && (
              <img
                src={`/api/files/image?path=${encodeURIComponent(playingImageRelPath)}`}
                alt={playingImageRelPath.split('/').pop()}
                className="max-h-[85vh] max-w-full object-contain rounded mx-auto"
              />
            )}
            {playingImageRelPath && imageSiblings.length > 1 && (
              <>
                <button
                  onClick={() => stepImage(-1)}
                  className="absolute left-2 top-1/2 -translate-y-1/2 z-10 bg-surface-2/90 hover:bg-surface-3 text-foreground rounded-full p-2 ring-1 ring-border transition-colors"
                  aria-label="Previous image" title="Previous (←)"
                >
                  <ChevronLeft className="w-5 h-5" />
                </button>
                <button
                  onClick={() => stepImage(1)}
                  className="absolute right-2 top-1/2 -translate-y-1/2 z-10 bg-surface-2/90 hover:bg-surface-3 text-foreground rounded-full p-2 ring-1 ring-border transition-colors"
                  aria-label="Next image" title="Next (→)"
                >
                  <ChevronRight className="w-5 h-5" />
                </button>
              </>
            )}
          </div>
        </div>
      )}

      <FileViewerModal file={textViewer} onClose={() => setTextViewer(null)} />

      {rrdWebViewer && (
        <RerunWebViewer
          key={rrdWebViewer.path}
          rrdPath={rrdWebViewer.path}
          fileName={rrdWebViewer.name}
          onClose={() => setRrdWebViewer(null)}
          onOpenNative={(fresh) => { const path = rrdWebViewer.path; setRrdWebViewer(null); openRrdNative(path, fresh); }}
        />
      )}

      {htmlViewer && (
        <HtmlViewer htmlPath={htmlViewer.path} fileName={htmlViewer.name} onClose={() => setHtmlViewer(null)} />
      )}

      {npzViewer && (
        <NpzViewer npzPath={npzViewer.path} fileName={npzViewer.name} onClose={() => setNpzViewer(null)} />
      )}
    </div>
  );
}
