import { useEffect, useState, useCallback } from 'react';
import { Download, Check, Loader2, AlertTriangle, ChevronDown, ChevronRight, Search } from 'lucide-react';
import { ExportPanel, jget, jpost, type Readiness, type Job } from './ExportPanel';
import { SequenceSelect } from './SequenceSelect';

/** SMPL-X Exporter tab: in-tab setup (Blender + add-on downloads) and readiness,
 *  a sequence source (detected results OR a custom path), and the shared export
 *  panel. The export core itself lives in ExportPanel so it's reused on results. */

interface Seq { tag: string; capture: string; seq: string; people: number; ma_3d_dir: string; ma_cap_dir: string; already_exported: boolean; mtime?: number; }

const seqKey = (s: Seq) => `${s.tag}/${s.capture}/${s.seq}::${s.ma_3d_dir}`;

export function Exporter() {
  const [ready, setReady] = useState<Readiness | null>(null);
  const [seqs, setSeqs] = useState<Seq[]>([]);
  const [source, setSource] = useState<'detected' | 'custom'>('detected');
  const [sels, setSels] = useState<string[]>([]);
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

  return (
    <div className="max-w-3xl mx-auto px-4 py-6 space-y-4">
      <div>
        <h1 className="text-foreground text-xl font-semibold">Exporter</h1>
        <p className="text-foreground-muted text-sm mt-1">Export SMPL-X fits to Blender / engine formats (npz, FBX, Alembic, BVH, USD).</p>
      </div>

      {/* ① Export tools */}
      <div className={card}>
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

      {/* ② What to export */}
      <div className={card}>
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-foreground">What to export</span>
          <div className="flex gap-1">{tab('detected', 'Detected results')}{tab('custom', 'Custom path')}</div>
        </div>

        {source === 'detected' ? (
          seqs.length === 0 ? (
            <p className="text-foreground-muted text-sm">No completed <code className="font-mono">ma_3d</code> results found. Run the pipeline, or use <button onClick={() => setSource('custom')} className="text-primary hover:underline">Custom path</button>.</p>
          ) : (
            <SequenceSelect seqs={seqs} values={sels} onChange={setSels} getKey={seqKey} showCapture />
          )
        ) : (
          <div className="space-y-2">
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
            {customSeqs.length > 0 && (
              <SequenceSelect seqs={customSeqs} values={customSels} onChange={setCustomSels} getKey={seqKey} showCapture
                placeholder={`Select sequences… (${customSeqs.length} found)`} />
            )}
          </div>
        )}
      </div>

      {/* ③ Formats & options + run (shared) */}
      <div className={card}>
        <div className="text-sm font-medium text-foreground mb-3">Formats & options</div>
        <ExportPanel targets={targets} readiness={ready} onNeedTools={() => setSetupOpen(true)} />
      </div>
    </div>
  );
}
