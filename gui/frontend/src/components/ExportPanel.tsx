import { useEffect, useState } from 'react';
import { Check, Loader2, AlertTriangle, Box, FolderOpen, Wrench } from 'lucide-react';

/** Shared SMPL-X export core — formats, options, run, and live status — reused by
 *  the Exporter tab and the inline export on a result. The caller supplies the
 *  target sequence; tool readiness is passed in (tab) or fetched here (results). */

export interface Readiness { blender: { present: boolean; path: string; version?: string; compat?: string }; addon: { present: boolean; path: string }; }
export interface ExportTarget { tag: string; capture: string; seq: string; ma_3d_dir: string; ma_cap_dir?: string; people?: number; }
export interface SeqResult { seq: string; capture: string; tag: string; ok: boolean; outputs: string[]; error: string | null; }
export interface Job {
  id: string; kind: string; state: 'running' | 'ready' | 'error';
  log_tail: string[]; outputs: string[]; error: string | null;
  progress?: { done: number; total: number; current: string | null } | null;
  results?: SeqResult[];
}

export const ALL_FORMATS = [
  { id: 'npz', label: 'npz', hint: 'SMPL-X Blender Add-on', blender: false },
  { id: 'fbx', label: 'FBX', hint: 'Rigged mesh + skeleton', blender: true },
  { id: 'abc', label: 'Alembic', hint: 'Vertex/geometry cache', blender: true },
  { id: 'bvh', label: 'BVH', hint: 'Skeleton motion only', blender: true },
  { id: 'usd', label: 'USD', hint: 'USD scene', blender: true },
];

export async function jget<T>(url: string): Promise<T> { const r = await fetch(url); return r.json(); }
export async function jpost<T>(url: string, body?: unknown): Promise<T> {
  const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : undefined });
  return r.json();
}

/** Run/output id padded to >=4 digits for a stable, sortable look (numeric tags only). */
export function fmtRun(tag: string): string {
  return /^\d+$/.test(tag) ? tag.padStart(4, '0') : tag;
}
/** Compact "when produced" hint from an epoch-seconds mtime; '' when missing. */
export function fmtWhen(mtime?: number): string {
  if (!mtime) return '';
  return new Date(mtime * 1000).toLocaleString(undefined,
    { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

export function ExportPanel({ targets, readiness, onNeedTools }: {
  /** Sequences to export — one runs a single export, many run a sequential batch. */
  targets: ExportTarget[];
  /** Pass readiness to share one source of truth (tab); omit to let the panel fetch it (results). */
  readiness?: Readiness | null;
  /** Called when the user needs to set up Blender (e.g. navigate to the Exporter tab). */
  onNeedTools?: () => void;
}) {
  const [ownReady, setOwnReady] = useState<Readiness | null>(null);
  const ready = readiness !== undefined ? readiness : ownReady;
  useEffect(() => { if (readiness === undefined) jget<Readiness>('/api/exporter/readiness').then(setOwnReady); }, [readiness]);
  const toolsReady = !!ready?.blender.present && !!ready?.addon.present;

  const [formats, setFormats] = useState<Record<string, boolean>>({ npz: true, fbx: false, abc: false, bvh: false, usd: false });
  const [ground, setGround] = useState(true);
  const [unit, setUnit] = useState('m');
  const [blenderFormat, setBlenderFormat] = useState('auto');
  const [fps, setFps] = useState('');
  const [outputDir, setOutputDir] = useState('');
  const [job, setJob] = useState<Job | null>(null);

  // Reset the job when the selection changes so stale results don't linger.
  const targetsKey = targets.map(t => `${t.tag}/${t.capture}/${t.seq}`).join('|');
  useEffect(() => { setJob(null); }, [targetsKey]);

  useEffect(() => {
    if (!job || job.state !== 'running') return;
    const t = setInterval(async () => setJob(await jget<Job>(`/api/exporter/job/${job.id}`)), 1500);
    return () => clearInterval(t);
  }, [job]);

  const chosen = ALL_FORMATS.filter(f => formats[f.id]).map(f => f.id);
  const needsBlender = chosen.some(f => ALL_FORMATS.find(x => x.id === f)?.blender);
  const canExport = targets.length > 0 && chosen.length > 0 && (!needsBlender || toolsReady) && job?.state !== 'running';

  const runExport = async () => {
    if (targets.length === 0) return;
    const r = await jpost<{ job_id: string }>('/api/exporter/export', {
      sequences: targets.map(t => ({
        tag: t.tag, capture: t.capture, seq: t.seq, ma_3d_dir: t.ma_3d_dir, ma_cap_dir: t.ma_cap_dir,
      })),
      formats: chosen, ground, unit, blender_format: blenderFormat, fps: fps ? Number(fps) : undefined,
      output_dir: outputDir.trim() || undefined,
    });
    setJob({ id: r.job_id, state: 'running', log_tail: [], outputs: [], error: null, kind: 'export' });
  };

  return (
    <div className="space-y-3">
      {/* formats */}
      <div className="flex flex-wrap gap-2">
        {ALL_FORMATS.map(f => {
          const disabled = f.blender && !toolsReady;
          const on = formats[f.id];
          return (
            <button key={f.id} disabled={disabled} title={disabled ? 'Needs Blender — set up the export tools' : f.hint}
              onClick={() => setFormats(p => ({ ...p, [f.id]: !p[f.id] }))}
              className={`inline-flex flex-col items-start px-3 py-1.5 rounded-md border text-xs transition-colors ${on ? 'bg-primary-muted border-primary/40 text-foreground' : 'bg-surface-2 border-border text-foreground-muted'} ${disabled ? 'opacity-40 cursor-not-allowed' : 'hover:border-primary/40'}`}>
              <span className="font-medium">{on ? '✓ ' : ''}{f.label}</span>
              <span className="text-[10px] text-foreground-faint">{disabled ? 'needs Blender' : f.hint}</span>
            </button>
          );
        })}
      </div>

      {/* options */}
      <div className="flex flex-wrap items-center gap-4 text-xs text-foreground-muted">
        <label className="flex items-center gap-1.5 cursor-pointer" title="Drop the feet to the floor (0 along the detected up-axis). The fit's axes are never changed.">
          <input type="checkbox" checked={ground} onChange={e => setGround(e.target.checked)} className="accent-primary" />
          Place on floor
        </label>
        <label className="flex items-center gap-1.5" title="Units for FBX/ABC/USD/BVH. Meters for Blender/Unity/Maya; centimeters for Unreal. The npz stays in meters.">Unit
          <select value={unit} onChange={e => setUnit(e.target.value)} className="bg-surface-2 border border-border rounded px-1.5 py-0.5 text-foreground">
            <option value="m">meters</option><option value="cm">centimeters</option>
          </select>
        </label>
        <label className="flex items-center gap-1.5" title="Prepares the npz so the add-on's Add Animation imports it upright with this Format. Auto keeps your data's axes and tells you which Format to pick.">Blender import
          <select value={blenderFormat} onChange={e => setBlenderFormat(e.target.value)} className="bg-surface-2 border border-border rounded px-1.5 py-0.5 text-foreground">
            <option value="auto">Auto (keep data axes)</option>
            <option value="amass">AMASS (Z-up)</option>
            <option value="smplx">SMPL-X (Y-up)</option>
          </select>
        </label>
        <label className="flex items-center gap-1.5">FPS
          <input value={fps} onChange={e => setFps(e.target.value)} placeholder="auto" className="w-16 bg-surface-2 border border-border rounded px-1.5 py-0.5 text-foreground" />
        </label>
      </div>

      {/* optional output folder */}
      <label className="flex items-center gap-2 text-xs text-foreground-muted"
        title="Where to write the exported files. Leave empty for the default output/export/. A custom folder keeps the same <run>/<capture>/<sequence>/ structure underneath.">
        <span className="shrink-0">Output folder</span>
        <input value={outputDir} onChange={e => setOutputDir(e.target.value)}
          placeholder="output/export  (default)"
          className="flex-1 min-w-0 bg-surface-2 border border-border rounded px-2 py-0.5 text-foreground font-mono" />
      </label>

      {/* export + status */}
      <div>
        <div className="flex items-center gap-3">
          <button onClick={runExport} disabled={!canExport}
            className="inline-flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-md text-sm font-medium disabled:opacity-40 disabled:cursor-not-allowed">
            <Box className="w-4 h-4" /> Export{targets.length > 1 ? ` ${targets.length}` : ''}
          </button>
          {targets.length === 0 && <span className="text-foreground-faint text-xs">choose at least one sequence</span>}
          {targets.length > 0 && needsBlender && !toolsReady && (
            onNeedTools
              ? <button onClick={onNeedTools} className="inline-flex items-center gap-1.5 text-status-pending text-xs hover:underline"><Wrench className="w-3.5 h-3.5" /> set up Blender for FBX/ABC/BVH/USD</button>
              : <span className="text-status-pending text-xs">set up the export tools for the Blender formats</span>
          )}
        </div>
        {ready?.blender.present && ready.blender.compat === 'too_old' && (
          <p className="mt-2 text-xs text-status-failed">
            <AlertTriangle className="w-3.5 h-3.5 inline mr-1 -mt-0.5" />
            Detected Blender {ready.blender.version} is too old — the SMPL-X add-on needs Blender 4.5+. Download the portable Blender.
          </p>
        )}
        {job && (
          <div className="mt-3 text-sm">
            {job.state === 'running' && (
              <span className="inline-flex items-center gap-2 text-status-running">
                <Loader2 className="w-4 h-4 animate-spin" />
                {job.progress && job.progress.total > 1
                  ? `exporting ${Math.min(job.progress.done + 1, job.progress.total)}/${job.progress.total}${job.progress.current ? ` — ${job.progress.current}` : ''}…`
                  : 'exporting…'}
              </span>
            )}
            {(job.state === 'ready' || job.state === 'error') && (() => {
              const res = job.results ?? [];
              const multi = res.length > 1;
              return (
                <div>
                  {job.state === 'error'
                    ? <span className="inline-flex items-center gap-2 text-status-failed"><AlertTriangle className="w-4 h-4" /> {job.error}</span>
                    : <span className="inline-flex items-center gap-2 text-status-completed"><Check className="w-4 h-4" /> {multi ? `exported ${res.filter(r => r.ok).length}/${res.length} sequences · ${job.outputs.length} file(s)` : `wrote ${job.outputs.length} file(s)`}</span>}
                  {multi ? (
                    <ul className="mt-2 space-y-0.5">
                      {res.map(r => (
                        <li key={`${r.tag}/${r.capture}/${r.seq}`} className={`flex items-center gap-1.5 text-xs ${r.ok ? 'text-foreground-faint' : 'text-status-failed'}`}>
                          {r.ok ? <Check className="w-3 h-3 flex-shrink-0" /> : <AlertTriangle className="w-3 h-3 flex-shrink-0" />}
                          <span className="truncate">{r.capture}/{r.seq} — {r.ok ? `${r.outputs.length} file(s)` : (r.error || 'failed')}</span>
                        </li>
                      ))}
                    </ul>
                  ) : job.outputs.length > 0 ? (
                    <ul className="mt-2 space-y-0.5">
                      {job.outputs.map(o => (
                        <li key={o} className="flex items-center gap-1.5 text-foreground-faint text-xs font-mono"><FolderOpen className="w-3 h-3 flex-shrink-0" /> {o}</li>
                      ))}
                    </ul>
                  ) : null}
                </div>
              );
            })()}
            {job.state === 'running' && job.log_tail.length > 0 && (
              <pre className="mt-2 text-[11px] text-foreground-faint font-mono max-h-32 overflow-auto whitespace-pre-wrap">{job.log_tail.slice(-8).join('\n')}</pre>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
