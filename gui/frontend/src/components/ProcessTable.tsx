import { useEffect, useMemo, useState } from 'react';
import { Search, X, Filter, Layers, Square, Trash2, RotateCcw, AlertTriangle, Clock, ListChecks } from 'lucide-react';
import { StatusBadge, statusKind, StatusKind, statusStyle, Dot } from './shared/StatusBadge';
import { stepLabel } from './shared/stepLabels';
import { formatTaskId } from './shared/formatTaskId';

export interface MatrixCell {
  processId: string;
  status: string;
  pid?: string | null;
  outFile?: string | null;
  errFile?: string | null;
  /** ISO-8601 UTC of the Running / terminal transition. Null for a
   *  cached/skipped step or a legacy row without timing. Drives the per-step
   *  duration shown under the status badge. */
  startedAt?: string | null;
  endedAt?: string | null;
}

export interface ProcessRow {
  taskId: string;
  seqName: string;
  /** Display name of the capture this run belongs to. Optional: present
   *  when the table renders rows from multiple captures (e.g. the Tasks
   *  page); absent for single-capture views. */
  captureName?: string;
  /** Absolute path to the capture.json this run was submitted against.
   *  Drives the dropdown filter below — path is more reliably unique than
   *  the (often-derived) display name. */
  captureJsonPath?: string;
  /** Source preset path the GUI used to materialize this run config, when
   *  recorded. Null for legacy rows (submitted before the preset_path
   *  column existed) and for CLI imports. Surfaced as a small badge so
   *  users can answer "which preset did I run on this capture?" without
   *  opening the side panel. */
  presetPath?: string | null;
  /** 1-indexed position in the task coordinator's queue. Present only
   *  for tasks whose runner hasn't been spawned yet (all cells are in
   *  the Queued process status). Surfaced as "Queued · #N" next to
   *  the row's status badge so the user sees their place in line. */
  queuePosition?: number;
  createdAt?: string;
  cells: Record<string, MatrixCell | undefined>;
}

/** Canonical body-branch step ordering. Matches backend ALL_STEPS. */
export const ALL_STEPS = ['ma_cap', 'ma_masks', 'ma_2d', 'ma_3d', 'ma_vis'];

/** Execution time of one step, in seconds, derived from the Running/terminal
 *  timestamps the runner stamps (measured outside the step subprocess, so it
 *  costs the pipeline nothing). Returns null when the step never started
 *  (cached/skipped, or a legacy row) so callers render "—" rather than 0. For
 *  an in-progress step (started, not ended) it measures up to `nowMs`, giving a
 *  live-ticking elapsed that advances on each poll. */
export function cellDurationSecs(cell?: MatrixCell, nowMs?: number): number | null {
  if (!cell?.startedAt) return null;
  const start = Date.parse(cell.startedAt);
  if (Number.isNaN(start)) return null;
  const end = cell.endedAt ? Date.parse(cell.endedAt) : (nowMs ?? Date.now());
  if (Number.isNaN(end)) return null;
  return Math.max(0, (end - start) / 1000);
}

/** Sum of all timed steps in a sequence row → that sequence's total execution
 *  time. Steps run sequentially in the DAG, so the sum is the wall time. Null
 *  when no step in the row has timing yet. */
export function rowDurationSecs(cells: ProcessRow['cells'], nowMs?: number): number | null {
  let total = 0, any = false;
  for (const k of Object.keys(cells)) {
    const d = cellDurationSecs(cells[k], nowMs);
    if (d != null) { total += d; any = true; }
  }
  return any ? total : null;
}

/** Compact human duration: "8s", "2m 45s", "1h 03m". */
export function formatDurationSecs(secs: number): string {
  const s = Math.round(secs);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) { const r = s % 60; return r ? `${m}m ${r}s` : `${m}m`; }
  const h = Math.floor(m / 60); const mm = m % 60;
  return mm ? `${h}h ${String(mm).padStart(2, '0')}m` : `${h}h`;
}

/** Row-level rollup. Mixed = at least one Completed AND at least one Failed/Pending. */
export type RowStatus = StatusKind | 'Mixed';

export function rowRollupStatus(cells: ProcessRow['cells']): RowStatus {
  let any = false, running = false, failed = false, completed = false, pending = false, queued = false;
  for (const k of Object.keys(cells)) {
    const c = cells[k];
    if (!c) continue;
    any = true;
    const s = statusKind(c.status);
    if (s === 'Running') running = true;
    else if (s === 'Failed') failed = true;
    else if (s === 'Completed') completed = true;
    else if (s === 'Queued') queued = true;
    else pending = true;
  }
  if (!any) return 'Pending';
  if (running) return 'Running';
  if (failed && completed) return 'Mixed';
  if (failed) return 'Failed';
  if (completed && !pending && !queued) return 'Completed';
  // Only collapse to "Queued" when every cell is queued — a Queued
  // task is one the coordinator hasn't spawned yet, so a mixed
  // queued+anything state shouldn't exist in practice. If it does,
  // fall through to plain Pending.
  if (queued && !running && !failed && !completed && !pending) return 'Queued';
  return 'Pending';
}

const STATUS_ORDER: RowStatus[] = ['Running', 'Queued', 'Failed', 'Mixed', 'Pending', 'Completed'];

function rowTokens(s: RowStatus): { pill: string; dot: string } {
  if (s === 'Mixed') return { pill: 'bg-status-mixed-bg border-status-mixed/35 text-status-mixed', dot: 'bg-status-mixed' };
  const k = statusStyle(s as StatusKind);
  return { pill: k.pill, dot: k.dot };
}

/** Status icon — same icon as the per-cell StatusBadge, plus an
 *  AlertTriangle for the row-level "Mixed" rollup (which has no
 *  per-cell equivalent). */
function rowIcon(s: RowStatus): typeof AlertTriangle {
  if (s === 'Mixed') return AlertTriangle;
  return statusStyle(s as StatusKind).icon;
}

interface Props {
  rows: ProcessRow[];
  /** Column ordering — typically `ALL_STEPS` filtered to in-use steps. */
  steps: string[];
  onCellClick?: (row: ProcessRow, stepName: string, cell: MatrixCell | null) => void;
  selected?: { taskId: string; seqName: string; stepName: string } | null;
  /** Jump to the Results page scoped to this row. Wired the same way the
   *  side panel's "Browse outputs" button is: passes
   *  (captureName, captureJsonPath, taskId, seqName, stepName) and lets the
   *  host route to CaptureResults. Pass to make the capture and sequence
   *  cells clickable; stepName is empty for these row-level jumps so the
   *  results page opens unscoped to a step. */
  onBrowseOutputs?: (
    captureName: string,
    captureJsonPath: string,
    taskId: string,
    seqName: string,
    stepName: string,
  ) => void;
  /** Cancel an in-flight task (kills the runner subprocess, marks
   *  remaining processes Cancelled). Pass to enable the Stop button in
   *  the Actions column. */
  onStopTask?: (taskId: string) => void | Promise<void>;
  /** Re-enqueue a stopped task. Resumes from where it left off — the
   *  runner's DONE-sentinel skip handles already-completed (step, seq)
   *  pairs. Pass to enable the Restart button. */
  onRestartTask?: (taskId: string) => void | Promise<void>;
  /** Remove a task from the DB. **Does not delete output files on disk.**
   *  Pass to enable the Delete button in the Actions column. */
  onDeleteTask?: (taskId: string) => void | Promise<void>;
}

/**
 * Per-(task, sequence) status table. Each row is one (taskId, seqName)
 * tuple; columns are pipeline steps. Multi-task history is visible at a
 * glance — e.g., "horse_01 succeeded in #11 but ma_3d failed in #10".
 *
 * Filters: status pills (multi-select with live counts), free-text search
 * across task # and sequence, and a "Latest run only" toggle that
 * collapses to one row per sequence (the most recent task for it).
 */
export function ProcessTable({ rows, steps, onCellClick, selected, onBrowseOutputs, onStopTask, onRestartTask, onDeleteTask }: Props) {
  const showActions = !!(onStopTask || onRestartTask || onDeleteTask);
  const [statusFilter, setStatusFilter] = useState<Set<RowStatus>>(new Set());
  /** Selected capture filter, keyed by capture JSON path (more specific than
   *  the display name in case two captures share a name). Empty = no filter. */
  const [capturePath, setCapturePath] = useState<string>('');
  const [search, setSearch] = useState('');
  const [latestOnly, setLatestOnly] = useState(false);
  // Status ⇄ Timing view. 'status' shows the badge grid; 'timing' swaps every
  // step cell for its execution time and the rollup column for the per-sequence
  // total. Persisted so the choice sticks across visits.
  const [view, setView] = useState<'status' | 'timing'>(() => {
    try { return localStorage.getItem('mamma.taskTableView') === 'timing' ? 'timing' : 'status'; }
    catch { return 'status'; }
  });
  useEffect(() => {
    try { localStorage.setItem('mamma.taskTableView', view); } catch { /* ignore */ }
  }, [view]);

  // Modal state for "peek at task config / preset" actions. Set by the
  // task-id and preset chip click handlers; cleared on close.
  const [peek, setPeek] = useState<{ title: string; path: string } | null>(null);

  const openPeek = async (taskId: string, kind: 'task' | 'preset') => {
    try {
      const res = await fetch(`/api/tasks/${encodeURIComponent(taskId)}/config-path`);
      if (!res.ok) return;
      const d = await res.json() as { path?: string; presetPath?: string | null };
      const path = kind === 'task' ? d.path : d.presetPath;
      if (!path) return;
      const file = path.split('/').pop() ?? path;
      setPeek({
        title: kind === 'task'
          ? `Task ${formatTaskId(taskId)} — ${file}`
          : `Preset — ${file}`,
        path,
      });
    } catch { /* swallow; user can retry */ }
  };

  // Show the Capture column + dropdown filter only when rows actually carry
  // capture identity AND span more than one capture; otherwise the filter is
  // visual noise.
  const showCapture = useMemo(
    () => rows.some(r => !!r.captureName) && new Set(rows.map(r => r.captureName).filter(Boolean)).size > 1,
    [rows]
  );

  // Distinct (path, name) pairs in row order — newest task first.
  const captureOptions = useMemo(() => {
    const seen = new Map<string, { path: string; name: string; count: number }>();
    for (const r of rows) {
      const path = r.captureJsonPath || r.captureName;
      if (!path) continue;
      const existing = seen.get(path);
      if (existing) {
        existing.count++;
      } else {
        seen.set(path, { path, name: r.captureName ?? path, count: 1 });
      }
    }
    return Array.from(seen.values()).sort((a, b) => a.name.localeCompare(b.name));
  }, [rows]);

  const annotated = useMemo(
    () => rows.map(r => ({ row: r, status: rowRollupStatus(r.cells) })),
    [rows]
  );

  const statusCounts = useMemo(() => {
    const c: Record<RowStatus, number> = { Running: 0, Failed: 0, Mixed: 0, Pending: 0, Completed: 0 };
    for (const a of annotated) c[a.status]++;
    return c;
  }, [annotated]);

  const filtered = useMemo(() => {
    let r = annotated;
    if (statusFilter.size > 0) r = r.filter(a => statusFilter.has(a.status));
    if (capturePath) {
      r = r.filter(a => (a.row.captureJsonPath || a.row.captureName) === capturePath);
    }
    const q = search.trim().toLowerCase();
    if (q) {
      r = r.filter(a =>
        a.row.seqName.toLowerCase().includes(q) ||
        a.row.taskId.toLowerCase().includes(q) ||
        `#${a.row.taskId}`.toLowerCase().includes(q) ||
        formatTaskId(a.row.taskId).toLowerCase().includes(q) ||
        (a.row.captureName?.toLowerCase().includes(q) ?? false) ||
        (a.row.captureJsonPath?.toLowerCase().includes(q) ?? false)
      );
    }
    if (latestOnly) {
      const best = new Map<string, typeof r[number]>();
      for (const a of r) {
        const key = a.row.captureName ? `${a.row.captureName}::${a.row.seqName}` : a.row.seqName;
        const cur = best.get(key);
        if (!cur || Number(a.row.taskId) > Number(cur.row.taskId)) best.set(key, a);
      }
      r = Array.from(best.values());
    }
    return [...r].sort((a, b) => {
      const t = Number(b.row.taskId) - Number(a.row.taskId);
      if (t !== 0) return t;
      const c = (a.row.captureName ?? '').localeCompare(b.row.captureName ?? '');
      if (c !== 0) return c;
      return a.row.seqName.localeCompare(b.row.seqName);
    });
  }, [annotated, statusFilter, capturePath, search, latestOnly]);

  const toggleStatus = (s: RowStatus) => {
    setStatusFilter(prev => {
      const next = new Set(prev);
      if (next.has(s)) next.delete(s); else next.add(s);
      return next;
    });
  };
  const anyFilter = statusFilter.size > 0 || capturePath !== '' || search !== '' || latestOnly;
  const clearAll = () => { setStatusFilter(new Set()); setCapturePath(''); setSearch(''); setLatestOnly(false); };

  return (
    <div className="bg-surface-1 border border-border-subtle rounded-xl overflow-hidden shadow-sm shadow-black/30 ring-1 ring-inset ring-white/[0.02]">
      {/* Filter bar */}
      <div className="px-4 py-3 border-b border-border-subtle bg-surface-2/40 flex flex-wrap items-center gap-x-2 gap-y-2">
        <div className="flex items-center gap-1.5 text-foreground-subtle text-[11px] font-medium uppercase tracking-wider mr-1">
          <Filter className="w-3.5 h-3.5" />
          Status
        </div>
        {STATUS_ORDER.map(s => (
          <FilterPill
            key={s}
            status={s}
            count={statusCounts[s]}
            active={statusFilter.has(s)}
            onClick={() => toggleStatus(s)}
          />
        ))}

        {showCapture && (
          <>
            <div className="w-px h-5 bg-border mx-1.5" />
            <div className="flex items-center gap-1.5 text-foreground-subtle text-[11px] font-medium uppercase tracking-wider mr-1">
              <Filter className="w-3.5 h-3.5" />
              Capture
            </div>
            <select
              value={capturePath}
              onChange={(e) => setCapturePath(e.target.value)}
              className="bg-surface-2 border border-border rounded-md px-2 py-1 text-foreground text-xs focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition-colors max-w-[18rem]"
              title={capturePath || 'All captures'}
            >
              <option value="">All ({rows.length})</option>
              {captureOptions.map(c => (
                // Use the backend-derived captureName (folder name when the
                // file is generically named "capture.json"); the raw file
                // basename is "capture" for every capture so it can't tell
                // them apart.
                <option key={c.path} value={c.path} className="bg-surface-2" title={c.path}>
                  {c.name} ({c.count})
                </option>
              ))}
            </select>
          </>
        )}

        <div className="w-px h-5 bg-border mx-1.5" />

        <label className="inline-flex items-center gap-2 text-foreground-muted text-xs cursor-pointer select-none hover:text-foreground transition-colors">
          <input
            type="checkbox"
            checked={latestOnly}
            onChange={(e) => setLatestOnly(e.target.checked)}
            className="w-3.5 h-3.5 accent-primary"
          />
          <Layers className="w-3.5 h-3.5 text-foreground-subtle" />
          Latest run only
        </label>

        <div className="ml-auto flex items-center gap-2">
          {/* Status ⇄ Timing view toggle. Segmented control; the active
              segment is filled. Keeps both views uncluttered — see one signal
              per cell at a time instead of stacking status + duration. */}
          <div
            className="inline-flex items-center rounded-md border border-border bg-surface-2 p-0.5"
            role="radiogroup"
            aria-label="Table view"
          >
            <button
              type="button"
              role="radio"
              aria-checked={view === 'status'}
              onClick={() => setView('status')}
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                view === 'status' ? 'bg-primary text-primary-foreground' : 'text-foreground-muted hover:text-foreground'
              }`}
              title="Show step statuses"
            >
              <ListChecks className="w-3.5 h-3.5" /> Status
            </button>
            <button
              type="button"
              role="radio"
              aria-checked={view === 'timing'}
              onClick={() => setView('timing')}
              className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium transition-colors ${
                view === 'timing' ? 'bg-primary text-primary-foreground' : 'text-foreground-muted hover:text-foreground'
              }`}
              title="Show per-step execution times"
            >
              <Clock className="w-3.5 h-3.5" /> Timing
            </button>
          </div>
          <div className="relative">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-foreground-subtle pointer-events-none" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search task # or sequence…"
              className="bg-surface-2 border border-border rounded-md pl-7 pr-7 py-1.5 text-foreground text-xs focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition w-60 placeholder:text-foreground-faint"
            />
            {search && (
              <button
                onClick={() => setSearch('')}
                className="absolute right-1.5 top-1/2 -translate-y-1/2 text-foreground-subtle hover:text-foreground transition-colors"
                aria-label="Clear search"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
          {anyFilter && (
            <button
              onClick={clearAll}
              className="text-xs text-foreground-muted hover:text-foreground px-2 py-1 rounded-md border border-border hover:border-border-strong transition-colors"
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Table */}
      {rows.length === 0 ? (
        <div className="p-12 text-center text-foreground-subtle text-sm">No runs yet for this capture.</div>
      ) : filtered.length === 0 ? (
        <div className="p-12 text-center text-foreground-subtle text-sm">
          No rows match these filters.{' '}
          <button onClick={clearAll} className="text-primary hover:opacity-80 underline underline-offset-2 ml-1">
            Clear filters
          </button>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="bg-surface-2/60 border-b border-border-subtle">
                <Th sticky="left-0">Task</Th>
                {showCapture && <Th>Capture</Th>}
                <Th>Sequence</Th>
                {steps.map(step => (
                  <th key={step} className="px-4 py-3 text-left whitespace-nowrap">
                    <div className="text-foreground text-sm font-medium">{stepLabel(step)}</div>
                    <div className="text-foreground-faint text-[10px] font-mono mt-0.5 tracking-wide">{step}</div>
                  </th>
                ))}
                <Th>{view === 'timing' ? 'Total' : 'Status'}</Th>
                {showActions && <Th>Actions</Th>}
              </tr>
            </thead>
            <tbody>
              {filtered.map(({ row, status }, idx) => {
                const rowKey = `${row.taskId}::${row.seqName}`;
                const stripeBg = idx % 2 === 0 ? 'bg-surface-1' : 'bg-surface-1/60';
                return (
                  <tr key={rowKey} className={`group border-b border-border-subtle/60 ${stripeBg} hover:bg-surface-3/40 transition-colors`}>
                    <td className={`sticky left-0 ${stripeBg} group-hover:bg-surface-3/40 px-4 py-3 whitespace-nowrap z-10 transition-colors`}>
                      <button
                        type="button"
                        onClick={(e) => { e.stopPropagation(); openPeek(row.taskId, 'task'); }}
                        className="font-mono text-sm text-primary rounded-md px-1.5 py-0.5 -mx-1.5 mamma-cell-clickable"
                        title={`View task config JSON`}
                      >
                        {formatTaskId(row.taskId)}
                      </button>
                      {row.presetPath && (() => {
                        const stem = row.presetPath.split('/').pop()?.replace(/\.(json|ya?ml)$/i, '') ?? '';
                        return stem ? (
                          <button
                            type="button"
                            onClick={(e) => { e.stopPropagation(); openPeek(row.taskId, 'preset'); }}
                            className="ml-1.5 inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-mono bg-surface-3/70 text-foreground-subtle border border-border-subtle align-middle mamma-cell-clickable"
                            title={`View preset: ${row.presetPath}`}
                          >
                            {stem}
                          </button>
                        ) : null;
                      })()}
                    </td>
                    {showCapture && (
                      <td className="px-4 py-3 whitespace-nowrap">
                        {onBrowseOutputs && row.captureName && row.captureJsonPath ? (
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              onBrowseOutputs(row.captureName!, row.captureJsonPath!, row.taskId, row.seqName, '');
                            }}
                            className="text-sm text-foreground-muted rounded-md px-2 py-1 -mx-2 mamma-cell-clickable"
                            title={`Browse results for capture '${row.captureName}'`}
                          >
                            {row.captureName}
                          </button>
                        ) : (
                          <span className="text-sm text-foreground-muted">{row.captureName ?? '—'}</span>
                        )}
                      </td>
                    )}
                    <td className="px-4 py-3 whitespace-nowrap">
                      {onBrowseOutputs && row.captureJsonPath ? (
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            onBrowseOutputs(row.captureName ?? '', row.captureJsonPath!, row.taskId, row.seqName, '');
                          }}
                          className="font-mono text-sm text-foreground rounded-md px-2 py-1 -mx-2 mamma-cell-clickable"
                          title={`Browse results for sequence '${row.seqName}' on task ${formatTaskId(row.taskId)}`}
                        >
                          {row.seqName}
                        </button>
                      ) : (
                        <span className="font-mono text-sm text-foreground">{row.seqName}</span>
                      )}
                    </td>
                    {steps.map(step => {
                      const cell = row.cells[step] ?? null;
                      const isSelected = selected?.taskId === row.taskId && selected.seqName === row.seqName && selected.stepName === step;
                      // A real cell is clickable; a Skipped placeholder isn't
                      // worth a side panel. Hover affordance only on real
                      // cells — that's how we signal "click this badge to
                      // see *its* logs / outputs", not "click anywhere on
                      // the row."
                      const clickable = !!cell;
                      const baseCls = 'px-4 py-3 transition-colors';
                      const stateCls = isSelected
                        ? 'bg-primary-muted ring-1 ring-inset ring-primary/40 cursor-pointer'
                        : clickable
                          ? 'mamma-cell-clickable'
                          : 'cursor-default';
                      const secs = cell ? cellDurationSecs(cell) : null;
                      const isRunning = cell ? statusKind(cell.status) === 'Running' : false;
                      return (
                        <td
                          key={step}
                          className={`${baseCls} ${stateCls}`}
                          onClick={() => clickable && onCellClick?.(row, step, cell)}
                          title={clickable ? `Open logs and outputs for ${step} on ${row.seqName} (${formatTaskId(row.taskId)})` : undefined}
                        >
                          {view === 'timing' ? (
                            secs != null ? (
                              <span
                                className="inline-flex items-center gap-1 text-xs tabular-nums text-foreground"
                                title="Step execution time"
                              >
                                {formatDurationSecs(secs)}
                                {isRunning && <span className="text-amber-500 animate-pulse" title="running">⟳</span>}
                              </span>
                            ) : (
                              <span
                                className="text-xs text-foreground-faint"
                                title={cell ? 'No timing recorded (cached or legacy run)' : 'Step not selected for this task'}
                              >
                                —
                              </span>
                            )
                          ) : cell ? (
                            <StatusBadge status={cell.status} compact />
                          ) : (
                            <span
                              className="inline-flex items-center px-2 py-0.5 rounded-full text-xs border border-border-subtle text-foreground-faint italic whitespace-nowrap"
                              title="This step was not selected for this task"
                            >
                              Skipped
                            </span>
                          )}
                        </td>
                      );
                    })}
                    <td className="px-4 py-3">
                      {view === 'timing' ? (() => {
                        const secs = rowDurationSecs(row.cells);
                        return secs != null && secs > 0 ? (
                          <span
                            className="text-sm tabular-nums text-foreground font-medium"
                            title="Total execution time for this sequence (sum of its steps)"
                          >
                            {formatDurationSecs(secs)}
                          </span>
                        ) : (
                          <span className="text-sm text-foreground-faint">—</span>
                        );
                      })() : (
                        <RowStatusBadge status={status} queuePosition={row.queuePosition} />
                      )}
                    </td>
                    {showActions && (
                      <td className="px-4 py-3 whitespace-nowrap">
                        <div className="flex items-center gap-1">
                          {onStopTask && (
                            <button
                              onClick={(e) => { e.stopPropagation(); onStopTask(row.taskId); }}
                              disabled={status !== 'Running' && status !== 'Pending' && status !== 'Queued'}
                              aria-label="Stop task"
                              className="inline-flex items-center justify-center p-1.5 text-foreground-muted bg-surface-2 border border-border hover:border-status-failed/55 hover:text-status-failed rounded-md transition-colors disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:border-border disabled:hover:text-foreground-muted"
                              title={status === 'Queued'
                                ? `Drop task ${formatTaskId(row.taskId)} from the queue — it will be cancelled before its runner starts.`
                                : (status === 'Running' || status === 'Pending')
                                ? `Stop task ${formatTaskId(row.taskId)} — kills the runner and marks remaining steps as Cancelled.`
                                : `Task ${formatTaskId(row.taskId)} is not running.`}
                            >
                              <Square className="w-3.5 h-3.5" />
                            </button>
                          )}
                          {onRestartTask && (
                            <button
                              onClick={(e) => { e.stopPropagation(); onRestartTask(row.taskId); }}
                              disabled={status !== 'Failed' && status !== 'Mixed'}
                              aria-label="Restart task"
                              className="inline-flex items-center justify-center p-1.5 text-foreground-muted bg-surface-2 border border-border hover:border-primary/55 hover:text-primary rounded-md transition-colors disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:border-border disabled:hover:text-foreground-muted"
                              title={status === 'Failed' || status === 'Mixed'
                                ? `Restart task ${formatTaskId(row.taskId)} — re-queues the task; the runner's DONE-sentinel skip resumes from where it left off (already-completed steps are not re-run).`
                                : `Restart is only available for failed or partially-failed tasks.`}
                            >
                              <RotateCcw className="w-3.5 h-3.5" />
                            </button>
                          )}
                          {onDeleteTask && (
                            <button
                              onClick={(e) => { e.stopPropagation(); onDeleteTask(row.taskId); }}
                              aria-label="Delete task"
                              className="inline-flex items-center justify-center p-1.5 text-foreground-muted bg-surface-2 border border-border hover:border-status-failed/55 hover:text-status-failed rounded-md transition-colors"
                              title={`Remove task ${formatTaskId(row.taskId)} from the database. Output files on disk are NOT deleted — only the DB record disappears, so this row stops cluttering the table.`}
                            >
                              <Trash2 className="w-3.5 h-3.5" />
                            </button>
                          )}
                        </div>
                      </td>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      {peek && (
        <ConfigPeekModal
          title={peek.title}
          path={peek.path}
          onClose={() => setPeek(null)}
        />
      )}
    </div>
  );
}

function ConfigPeekModal({
  title, path, onClose,
}: {
  title: string;
  path: string;
  onClose: () => void;
}) {
  const [content, setContent] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setContent(null);
    setError(null);
    fetch(`/api/files/content?path=${encodeURIComponent(path)}`)
      .then(r => r.ok ? r.json() : Promise.reject(r.statusText))
      .then((d: { content: string }) => { if (!cancelled) setContent(d.content); })
      .catch(e => { if (!cancelled) setError(String(e)); });
    return () => { cancelled = true; };
  }, [path]);

  // Pretty-print JSON when we recognise it; fall back to raw text otherwise.
  const display = useMemo(() => {
    if (content == null) return null;
    try {
      return JSON.stringify(JSON.parse(content), null, 2);
    } catch {
      return content;
    }
  }, [content]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/65 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-3xl max-h-[85vh] flex flex-col bg-surface-1 border border-border rounded-xl shadow-2xl shadow-black/50 ring-1 ring-inset ring-white/[0.03]"
      >
        <div className="flex items-start justify-between px-5 py-3 border-b border-border-subtle gap-3">
          <div className="min-w-0">
            <div className="text-foreground text-sm font-medium truncate">{title}</div>
            <div className="text-foreground-faint text-[11px] font-mono truncate mt-0.5" title={path}>
              {path}
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="p-1.5 -mt-0.5 -mr-1 rounded-md text-foreground-muted hover:text-foreground hover:bg-surface-2 transition-colors flex-shrink-0"
            aria-label="Close"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
        <div className="flex-1 overflow-auto p-4">
          {error && (
            <div className="text-status-failed text-sm">Failed to load: {error}</div>
          )}
          {!error && display == null && (
            <div className="text-foreground-muted text-sm">Loading…</div>
          )}
          {!error && display != null && (
            <pre className="text-foreground text-xs font-mono whitespace-pre leading-relaxed">
              {display}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}

function Th({ children, sticky }: { children: React.ReactNode; sticky?: string }) {
  const stickyCls = sticky ? `sticky ${sticky} bg-surface-2` : '';
  return (
    <th className={`${stickyCls} px-4 py-3 text-left text-foreground-muted text-[11px] font-semibold uppercase tracking-wider z-10`}>
      {children}
    </th>
  );
}

function FilterPill({ status, count, active, onClick }: { status: RowStatus; count: number; active: boolean; onClick: () => void }) {
  const tok = rowTokens(status);
  const disabled = count === 0 && !active;
  const base = 'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs border transition-all';
  let cls: string;
  if (disabled) {
    cls = `${base} bg-transparent border-border-subtle text-foreground-faint cursor-not-allowed`;
  } else if (active) {
    cls = `${base} ${tok.pill} ring-1 ring-inset ring-white/15 shadow-sm`;
  } else {
    cls = `${base} bg-surface-2 border-border text-foreground-muted hover:border-border-strong hover:bg-surface-3 hover:text-foreground`;
  }
  return (
    <button onClick={onClick} disabled={disabled} className={cls} title={`${status} runs`}>
      <Dot className={tok.dot} pulsing={active && status === 'Running'} />
      <span className="font-medium">{status}</span>
      <span className={`font-mono text-[11px] tabular-nums ${active ? 'opacity-80' : 'opacity-70'}`}>{count}</span>
    </button>
  );
}

function RowStatusBadge({ status, queuePosition }: { status: RowStatus; queuePosition?: number }) {
  const tok = rowTokens(status);
  const Icon = rowIcon(status);
  const label = status === 'Queued' && queuePosition
    ? `Queued · #${queuePosition}`
    : status;
  const isRunning = status === 'Running';
  return (
    <span
      title={label}
      aria-label={label}
      className={`inline-flex items-center gap-1 px-1.5 py-1 rounded-full border ${tok.pill} whitespace-nowrap`}
    >
      <Icon className={`w-3.5 h-3.5 ${isRunning ? 'animate-pulse' : ''}`} />
      {status === 'Queued' && queuePosition !== undefined && (
        <span className="text-[11px] font-mono pr-0.5 tabular-nums">#{queuePosition}</span>
      )}
    </span>
  );
}

/**
 * Pivots flat process rows from /api/tasks/history (or live polling)
 * into per-(task, sequence) ProcessRow shape the table consumes.
 */
export function buildProcessRows<T extends {
  taskId: string;
  seqName: string;
  captureName?: string;
  captureJsonPath?: string;
  presetPath?: string | null;
  queuePosition?: number;
  createdAt?: string;
  processType: string;
  processId: string;
  status: string;
  pid?: string | null;
  outFile?: string | null;
  errFile?: string | null;
  startedAt?: string | null;
  endedAt?: string | null;
}>(records: T[]): { rows: ProcessRow[]; stepsInUse: Set<string> } {
  const byKey = new Map<string, ProcessRow>();
  const stepsInUse = new Set<string>();
  for (const r of records) {
    const key = `${r.taskId}::${r.seqName}`;
    let row = byKey.get(key);
    if (!row) {
      row = {
        taskId: r.taskId,
        seqName: r.seqName,
        captureName: r.captureName,
        captureJsonPath: r.captureJsonPath,
        presetPath: r.presetPath ?? null,
        queuePosition: r.queuePosition,
        createdAt: r.createdAt,
        cells: {},
      };
      byKey.set(key, row);
    }
    row.cells[r.processType] = {
      processId: r.processId,
      status: r.status,
      pid: r.pid ?? null,
      outFile: r.outFile ?? null,
      errFile: r.errFile ?? null,
      startedAt: r.startedAt ?? null,
      endedAt: r.endedAt ?? null,
    };
    stepsInUse.add(r.processType);
  }
  return { rows: Array.from(byKey.values()), stepsInUse };
}
