import { useEffect, useMemo, useState } from 'react';
import { Plus, ArrowLeft, Activity, Upload } from 'lucide-react';
import { toast } from 'sonner';
import { ProcessTable, MatrixCell, ALL_STEPS, buildProcessRows } from './ProcessTable';
import { PipelineOverview } from './PipelineOverview';
import { CellSidePanel } from './CellSidePanel';
import { NewTaskForm } from './NewTaskForm';
import { RunModeBadge } from './RunModeBadge';
import { ImportPreviousRunModal } from './ImportPreviousRunModal';
import { useTaskPolling } from './shared/useTaskPolling';
import { formatTaskId } from './shared/formatTaskId';

interface HistoryProcess {
  processId: string;
  processType: string;
  status: string;
  pid?: string | null;
  outFile?: string | null;
  errFile?: string | null;
  /** ISO-8601 UTC instant the step transitioned to Running / to a terminal
   *  state. Null when the step never ran (cached/skipped) or for legacy rows
   *  recorded before the timing columns existed. */
  startedAt?: string | null;
  endedAt?: string | null;
}
interface HistorySequence { seqName: string; processes: HistoryProcess[]; numFrames?: number | null; numCameras?: number | null; }
interface HistoryTask {
  taskId: string;
  captureName: string;
  captureJsonPath?: string;
  /** Path of the preset the GUI used to materialize this run, when
   *  recorded. Null/undefined for legacy rows and CLI imports. */
  presetPath?: string | null;
  /** 1-indexed position in the task-coordinator's queue. Present only
   *  for tasks whose runner hasn't been spawned yet. Backend computes
   *  this from the in-memory queue and decorates the history response. */
  queuePosition?: number;
  username: string;
  createdAt: string;
  sequences: HistorySequence[];
}
interface ActiveTask {
  taskId: string;
  captureName: string;
  captureJsonPath?: string;
  username: string;
  createdAt: string;
  runnerPid?: string | null;
  processes: Array<HistoryProcess & { sequenceName: string }>;
}

interface Props {
  /** Called after a task is successfully submitted, with its id. */
  onSubmitted?: (taskId: number) => void;
  /** Which sub-view to render on first mount. 'list' is the default
   *  cross-capture matrix; 'submit' lands directly on NewTaskForm so
   *  the Home "Submit a task" CTA can deep-link into the form without
   *  the user hunting for the New-task button. Read once at mount;
   *  changes after mount are ignored (the user can navigate within
   *  the tab freely). */
  initialSubView?: 'list' | 'submit';
  /** Cross-tab jump from a side-panel cell to the Results detail page,
   *  deep-linked into the cell's (task, sequence, step). */
  onBrowseOutputs?: (
    captureName: string,
    captureJsonPath: string,
    taskId: string,
    seqName: string,
    stepName: string
  ) => void;
}

/**
 * The Tasks tab. Two sub-views, both lightweight:
 *  - "list": the cross-capture monitor (ProcessTable).
 *  - "submit": the NewTaskForm. Submitting flips back to list.
 *
 * The matrix here is unscoped — every task across every capture appears.
 * Users can narrow with the table's Capture / Status / Sequence filters.
 */
export function Tasks({ onSubmitted, onBrowseOutputs, initialSubView }: Props) {
  const [subView, setSubView] = useState<'list' | 'submit'>(initialSubView ?? 'list');
  const [importOpen, setImportOpen] = useState(false);

  // Poll fast only while a run is actually in flight. When idle we drop the
  // /active poll entirely and let /history tick slowly — enough to notice a run
  // started from the CLI or a status change made elsewhere, without hammering the
  // dev server every 2s when nothing is running. `anyActive` (recomputed below
  // from the latest polls) drives the cadence; a run appearing ramps it back up.
  const [anyActive, setAnyActive] = useState(false);
  // /active: 2s while running, off when idle (the merge below just no-ops then).
  const { data: activeData } = useTaskPolling<ActiveTask[]>('/api/processes/active', { intervalMs: anyActive ? 2000 : 0 });
  // /history catches queue/completion transitions the /active overlay misses
  // (queued tasks aren't "active" yet; completed tasks aren't any more; quick
  // steps can finish between two /active polls). 4s while running; a slow 15s
  // heartbeat when idle so a newly-started run is still noticed and ramps polling.
  const { data: historyData, refresh: refreshHistory } = useTaskPolling<HistoryTask[]>('/api/tasks/history', { intervalMs: anyActive ? 4000 : 15000 });

  // Is anything in flight? An active /active payload, or an in-flight status in
  // history, both count. Drives the adaptive cadence above.
  useEffect(() => {
    const inFlight = new Set(['running', 'queued', 'pending', 'retrying', 'cancelling', 'starting']);
    const active =
      (activeData?.length ?? 0) > 0 ||
      (historyData ?? []).some(t => t.sequences.some(s => s.processes.some(p => inFlight.has((p.status || '').toLowerCase()))));
    setAnyActive(active);
  }, [activeData, historyData]);

  // Merge live tasks on top of history when a task is currently active so its cells
  // reflect the latest status without waiting for a refresh.
  const allRuns = useMemo<HistoryTask[]>(() => {
    const fromHistory = historyData ?? [];
    if (!activeData) return fromHistory;
    const liveByTaskId = new Map<string, ActiveTask>();
    for (const t of activeData) liveByTaskId.set(t.taskId, t);
    return fromHistory.map(h => {
      const live = liveByTaskId.get(h.taskId);
      if (!live) return h;
      const seqMap: Record<string, HistoryProcess[]> = {};
      for (const p of live.processes) {
        (seqMap[p.sequenceName] ||= []).push({
          processId: p.processId,
          processType: p.processType,
          status: p.status,
          pid: p.pid,
          outFile: p.outFile,
          errFile: p.errFile,
          startedAt: p.startedAt,
          endedAt: p.endedAt,
        });
      }
      // Preserve the per-sequence frame/camera counts from history — the live
      // payload doesn't carry them, so rebuilding from it alone would blank
      // them out for a task that's still running after ma_cap finished.
      const countsBySeq = new Map(h.sequences.map(s => [s.seqName, s]));
      return {
        ...h,
        sequences: Object.entries(seqMap).map(([seqName, processes]) => ({
          seqName,
          processes,
          numFrames: countsBySeq.get(seqName)?.numFrames ?? null,
          numCameras: countsBySeq.get(seqName)?.numCameras ?? null,
        })),
      };
    });
  }, [historyData, activeData]);

  // Flatten into per-(task, seq) rows for the table; carry captureName +
  // captureJsonPath so the dropdown filter can key on the precise source.
  const tableData = useMemo(() => {
    const flat: Array<{
      taskId: string; seqName: string; captureName: string; captureJsonPath?: string;
      presetPath?: string | null;
      queuePosition?: number;
      createdAt?: string;
      processType: string; processId: string; status: string;
      pid?: string | null; outFile?: string | null; errFile?: string | null;
      startedAt?: string | null; endedAt?: string | null;
      numFrames?: number | null; numCameras?: number | null;
    }> = [];
    for (const run of allRuns) {
      for (const seq of run.sequences) {
        for (const p of seq.processes) {
          flat.push({
            taskId: run.taskId,
            seqName: seq.seqName,
            captureName: run.captureName,
            captureJsonPath: run.captureJsonPath,
            presetPath: run.presetPath ?? null,
            queuePosition: run.queuePosition,
            createdAt: run.createdAt,
            processType: p.processType,
            processId: p.processId,
            status: p.status,
            pid: p.pid,
            outFile: p.outFile,
            errFile: p.errFile,
            startedAt: p.startedAt,
            endedAt: p.endedAt,
            numFrames: seq.numFrames,
            numCameras: seq.numCameras,
          });
        }
      }
    }
    const { rows, stepsInUse } = buildProcessRows(flat);
    const orderedSteps = ALL_STEPS.filter(s => stepsInUse.has(s));
    return { rows, steps: orderedSteps };
  }, [allRuns]);

  // Side panel selection (a single cell click opens the panel).
  // Capture context is carried alongside so the side panel can offer
  // cross-tab navigation to the Results outputs explorer.
  const [selectedCell, setSelectedCell] = useState<{
    taskId: string;
    seqName: string;
    stepName: string;
    captureName?: string;
    captureJsonPath?: string;
    cell: MatrixCell | null;
  } | null>(null);

  const handleStopCell = async (cell: MatrixCell) => {
    try { await fetch(`/api/processes/${cell.processId}/stop`, { method: 'POST' }); }
    catch (e) { console.error('stop failed', e); }
  };

  // Stop a whole task (kills the runner subprocess + cancels remaining
  // processes server-side). Used by the row-level Stop button. A task's
  // sequences all run in ONE process, so Stop is whole-task by nature —
  // when the run has more than one sequence we confirm first so it's clear
  // this cancels every sequence, not just the row that was clicked.
  const handleStopTask = async (taskId: string) => {
    const seqNames = tableData.rows.filter(r => r.taskId === taskId).map(r => r.seqName);
    if (seqNames.length > 1) {
      const ok = window.confirm(
        `Stop task ${formatTaskId(taskId)}?\n\n` +
        `This run's ${seqNames.length} sequences all execute in a single ` +
        `process, so stopping it cancels ALL of them:\n` +
        `  • ${seqNames.join('\n  • ')}\n\n` +
        `Finished steps keep their outputs (DONE sentinels), so a later ` +
        `Restart resumes from where it left off.`
      );
      if (!ok) return;
    }
    try {
      const res = await fetch(`/api/tasks/${encodeURIComponent(taskId)}/stop`, { method: 'POST' });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        toast.error(err.error || `Failed to stop task ${formatTaskId(taskId)} (${res.status})`);
        return;
      }
      toast.success(`Stopped task ${formatTaskId(taskId)}`);
      refreshHistory();
    } catch (e) {
      console.error(e);
      toast.error(`Failed to reach the backend.`);
    }
  };

  /** Re-enqueue a stopped task. The runner reuses the existing
   *  run_<id>.json + DONE sentinels, so already-completed (step, seq)
   *  pairs are skipped — execution effectively continues from where
   *  the previous run halted. */
  const handleRestartTask = async (taskId: string) => {
    try {
      const res = await fetch(`/api/tasks/${encodeURIComponent(taskId)}/restart`, { method: 'POST' });
      const d = await res.json().catch(() => ({}));
      if (!res.ok) {
        toast.error(d.error || `Failed to restart task ${formatTaskId(taskId)} (${res.status})`);
        return;
      }
      if (d.resetCount === 0) {
        toast.info(d.message || `Task ${formatTaskId(taskId)} is already fully completed.`);
      } else {
        toast.success(`Restarted task ${formatTaskId(taskId)}`);
        setAnyActive(true);  // re-enqueued work — ramp polling up now
      }
      refreshHistory();
    } catch (e) {
      console.error(e);
      toast.error(`Failed to reach the backend.`);
    }
  };

  /** Remove a single sequence from a task. Each Tasks-table row is one
   *  (task, sequence), so delete is scoped to that sequence — the other
   *  sequences in the same task are untouched. If it was the task's last
   *  sequence, the empty task row disappears too. **Does not delete output
   *  files on disk** — we surface that caveat so re-submitting against the
   *  same `output_id` still skips finished work via DONE sentinels. */
  const handleDeleteSequence = async (taskId: string, seqName: string) => {
    const ok = window.confirm(
      `Remove sequence '${seqName}' from task ${formatTaskId(taskId)}?\n\n` +
      `Only this sequence is removed from the Tasks table — other sequences ` +
      `in the same task stay. If it's the task's last sequence, the task row ` +
      `goes too.\n\n` +
      `Output files on disk are NOT deleted: logs and outputs (under ` +
      `output/<step>/<output_id>/...) stay where they are, and re-submitting ` +
      `against the same Output ID will still skip finished steps via DONE ` +
      `sentinels. If you want to wipe the files too, delete them on disk afterwards.`
    );
    if (!ok) return;
    try {
      const res = await fetch(
        `/api/tasks/${encodeURIComponent(taskId)}/sequence?name=${encodeURIComponent(seqName)}`,
        { method: 'DELETE' },
      );
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        toast.error(err.error || `Failed to remove '${seqName}' from task ${formatTaskId(taskId)} (${res.status})`);
        return;
      }
      const d = await res.json().catch(() => ({} as { taskRemoved?: boolean }));
      toast.success(d.taskRemoved
        ? `Removed sequence '${seqName}' — task ${formatTaskId(taskId)} had no sequences left, so it's gone too.`
        : `Removed sequence '${seqName}' from task ${formatTaskId(taskId)} (files on disk untouched).`);
      // Close the side panel if it was pointing at the deleted (task, sequence).
      setSelectedCell(prev => (prev?.taskId === taskId && prev?.seqName === seqName ? null : prev));
      refreshHistory();
    } catch (e) {
      console.error(e);
      toast.error(`Failed to reach the backend.`);
    }
  };

  if (subView === 'submit') {
    return (
      <div className="px-6 py-8">
        <div className="max-w-3xl mx-auto mb-3">
          <button
            onClick={() => setSubView('list')}
            className="inline-flex items-center gap-1.5 text-foreground-muted hover:text-foreground text-sm transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to tasks
          </button>
        </div>
        <NewTaskForm
          onSubmitted={(taskId) => {
            setSubView('list');
            setAnyActive(true);  // ramp polling up now, don't wait for the idle heartbeat
            refreshHistory();
            onSubmitted?.(taskId);
          }}
        />
      </div>
    );
  }

  return (
    <div className="px-6 py-8">
      <div className="max-w-[1600px] mx-auto">
        <div className="mb-6">
          <PipelineOverview />
        </div>
        <div className="flex items-end justify-between gap-4 mb-6 flex-wrap">
          <div>
            <h2 className="text-3xl text-foreground tracking-tight font-medium mb-1 flex items-center gap-2">
              <Activity className="w-6 h-6 text-primary" />
              Tasks
            </h2>
            <p className="text-foreground-muted text-sm">
              Submitted runs. Click a cell to inspect that step.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <div className="text-foreground-subtle text-xs flex items-center gap-1.5">
              <span className="inline-block w-1.5 h-1.5 rounded-full bg-status-completed animate-pulse" />
              Live · refreshes every 2s
            </div>
            <RunModeBadge />
            <button
              onClick={() => setImportOpen(true)}
              title="Register pipeline runs that completed outside the GUI (e.g. via the terminal)."
              className="inline-flex items-center gap-1.5 px-3.5 py-2 bg-surface-2 border border-border text-foreground-muted hover:text-foreground hover:border-border-strong rounded-md text-sm font-medium transition-colors"
            >
              <Upload className="w-4 h-4" />
              Import previous runs
            </button>
            <button
              onClick={() => setSubView('submit')}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 bg-primary text-primary-foreground hover:opacity-90 rounded-md text-sm font-medium transition-opacity shadow-sm shadow-black/30"
            >
              <Plus className="w-4 h-4" />
              New task
            </button>
          </div>
        </div>
        <ImportPreviousRunModal
          open={importOpen}
          onClose={() => setImportOpen(false)}
          onMutated={refreshHistory}
        />

        <ProcessTable
          rows={tableData.rows}
          steps={tableData.steps}
          onCellClick={(row, stepName, cell) =>
            setSelectedCell({
              taskId: row.taskId,
              seqName: row.seqName,
              stepName,
              captureName: row.captureName,
              captureJsonPath: row.captureJsonPath,
              cell,
            })
          }
          selected={selectedCell ? {
            taskId: selectedCell.taskId, seqName: selectedCell.seqName, stepName: selectedCell.stepName,
          } : null}
          onBrowseOutputs={onBrowseOutputs}
          onStopTask={handleStopTask}
          onRestartTask={handleRestartTask}
          onDeleteSequence={handleDeleteSequence}
        />
      </div>

      <CellSidePanel
        selection={selectedCell}
        onClose={() => setSelectedCell(null)}
        onStop={handleStopCell}
        onBrowseOutputs={onBrowseOutputs}
      />
    </div>
  );
}
