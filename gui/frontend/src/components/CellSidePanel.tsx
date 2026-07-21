import { useEffect, useState } from 'react';
import { X, FileText, StopCircle, Folder, FileJson, Maximize2 } from 'lucide-react';
import { toast } from 'sonner';
import { StatusBadge, statusKind } from './shared/StatusBadge';
import { FileViewerModal } from './shared/FileViewerModal';
import { stepLabel } from './shared/stepLabels';
import { formatTaskId } from './shared/formatTaskId';
import { MatrixCell } from './ProcessTable';
import { MeshPreview, MeshViewer } from './MeshPreview';

interface Props {
  /** When null the panel is closed. */
  selection: {
    taskId: string;
    seqName: string;
    stepName: string;
    /** Capture context — required for cross-tab "Browse outputs" navigation
     *  when the panel is used in the Tasks tab. Optional in single-capture
     *  views where the parent handles output paths internally. */
    captureName?: string;
    captureJsonPath?: string;
    /** Output root + id of the run, used to locate its ma_3d results for the
     *  inline 3D viewer. Optional — the viewer is skipped when absent. */
    outputPath?: string | null;
    outputId?: string | null;
    cell: MatrixCell | null;
  } | null;
  onClose: () => void;
  /** Called when the user clicks Stop on a Running cell. */
  onStop?: (cell: MatrixCell) => void;
  /** Called when the user clicks "Browse outputs". When the parent is on the
   *  Tasks tab, this jumps to the Results detail page deep-linked into the
   *  cell's task/sequence/step. When omitted, the button is hidden. */
  onBrowseOutputs?: (
    captureName: string,
    captureJsonPath: string,
    taskId: string,
    seqName: string,
    stepName: string
  ) => void;
}

/**
 * Right-side slide-in panel showing one (task, sequence, step) cell's
 * details — status, log/out/err viewers, stop, reveal-output.
 */
export function CellSidePanel({ selection, onClose, onStop, onBrowseOutputs }: Props) {
  const [viewer, setViewer] = useState<{ name: string; path: string } | null>(null);
  /** ma_3d fitted-body target for the inline 3D viewer, or null when none exist. */
  const [viewer3dTarget, setViewer3dTarget] = useState<{ tag: string; capture: string; seq: string; ma_3d_dir: string; ma_cap_dir?: string } | null>(null);
  /** When set, the fullscreen 3D mesh preview modal is open for this target. */
  const [meshModal, setMeshModal] = useState<{ tag: string; capture: string; seq: string; ma_3d_dir: string; ma_cap_dir?: string } | null>(null);

  // Probe for this (task, sequence)'s ma_3d fitted SMPL-X params so the panel can
  // embed an inline 3D viewer when results exist. Mirrors CaptureDetail's probe.
  // Declared before the early return below so the hook order stays stable across
  // open/closed panel states; it bails internally when there's no selection.
  useEffect(() => {
    setViewer3dTarget(null);
    if (!selection) return;
    const outRoot = selection.outputPath || 'output';
    const outId = selection.outputId || selection.taskId;
    const dataset = selection.captureName || '';   // dataset defaults to the capture name in GUI runs
    if (!dataset) return;
    const seqName = selection.seqName;
    const controller = new AbortController();
    fetch(`/api/files/list?path=${encodeURIComponent(`${outRoot}/ma_3d/${outId}/${dataset}/${seqName}`)}`, { signal: controller.signal })
      .then(r => (r.ok ? r.json() : null))
      .then(d => {
        if (d && Array.isArray(d.files) && d.files.some((f: { name: string }) => f.name.startsWith('smplx_params_body_id-'))) {
          setViewer3dTarget({
            tag: String(outId), capture: dataset, seq: seqName,
            ma_3d_dir: `${outRoot}/ma_3d/${outId}/${dataset}`,
            ma_cap_dir: `${outRoot}/ma_cap/${outId}/${dataset}`,   // lets the backend read the recorded fps
          });
        }
      })
      .catch(() => {});
    return () => controller.abort();
    // selection.output*/captureName all move together with taskId/seqName.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selection?.taskId, selection?.seqName]);

  if (!selection) return null;

  const { taskId, seqName, stepName, captureName, captureJsonPath, cell } = selection;
  const canBrowseOutputs = !!onBrowseOutputs && !!captureName && !!captureJsonPath;
  const canStop = cell ? statusKind(cell.status) === 'Running' : false;
  const labelPrefix = `task${taskId}_${seqName}_${stepName}`;

  return (
    <>
      <div
        className="fixed inset-0 bg-black/55 backdrop-blur-sm z-30"
        onClick={onClose}
        aria-hidden
      />
      <aside
        className="fixed right-0 top-0 bottom-0 w-[28rem] max-w-[92vw] bg-surface-1 border-l border-border-subtle z-40 flex flex-col shadow-2xl shadow-black/60"
        role="complementary"
      >
        {/* Header */}
        <header className="flex items-start justify-between gap-3 p-5 border-b border-border-subtle bg-surface-2/40">
          <div className="min-w-0">
            <div className="flex items-center gap-2 text-[11px] uppercase tracking-wider mb-1.5">
              <span className="text-primary font-mono">{formatTaskId(taskId)}</span>
              <span className="text-foreground-faint">/</span>
              <span className="text-foreground-muted font-mono normal-case truncate">{seqName}</span>
            </div>
            <div className="text-foreground text-base font-medium leading-tight">
              {stepLabel(stepName)}
            </div>
            <div className="text-foreground-subtle text-xs font-mono mt-0.5">{stepName}</div>
          </div>
          <button
            onClick={onClose}
            className="flex-shrink-0 text-foreground-subtle hover:text-foreground p-1.5 -m-1.5 rounded-md hover:bg-surface-3/60 transition-colors"
            aria-label="Close panel"
          >
            <X className="w-5 h-5" />
          </button>
        </header>

        {/* Scrollable middle — status, logs, and the inline 3D viewer scroll as
            a unit so the header and footer actions stay pinned on short viewports. */}
        <div className="flex-1 min-h-0 overflow-y-auto">
        {/* Status */}
        <section className="p-5 border-b border-border-subtle space-y-3">
          {cell ? (
            <>
              <div className="flex items-center gap-3">
                <StatusBadge status={cell.status} />
              </div>
              {cell.pid && (
                <div className="text-xs text-foreground-subtle">
                  PID <span className="font-mono text-foreground tabular-nums">{cell.pid}</span>
                </div>
              )}
            </>
          ) : (
            <div className="text-foreground-subtle text-sm">
              This step hasn't been registered for this sequence yet.
            </div>
          )}
        </section>

        {/* Log files */}
        {cell && (
          <section className="p-5 border-b border-border-subtle">
            <div className="text-[11px] text-foreground-subtle uppercase tracking-wider font-medium mb-2.5">
              Log files
            </div>
            <div className="flex flex-col gap-2">
              {cell.outFile && (
                <FileButton
                  iconClass="text-status-completed"
                  label=".out (stdout)"
                  path={cell.outFile}
                  onClick={() => setViewer({ name: `${labelPrefix}.out`, path: cell.outFile! })}
                />
              )}
              {cell.errFile && (
                <FileButton
                  iconClass="text-status-failed"
                  label=".err (stderr)"
                  path={cell.errFile}
                  onClick={() => setViewer({ name: `${labelPrefix}.err`, path: cell.errFile! })}
                />
              )}
              {!cell.outFile && !cell.errFile && (
                <div className="text-foreground-subtle text-xs italic">
                  No log files yet — process likely hasn't started.
                </div>
              )}
            </div>
          </section>
        )}

        {/* 3D viewer — inline SMPL-X body preview, shown when ma_3d fitted
            params exist for this (task, sequence). Maximize opens the same
            fullscreen modal used elsewhere. */}
        {viewer3dTarget && (
          <section className="p-5 border-b border-border-subtle">
            <div className="flex items-center justify-between mb-2.5">
              <div className="text-[11px] text-foreground-subtle uppercase tracking-wider font-medium">3D viewer</div>
              <button onClick={() => setMeshModal(viewer3dTarget)} title="Open fullscreen"
                      className="p-1 -m-1 rounded-md text-foreground-subtle hover:text-foreground hover:bg-surface-3/60 transition-colors">
                <Maximize2 className="w-3.5 h-3.5" />
              </button>
            </div>
            <div className="h-[230px] rounded-md overflow-hidden border border-border-subtle">
              <MeshViewer key={`${viewer3dTarget.tag}::${viewer3dTarget.seq}`} target={viewer3dTarget} className="h-full" />
            </div>
          </section>
        )}
        </div>

        {/* Actions */}
        <section className="p-5 flex flex-col gap-2 border-t border-border-subtle bg-surface-2/30">
          <button
            onClick={async () => {
              try {
                const res = await fetch(`/api/tasks/${encodeURIComponent(taskId)}/config-path`);
                if (!res.ok) {
                  const err = await res.json().catch(() => ({}));
                  toast.error(err.error || `Failed to locate run config (${res.status})`);
                  return;
                }
                const data = await res.json();
                setViewer({ name: `run_${taskId}.json`, path: data.path });
              } catch (e) {
                console.error(e);
                toast.error('Failed to load run config. See console.');
              }
            }}
            className="flex items-center justify-center gap-2 px-4 py-2.5 bg-surface-3 hover:bg-surface-4 border border-border hover:border-border-strong rounded-md text-foreground text-sm transition-colors"
            title="View the frozen run_<id>.json that this run executed against"
          >
            <FileJson className="w-4 h-4 text-primary" />
            View run config
          </button>
          <button
            onClick={async () => {
              try {
                const res = await fetch(`/api/tasks/${encodeURIComponent(taskId)}/config-path`);
                if (!res.ok) {
                  const err = await res.json().catch(() => ({}));
                  toast.error(err.error || `Failed to locate preset (${res.status})`);
                  return;
                }
                const data = await res.json();
                if (!data.presetPath) {
                  toast.info('No preset lineage recorded for this run (likely submitted before preset tracking, or imported from CLI).');
                  return;
                }
                const presetName = (data.presetPath as string).split('/').pop() || 'preset';
                setViewer({ name: presetName, path: data.presetPath });
              } catch (e) {
                console.error(e);
                toast.error('Failed to load preset. See console.');
              }
            }}
            className="flex items-center justify-center gap-2 px-4 py-2.5 bg-surface-3 hover:bg-surface-4 border border-border hover:border-border-strong rounded-md text-foreground text-sm transition-colors"
            title="Open the source preset (capture-independent template) this run was built from"
          >
            <FileJson className="w-4 h-4 text-foreground-subtle" />
            View source preset
          </button>
          {canBrowseOutputs && (
            <button
              onClick={() => onBrowseOutputs!(captureName!, captureJsonPath!, taskId, seqName, stepName)}
              className="flex items-center justify-center gap-2 px-4 py-2.5 bg-surface-3 hover:bg-surface-4 border border-border hover:border-border-strong rounded-md text-foreground text-sm transition-colors"
              title="Open the Outputs explorer for this cell in the Results tab"
            >
              <Folder className="w-4 h-4 text-status-pending" />
              Browse outputs
            </button>
          )}
          {canStop && cell && (
            <button
              onClick={() => onStop?.(cell)}
              className="flex items-center justify-center gap-2 px-4 py-2.5 bg-destructive/90 hover:bg-destructive rounded-md text-destructive-foreground text-sm font-medium transition-colors shadow-sm shadow-black/30"
            >
              <StopCircle className="w-4 h-4" />
              Stop this step
            </button>
          )}
        </section>
      </aside>

      <FileViewerModal file={viewer} onClose={() => setViewer(null)} />
      {meshModal && <MeshPreview target={meshModal} onClose={() => setMeshModal(null)} />}
    </>
  );
}

function FileButton({
  iconClass, label, path, onClick,
}: {
  iconClass: string;
  label: string;
  path: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="group flex items-start gap-2.5 p-2.5 bg-surface-2 hover:bg-surface-3 border border-border-subtle hover:border-border rounded-md text-left transition-colors"
    >
      <FileText className={`w-4 h-4 mt-0.5 flex-shrink-0 ${iconClass}`} />
      <div className="flex-1 min-w-0">
        <div className="text-foreground text-sm">{label}</div>
        <div className="text-foreground-subtle text-[11px] font-mono truncate">{path}</div>
      </div>
    </button>
  );
}
