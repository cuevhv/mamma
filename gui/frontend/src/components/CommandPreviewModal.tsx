/*
 * Read-only modal showing the EXACT shell command the runner will execute for
 * each enabled pipeline step, resolved by the backend
 * (POST /api/tasks/preview-commands) via the same code path a real run uses.
 * A confirmation that the user's flags/overrides landed correctly.
 */
import { forwardRef, useEffect, useRef, useState } from 'react';
import { X, Loader2, AlertCircle, Terminal, Check, Copy } from 'lucide-react';
import { stepLabel } from './shared/stepLabels';

export interface StepCommand {
  command?: string;
  cwd?: string | null;
  engine?: string;
  error?: string;
}

export interface CommandPreview {
  commands: Record<string, StepCommand>;
  seqName?: string;
  outputIdPlaceholder?: string | null;
}

const STEP_ORDER = ['ma_cap', 'ma_masks', 'ma_2d', 'ma_3d', 'ma_vis'];

export function CommandPreviewModal({
  open, loading, error, data, focusStep, onClose,
}: {
  open: boolean;
  loading: boolean;
  error: string | null;
  data: CommandPreview | null;
  focusStep: string | null;
  onClose: () => void;
}) {
  const focusRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  // Scroll the clicked step into view once its command is rendered.
  useEffect(() => {
    if (open && data && focusRef.current) {
      focusRef.current.scrollIntoView({ block: 'nearest' });
    }
  }, [open, data, focusStep]);

  if (!open) return null;

  const steps = data
    ? STEP_ORDER.filter((s) => s in data.commands)
    : [];

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/65 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-4xl max-h-[85vh] bg-surface-1 border border-border rounded-xl shadow-2xl shadow-black/50 ring-1 ring-inset ring-white/[0.03] overflow-hidden flex flex-col"
      >
        <div className="flex items-start justify-between gap-3 px-5 py-3.5 border-b border-border-subtle">
          <div className="min-w-0">
            <div className="text-foreground text-base font-medium flex items-center gap-2">
              <Terminal className="w-4 h-4 text-foreground-faint" />
              Commands to run
            </div>
            {data?.seqName && (
              <div className="text-foreground-subtle text-[11px] mt-0.5">
                The exact commands the runner will execute, for sequence{' '}
                <span className="font-mono text-foreground-muted">{data.seqName}</span>.
                {data.outputIdPlaceholder && (
                  <> Output id shown as{' '}
                    <code className="font-mono text-foreground-muted">{data.outputIdPlaceholder}</code>
                    {' '}until the task is created.</>
                )}
              </div>
            )}
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="flex-shrink-0 p-1.5 rounded-md text-foreground-muted hover:text-foreground hover:bg-surface-2 transition-colors"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="flex-1 overflow-auto px-4 py-3 space-y-3">
          {loading && (
            <div className="text-foreground-subtle text-xs px-1 py-4 flex items-center gap-2">
              <Loader2 className="w-3.5 h-3.5 animate-spin" /> Resolving commands…
            </div>
          )}
          {error && (
            <div className="p-2.5 text-status-failed text-xs bg-status-failed-bg border border-status-failed/35 rounded flex items-start gap-2">
              <AlertCircle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
              <div>
                <div className="font-medium mb-0.5">Couldn't resolve commands</div>
                <div className="text-foreground-muted break-words">{error}</div>
              </div>
            </div>
          )}
          {data && steps.map((s) => (
            <StepCommandBlock
              key={s}
              ref={s === focusStep ? focusRef : undefined}
              stepName={s}
              cmd={data.commands[s]}
              focused={s === focusStep}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

const StepCommandBlock = forwardRef<HTMLDivElement, {
  stepName: string;
  cmd: StepCommand;
  focused: boolean;
}>(function StepCommandBlock({ stepName, cmd, focused }, ref) {
  const [copied, setCopied] = useState(false);
  const copy = () => {
    if (!cmd.command) return;
    navigator.clipboard?.writeText(cmd.command).then(() => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    }).catch(() => {});
  };

  return (
    <div
      ref={ref}
      className={`rounded-lg border ${focused ? 'border-primary/50 ring-1 ring-primary/30' : 'border-border-subtle'} bg-surface-2/40 overflow-hidden`}
    >
      <div className="flex items-center gap-2 px-3 py-2 border-b border-border-subtle">
        <span className="text-foreground text-[13px] font-medium">{stepLabel(stepName)}</span>
        <span className="text-foreground-subtle font-mono text-[11px]">{stepName}</span>
        {cmd.engine && (
          <span className="text-foreground-faint text-[10px] uppercase tracking-wide border border-border-subtle rounded px-1.5 py-0.5">
            {cmd.engine}
          </span>
        )}
        {cmd.command && (
          <button
            type="button"
            onClick={copy}
            className="ml-auto inline-flex items-center gap-1 text-[10.5px] text-foreground-subtle hover:text-primary transition-colors"
            title="Copy command"
          >
            {copied ? <Check className="w-3 h-3 text-status-completed" /> : <Copy className="w-3 h-3" />}
            {copied ? 'Copied' : 'Copy'}
          </button>
        )}
      </div>
      {cmd.error ? (
        <div className="px-3 py-2 text-[11px] text-status-failed flex items-start gap-2">
          <AlertCircle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
          <span className="break-words">{cmd.error}</span>
        </div>
      ) : (
        <>
          {cmd.cwd && (
            <div className="px-3 pt-2 text-[10.5px] text-foreground-faint font-mono truncate" title={cmd.cwd}>
              cwd: {cmd.cwd}
            </div>
          )}
          <pre className="px-3 py-2 text-[11px] text-foreground-muted whitespace-pre-wrap break-words font-mono leading-relaxed">
            {cmd.command}
          </pre>
        </>
      )}
    </div>
  );
});
