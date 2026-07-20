import { HelpCircle } from 'lucide-react';
import type { ReactNode } from 'react';
import { CalibrationFormatDocs } from './CalibrationFormatDocs';

/**
 * Shared "new capture" field primitives + the footage/calibration hints.
 *
 * Used by BOTH capture-creation entry points so their guidance and the
 * (?) hint toggle stay identical:
 *   - New Task → Step 1 → "Create capture" (NewTaskForm's CreateCapturePanel)
 *   - Captures tab → "New capture" (InlineCaptureJsonForm)
 */

// ─── Field row with an inline hint icon next to the label ─────────────
//
// Left-rail label + a small (?) button that toggles an inline hint card
// under the input. Used for the two path inputs (footage root, calibration)
// so each gets its own focused, dismissable "what goes here?" disclosure.
export function CreateField({
  label, hintOpen, onHintToggle, hint, children,
}: {
  label: string;
  hintOpen: boolean;
  onHintToggle: () => void;
  hint: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="flex items-start gap-4">
      <div className="w-32 flex-shrink-0 pt-2.5 flex items-center justify-end gap-1.5">
        <label className="text-foreground-muted text-xs uppercase tracking-wider font-medium">
          {label}
        </label>
        <button
          type="button"
          onClick={onHintToggle}
          title={hintOpen ? 'Hide hint' : 'What goes here?'}
          aria-expanded={hintOpen}
          className={
            'text-foreground-faint hover:text-foreground transition-colors ' +
            (hintOpen ? 'text-primary' : '')
          }
        >
          <HelpCircle className="w-3.5 h-3.5" />
        </button>
      </div>
      <div className="flex-1 min-w-0">
        {children}
        {hintOpen && (
          <div className="mt-2 rounded-md border border-border-subtle bg-surface-2/40 p-3 text-[11px] text-foreground-muted leading-relaxed">
            {hint}
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Plain label + input row (no hint) ────────────────────────────────
export function FieldRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex items-start gap-4">
      <label className="text-foreground-muted text-xs uppercase tracking-wider font-medium w-32 text-right pt-2.5 flex-shrink-0">{label}</label>
      <div className="flex-1 min-w-0">{children}</div>
    </div>
  );
}

export function FootageHint() {
  return (
    <div className="space-y-2">
      <div className="text-foreground-faint uppercase tracking-[0.14em] text-[10px]">
        Expected layout — one of:
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <div>
          <div className="text-foreground uppercase tracking-[0.14em] text-[10px] mb-1">Videos</div>
          <pre className="font-mono text-foreground-subtle leading-relaxed whitespace-pre text-[10.5px]">{`<footage_root>/
  <seq>/
    videos/
      <cam>.mp4`}</pre>
        </div>
        <div>
          <div className="text-foreground uppercase tracking-[0.14em] text-[10px] mb-1">Images</div>
          <pre className="font-mono text-foreground-subtle leading-relaxed whitespace-pre text-[10.5px]">{`<footage_root>/
  <seq>/
    <cam>/
      *.jpg/png`}</pre>
        </div>
      </div>
      <p className="text-foreground-faint">
        Each <span className="font-mono">&lt;seq&gt;</span> is a single recording.
        Camera names are taken from the file basenames (videos layout) or
        the subdirectory names (images layout). The footage layout is
        auto-detected — the badge above the input tells you which one
        was found.
      </p>
    </div>
  );
}

export function CalibrationHint() {
  return (
    <div className="space-y-2">
      <p className="text-foreground-faint">
        Multi-camera calibration (intrinsics and extrinsics). Camera names must
        match those detected in the footage root.
      </p>
      <CalibrationFormatDocs />
      <p className="text-foreground-faint">
        A wrong extrinsics convention won't error — it silently produces bad 3D. Use{' '}
        <strong>Preview camera rig</strong> to confirm cameras ring the subject, looking inward.
      </p>
    </div>
  );
}
