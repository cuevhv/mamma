/*
 * ConfirmModal — a small, styled confirmation dialog that replaces
 * `window.confirm()` for destructive / consequential actions.
 *
 * Controlled: render it always and toggle `open`; it returns null when
 * closed. The backdrop, Esc, and the Cancel button all fire `onCancel`
 * (suppressed while `busy` so an in-flight action can't be dismissed
 * out from under itself). `onConfirm` runs the action — keep the modal
 * mounted and pass `busy` while it's awaiting so the primary button
 * shows a spinner; close by flipping `open` once it resolves.
 *
 * The body copy is arbitrary JSX via `children`, so callers can lay out
 * whatever context the decision needs.
 */
import { useEffect, useRef, type ReactNode } from 'react';
import { AlertTriangle, Loader2, X } from 'lucide-react';

type Tone = 'danger' | 'primary';

interface Props {
  open: boolean;
  /** Short headline; also the dialog's accessible name. */
  title: string;
  /** Body copy / context. Rendered muted below the title. */
  children?: ReactNode;
  confirmLabel?: string;
  cancelLabel?: string;
  /** 'danger' → red confirm button + warning badge (default).
   *  'primary' → the app's primary accent. */
  tone?: Tone;
  /** Overrides the header badge icon. Omit for the per-tone default
   *  (a warning triangle for danger, no badge for primary). */
  icon?: ReactNode;
  /** Action in flight — spins the confirm button and locks dismissal. */
  busy?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

export function ConfirmModal({
  open,
  title,
  children,
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  tone = 'danger',
  icon,
  busy = false,
  onConfirm,
  onCancel,
}: Props) {
  const confirmRef = useRef<HTMLButtonElement>(null);

  // Esc cancels, but never while an action is mid-flight.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !busy) onCancel();
    };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open, busy, onCancel]);

  // Focus the confirm button on open so Enter confirms — parity with the
  // native confirm() this replaces.
  useEffect(() => {
    if (open) confirmRef.current?.focus();
  }, [open]);

  if (!open) return null;

  const isDanger = tone === 'danger';
  const badge = icon ?? (isDanger ? <AlertTriangle className="w-4 h-4" /> : null);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/65 backdrop-blur-sm animate-overlay-in"
      onClick={() => { if (!busy) onCancel(); }}
      role="presentation"
    >
      <div
        role="alertdialog"
        aria-modal="true"
        aria-label={title}
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-md bg-surface-1 border border-border rounded-xl shadow-2xl shadow-black/50 ring-1 ring-inset ring-white/[0.03] overflow-hidden animate-modal-in"
      >
        {/* Header */}
        <div className="flex items-start gap-3 px-5 pt-5 pb-3">
          {badge && (
            <div
              className={
                'flex-shrink-0 flex items-center justify-center w-9 h-9 rounded-full ' +
                (isDanger ? 'bg-status-failed-bg text-status-failed' : 'bg-primary-muted text-primary')
              }
              aria-hidden
            >
              {badge}
            </div>
          )}
          <h2 className="min-w-0 flex-1 text-foreground text-[15px] font-medium tracking-tight leading-snug break-words pt-1">
            {title}
          </h2>
          <button
            type="button"
            onClick={() => { if (!busy) onCancel(); }}
            disabled={busy}
            aria-label="Cancel"
            className="flex-shrink-0 -mt-1 -mr-1 p-1.5 rounded-md text-foreground-muted hover:text-foreground hover:bg-surface-2 transition-colors disabled:opacity-40 disabled:pointer-events-none"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Body */}
        {children && (
          <div className="px-5 pb-4 text-[12.5px] text-foreground-muted leading-relaxed">
            {children}
          </div>
        )}

        {/* Footer */}
        <div className="flex justify-end gap-2 px-5 py-3.5 border-t border-border-subtle bg-surface-2/30">
          <button
            type="button"
            onClick={onCancel}
            disabled={busy}
            className="inline-flex items-center px-3 py-1.5 text-[12px] rounded-md border border-border text-foreground-muted hover:text-foreground hover:border-border-strong transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {cancelLabel}
          </button>
          <button
            ref={confirmRef}
            type="button"
            onClick={onConfirm}
            disabled={busy}
            className={
              'inline-flex items-center gap-1.5 px-3 py-1.5 text-[12px] font-medium rounded-md transition-colors disabled:opacity-60 disabled:cursor-not-allowed shadow-sm shadow-black/30 ' +
              (isDanger
                ? 'bg-status-failed/85 hover:bg-status-failed text-background'
                : 'bg-primary text-primary-foreground hover:opacity-90')
            }
          >
            {busy && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}

export default ConfirmModal;
