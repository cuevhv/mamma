import { useEffect, useRef, useState } from 'react';
import { Layers, X } from 'lucide-react';
import { toast } from 'sonner';

interface ConcurrencySetting {
  limit: number;
  mode: 'sequential' | 'parallel';
}

/**
 * Pill button + popover for the task-queue concurrency setting.
 *
 * Sits in the Tasks page header. Reflects today's mode (Sequential
 * default, or Parallel with a max-N picker) and PUTs the new value
 * to /api/settings/concurrency on save. The backend's coordinator
 * thread reads this value on every spawn decision, so a bump takes
 * effect for already-queued tasks within ~1 tick.
 *
 * Self-contained: no shared state, no context. Mounts once at the
 * top of Tasks.tsx, polls /api/settings/concurrency on open + after
 * save, otherwise renders from local state.
 */
export function RunModeBadge() {
  const [setting, setSetting] = useState<ConcurrencySetting | null>(null);
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const fetchSetting = () => {
    fetch('/api/settings/concurrency')
      .then(r => r.ok ? r.json() : null)
      .then((d: ConcurrencySetting | null) => { if (d) setSetting(d); })
      .catch(() => {});
  };

  useEffect(() => { fetchSetting(); }, []);

  // Click-outside to close the popover. Skip the click that opens
  // the popover itself (it bubbles to the document otherwise).
  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (!containerRef.current?.contains(e.target as Node)) setOpen(false);
    };
    // mousedown rather than click so a "click outside" closes BEFORE
    // any other click handler fires on the underlying element.
    document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, [open]);

  const mode = setting?.mode ?? 'sequential';
  const limit = setting?.limit ?? 1;
  const label = mode === 'sequential' ? 'Sequential' : `Parallel · max ${limit}`;
  const dotCls = mode === 'sequential' ? 'bg-status-completed' : 'bg-status-running';

  return (
    <div ref={containerRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen(v => !v)}
        className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border border-border bg-surface-2 hover:border-border-strong hover:bg-surface-3 text-foreground-muted hover:text-foreground transition-colors text-xs"
        title="Task run mode — how many tasks may run at once."
      >
        <Layers className="w-3.5 h-3.5" />
        <span>Tasks Run mode:</span>
        <span className="inline-flex items-center gap-1.5">
          <span className={`w-1.5 h-1.5 rounded-full ${dotCls}`} />
          <span className="text-foreground">{label}</span>
        </span>
      </button>

      {open && setting && (
        <RunModePopover
          setting={setting}
          onClose={() => setOpen(false)}
          onSaved={(next) => { setSetting(next); setOpen(false); }}
        />
      )}
    </div>
  );
}

function RunModePopover({
  setting, onClose, onSaved,
}: {
  setting: ConcurrencySetting;
  onClose: () => void;
  onSaved: (next: ConcurrencySetting) => void;
}) {
  const [mode, setMode] = useState<'sequential' | 'parallel'>(setting.mode);
  const [maxN, setMaxN] = useState<number>(Math.max(2, setting.limit));
  const [saving, setSaving] = useState(false);

  const save = async () => {
    if (saving) return;
    setSaving(true);
    const limit = mode === 'sequential' ? 1 : Math.max(2, Math.min(maxN, 8));
    try {
      const res = await fetch('/api/settings/concurrency', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ limit }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        toast.error(err.error || `Failed to update run mode (${res.status})`);
        return;
      }
      const next: ConcurrencySetting = await res.json();
      toast.success(`Run mode: ${next.mode === 'sequential' ? 'Sequential' : `Parallel (max ${next.limit})`}`);
      onSaved(next);
    } catch (e) {
      console.error(e);
      toast.error('Failed to reach the backend.');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div
      className="absolute right-0 mt-2 w-80 z-30 bg-surface-1 border border-border rounded-xl shadow-xl shadow-black/40 ring-1 ring-inset ring-white/[0.03] overflow-hidden"
      onClick={(e) => e.stopPropagation()}
    >
      <div className="flex items-center justify-between px-4 py-3 border-b border-border-subtle">
        <div className="text-foreground text-sm font-medium">Tasks Run mode</div>
        <button
          type="button"
          onClick={onClose}
          className="p-1 -mr-1 rounded-md text-foreground-muted hover:text-foreground hover:bg-surface-2"
          aria-label="Close"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      <div className="px-4 py-3 space-y-3 text-sm">
        <label className={`flex items-start gap-2.5 p-2.5 rounded-md border cursor-pointer transition-colors ${
          mode === 'sequential'
            ? 'border-primary/55 bg-primary-muted ring-1 ring-inset ring-white/10'
            : 'border-border bg-surface-2 hover:border-border-strong'
        }`}>
          <input
            type="radio"
            name="runmode"
            checked={mode === 'sequential'}
            onChange={() => setMode('sequential')}
            className="mt-0.5 accent-primary"
          />
          <div className="flex-1 min-w-0">
            <div className="text-foreground font-medium">Sequential <span className="text-foreground-subtle font-normal">(recommended)</span></div>
            <div className="text-foreground-muted text-xs mt-0.5 leading-relaxed">
              Runs one task at a time. Best for single-GPU machines — avoids VRAM
              contention and gives each task the full device.
            </div>
          </div>
        </label>

        <label className={`flex items-start gap-2.5 p-2.5 rounded-md border cursor-pointer transition-colors ${
          mode === 'parallel'
            ? 'border-primary/55 bg-primary-muted ring-1 ring-inset ring-white/10'
            : 'border-border bg-surface-2 hover:border-border-strong'
        }`}>
          <input
            type="radio"
            name="runmode"
            checked={mode === 'parallel'}
            onChange={() => setMode('parallel')}
            className="mt-0.5 accent-primary"
          />
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <div className="text-foreground font-medium">Parallel</div>
              <div className="flex items-center gap-1.5">
                <span className="text-foreground-subtle text-xs">max:</span>
                <select
                  value={Math.max(2, maxN)}
                  onChange={e => setMaxN(parseInt(e.target.value, 10))}
                  onClick={() => setMode('parallel')}
                  disabled={mode !== 'parallel'}
                  className="bg-surface-2 border border-border rounded px-1.5 py-0.5 text-foreground text-xs disabled:opacity-50 focus:outline-none focus:border-primary/60"
                >
                  {[2, 3, 4, 5, 6, 7, 8].map(n => (
                    <option key={n} value={n}>{n}</option>
                  ))}
                </select>
              </div>
            </div>
            <div className="text-foreground-muted text-xs mt-0.5 leading-relaxed">
              Runs up to N tasks concurrently on the same GPU. Each pipeline
              step uses several GB of VRAM, so values above 1 risk
              out-of-memory failures. Tasks beyond the limit queue.
            </div>
          </div>
        </label>
      </div>

      <div className="flex items-center justify-end gap-2 px-4 py-3 border-t border-border-subtle bg-surface-1/80">
        <button
          type="button"
          onClick={onClose}
          className="px-3 py-1.5 text-xs text-foreground-muted hover:text-foreground transition-colors"
        >
          Cancel
        </button>
        <button
          type="button"
          onClick={save}
          disabled={saving}
          className="px-3 py-1.5 bg-primary text-primary-foreground hover:opacity-90 rounded-md text-xs font-medium transition-opacity shadow-sm shadow-black/30 disabled:opacity-50 disabled:cursor-wait"
        >
          {saving ? 'Saving…' : 'Save'}
        </button>
      </div>
    </div>
  );
}
