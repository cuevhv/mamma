import { useState, useRef, useEffect } from 'react';
import { ChevronDown, Check, Box } from 'lucide-react';
import { fmtRun, fmtWhen } from './ExportPanel';

export interface SeqItem {
  tag: string; capture: string; seq: string; people: number;
  ma_3d_dir: string; already_exported: boolean; mtime?: number;
}

/** Styled single-select for export sequences. Native <option> can't show rich
 *  text, so this is a light custom listbox: a prominent name, a muted metadata
 *  line (people · run · time), and an "exported" chip. Same value/onChange
 *  contract as the <select> it replaces; click-outside + Escape to close. */
export function SequenceSelect<T extends SeqItem>({
  seqs, value, onChange, getKey, showCapture = false, placeholder = 'Select a sequence…',
}: {
  seqs: T[];
  value: string;
  onChange: (v: string) => void;
  getKey: (s: T) => string;
  showCapture?: boolean;
  placeholder?: string;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onDoc);
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('mousedown', onDoc); document.removeEventListener('keydown', onKey); };
  }, []);

  const name = (s: SeqItem) => (showCapture ? `${s.capture} / ${s.seq}` : s.seq);
  const meta = (s: SeqItem) =>
    `${s.people} ${s.people === 1 ? 'person' : 'people'} · run ${fmtRun(s.tag)}${s.mtime ? ` · ${fmtWhen(s.mtime)}` : ''}`;
  const sel = seqs.find(s => getKey(s) === value) ?? null;

  // Shared row body — icon + (name over muted metadata) + "exported" chip.
  // Reused in the trigger (selected item) and the list so both show full info.
  const body = (s: T) => (
    <>
      <Box className="w-4 h-4 text-foreground-subtle flex-shrink-0" />
      <div className="min-w-0 flex-1 text-left">
        <div className="text-foreground text-sm font-medium truncate">{name(s)}</div>
        <div className="text-foreground-muted text-xs font-mono truncate">{meta(s)}</div>
      </div>
      {s.already_exported && (
        <span className="flex-shrink-0 text-[10px] font-medium text-emerald-400 bg-emerald-400/10 border border-emerald-400/25 rounded px-1.5 py-0.5">
          exported
        </span>
      )}
    </>
  );

  return (
    <div className="relative" ref={ref}>
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        aria-haspopup="listbox"
        aria-expanded={open}
        className="w-full bg-surface-2 border border-border rounded-md px-3 py-2 flex items-center gap-2.5 hover:border-border-strong focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition-colors"
      >
        {sel ? body(sel) : <span className="flex-1 text-left text-sm text-foreground-faint">{placeholder}</span>}
        <ChevronDown className={`w-4 h-4 text-foreground-subtle flex-shrink-0 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && (
        <ul role="listbox" className="absolute z-30 mt-1 w-full bg-surface-2 border border-border rounded-md shadow-2xl shadow-black/60 max-h-72 overflow-y-auto p-1">
          {seqs.length === 0 && <li className="px-3 py-2 text-foreground-faint text-sm">No sequences</li>}
          {seqs.map(s => {
            const k = getKey(s);
            const isSel = k === value;
            return (
              <li
                key={k}
                role="option"
                aria-selected={isSel}
                onClick={() => { onChange(k); setOpen(false); }}
                className={`flex items-center gap-2.5 rounded px-2.5 py-2 cursor-pointer select-none ${isSel ? 'bg-primary-muted' : 'hover:bg-surface-3'}`}
              >
                {body(s)}
                {isSel && <Check className="w-4 h-4 text-primary flex-shrink-0" />}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
