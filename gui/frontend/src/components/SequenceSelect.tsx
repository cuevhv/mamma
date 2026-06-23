import { useState, useRef, useEffect } from 'react';
import { ChevronDown, Box } from 'lucide-react';
import { fmtRun, fmtWhen } from './ExportPanel';

export interface SeqItem {
  tag: string; capture: string; seq: string; people: number;
  ma_3d_dir: string; already_exported: boolean; mtime?: number;
}

/** Styled MULTI-select for export sequences. Native <option> can't show rich text
 *  or multi-select cleanly, so this is a light custom listbox: each row has a
 *  checkbox, a prominent name, a muted metadata line (people · run · time) and an
 *  "exported" chip. Toggling keeps the menu open; click-outside / Escape closes. */
export function SequenceSelect<T extends SeqItem>({
  seqs, values, onChange, getKey, showCapture = false, placeholder = 'Select sequences…',
}: {
  seqs: T[];
  values: string[];
  onChange: (vals: string[]) => void;
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

  const selected = new Set(values);
  const toggle = (k: string) => {
    const next = new Set(selected);
    next.has(k) ? next.delete(k) : next.add(k);
    onChange([...next]);
  };
  const allKeys = seqs.map(getKey);
  const allOn = allKeys.length > 0 && allKeys.every(k => selected.has(k));
  const firstSel = seqs.find(s => selected.has(getKey(s)));
  const triggerText = values.length === 0 ? placeholder
    : values.length === 1 && firstSel ? name(firstSel)
    : `${values.length} sequences selected`;

  return (
    <div className="relative" ref={ref}>
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        aria-haspopup="listbox"
        aria-expanded={open}
        className="w-full bg-surface-2 border border-border rounded-md px-3 py-2 text-sm flex items-center gap-2 hover:border-border-strong focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition-colors"
      >
        <span className={`flex-1 text-left truncate ${values.length ? 'text-foreground' : 'text-foreground-faint'}`}>
          {triggerText}
        </span>
        <ChevronDown className={`w-4 h-4 text-foreground-subtle flex-shrink-0 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && (
        <ul role="listbox" aria-multiselectable="true" className="absolute z-30 mt-1 w-full bg-surface-2 border border-border rounded-md shadow-2xl shadow-black/60 max-h-72 overflow-y-auto p-1">
          {seqs.length === 0 && <li className="px-3 py-2 text-foreground-faint text-sm">No sequences</li>}
          {seqs.length > 1 && (
            <li
              onClick={() => onChange(allOn ? [] : allKeys)}
              className="flex items-center gap-2.5 rounded px-2.5 py-2 cursor-pointer select-none hover:bg-surface-3 border-b border-border-subtle mb-1"
            >
              <input type="checkbox" readOnly checked={allOn} className="w-3.5 h-3.5 accent-primary flex-shrink-0" />
              <span className="text-foreground text-sm font-medium">{allOn ? 'Deselect all' : 'Select all'}</span>
            </li>
          )}
          {seqs.map(s => {
            const k = getKey(s);
            const on = selected.has(k);
            return (
              <li
                key={k}
                role="option"
                aria-selected={on}
                onClick={() => toggle(k)}
                className={`flex items-center gap-2.5 rounded px-2.5 py-2 cursor-pointer select-none ${on ? 'bg-primary-muted' : 'hover:bg-surface-3'}`}
              >
                <input type="checkbox" readOnly checked={on} className="w-3.5 h-3.5 accent-primary flex-shrink-0" />
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
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
