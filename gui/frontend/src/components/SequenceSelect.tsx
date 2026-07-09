import { useState, useRef, useEffect, useMemo } from 'react';
import { ChevronDown, ChevronRight, Box, Search } from 'lucide-react';
import { fmtRun, fmtWhen } from './ExportPanel';

export interface SeqItem {
  tag: string; capture: string; seq: string; people: number;
  ma_3d_dir: string; already_exported: boolean; mtime?: number;
}

/** Styled MULTI-select for export sequences. Native <option> can't show rich text
 *  or multi-select cleanly, so this is a light custom listbox: each row has a
 *  checkbox, a prominent name, a muted metadata line (people · run · time) and an
 *  "exported" chip. Long lists get a type-to-filter box, and with `showCapture`
 *  rows are grouped per capture behind collapsed-by-default headers (checkbox
 *  toggles the whole capture; filtering auto-expands the matching groups).
 *  Toggling keeps the menu open; click-outside / Escape closes. */
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
  const [query, setQuery] = useState('');
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const ref = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onDoc);
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('mousedown', onDoc); document.removeEventListener('keydown', onKey); };
  }, []);

  useEffect(() => {
    if (open) inputRef.current?.focus();
    else { setQuery(''); setExpanded(new Set()); }
  }, [open]);

  const meta = (s: SeqItem) =>
    `${s.people} ${s.people === 1 ? 'person' : 'people'} · run ${fmtRun(s.tag)}${s.mtime ? ` · ${fmtWhen(s.mtime)}` : ''}`;

  const q = query.trim().toLowerCase();
  const visible = useMemo(
    () => (q ? seqs.filter(s => `${s.capture} ${s.seq}`.toLowerCase().includes(q)) : seqs),
    [seqs, q],
  );
  // Group by capture (insertion order) when the list spans several captures.
  const groups = useMemo(() => {
    const m = new Map<string, T[]>();
    for (const s of visible) (m.get(s.capture) ?? m.set(s.capture, []).get(s.capture)!).push(s);
    return [...m.entries()];
  }, [visible]);
  const grouped = showCapture && new Set(seqs.map(s => s.capture)).size > 1;
  const hasFilter = seqs.length > 5;

  const selected = new Set(values);
  const setKeys = (keys: string[], on: boolean) => {
    const next = new Set(selected);
    keys.forEach(k => (on ? next.add(k) : next.delete(k)));
    onChange([...next]);
  };
  const toggle = (k: string) => setKeys([k], !selected.has(k));
  const visKeys = visible.map(getKey);
  const allOn = visKeys.length > 0 && visKeys.every(k => selected.has(k));
  const firstSel = seqs.find(s => selected.has(getKey(s)));
  const triggerText = values.length === 0 ? placeholder
    : values.length === 1 && firstSel ? (showCapture ? `${firstSel.capture} / ${firstSel.seq}` : firstSel.seq)
    : `${values.length} sequences selected`;

  const row = (s: T) => {
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
          <div className="text-foreground text-sm font-medium truncate">{grouped || !showCapture ? s.seq : `${s.capture} / ${s.seq}`}</div>
          <div className="text-foreground-muted text-xs font-mono truncate">{meta(s)}</div>
        </div>
        {s.already_exported && (
          <span className="flex-shrink-0 text-[10px] font-medium text-emerald-400 bg-emerald-400/10 border border-emerald-400/25 rounded px-1.5 py-0.5">
            exported
          </span>
        )}
      </li>
    );
  };

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
        <div className="absolute z-30 mt-1 w-full bg-surface-2 border border-border rounded-md shadow-2xl shadow-black/60">
          {hasFilter && (
            <div className="flex items-center gap-2 px-2.5 py-2 border-b border-border-subtle">
              <Search className="w-3.5 h-3.5 text-foreground-subtle flex-shrink-0" />
              <input
                ref={inputRef}
                value={query}
                onChange={e => setQuery(e.target.value)}
                placeholder="Filter by capture or sequence…"
                className="flex-1 bg-transparent text-sm text-foreground placeholder:text-foreground-faint focus:outline-none"
              />
            </div>
          )}
          <ul role="listbox" aria-multiselectable="true" className="max-h-[65vh] overflow-y-auto p-1">
            {visible.length === 0 && <li className="px-3 py-2 text-foreground-faint text-sm">{q ? 'No matches' : 'No sequences'}</li>}
            {visible.length > 1 && (
              <li
                onClick={() => setKeys(visKeys, !allOn)}
                className="flex items-center gap-2.5 rounded px-2.5 py-2 cursor-pointer select-none hover:bg-surface-3 border-b border-border-subtle mb-1"
              >
                <input type="checkbox" readOnly checked={allOn} className="w-3.5 h-3.5 accent-primary flex-shrink-0" />
                <span className="text-foreground text-sm font-medium">{allOn ? 'Deselect all' : 'Select all'}{q ? ` (${visible.length} shown)` : ''}</span>
              </li>
            )}
            {grouped
              ? groups.map(([capture, items]) => {
                  const keys = items.map(getKey);
                  const groupOn = keys.every(k => selected.has(k));
                  const nSel = keys.filter(k => selected.has(k)).length;
                  const isOpen = !!q || expanded.has(capture);
                  const flip = () => setExpanded(prev => {
                    const next = new Set(prev);
                    next.has(capture) ? next.delete(capture) : next.add(capture);
                    return next;
                  });
                  return (
                    <li key={capture}>
                      <div
                        onClick={flip}
                        className="sticky top-0 z-10 bg-surface-2 flex items-center gap-2.5 rounded px-2.5 py-2 cursor-pointer select-none hover:bg-surface-3"
                      >
                        <input
                          type="checkbox" readOnly checked={groupOn}
                          onClick={e => { e.stopPropagation(); setKeys(keys, !groupOn); }}
                          className="w-3.5 h-3.5 accent-primary flex-shrink-0 cursor-pointer"
                        />
                        {isOpen ? <ChevronDown className="w-3.5 h-3.5 text-foreground-subtle flex-shrink-0" /> : <ChevronRight className="w-3.5 h-3.5 text-foreground-subtle flex-shrink-0" />}
                        <span className="min-w-0 flex-1 text-left text-foreground text-sm font-medium truncate">{capture}</span>
                        <span className="flex-shrink-0 text-[10px] text-foreground-faint font-mono">
                          {nSel ? `${nSel}/${items.length} selected` : `${items.length} seq${items.length === 1 ? '' : 's'}`}
                        </span>
                      </div>
                      {isOpen && <ul className="pl-4">{items.map(row)}</ul>}
                    </li>
                  );
                })
              : visible.map(row)}
          </ul>
        </div>
      )}
    </div>
  );
}
