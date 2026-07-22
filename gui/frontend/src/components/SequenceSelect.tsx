import { useState, useRef, useEffect, useMemo } from 'react';
import { ChevronDown, ChevronRight, Box, Search, Eye } from 'lucide-react';
import { fmtRun, fmtWhen } from './ExportPanel';

export interface SeqItem {
  tag: string; capture: string; seq: string; people: number;
  /** ma_cap output dir for this sequence, when one exists on disk — the
   *  backend emits "" when absent. Carried through so a picked row's
   *  `onPreview` target can feed recorded fps to the mesh preview. */
  ma_3d_dir: string; ma_cap_dir?: string; already_exported: boolean; mtime?: number;
}

/** Styled MULTI-select for export sequences. Native <option> can't show rich text
 *  or multi-select cleanly, so this is a light custom listbox: each row has a
 *  checkbox, a prominent name, a muted metadata line (people · run · time) and an
 *  "exported" chip. Long lists get a type-to-filter box, and with `showCapture`
 *  rows are grouped per capture behind collapsed-by-default headers (checkbox
 *  toggles the whole capture; filtering auto-expands the matching groups).
 *
 *  Two shapes share this component:
 *   - `dropdown` (default): a trigger button + absolute popover; a row click
 *     toggles selection; each row can carry a "Preview" button. Toggling keeps
 *     the menu open; click-outside / Escape closes.
 *   - `panel`: an always-open list that fills its parent (workbench layout). A
 *     row click ACTIVATES it (`onActivate`) for an external viewer while the
 *     checkbox owns selection; the active row is highlighted via `activeKey`. */
export function SequenceSelect<T extends SeqItem>({
  seqs, values, onChange, getKey, showCapture = false, placeholder = 'Select sequences…', onPreview, previewOpen = false,
  variant = 'dropdown', activeKey = null, onActivate,
}: {
  seqs: T[];
  values: string[];
  onChange: (vals: string[]) => void;
  getKey: (s: T) => string;
  showCapture?: boolean;
  placeholder?: string;
  /** When set, each row gets a "Preview" button that opens a quick 3D viewer.
   *  Dropdown variant only — ignored in panel mode (a row click previews). */
  onPreview?: (s: T) => void;
  /** True while the host's preview modal is open. Suspends click-outside /
   *  Escape closing, so the list (and the user's place in it) survives the
   *  modal — otherwise closing the preview would collapse the dropdown. */
  previewOpen?: boolean;
  /** `dropdown` (trigger + popover) or `panel` (always-open, fills parent). */
  variant?: 'dropdown' | 'panel';
  /** Panel variant: key of the externally-controlled active (previewed) row. */
  activeKey?: string | null;
  /** Panel variant: a row click activates it (instead of toggling selection). */
  onActivate?: (s: T) => void;
}) {
  const panel = variant === 'panel';
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  /** Key of the row whose Preview was clicked last — highlighted so the user
   *  can tell which sequence they just looked at. */
  const [lastPreviewed, setLastPreviewed] = useState<string | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  // Close handlers are bound once; they read the modal state through a ref so
  // the same Escape/click that dismisses the preview can't also close the list.
  const previewOpenRef = useRef(previewOpen);
  useEffect(() => { previewOpenRef.current = previewOpen; }, [previewOpen]);

  // Panel variant: an all-collapsed group list looks empty on first load, so
  // expand the first capture group once when data arrives. One-shot (ref-guarded)
  // so a user who later collapses everything isn't fought.
  const didInitExpand = useRef(false);
  useEffect(() => {
    if (panel && !didInitExpand.current && seqs.length > 0) {
      didInitExpand.current = true;
      setExpanded(prev => (prev.size ? prev : new Set([seqs[0].capture])));
    }
  }, [panel, seqs]);

  useEffect(() => {
    // Panel variant is always-open and has no popover — no close handling.
    if (panel) return;
    const onDoc = (e: MouseEvent) => {
      if (previewOpenRef.current) return;
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !previewOpenRef.current) setOpen(false);
    };
    document.addEventListener('mousedown', onDoc);
    document.addEventListener('keydown', onKey);
    return () => { document.removeEventListener('mousedown', onDoc); document.removeEventListener('keydown', onKey); };
  }, [panel]);

  useEffect(() => {
    if (open) inputRef.current?.focus();
    else { setQuery(''); setExpanded(new Set()); }
    // Bring the last-previewed row back into view (expanding its capture
    // group first) so reopening the list lands where the user left off.
    if (open && lastPreviewed) {
      const s = seqs.find(x => getKey(x) === lastPreviewed);
      if (s) setExpanded(prev => new Set(prev).add(s.capture));
      requestAnimationFrame(() => {
        ref.current?.querySelector('[data-last-previewed]')?.scrollIntoView({ block: 'nearest' });
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
    // Panel mode tracks the externally-controlled active row; dropdown mode
    // highlights the row whose Preview button was clicked last.
    const viewed = panel ? k === activeKey : k === lastPreviewed;
    return (
      <li
        key={k}
        role="option"
        aria-selected={on}
        onClick={() => (panel ? onActivate?.(s) : toggle(k))}
        data-last-previewed={(!panel && viewed) || undefined}
        className={`flex items-center gap-2.5 rounded px-2.5 py-2 cursor-pointer select-none ${on ? 'bg-primary-muted' : 'hover:bg-surface-3'} ${
          panel
            ? (viewed ? `ring-1 ring-inset ring-primary/50 ${on ? '' : 'bg-surface-3'}` : '')
            : (viewed ? 'ring-1 ring-inset ring-primary/35' : '')
        }`}
      >
        {/* Panel mode: the checkbox owns selection (row click previews), so it
            handles its own click and stops it reaching the row's activate. */}
        <input
          type="checkbox"
          readOnly
          checked={on}
          onClick={panel ? (e) => { e.stopPropagation(); toggle(k); } : undefined}
          className={`w-3.5 h-3.5 accent-primary flex-shrink-0${panel ? ' cursor-pointer' : ''}`}
        />
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
        {!panel && onPreview && (
          <button
            type="button"
            onClick={(e) => { e.stopPropagation(); setLastPreviewed(k); onPreview(s); }}
            title={viewed ? 'Previewed last — open again' : 'Quick 3D preview of the fitted bodies'}
            className={`flex-shrink-0 text-[10px] font-medium rounded px-1.5 py-0.5 inline-flex items-center gap-1 transition-colors border ${
              viewed
                ? 'text-primary bg-primary-muted border-primary/40 hover:bg-primary-muted-strong'
                : 'text-foreground-muted hover:text-foreground bg-surface-2 hover:bg-surface-3 border-border'
            }`}
          >
            <Eye className="w-3 h-3" /> {viewed ? 'Viewed' : 'Preview'}
          </button>
        )}
      </li>
    );
  };

  // The list contents (empty state, select-all, groups/rows) are shared verbatim
  // between the dropdown popover and the always-open panel.
  const listBody = (
    <>
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
    </>
  );

  // Panel variant: an always-open list that fills its parent (workbench layout).
  if (panel) {
    return (
      <div className="h-full min-h-0 flex flex-col bg-surface-2 border border-border rounded-md overflow-hidden">
        {hasFilter && (
          <div className="flex items-center gap-2 px-2.5 py-2 border-b border-border-subtle flex-shrink-0">
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
        <ul role="listbox" aria-multiselectable="true" className="flex-1 min-h-0 overflow-y-auto p-1">
          {listBody}
        </ul>
      </div>
    );
  }

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
            {listBody}
          </ul>
        </div>
      )}
    </div>
  );
}
