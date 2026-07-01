/*
 * Richer capture picker for the New Task form's Step 1 (Pick mode): a custom
 * dropdown that shows each capture's thumbnail + name + quick info (sequence /
 * camera counts) instead of a flat native <select> of names.
 *
 * The option list renders in a body-level portal (so an ancestor's
 * overflow-hidden can't clip it) and closes on outside-click / Esc / background
 * scroll / resize — while scrolls INSIDE the list are ignored. Reuses the
 * shared <Thumbnail> (served via /api/files/image).
 */
import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { ChevronDown, Check } from 'lucide-react';
import { Thumbnail } from './shared/Thumbnail';

export interface CaptureOption {
  id: string;
  captureName: string;
  jsonPath: string;
  seqNames: string[];
  cams: string[];
  thumbnailPath: string | null;
}

function quickInfo(c: CaptureOption): string {
  const s = c.seqNames.length;
  const parts = [`${s} seq${s === 1 ? '' : 's'}`];
  if (c.cams.length) parts.push(`${c.cams.length} cam${c.cams.length === 1 ? '' : 's'}`);
  return parts.join(' · ');
}

export function CaptureSelect({ captures, value, onPick }: {
  captures: CaptureOption[];
  value: string;                 // currently-picked jsonPath
  onPick: (jsonPath: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLUListElement>(null);
  const [rect, setRect] = useState<DOMRect | null>(null);

  const openMenu = () => {
    if (triggerRef.current) setRect(triggerRef.current.getBoundingClientRect());
    setOpen(true);
  };

  useEffect(() => {
    if (!open) return;
    const close = () => setOpen(false);
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false); };
    // Close on background scroll (the menu is fixed at the trigger), but ignore
    // scrolls inside the menu's own list.
    const onScroll = (e: Event) => {
      if (menuRef.current && e.target instanceof Node && menuRef.current.contains(e.target)) return;
      setOpen(false);
    };
    window.addEventListener('scroll', onScroll, true);
    window.addEventListener('resize', close);
    document.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('scroll', onScroll, true);
      window.removeEventListener('resize', close);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  const selected = captures.find((c) => c.jsonPath === value) ?? null;
  const empty = captures.length === 0;

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => (open ? setOpen(false) : openMenu())}
        className="w-full flex items-center gap-2.5 bg-surface-2 border border-border rounded-md px-2.5 py-2 text-left hover:border-border-strong focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition-colors"
      >
        {selected ? (
          <>
            <Thumbnail path={selected.thumbnailPath} alt={selected.captureName} className="w-12 h-8 flex-shrink-0" />
            <div className="min-w-0 flex-1">
              <div className="text-foreground text-sm truncate">{selected.captureName}</div>
              <div className="text-foreground-subtle text-[11px] truncate">{quickInfo(selected)}</div>
            </div>
          </>
        ) : (
          <span className="flex-1 text-foreground-subtle text-sm">
            {empty ? 'No captures yet — switch to "Create new capture"' : 'Pick a capture…'}
          </span>
        )}
        <ChevronDown className={`w-4 h-4 flex-shrink-0 text-foreground-faint transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && rect && createPortal(
        <>
          <div className="fixed inset-0 z-40" onMouseDown={() => setOpen(false)} />
          <ul
            ref={menuRef}
            role="listbox"
            style={{ position: 'fixed', top: rect.bottom + 4, left: rect.left, width: rect.width }}
            className="z-50 max-h-80 overflow-auto rounded-md border border-border bg-surface-1 shadow-xl shadow-black/40 ring-1 ring-inset ring-white/[0.04] py-1"
          >
            {empty ? (
              <li className="px-3 py-3 text-xs text-foreground-subtle">
                No captures yet — switch back to "Create new capture" to make one.
              </li>
            ) : (
              captures.map((c) => {
                const active = c.jsonPath === value;
                return (
                  <li key={c.id}>
                    <button
                      type="button"
                      role="option"
                      aria-selected={active}
                      onClick={() => { onPick(c.jsonPath); setOpen(false); }}
                      className={`w-full flex items-center gap-2.5 px-2.5 py-2 text-left transition-colors ${active ? 'bg-primary-muted' : 'hover:bg-surface-2'}`}
                    >
                      <Thumbnail path={c.thumbnailPath} alt={c.captureName} className="w-14 h-9 flex-shrink-0" />
                      <div className="min-w-0 flex-1">
                        <div className={`text-sm truncate ${active ? 'text-primary font-medium' : 'text-foreground'}`}>{c.captureName}</div>
                        <div className="text-foreground-subtle text-[11px] truncate">{quickInfo(c)}</div>
                        <div className="text-foreground-faint text-[10px] font-mono truncate" title={c.jsonPath}>{c.jsonPath}</div>
                      </div>
                      {active && <Check className="w-4 h-4 text-primary flex-shrink-0" />}
                    </button>
                  </li>
                );
              })
            )}
          </ul>
        </>,
        document.body,
      )}
    </>
  );
}
