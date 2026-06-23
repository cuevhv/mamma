import { useState, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { Monitor, ChevronDown, RotateCcw } from 'lucide-react';

const MENU_W = 240; // px — matches the menu width below

/** Split button for launching the native Rerun viewer: the main part opens
 *  keeping the viewer's saved layout; the caret reveals "fresh layout", which
 *  asks the backend to clear this recording's cached blueprint before opening.
 *  The menu is portaled to <body> with fixed positioning so it isn't clipped by
 *  the scrollable step/output boxes it lives in. Used in the file list and the
 *  web-viewer header. */
export function NativeOpenButton({
  onOpen, label = 'Native', size = 'sm', title,
}: {
  /** fresh=false keeps the saved layout; fresh=true resets it first. */
  onOpen: (fresh: boolean) => void;
  label?: string;
  size?: 'sm' | 'md';
  title?: string;
}) {
  const [pos, setPos] = useState<{ top: number; left: number } | null>(null);
  const triggerRef = useRef<HTMLDivElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const open = pos !== null;

  const openMenu = () => {
    const r = triggerRef.current?.getBoundingClientRect();
    if (r) setPos({ top: r.bottom + 4, left: Math.max(8, r.right - MENU_W) });
  };

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      const t = e.target as Node;
      if (triggerRef.current?.contains(t) || menuRef.current?.contains(t)) return;
      setPos(null);
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setPos(null); };
    // Close on any scroll/resize so the fixed menu never floats away from its button.
    const close = () => setPos(null);
    document.addEventListener('mousedown', onDoc);
    document.addEventListener('keydown', onKey);
    window.addEventListener('scroll', close, true); // capture → catches inner scrollers
    window.addEventListener('resize', close);
    return () => {
      document.removeEventListener('mousedown', onDoc);
      document.removeEventListener('keydown', onKey);
      window.removeEventListener('scroll', close, true);
      window.removeEventListener('resize', close);
    };
  }, [open]);

  const pad = size === 'md' ? 'px-2.5 py-1.5 gap-1.5' : 'px-2 py-1 gap-1';
  const icon = size === 'md' ? 'w-3.5 h-3.5' : 'w-3 h-3';

  return (
    <div className="inline-flex" ref={triggerRef}>
      <div className="inline-flex items-stretch rounded-md border border-border hover:border-border-strong overflow-hidden bg-surface-2">
        <button
          onClick={() => onOpen(false)}
          title={title ?? 'Launch the native Rerun desktop viewer (keeps your saved layout).'}
          className={`inline-flex items-center ${pad} text-xs text-foreground-muted hover:text-foreground hover:bg-surface-3 transition-colors`}
        >
          <Monitor className={icon} />
          {label}
        </button>
        <button
          onClick={() => (open ? setPos(null) : openMenu())}
          aria-haspopup="menu"
          aria-expanded={open}
          title="More open options"
          className="px-1 text-foreground-muted hover:text-foreground hover:bg-surface-3 border-l border-border transition-colors"
        >
          <ChevronDown className={`${icon} transition-transform ${open ? 'rotate-180' : ''}`} />
        </button>
      </div>
      {open && pos && createPortal(
        <div
          ref={menuRef}
          role="menu"
          style={{ position: 'fixed', top: pos.top, left: pos.left, width: MENU_W }}
          className="z-50 bg-surface-2 border border-border rounded-md shadow-2xl shadow-black/60 p-1"
        >
          <button
            role="menuitem"
            onClick={() => { setPos(null); onOpen(true); }}
            className="w-full flex items-start gap-2 rounded px-2.5 py-2 text-left hover:bg-surface-3 transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5 mt-0.5 text-foreground-subtle flex-shrink-0" />
            <span className="min-w-0">
              <span className="block text-foreground text-xs font-medium">Open native (fresh layout)</span>
              <span className="block text-foreground-faint text-[11px] leading-snug">Reset this sequence's saved layout, then open.</span>
            </span>
          </button>
        </div>,
        document.body,
      )}
    </div>
  );
}
