import { useEffect } from 'react';
import { X, FileCode2, ExternalLink } from 'lucide-react';

interface Props {
  /** Absolute path of the HTML file. Streamed via /api/files/html. */
  htmlPath: string;
  fileName: string;
  onClose: () => void;
}

/**
 * Modal-embedded viewer for HTML output files (Plotly, Bokeh,
 * pandas-profiling, etc.).
 *
 * Loading model:
 *   - The iframe `src` points at `/api/files/html?path=<absolute>` which
 *     serves the HTML inline as `text/html`. Same-origin via the Vite
 *     proxy in dev.
 *   - `sandbox="allow-scripts"` (no `allow-same-origin`) means the
 *     loaded HTML lives in an opaque origin: its JS can run for
 *     interactivity, but it can't read cookies or make authenticated
 *     calls back into our API.
 *   - For Plotly/Bokeh-style self-contained HTML this just works. HTML
 *     that references sibling resources (`./style.css`) won't be able
 *     to load them — that's the known limitation.
 *
 * "Open in new tab" is offered for HTML that only renders well at full
 * width. The backend sends `Content-Security-Policy: sandbox` with the
 * same flags, so the tab gets the same opaque origin as the iframe.
 */
export function HtmlViewer({ htmlPath, fileName, onClose }: Props) {
  // Close on Escape — matches the rest of the app's modal affordances.
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onClose]);

  const src = `/api/files/html?path=${encodeURIComponent(htmlPath)}`;

  return (
    <div
      className="fixed inset-0 bg-black/85 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="bg-surface-1 border border-border rounded-xl w-[95vw] h-[90vh] flex flex-col shadow-2xl shadow-black/60 ring-1 ring-inset ring-white/[0.03] overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="flex items-center justify-between gap-3 p-3 border-b border-border-subtle bg-surface-2/40">
          <div className="min-w-0 flex items-center gap-2">
            <FileCode2 className="w-4 h-4 text-status-completed flex-shrink-0" />
            <span className="text-foreground text-sm font-medium truncate" title={fileName}>{fileName}</span>
            <span className="text-foreground-faint text-[11px] font-mono truncate hidden md:inline" title={htmlPath}>{htmlPath}</span>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            <a
              href={src}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 px-2.5 py-1.5 bg-surface-2 hover:bg-surface-3 border border-border hover:border-border-strong text-foreground-muted hover:text-foreground rounded-md text-xs transition-colors"
              title="Open the HTML in its own tab — useful if the embedded view is cramped or relative resources fail to load."
            >
              <ExternalLink className="w-3.5 h-3.5" />
              Open in new tab
            </a>
            <button
              onClick={onClose}
              className="text-foreground-subtle hover:text-foreground p-1.5 -m-1.5 rounded-md hover:bg-surface-3 transition-colors"
              aria-label="Close"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </header>

        <div className="relative flex-1 bg-white">
          <iframe
            src={src}
            title={fileName}
            sandbox="allow-scripts allow-popups allow-forms"
            className="w-full h-full border-0 bg-white"
          />
        </div>
      </div>
    </div>
  );
}
