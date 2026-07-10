import { useEffect, useMemo, useState } from 'react';
import { X, Database, Copy, Check, AlertCircle } from 'lucide-react';
import { Skeleton } from './shared/Skeleton';

interface NpzArray {
  name: string;
  shape?: number[];
  dtype?: string;
  sizeBytes?: number;
  compressedSize?: number;
  error?: string;
}

interface NpzMeta {
  path: string;
  fileSize: number;
  arrays: NpzArray[];
}

interface Props {
  /** Absolute path of the .npz file. */
  npzPath: string;
  fileName: string;
  onClose: () => void;
}

function formatSize(bytes: number | undefined): string {
  if (bytes == null) return '—';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function formatShape(shape: number[] | undefined): string {
  if (!shape || shape.length === 0) return '()';
  return `(${shape.join(', ')})`;
}

/**
 * Modal-embedded inspector for `.npz` (numpy archive) files.
 *
 * The backend reads only `.npy` headers — never array bodies — so this
 * is instant on multi-GB files. We render a compact arrays-table
 * (key / shape / dtype / size / compressed) plus a "Copy Python
 * snippet" affordance so users can move from "what's in here" to
 * "load it" in one keystroke.
 */
export function NpzViewer({ npzPath, fileName, onClose }: Props) {
  const [meta, setMeta] = useState<NpzMeta | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [snippetCopied, setSnippetCopied] = useState(false);

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onClose]);

  useEffect(() => {
    let cancelled = false;
    setMeta(null);
    setLoadError(null);
    fetch(`/api/files/npz-meta?path=${encodeURIComponent(npzPath)}`)
      .then(async res => {
        if (cancelled) return;
        if (!res.ok) {
          const err = await res.json().catch(() => ({ error: res.statusText }));
          setLoadError(err.error || `HTTP ${res.status}`);
          return;
        }
        const data = await res.json();
        setMeta(data);
      })
      .catch((e) => { if (!cancelled) setLoadError(String(e?.message ?? e)); });
    return () => { cancelled = true; };
  }, [npzPath]);

  const summary = useMemo(() => {
    if (!meta) return '';
    const n = meta.arrays.length;
    return `${n} array${n === 1 ? '' : 's'} · ${formatSize(meta.fileSize)} on disk`;
  }, [meta]);

  const pythonSnippet = useMemo(() => (
    `import numpy as np\n` +
    `data = np.load("${npzPath}")\n` +
    `print(data.files)\n` +
    `# example: arr = data["${meta?.arrays.find(a => !a.error)?.name ?? '<key>'}"]`
  ), [npzPath, meta]);

  const copySnippet = async () => {
    try {
      await navigator.clipboard.writeText(pythonSnippet);
      setSnippetCopied(true);
      setTimeout(() => setSnippetCopied(false), 2000);
    } catch {
      /* clipboard may be unavailable in cross-origin contexts; ignore. */
    }
  };

  return (
    <div
      className="fixed inset-0 bg-black/85 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="bg-surface-1 border border-border rounded-xl w-[95vw] h-[90vh] max-w-5xl flex flex-col shadow-2xl shadow-black/60 ring-1 ring-inset ring-white/[0.03] overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <header className="flex items-center justify-between gap-3 p-3 border-b border-border-subtle bg-surface-2/40">
          <div className="min-w-0 flex items-center gap-2">
            <Database className="w-4 h-4 text-status-completed flex-shrink-0" />
            <span className="text-foreground text-sm font-medium truncate" title={fileName}>{fileName}</span>
            <span className="text-foreground-faint text-[11px] font-mono truncate hidden md:inline" title={npzPath}>{npzPath}</span>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            <button
              onClick={copySnippet}
              disabled={!meta}
              className="inline-flex items-center gap-1.5 px-2.5 py-1.5 bg-surface-2 hover:bg-surface-3 border border-border hover:border-border-strong text-foreground-muted hover:text-foreground rounded-md text-xs transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              title="Copy a Python snippet that loads this file with numpy."
            >
              {snippetCopied ? (
                <><Check className="w-3.5 h-3.5 text-status-completed" /><span className="text-status-completed">Copied</span></>
              ) : (
                <><Copy className="w-3.5 h-3.5" />Copy Python snippet</>
              )}
            </button>
            <button
              onClick={onClose}
              className="text-foreground-subtle hover:text-foreground p-1.5 -m-1.5 rounded-md hover:bg-surface-3 transition-colors"
              aria-label="Close"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </header>

        <div className="px-4 pt-3 pb-1 text-foreground-muted text-xs min-h-8">
          {meta ? <span>{summary}</span> : (loadError ? '' : <Skeleton className="h-3 w-44" />)}
        </div>

        <div className="flex-1 overflow-auto px-4 pb-4">
          {loadError && (
            <div className="flex items-start gap-2 px-3 py-2 bg-status-failed-bg border border-status-failed/35 rounded-md text-status-failed text-xs">
              <AlertCircle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
              <span>Failed to read .npz: {loadError}</span>
            </div>
          )}
          {!meta && !loadError && (
            <div className="bg-background border border-border-subtle rounded-md overflow-hidden">
              <div className="bg-surface-2/60 border-b border-border-subtle flex gap-3 px-3 py-2">
                {['w-12', 'w-16', 'w-12', 'w-12', 'w-20'].map((w, i) => (
                  <Skeleton key={i} className={`h-3 ${w}`} />
                ))}
              </div>
              {Array.from({ length: 5 }).map((_, ri) => (
                <div key={ri} className={`flex gap-3 px-3 py-2 border-b border-border-subtle/60 ${ri % 2 === 0 ? 'bg-surface-1' : 'bg-surface-1/60'}`}>
                  <Skeleton className="h-3 w-32" />
                  <Skeleton className="h-3 w-24" />
                  <Skeleton className="h-3 w-16" />
                  <div className="flex-1" />
                  <Skeleton className="h-3 w-14" />
                  <Skeleton className="h-3 w-14" />
                </div>
              ))}
            </div>
          )}
          {meta && meta.arrays.length === 0 && (
            <div className="text-foreground-subtle text-sm text-center py-8">This .npz contains no arrays.</div>
          )}
          {meta && meta.arrays.length > 0 && (
            <div className="bg-background border border-border-subtle rounded-md overflow-hidden">
              <table className="w-full text-xs">
                <thead>
                  <tr className="bg-surface-2/60 border-b border-border-subtle">
                    <th className="px-3 py-2 text-left text-foreground-muted text-[11px] font-semibold uppercase tracking-wider">Key</th>
                    <th className="px-3 py-2 text-left text-foreground-muted text-[11px] font-semibold uppercase tracking-wider">Shape</th>
                    <th className="px-3 py-2 text-left text-foreground-muted text-[11px] font-semibold uppercase tracking-wider">dtype</th>
                    <th className="px-3 py-2 text-right text-foreground-muted text-[11px] font-semibold uppercase tracking-wider">Size</th>
                    <th className="px-3 py-2 text-right text-foreground-muted text-[11px] font-semibold uppercase tracking-wider">Compressed</th>
                  </tr>
                </thead>
                <tbody>
                  {meta.arrays.map((a, idx) => {
                    const stripe = idx % 2 === 0 ? 'bg-surface-1' : 'bg-surface-1/60';
                    if (a.error) {
                      return (
                        <tr key={a.name} className={`${stripe} border-b border-border-subtle/60`}>
                          <td className="px-3 py-2 font-mono text-foreground">{a.name}</td>
                          <td colSpan={3} className="px-3 py-2 text-status-failed">
                            <span className="inline-flex items-center gap-1">
                              <AlertCircle className="w-3 h-3" />
                              {a.error}
                            </span>
                          </td>
                          <td className="px-3 py-2 text-right text-foreground-subtle font-mono tabular-nums">{formatSize(a.compressedSize)}</td>
                        </tr>
                      );
                    }
                    return (
                      <tr key={a.name} className={`${stripe} border-b border-border-subtle/60`}>
                        <td className="px-3 py-2 font-mono text-foreground break-all">{a.name}</td>
                        <td className="px-3 py-2 font-mono text-foreground-muted whitespace-nowrap">{formatShape(a.shape)}</td>
                        <td className="px-3 py-2 font-mono text-foreground-muted">{a.dtype ?? '—'}</td>
                        <td className="px-3 py-2 text-right text-foreground-muted font-mono tabular-nums">{formatSize(a.sizeBytes)}</td>
                        <td className="px-3 py-2 text-right text-foreground-subtle font-mono tabular-nums">{formatSize(a.compressedSize)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
