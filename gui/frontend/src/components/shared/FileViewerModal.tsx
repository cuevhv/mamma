import { useState, useEffect, useMemo } from 'react';
import { Eye, Copy, Check, X, Info, Code2, Table as TableIcon, AlertCircle } from 'lucide-react';
import { Skeleton } from './Skeleton';
import { CodeBlock } from './CodeBlock';

interface Props {
  /** File to load. When null, the modal is hidden. */
  file: { name: string; path: string } | null;
  onClose: () => void;
}

type FileKind = 'json' | 'csv' | 'tsv' | 'yaml' | 'text';

function detectKind(name: string): FileKind {
  const lower = name.toLowerCase();
  if (lower.endsWith('.json') || lower.endsWith('.jsonl')) return 'json';
  if (lower.endsWith('.csv')) return 'csv';
  if (lower.endsWith('.tsv')) return 'tsv';
  if (lower.endsWith('.yaml') || lower.endsWith('.yml')) return 'yaml';
  return 'text';
}

/** Tiny RFC-4180-style CSV/TSV parser. Handles quoted fields with
 *  embedded delimiters and `""`-escaped quotes; tolerant of trailing
 *  newlines. Synchronous, in-memory — fine for the file sizes the
 *  Outputs explorer surfaces; if we ever hit multi-MB CSVs we'd swap
 *  this for a streaming parser. */
function parseDelimited(text: string, delim: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = '';
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; continue; }
        inQuotes = false;
        continue;
      }
      field += ch;
    } else {
      if (ch === '"' && field.length === 0) { inQuotes = true; continue; }
      if (ch === delim) { row.push(field); field = ''; continue; }
      if (ch === '\n' || ch === '\r') {
        row.push(field); field = '';
        rows.push(row); row = [];
        if (ch === '\r' && text[i + 1] === '\n') i++;
        continue;
      }
      field += ch;
    }
  }
  if (field !== '' || row.length > 0) { row.push(field); rows.push(row); }
  // Drop a trailing empty row from a final newline.
  if (rows.length > 0 && rows[rows.length - 1].length === 1 && rows[rows.length - 1][0] === '') {
    rows.pop();
  }
  return rows;
}

const CSV_ROW_CAP = 1000;  // soft cap so a 100k-row CSV doesn't kill the DOM

/**
 * Shared file-content viewer. Fetches `/api/files/content?path=...` and
 * renders the body type-aware:
 *   - `.json` / `.jsonl` → pretty-printed (falls back to raw if invalid)
 *   - `.csv` / `.tsv`    → header-aware table
 *   - `.yaml` / `.yml`   → monospace (already well-formatted by design)
 *   - everything else    → monospace `<pre>` (logs, txt, md, …)
 *
 * A "Raw" toggle in the header lets the user switch back to the
 * on-disk text for any of the parsed views.
 *
 * The backend may return a `synthetic: true` body when the requested
 * file legitimately doesn't exist (e.g. the step was skipped via a DONE
 * sentinel from a previous submission). We surface that with a notice
 * banner instead of pretending it's the real log.
 */
export function FileViewerModal({ file, onClose }: Props) {
  const [content, setContent] = useState<string>('');
  const [synthetic, setSynthetic] = useState(false);
  const [copied, setCopied] = useState(false);
  const [raw, setRaw] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!file) { setContent(''); setSynthetic(false); setCopied(false); setRaw(false); setLoading(false); return; }
    setContent('');
    setSynthetic(false);
    setRaw(false);
    setLoading(true);
    let cancelled = false;
    fetch(`/api/files/content?path=${encodeURIComponent(file.path)}`)
      .then(async res => {
        if (cancelled) return;
        if (res.ok) {
          const data = await res.json();
          setContent(data.content || '');
          setSynthetic(!!data.synthetic);
        } else {
          const err = await res.json().catch(() => ({ error: res.statusText }));
          setContent(`Error loading file content: ${err.error || res.statusText}`);
          setSynthetic(false);
        }
      })
      .catch(() => { if (!cancelled) { setContent('Error loading file content'); setSynthetic(false); } })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [file]);

  const kind = useMemo<FileKind>(() => file ? detectKind(file.name) : 'text', [file]);

  // Pretty-print JSON; on parse failure, we render raw and surface a
  // note so the user sees both that we tried and why we couldn't.
  const jsonView = useMemo(() => {
    if (kind !== 'json' || raw || !content) return null;
    try {
      const parsed = JSON.parse(content);
      return { ok: true as const, text: JSON.stringify(parsed, null, 2) };
    } catch (e) {
      return { ok: false as const, error: e instanceof Error ? e.message : String(e) };
    }
  }, [kind, raw, content]);

  const csvView = useMemo(() => {
    if ((kind !== 'csv' && kind !== 'tsv') || raw || !content) return null;
    const delim = kind === 'tsv' ? '\t' : ',';
    const rows = parseDelimited(content, delim);
    if (rows.length === 0) return { rows: [], header: [] as string[], total: 0, capped: false };
    const header = rows[0];
    const body = rows.slice(1);
    const capped = body.length > CSV_ROW_CAP;
    return {
      header,
      rows: capped ? body.slice(0, CSV_ROW_CAP) : body,
      total: body.length,
      capped,
    };
  }, [kind, raw, content]);

  if (!file) return null;

  const copyPath = () => {
    navigator.clipboard.writeText(file.path);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const canToggleRaw = kind === 'json' || kind === 'csv' || kind === 'tsv';

  return (
    <div className="fixed inset-0 bg-black/65 backdrop-blur-sm flex items-center justify-center z-50 p-4" onClick={onClose}>
      <div
        className="bg-surface-1 border border-border rounded-xl max-w-6xl w-full max-h-[90vh] flex flex-col shadow-2xl shadow-black/60 ring-1 ring-inset ring-white/[0.03]"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between gap-3 p-4 border-b border-border-subtle bg-surface-2/40">
          <h3 className="text-foreground flex items-center gap-2 text-base font-medium min-w-0">
            <Eye className="w-4 h-4 text-primary flex-shrink-0" />
            <span className="truncate" title={file.name}>{file.name}</span>
          </h3>
          <div className="flex items-center gap-2 flex-shrink-0">
            {canToggleRaw && (
              <button
                onClick={() => setRaw(r => !r)}
                className="inline-flex items-center gap-1.5 px-2.5 py-1.5 bg-surface-2 hover:bg-surface-3 border border-border hover:border-border-strong text-foreground-muted hover:text-foreground rounded-md text-xs transition-colors"
                title={raw ? 'Switch back to the parsed view' : 'Show the raw file as-is'}
              >
                {raw ? <TableIcon className="w-3.5 h-3.5" /> : <Code2 className="w-3.5 h-3.5" />}
                {raw ? (kind === 'json' ? 'Pretty' : 'Table') : 'Raw'}
              </button>
            )}
            <button
              onClick={onClose}
              className="text-foreground-subtle hover:text-foreground p-1.5 -m-1.5 rounded-md hover:bg-surface-3 transition-colors"
              aria-label="Close"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        <div className="px-4 pt-3 pb-2">
          <div className="flex items-center gap-2 bg-surface-2 border border-border-subtle rounded-md p-3">
            <div className="flex-1 min-w-0">
              <div className="text-foreground-subtle text-[11px] uppercase tracking-wider mb-1">File path</div>
              <code className="text-foreground-muted text-sm font-mono break-all">{file.path}</code>
            </div>
            <button
              onClick={copyPath}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-surface-3 hover:bg-surface-4 border border-border rounded-md text-sm text-foreground-muted hover:text-foreground transition-colors whitespace-nowrap"
              title="Copy path to clipboard"
            >
              {copied ? (
                <>
                  <Check className="w-4 h-4 text-status-completed" />
                  <span className="text-status-completed">Copied</span>
                </>
              ) : (
                <>
                  <Copy className="w-4 h-4" />
                  <span>Copy</span>
                </>
              )}
            </button>
          </div>
        </div>

        <div className="p-4 overflow-auto flex-1">
          {synthetic && (
            <div className="mb-3 flex items-start gap-2 px-3 py-2 bg-status-pending-bg border border-status-pending/35 rounded-md text-status-pending text-xs">
              <Info className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
              <span>
                This file isn't on disk. The runner skipped this step (DONE sentinel from a prior run),
                so no output was written. The body below explains what happened and how to force a re-run.
              </span>
            </div>
          )}

          {loading && (
            <div className="bg-background border border-border-subtle p-4 rounded-md space-y-2">
              {['w-3/4', 'w-5/6', 'w-1/2', 'w-2/3', 'w-4/5', 'w-3/5', 'w-3/4'].map((w, i) => (
                <Skeleton key={i} className={`h-3 ${w}`} />
              ))}
            </div>
          )}

          {!loading && jsonView && jsonView.ok && (
            <CodeBlock
              code={jsonView.text}
              language="json"
              className="text-foreground text-xs font-mono whitespace-pre-wrap bg-background border border-border-subtle p-4 rounded-md"
            />
          )}

          {!loading && jsonView && !jsonView.ok && (
            <div>
              <div className="mb-3 flex items-start gap-2 px-3 py-2 bg-status-failed-bg border border-status-failed/35 rounded-md text-status-failed text-xs">
                <AlertCircle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
                <span>Couldn't parse as JSON ({jsonView.error}). Showing the raw file.</span>
              </div>
              <pre className="text-foreground-muted text-xs font-mono whitespace-pre-wrap bg-background border border-border-subtle p-4 rounded-md">
                {content || 'File is empty'}
              </pre>
            </div>
          )}

          {!loading && csvView && (
            <div className="bg-background border border-border-subtle rounded-md overflow-hidden">
              {csvView.rows.length === 0 ? (
                <div className="p-4 text-foreground-subtle text-xs text-center">No rows.</div>
              ) : (
                <>
                  <div className="overflow-auto">
                    <table className="w-full text-xs">
                      <thead className="sticky top-0 z-10">
                        <tr className="bg-surface-2/90 backdrop-blur border-b border-border-subtle">
                          {csvView.header.map((h, i) => (
                            <th key={i} className="px-3 py-2 text-left text-foreground text-[11px] font-semibold whitespace-nowrap">
                              {h || <span className="text-foreground-faint italic">col {i + 1}</span>}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {csvView.rows.map((r, ri) => {
                          const stripe = ri % 2 === 0 ? 'bg-surface-1' : 'bg-surface-1/60';
                          return (
                            <tr key={ri} className={`${stripe} border-b border-border-subtle/60`}>
                              {csvView.header.map((_, ci) => (
                                <td key={ci} className="px-3 py-1.5 font-mono text-foreground-muted whitespace-nowrap">
                                  {r[ci] ?? ''}
                                </td>
                              ))}
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  {csvView.capped && (
                    <div className="px-3 py-2 bg-surface-2/40 border-t border-border-subtle text-foreground-subtle text-[11px] flex items-center gap-1.5">
                      <Info className="w-3 h-3" />
                      Showing first {CSV_ROW_CAP.toLocaleString()} of {csvView.total.toLocaleString()} rows. Switch to Raw to see the rest.
                    </div>
                  )}
                </>
              )}
            </div>
          )}

          {!loading && !jsonView && !csvView && (
            kind === 'yaml' && content ? (
              <CodeBlock
                code={content}
                language="yaml"
                className="text-foreground text-xs font-mono whitespace-pre-wrap bg-background border border-border-subtle p-4 rounded-md"
              />
            ) : (
              <pre className="text-foreground-muted text-xs font-mono whitespace-pre-wrap bg-background border border-border-subtle p-4 rounded-md">
                {content || 'File is empty'}
              </pre>
            )
          )}
        </div>
      </div>
    </div>
  );
}
