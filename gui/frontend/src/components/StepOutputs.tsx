import { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowLeft, Folder, FileVideo, FileImage, File, Sparkles, Globe, ChevronDown, ChevronRight, FileCode2, Database, FileJson, Sheet, FileText } from 'lucide-react';
import { stepLabel } from './shared/stepLabels';
import { NativeOpenButton } from './NativeOpenButton';
import { FileRowsSkeleton } from './shared/Skeleton';

interface FileEntry { name: string; size: number; }
interface DirEntry { name: string; }

interface StepOutputsProps {
  step: string;
  baseRelPath: string;
  /** Auto-expand on mount — used by the deep-link-from-Tasks flow so the
   *  user lands directly on the right step. */
  defaultOpen?: boolean;
  /** Scroll into view on mount. Same use-case as `defaultOpen`. */
  scrollIntoViewOnMount?: boolean;
  onPlayVideo: (relPath: string) => void;
  onPlayImage: (relPath: string) => void;
  onOpenRrdBrowser: (relPath: string, name: string) => void;
  onOpenRrdNative: (relPath: string, fresh?: boolean) => void;
  onOpenHtml: (relPath: string, name: string) => void;
  onOpenNpz: (relPath: string, name: string) => void;
  /** JSON, CSV/TSV, YAML, plain text — all routed through the shared
   *  FileViewerModal which renders type-aware (pretty-printed JSON,
   *  table CSV, monospace text). */
  onOpenText: (relPath: string, name: string) => void;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

function isImageFile(name: string): boolean {
  const lower = name.toLowerCase();
  return lower.endsWith('.png') || lower.endsWith('.jpg') || lower.endsWith('.jpeg')
      || lower.endsWith('.webp') || lower.endsWith('.gif') || lower.endsWith('.bmp');
}

/**
 * One step's slice of the Outputs explorer. Owns its own breadcrumb/listing
 * state so the parent can render N of these stacked vertically without each
 * step's navigation clobbering the others.
 *
 * The parent passes in a pre-computed `baseRelPath` (the canonical
 * `<output_path>/<step>/<output_id>/<dataset>/<seq>` shape) and we manage
 * the in-section navigation from there.
 */
export function StepOutputs({
  step,
  baseRelPath,
  defaultOpen = true,
  scrollIntoViewOnMount = false,
  onPlayVideo,
  onPlayImage,
  onOpenRrdBrowser,
  onOpenRrdNative,
  onOpenHtml,
  onOpenNpz,
  onOpenText,
}: StepOutputsProps) {
  const [open, setOpen] = useState(defaultOpen);
  const [relPath, setRelPath] = useState(baseRelPath);
  const [entries, setEntries] = useState<{ dirs: DirEntry[]; files: FileEntry[] } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sectionRef = useRef<HTMLDivElement | null>(null);

  // Reset back to the step's root when the surrounding selection (task /
  // sequence / output_id) changes — `baseRelPath` carries all of those.
  useEffect(() => {
    setRelPath(baseRelPath);
  }, [baseRelPath]);

  // Optional scroll-to-this-step on mount (deep-link from Tasks tab).
  useEffect(() => {
    if (scrollIntoViewOnMount && sectionRef.current) {
      sectionRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    // Intentional: only fire once on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!open || !relPath) {
      setEntries(null);
      return;
    }
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    fetch(`/api/files/list?path=${encodeURIComponent(relPath)}`, { signal: controller.signal })
      .then(res => res.json())
      .then(data => {
        if (data.error) throw new Error(data.error);
        setEntries(data);
      })
      .catch(err => {
        if (err.name !== 'AbortError') setError(err.message);
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [relPath, open]);

  const breadcrumbs = useMemo(() => {
    if (!baseRelPath || !relPath) return [];
    const baseParts = baseRelPath.split('/');
    const currentParts = relPath.split('/');
    const result: { name: string; path: string }[] = [
      { name: baseParts[baseParts.length - 1], path: baseRelPath },
    ];
    for (let i = baseParts.length; i < currentParts.length; i++) {
      result.push({ name: currentParts[i], path: currentParts.slice(0, i + 1).join('/') });
    }
    return result;
  }, [baseRelPath, relPath]);

  const canGoUp = relPath !== baseRelPath && relPath.startsWith(baseRelPath + '/');

  const goUp = () => {
    if (!canGoUp) return;
    const parts = relPath.split('/');
    parts.pop();
    setRelPath(parts.join('/'));
  };

  const fileCount = entries ? entries.dirs.length + entries.files.length : null;

  return (
    <div ref={sectionRef} className="bg-background border border-border-subtle rounded-lg overflow-hidden">
      {/* Step header — clickable to collapse/expand. The label uses the
          same stepLabel() helper as the matrix so naming stays consistent. */}
      <button
        type="button"
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-2 px-3 py-2 bg-surface-1 border-b border-border-subtle text-left hover:bg-surface-2/60 transition-colors"
      >
        {open ? <ChevronDown className="w-4 h-4 text-foreground-muted" /> : <ChevronRight className="w-4 h-4 text-foreground-muted" />}
        <span className="text-foreground text-sm font-medium">{stepLabel(step)}</span>
        <span className="text-foreground-faint text-xs font-mono">({step})</span>
        {open && fileCount !== null && (
          <span className="ml-auto text-foreground-subtle text-xs">
            {fileCount === 0 ? 'empty' : `${fileCount} item${fileCount === 1 ? '' : 's'}`}
          </span>
        )}
      </button>

      {open && (
        <>
          {/* Breadcrumbs + Up */}
          <div className="flex items-center gap-2 px-3 py-1.5 bg-surface-1/40 border-b border-border-subtle">
            <button
              onClick={goUp}
              disabled={!canGoUp}
              title="Go up"
              className="p-1 rounded-md text-foreground-muted hover:text-foreground hover:bg-surface-3 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
            >
              <ArrowLeft className="w-4 h-4" />
            </button>
            <div className="flex items-center gap-1 text-xs font-mono text-foreground-muted overflow-x-auto min-w-0 flex-1">
              {!baseRelPath ? (
                <span className="text-foreground-subtle italic">Select a sequence</span>
              ) : (
                breadcrumbs.map((crumb, i) => (
                  <span key={crumb.path} className="flex items-center gap-1 shrink-0">
                    {i > 0 && <span className="text-foreground-faint">/</span>}
                    {i < breadcrumbs.length - 1 ? (
                      <button
                        onClick={() => setRelPath(crumb.path)}
                        className="hover:text-foreground hover:underline px-0.5 transition-colors"
                      >
                        {crumb.name}
                      </button>
                    ) : (
                      <span className="text-foreground font-medium">{crumb.name}</span>
                    )}
                  </span>
                ))
              )}
            </div>
          </div>

          {/* File listing — capped height so 5 stacked steps stay scannable
              without forcing the whole page to scroll past one giant step. */}
          <div className="min-h-24 max-h-[260px] overflow-y-auto">
            {loading && (
              <div className="py-1">
                <FileRowsSkeleton count={4} />
              </div>
            )}
            {error && <div className="p-4 text-status-failed text-xs text-center">{error}</div>}
            {!loading && !error && !entries && (
              <div className="p-4 text-foreground-subtle text-xs text-center">No path selected.</div>
            )}
            {!loading && !error && entries && (
              <div>
                {entries.dirs.length === 0 && entries.files.length === 0 && (
                  <div className="p-4 text-foreground-subtle text-xs text-center">Empty directory</div>
                )}
                {entries.dirs.map(dir => (
                  <button
                    key={dir.name}
                    onClick={() => setRelPath(`${relPath}/${dir.name}`)}
                    className="w-full flex items-center gap-3 px-4 py-1.5 hover:bg-surface-3/50 text-left transition-colors"
                  >
                    <Folder className="w-4 h-4 text-status-pending shrink-0" />
                    <span className="text-foreground text-sm font-mono">{dir.name}</span>
                  </button>
                ))}
                {entries.files.map(file => {
                  const lower = file.name.toLowerCase();
                  const isMP4 = lower.endsWith('.mp4');
                  const isImage = isImageFile(file.name);
                  const isRrd = lower.endsWith('.rrd');
                  const isHtml = lower.endsWith('.html') || lower.endsWith('.htm');
                  const isNpz = lower.endsWith('.npz');
                  const isJson = lower.endsWith('.json') || lower.endsWith('.jsonl');
                  const isCsv = lower.endsWith('.csv') || lower.endsWith('.tsv');
                  const isYaml = lower.endsWith('.yaml') || lower.endsWith('.yml');
                  const isText = isJson || isCsv || isYaml;
                  const filePath = `${relPath}/${file.name}`;

                  if (isRrd) {
                    return (
                      <div
                        key={file.name}
                        className="w-full flex items-center gap-3 px-4 py-1.5 hover:bg-surface-3/30 transition-colors"
                        title={file.name}
                      >
                        <Sparkles className="w-4 h-4 text-status-mixed shrink-0" />
                        <div className="min-w-0 flex-1 flex items-center gap-3">
                          <button
                            onClick={() => onOpenRrdBrowser(filePath, file.name)}
                            className="min-w-0 break-all text-left text-sm font-mono leading-snug text-foreground hover:text-status-mixed hover:underline cursor-pointer transition-colors"
                            title="Open in an embedded browser viewer (good for files up to a few hundred MB)"
                          >
                            {file.name}
                          </button>
                          <span className="shrink-0 text-foreground-subtle text-xs font-mono tabular-nums">{formatSize(file.size)}</span>
                        </div>
                        <div className="flex items-center gap-1 flex-shrink-0">
                          <button
                            onClick={() => onOpenRrdBrowser(filePath, file.name)}
                            className="inline-flex items-center gap-1 px-2 py-1 text-xs text-status-mixed bg-status-mixed-bg border border-status-mixed/35 hover:border-status-mixed/55 rounded-md transition-colors"
                            title="Open in an embedded browser viewer (good for files up to a few hundred MB)"
                          >
                            <Globe className="w-3 h-3" />
                            Browser
                          </button>
                          <NativeOpenButton
                            label="Native"
                            size="sm"
                            title="Launch the native Rerun desktop viewer — better for large recordings (1GB+). Keeps your saved layout."
                            onOpen={(fresh) => onOpenRrdNative(filePath, fresh)}
                          />
                        </div>
                      </div>
                    );
                  }

                  const isActionable = isMP4 || isImage || isHtml || isNpz || isText;
                  return (
                    <button
                      key={file.name}
                      onClick={() => {
                        if (!isActionable) return;
                        if (isMP4) onPlayVideo(filePath);
                        else if (isImage) onPlayImage(filePath);
                        else if (isHtml) onOpenHtml(filePath, file.name);
                        else if (isNpz) onOpenNpz(filePath, file.name);
                        else if (isText) onOpenText(filePath, file.name);
                      }}
                      disabled={!isActionable}
                      className={`w-full flex items-center gap-3 px-4 py-1.5 text-left transition-colors ${
                        isActionable ? 'hover:bg-surface-3/50 cursor-pointer' : 'cursor-default opacity-50'
                      }`}
                      title={file.name}
                    >
                      {isMP4 && <FileVideo className="w-4 h-4 text-primary shrink-0" />}
                      {!isMP4 && isImage && <FileImage className="w-4 h-4 text-status-completed shrink-0" />}
                      {!isMP4 && !isImage && isHtml && <FileCode2 className="w-4 h-4 text-status-completed shrink-0" />}
                      {!isMP4 && !isImage && !isHtml && isNpz && <Database className="w-4 h-4 text-status-completed shrink-0" />}
                      {!isMP4 && !isImage && !isHtml && !isNpz && isJson && <FileJson className="w-4 h-4 text-foreground-muted shrink-0" />}
                      {!isMP4 && !isImage && !isHtml && !isNpz && !isJson && isCsv && <Sheet className="w-4 h-4 text-foreground-muted shrink-0" />}
                      {!isMP4 && !isImage && !isHtml && !isNpz && !isJson && !isCsv && isYaml && <FileText className="w-4 h-4 text-foreground-muted shrink-0" />}
                      {!isMP4 && !isImage && !isHtml && !isNpz && !isText && <File className="w-4 h-4 text-foreground-faint shrink-0" />}
                      <div className="min-w-0 flex-1 flex items-center gap-3">
                        <span className={`min-w-0 break-all text-sm font-mono leading-snug ${isActionable ? 'text-foreground' : 'text-foreground-subtle'}`}>
                          {file.name}
                        </span>
                        <span className="shrink-0 text-foreground-subtle text-xs font-mono tabular-nums">{formatSize(file.size)}</span>
                      </div>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
