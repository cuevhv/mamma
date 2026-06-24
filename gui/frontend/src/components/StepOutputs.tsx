import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { ArrowLeft, Folder, FileVideo, FileImage, File, Sparkles, Globe, ChevronDown, ChevronRight, FileCode2, Database, FileJson, Sheet, FileText, Users, ScrollText } from 'lucide-react';
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
  /** `siblings` is the ordered list of image relPaths in the same folder,
   *  so the lightbox can offer prev/next within the directory. */
  onPlayImage: (relPath: string, siblings: string[]) => void;
  onOpenRrdBrowser: (relPath: string, name: string) => void;
  onOpenRrdNative: (relPath: string, fresh?: boolean) => void;
  onOpenHtml: (relPath: string, name: string) => void;
  onOpenNpz: (relPath: string, name: string) => void;
  /** JSON, CSV/TSV, YAML, plain text — all routed through the shared
   *  FileViewerModal which renders type-aware (pretty-printed JSON,
   *  table CSV, monospace text). Also used for the .out/.err log buttons. */
  onOpenText: (relPath: string, name: string) => void;
  /** Absolute paths to this step's stdout/stderr logs for the current
   *  task+sequence (live in jobs_log_dir, not the output dir). When present,
   *  shown as `out`/`err` buttons in the header. */
  outLog?: string | null;
  errLog?: string | null;
  /** Extra buttons rendered in the header action area (left of out/err).
   *  Used to attach run-level config shortcuts (preset / task JSON) to the
   *  entry step. */
  headerExtras?: ReactNode;
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

// Per-person mask images are named `<...>_<frame>_<person>.<ext>` (e.g.
// `mask_0325_04.png`). Two trailing underscore-separated numeric groups —
// the last is the 1-based person id. We key the person facet off this so the
// pattern is generic to any step that emits per-person frames, not just masks.
const PERSON_FILE_RE = /_(\d+)_(\d+)\.(?:png|jpe?g|webp|gif|bmp)$/i;

/** Returns the person id (as it appears in the filename, e.g. "04") or null. */
function personIdOf(name: string): string | null {
  const m = name.match(PERSON_FILE_RE);
  return m ? m[2] : null;
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
  outLog,
  errLog,
  headerExtras,
}: StepOutputsProps) {
  const [open, setOpen] = useState(defaultOpen);
  const [relPath, setRelPath] = useState(baseRelPath);
  const [entries, setEntries] = useState<{ dirs: DirEntry[]; files: FileEntry[] } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  /** Active person facet ("04") or null for "all". Reset on folder change. */
  const [personFilter, setPersonFilter] = useState<string | null>(null);
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

  // Drop any active person filter when navigating to a different folder.
  useEffect(() => { setPersonFilter(null); }, [relPath]);

  // Person facets present in this folder: [["01", count], ["02", count], ...]
  // sorted by id. Drives the filter bar; only shown when ≥2 persons exist.
  const personFacets = useMemo(() => {
    if (!entries) return [] as [string, number][];
    const counts = new Map<string, number>();
    for (const f of entries.files) {
      const pid = personIdOf(f.name);
      if (pid) counts.set(pid, (counts.get(pid) ?? 0) + 1);
    }
    return [...counts.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  }, [entries]);

  // Files actually rendered: when a person is selected, hide other persons'
  // mask images but keep non-mask files (configs, summaries, etc.) visible.
  const visibleFiles = useMemo(() => {
    if (!entries) return [] as FileEntry[];
    if (!personFilter) return entries.files;
    return entries.files.filter(f => {
      const pid = personIdOf(f.name);
      return pid === null || pid === personFilter;
    });
  }, [entries, personFilter]);

  return (
    <div ref={sectionRef} className="bg-background border border-border-subtle rounded-lg overflow-hidden flex flex-col">
      {/* Step header — the title area toggles collapse/expand; the .out/.err
          log buttons sit alongside it (kept outside the toggle button since
          buttons can't nest). */}
      <div className="flex items-stretch bg-surface-1 border-b border-border-subtle">
        <button
          type="button"
          onClick={() => setOpen(o => !o)}
          className="flex-1 min-w-0 flex items-center gap-2 px-3 py-2 text-left hover:bg-surface-2/60 transition-colors"
        >
          {open ? <ChevronDown className="w-4 h-4 text-foreground-muted shrink-0" /> : <ChevronRight className="w-4 h-4 text-foreground-muted shrink-0" />}
          <span className="text-foreground text-sm font-medium">{stepLabel(step)}</span>
          <span className="text-foreground-faint text-xs font-mono">({step})</span>
          {open && fileCount !== null && (
            <span className="ml-auto text-foreground-subtle text-xs">
              {fileCount === 0 ? 'empty' : `${fileCount} item${fileCount === 1 ? '' : 's'}`}
            </span>
          )}
        </button>
        {(headerExtras || outLog || errLog) && (
          <div className="flex items-center gap-1 pl-1 pr-2 shrink-0">
            {headerExtras}
            {outLog && (
              <button
                type="button"
                onClick={() => onOpenText(outLog, outLog.split('/').pop() || 'stdout.out')}
                title="View stdout log (.out)"
                className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-mono text-foreground-muted hover:text-foreground hover:bg-surface-3 ring-1 ring-inset ring-border transition-colors"
              >
                <ScrollText className="w-3 h-3" /> out
              </button>
            )}
            {errLog && (
              <button
                type="button"
                onClick={() => onOpenText(errLog, errLog.split('/').pop() || 'stderr.err')}
                title="View stderr log (.err)"
                className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-mono text-foreground-muted hover:text-foreground hover:bg-surface-3 ring-1 ring-inset ring-border transition-colors"
              >
                <ScrollText className="w-3 h-3" /> err
              </button>
            )}
          </div>
        )}
      </div>

      {open && (
        // Fixed-height open region so every step card is the same size in the
        // Results grid; the file listing flexes to fill whatever space the
        // breadcrumbs (and optional person-filter bar) leave.
        <div className="flex flex-col h-[300px]">
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

          {/* Person filter — appears only when the folder holds per-person
              mask frames (≥2 persons). Scopes the listing (and therefore the
              lightbox prev/next) to a single person so masks are easy to scan. */}
          {personFacets.length >= 2 && (
            <div className="flex items-center gap-1.5 px-3 py-2 bg-surface-1/40 border-b border-border-subtle overflow-x-auto">
              <Users className="w-3.5 h-3.5 text-foreground-faint shrink-0 mr-0.5" />
              <FacetPill
                label="All"
                active={personFilter === null}
                onClick={() => setPersonFilter(null)}
              />
              {personFacets.map(([pid, count]) => (
                <FacetPill
                  key={pid}
                  label={`Person ${pid}`}
                  count={count}
                  active={personFilter === pid}
                  onClick={() => setPersonFilter(p => (p === pid ? null : pid))}
                />
              ))}
            </div>
          )}

          {/* File listing — capped height so 5 stacked steps stay scannable
              without forcing the whole page to scroll past one giant step. */}
          <div className="flex-1 min-h-0 overflow-y-auto">
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
                {visibleFiles.map(file => {
                  const lower = file.name.toLowerCase();
                  const isMP4 = lower.endsWith('.mp4');
                  const isImage = isImageFile(file.name);
                  const isRrd = lower.endsWith('.rrd');
                  const isHtml = lower.endsWith('.html') || lower.endsWith('.htm');
                  const isNpz = lower.endsWith('.npz');
                  const isJson = lower.endsWith('.json') || lower.endsWith('.jsonl');
                  const isCsv = lower.endsWith('.csv') || lower.endsWith('.tsv');
                  const isYaml = lower.endsWith('.yaml') || lower.endsWith('.yml');
                  const isLog = lower.endsWith('.out') || lower.endsWith('.err') || lower.endsWith('.log') || lower.endsWith('.txt');
                  const isText = isJson || isCsv || isYaml || isLog;
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
                        else if (isImage) onPlayImage(
                          filePath,
                          visibleFiles
                            .filter(f => isImageFile(f.name))
                            .map(f => `${relPath}/${f.name}`),
                        );
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
                      {!isMP4 && !isImage && !isHtml && !isNpz && !isJson && !isCsv && !isYaml && isLog && <ScrollText className="w-4 h-4 text-foreground-muted shrink-0" />}
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
        </div>
      )}
    </div>
  );
}

/** A single pill in the person-filter bar. Active pill uses the primary
 *  accent; the optional count sits in a subtle inset badge. */
function FacetPill({
  label, count, active, onClick,
}: {
  label: string;
  count?: number;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`inline-flex items-center gap-1.5 shrink-0 rounded-full px-2.5 py-1 text-xs font-medium transition-colors ${
        active
          ? 'bg-primary text-primary-foreground ring-1 ring-inset ring-white/10'
          : 'bg-surface-2 text-foreground-muted ring-1 ring-inset ring-border hover:bg-surface-3 hover:text-foreground'
      }`}
    >
      <span>{label}</span>
      {count !== undefined && (
        <span className={`tabular-nums rounded px-1 text-[10px] leading-tight ${
          active ? 'bg-white/15 text-primary-foreground' : 'bg-surface-1 text-foreground-faint'
        }`}>
          {count}
        </span>
      )}
    </button>
  );
}
