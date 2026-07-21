import { Search, X, FolderOpen, Trash2, Info, Check } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { toast } from 'sonner';
import { Thumbnail } from './shared/Thumbnail';
import { formatRelativeTime } from './shared/relativeTime';
import { ConfirmModal } from './shared/ConfirmModal';

interface Capture {
  id: string;
  captureName: string;
  jsonPath: string;
  datasetName: string | null;
  thumbnailPath: string | null;
  taskCount: number;
  lastTaskAt: string | null;
  // Other fields from /api/captures intentionally ignored — Results is a
  // visual browser for outputs of actual pipeline runs, not a dataset
  // browser. Captures with no runs are filtered out before render.
}

interface Props {
  /** Click on a card → caller decides where to navigate (CaptureDetail). */
  onOpen?: (captureName: string, jsonPath: string) => void;
}

/**
 * Output-side capture browser. Renders one card per capture in a
 * responsive grid (1/2/3 columns by viewport). Each card is a glanceable
 * identifier — large thumbnail + name + dataset + run summary — that
 * navigates to the Outputs explorer when clicked.
 *
 * Distinct from the Captures table by design: visual rather than
 * tabular, single click rather than multi-button row, no input metadata
 * (sequences/cameras live in the Outputs explorer dropdowns).
 */
export function ResultsList({ onOpen }: Props) {
  const [captures, setCaptures] = useState<Capture[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  // Delete-capture confirmation (DB-only). A non-null value opens the modal;
  // `deleting` spins its confirm button while the DELETE is in flight.
  const [pendingDelete, setPendingDelete] = useState<Capture | null>(null);
  const [deleting, setDeleting] = useState(false);

  const refresh = () => {
    setError(null);
    fetch('/api/captures')
      .then(res => res.ok ? res.json() : Promise.reject(new Error('fetch failed')))
      // Hide captures that have no pipeline runs yet — Results is for
      // browsing actual run outputs, not the released dataset's
      // preprocessed inputs/ground truth.
      .then((data: Capture[]) => setCaptures(data.filter(c => c.taskCount > 0)))
      .catch(err => { console.error('Error fetching captures:', err); setError(String(err.message ?? err)); setCaptures([]); });
  };

  useEffect(() => { refresh(); }, []);

  /** Remove a capture's DB row (and cascading tasks/processes) without
   *  touching any files. The confirm modal spells this out so the user
   *  doesn't think their artifacts are about to disappear. */
  const handleDelete = (capture: Capture) => {
    setPendingDelete(capture);
  };

  /** Runs the DB-only DELETE for the capture the modal is confirming. The
   *  finally closes the modal on both success and handled failure; feedback
   *  is by toast. */
  const confirmDelete = async () => {
    if (!pendingDelete || deleting) return;
    const capture = pendingDelete;
    setDeleting(true);
    try {
      const res = await fetch(
        `/api/captures/db?path=${encodeURIComponent(capture.jsonPath)}`,
        { method: 'DELETE' },
      );
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        toast.error(err.error || `Failed to delete (${res.status})`);
        return;
      }
      toast.success(`Removed “${capture.captureName}” from the database (files on disk untouched).`);
      refresh();
    } catch (e) {
      console.error(e);
      toast.error('Failed to reach the backend.');
    } finally {
      setDeleting(false);
      setPendingDelete(null);
    }
  };

  const filtered = useMemo(() => {
    if (!captures) return null;
    const q = search.trim().toLowerCase();
    if (!q) return captures;
    return captures.filter(c =>
      c.captureName.toLowerCase().includes(q) ||
      (c.datasetName?.toLowerCase().includes(q) ?? false) ||
      (c.jsonPath?.toLowerCase().includes(q) ?? false)
    );
  }, [captures, search]);

  return (
    <div className="w-full max-w-7xl mx-auto px-6 py-8">
      <div className="flex items-end justify-between gap-4 mb-8 flex-wrap">
        <div>
          <h2 className="text-3xl text-foreground tracking-tight font-medium mb-1 flex items-center gap-2">
            <FolderOpen className="w-6 h-6 text-primary" />
            Results
          </h2>
          <p className="text-foreground-muted text-sm">
            Browse outputs. Pick a capture to walk through its produced files.
          </p>
        </div>
        {captures && captures.length > 0 && <SearchBox value={search} onChange={setSearch} />}
      </div>

      {error && (
        <div className="bg-status-failed-bg border border-status-failed/35 rounded-lg p-4 text-status-failed text-sm mb-6">
          {error}
        </div>
      )}

      {captures === null ? (
        <SkeletonGrid />
      ) : captures.length === 0 ? (
        <div className="text-center py-16 bg-surface-1 border border-border-subtle rounded-xl shadow-sm shadow-black/30">
          <div className="text-foreground text-base mb-1">No results yet</div>
          <div className="text-foreground-subtle text-sm">Submit a task from the Tasks tab to produce results.</div>
        </div>
      ) : filtered && filtered.length === 0 ? (
        <div className="text-center py-12 bg-surface-1 border border-border-subtle rounded-xl">
          <div className="text-foreground-subtle text-sm">
            No captures match “{search}”.{' '}
            <button onClick={() => setSearch('')} className="text-primary hover:underline ml-1">Clear</button>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {filtered!.map(capture => (
            <CaptureCard
              key={capture.id}
              capture={capture}
              onClick={() => onOpen?.(capture.captureName, capture.jsonPath)}
              onDelete={() => handleDelete(capture)}
            />
          ))}
        </div>
      )}

      <ConfirmModal
        open={!!pendingDelete}
        tone="danger"
        title={pendingDelete ? `Remove “${pendingDelete.captureName}” from the database?` : ''}
        confirmLabel="Remove from database"
        cancelLabel="Cancel"
        busy={deleting}
        onConfirm={confirmDelete}
        onCancel={() => { if (!deleting) setPendingDelete(null); }}
      >
        {pendingDelete && (
          <>
            <p>This is a database-only delete — no files on disk are touched.</p>
            <ul className="mt-2.5 space-y-1.5">
              <li className="flex items-start gap-2">
                <Check className="w-3.5 h-3.5 text-status-completed flex-shrink-0 mt-0.5" aria-hidden />
                <span>
                  The <code className="font-mono text-foreground-muted">capture.json</code> stays on disk:{' '}
                  <span className="font-mono text-foreground-subtle break-all">{pendingDelete.jsonPath}</span>
                </span>
              </li>
              <li className="flex items-start gap-2">
                <Check className="w-3.5 h-3.5 text-status-completed flex-shrink-0 mt-0.5" aria-hidden />
                <span>
                  Output files on disk stay — logs,{' '}
                  <code className="font-mono text-foreground-muted">.rrd</code>,{' '}
                  <code className="font-mono text-foreground-muted">.mp4</code>, etc.
                </span>
              </li>
              <li className="flex items-start gap-2">
                <X className="w-3.5 h-3.5 text-status-failed flex-shrink-0 mt-0.5" aria-hidden />
                <span>
                  This capture's {pendingDelete.taskCount} task row{pendingDelete.taskCount === 1 ? '' : 's'}{' '}
                  {pendingDelete.taskCount === 1 ? 'is' : 'are'} removed from the Tasks table, along with the
                  capture row itself.
                </span>
              </li>
            </ul>
            <div className="mt-3 flex items-start gap-2 rounded-md border border-border-subtle bg-surface-2/50 px-3 py-2">
              <Info className="w-3.5 h-3.5 text-foreground-faint flex-shrink-0 mt-0.5" aria-hidden />
              <span className="text-[12px] text-foreground-subtle leading-relaxed">
                You can re-add it later by submitting against the same{' '}
                <code className="font-mono text-foreground-muted">capture.json</code> — DONE sentinels under
                existing output dirs will still skip finished steps.
              </span>
            </div>
          </>
        )}
      </ConfirmModal>
    </div>
  );
}

// ====================================================================
// Card
// ====================================================================

function CaptureCard({ capture, onClick, onDelete }: { capture: Capture; onClick: () => void; onDelete: () => void }) {
  // taskCount is always > 0 here — captures with no runs are filtered
  // out before render.
  const runsLabel = `${capture.taskCount} run${capture.taskCount === 1 ? '' : 's'}`;
  const lastRun = capture.lastTaskAt ? formatRelativeTime(capture.lastTaskAt) : null;

  // The card itself is a `<div role=button>` rather than a real `<button>`
  // because nesting a Delete `<button>` inside a parent `<button>` is
  // invalid HTML and Firefox/Safari surface the inner click on the outer.
  // We get keyboard activation back via tabIndex + onKeyDown.
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onClick}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onClick(); } }}
      className="group relative bg-surface-1 border border-border-subtle rounded-xl overflow-hidden shadow-sm shadow-black/30 ring-1 ring-inset ring-white/[0.02] hover:border-primary/40 hover:ring-primary/20 focus-visible:border-primary/60 focus-visible:ring-primary/30 outline-none transition-all text-left flex flex-col cursor-pointer"
    >
      <div className="aspect-video w-full overflow-hidden bg-surface-2">
        <Thumbnail
          path={capture.thumbnailPath}
          alt={capture.captureName}
          className="w-full h-full"
          fit="cover"
        />
      </div>
      <div className="p-4 flex-1 flex flex-col gap-1">
        <div className="text-foreground text-base font-medium tracking-tight truncate" title={capture.captureName}>
          {capture.captureName}
        </div>
        {capture.datasetName && capture.datasetName !== capture.captureName && (
          <div className="text-foreground-subtle text-xs font-mono truncate">{capture.datasetName}</div>
        )}
        <div className="text-foreground-muted text-xs mt-0.5">
          {runsLabel}
          {lastRun && <span className="text-foreground-subtle"> · last {lastRun}</span>}
        </div>
      </div>

      {/* Hover-only delete affordance. `stopPropagation` on the click so
          deleting doesn't also navigate into the capture. */}
      <button
        type="button"
        onClick={(e) => { e.stopPropagation(); onDelete(); }}
        className="absolute top-2 right-2 inline-flex items-center justify-center w-7 h-7 rounded-md bg-surface-2/90 backdrop-blur border border-border text-foreground-muted opacity-0 group-hover:opacity-100 hover:text-status-failed hover:border-status-failed/55 focus:opacity-100 transition-all"
        title="Remove this capture from the database (files on disk are NOT deleted)."
        aria-label={`Remove ${capture.captureName} from the database`}
      >
        <Trash2 className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}

// ====================================================================
// Search + skeleton primitives
// ====================================================================

function SearchBox({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <div className="relative">
      <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-foreground-subtle pointer-events-none" />
      <input
        type="text"
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder="Filter by name, dataset, or path…"
        className="bg-surface-2 border border-border rounded-md pl-7 pr-7 py-1.5 text-foreground text-xs focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition-colors w-72 placeholder:text-foreground-faint"
      />
      {value && (
        <button
          onClick={() => onChange('')}
          className="absolute right-1.5 top-1/2 -translate-y-1/2 text-foreground-subtle hover:text-foreground transition-colors"
          aria-label="Clear search"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      )}
    </div>
  );
}

/** Card-shaped skeletons during the initial fetch. Same grid + aspect as
 *  the live cards so the layout doesn't jump on swap-in. */
function SkeletonGrid() {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
      {Array.from({ length: 6 }).map((_, idx) => (
        <div
          key={idx}
          className="bg-surface-1 border border-border-subtle rounded-xl overflow-hidden shadow-sm shadow-black/30"
        >
          <div className="aspect-video w-full bg-surface-2 animate-pulse" />
          <div className="p-4 space-y-2">
            <div className="h-4 w-32 rounded bg-surface-2 animate-pulse" />
            <div className="h-3 w-24 rounded bg-surface-2 animate-pulse" />
            <div className="h-3 w-40 rounded bg-surface-2 animate-pulse" />
          </div>
        </div>
      ))}
    </div>
  );
}
