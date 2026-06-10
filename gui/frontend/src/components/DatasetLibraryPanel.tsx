import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertCircle,
  Check,
  CheckCircle2,
  ChevronDown,
  Download,
  ExternalLink,
  Eye,
  EyeOff,
  KeyRound,
  Loader2,
  LogIn,
  RotateCcw,
  Search,
  X,
} from 'lucide-react';
import { DOMAIN_LABEL, REGISTER_URL, useCredentials, verifyMpiCredentials } from './CredentialsContext';

/*
 * Dataset library — second card on Home, below the body-models / weights
 * readiness panel. Single source of truth for "download MAMMA datasets"
 * inside the GUI; wraps the data/download_mamma_*.sh scripts behind a
 * single form-driven flow.
 *
 * Two principles drive the UX:
 *
 *   1. **Defaults work.** Each family has a usable default selection
 *      (a common-sense subset of groups + assets), so a user who just
 *      wants Bachata + meta + pred + CRF24 videos can hit Sign in &
 *      download without touching anything. Power users can de-select.
 *
 *   2. **Plan-before-creds.** As the selection changes, we POST it to
 *      /api/datasets/plan (no creds) and surface the live file count.
 *      Users see exactly how big the download is before they enter a
 *      password — fewer surprise multi-TB pulls.
 *
 * Credentials follow the same single-shot contract as the readiness
 * panel: one POST body to /api/datasets/start, used once by the
 * backend, never persisted. See gui/backend/dataset_downloader.py.
 */

// ───────────────────────── types ───────────────────────────────────────

interface Group { id: string; label: string; count: number }
interface AssetType { id: string; label: string; description: string }
interface VideoVariant { id: string; label: string; description: string }

interface FamilyInfo {
  id: string;
  label: string;
  description: string;
  script: string;
  content_groups: Group[];
  asset_types: AssetType[];
  video_variants: VideoVariant[];
  camera_kind: 'ioi32' | 'ioi16-or-32' | 'iphone4' | 'none';
  default_cameras: string[];
  max_ioi_cameras: number;
  notes: string;
}

interface Catalog { families: FamilyInfo[] }

interface Selection {
  groups: string[];
  asset_types: string[];
  video_variant: string;       // 'none' to skip videos
  cameras: string[];
  sequences: string[];         // empty = no filter; non-empty = explicit subset
}

interface PlanPreview { sfile: string; local: string }
interface PlanResponse { family: string; files_total: number; preview: PlanPreview[] }

interface Job {
  id: string;
  family: string;
  state: 'downloading' | 'done' | 'error' | 'cancelled';
  files_total: number;
  files_done: number;
  files_failed: number;
  current_label: string | null;
  current_bytes: number;
  current_total: number;
  bytes_done_total: number;
  error: string | null;
  started_at: number;
}

interface SequencesResponse {
  sequences: { path: string; group: string | null }[];
}

const POLL_MS = 1500;

// ───────────────────────── helpers ─────────────────────────────────────

function fmtBytes(b: number): string {
  if (!b) return '0';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0, n = b;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n >= 100 || i === 0 ? n.toFixed(0) : n.toFixed(1)} ${units[i]}`;
}

function fmtFiles(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(1).replace(/\.0$/, '')}k`;
  return String(n);
}

function defaultSelection(f: FamilyInfo): Selection {
  // Defaults bias toward a SMALL, deliberate first download. The user
  // can opt into more by checking additional groups. This avoids the
  // failure mode where someone clicks Download immediately and silently
  // queues ~4000 files because everything was pre-checked.
  const smallestGroup = f.content_groups.length === 0 ? [] : [
    f.content_groups.reduce((a, b) => (a.count <= b.count ? a : b)).id,
  ];
  const isMarkerless = f.id === 'dance' || f.id === 'multi' || f.id === 'iphone';
  const isEval = f.id === 'eval';
  const isSyn = f.id === 'syn';

  const allAssets = f.asset_types.map(a => a.id);
  let defaultAssets: string[] = [];
  if (isMarkerless) defaultAssets = ['meta', 'pred'].filter(x => allAssets.includes(x));
  else if (isEval) defaultAssets = ['gt'].filter(x => allAssets.includes(x));
  else defaultAssets = allAssets;  // syn has none

  // Pick the smallest video variant that isn't "none" — typically *_crf24
  // or *_light. This biases users toward a workable demo download.
  const nonNone = f.video_variants.filter(v => v.id !== 'none').map(v => v.id);
  const defaultVideo = (
    nonNone.find(v => v === 'videos_crf24' || v === 'videos_light')
    || nonNone[0]
    || 'none'
  );

  return {
    groups: smallestGroup,
    asset_types: defaultAssets,
    video_variant: isSyn ? 'none' : defaultVideo,
    cameras: f.default_cameras.slice(),
    sequences: [],   // no per-sequence filter by default
  };
}

// ───────────────────────── component ───────────────────────────────────

export function DatasetLibraryPanel() {
  const ctx = useCredentials();
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [activeFamily, setActiveFamily] = useState<string>('');
  const [selections, setSelections] = useState<Record<string, Selection>>({});
  const [planCounts, setPlanCounts] = useState<Record<string, number>>({});
  const [signInOpen, setSignInOpen] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [collapsed, setCollapsed] = useState<boolean | null>(null);
  const [startError, setStartError] = useState<string | null>(null);

  // Initial catalog load.
  useEffect(() => {
    let cancelled = false;
    fetch('/api/datasets/catalog')
      .then(r => r.json())
      .then((cat: Catalog) => {
        if (cancelled) return;
        setCatalog(cat);
        if (cat.families.length > 0) setActiveFamily(cat.families[0].id);
        const init: Record<string, Selection> = {};
        cat.families.forEach(f => { init[f.id] = defaultSelection(f); });
        setSelections(init);
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, []);

  // First-time collapse choice: start collapsed by default — the panel
  // is opt-in, not a chore-list. User opens it when they want a dataset.
  useEffect(() => {
    if (catalog && collapsed === null) setCollapsed(true);
  }, [catalog, collapsed]);

  // Whenever the active family's selection changes, refetch the plan
  // (file count + first few preview entries). Debounced ~200ms.
  useEffect(() => {
    if (!catalog || !activeFamily) return;
    const sel = selections[activeFamily];
    if (!sel) return;
    let cancelled = false;
    const id = window.setTimeout(async () => {
      try {
        const r = await fetch('/api/datasets/plan', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ family: activeFamily, ...sel }),
        });
        if (!cancelled && r.ok) {
          const p: PlanResponse = await r.json();
          setPlanCounts(prev => ({ ...prev, [activeFamily]: p.files_total }));
        }
      } catch { /* ignore */ }
    }, 250);
    return () => { cancelled = true; window.clearTimeout(id); };
  }, [catalog, activeFamily, selections]);

  // Poll the active job while it's running.
  useEffect(() => {
    if (!job || job.state !== 'downloading') return;
    const id = window.setInterval(async () => {
      try {
        const r = await fetch(`/api/datasets/job/${job.id}`);
        if (r.ok) {
          const next: Job = await r.json();
          setJob(next);
        }
      } catch { /* ignore */ }
    }, POLL_MS);
    return () => window.clearInterval(id);
  }, [job?.id, job?.state]);

  const setSelection = useCallback((familyId: string, mut: (s: Selection) => Selection) => {
    setSelections(prev => ({ ...prev, [familyId]: mut(prev[familyId]) }));
  }, []);

  const start = async (username: string, password: string) => {
    if (!catalog || !activeFamily) return;
    const sel = selections[activeFamily];
    if (!sel) return;
    setStartError(null);
    try {
      const r = await fetch('/api/datasets/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          family: activeFamily, ...sel, username, password,
        }),
      });
      if (!r.ok) {
        const detail = await r.json().catch(() => ({} as { error?: string }));
        setStartError(detail.error || `HTTP ${r.status}`);
        return;
      }
      const out = await r.json();
      setSignInOpen(false);
      setJob({
        id: out.job_id, family: activeFamily, state: 'downloading',
        files_total: out.files_total, files_done: 0, files_failed: 0,
        current_label: null, current_bytes: 0, current_total: 0,
        bytes_done_total: 0, error: null, started_at: Date.now() / 1000,
      });
    } catch (e) {
      setStartError('Network error');
    }
  };

  // Inline-form path only: verify the typed credentials before kicking off
  // a (multi-file) download, so a wrong password fails fast with a clear
  // message instead of queuing a job that fails file-by-file. On success,
  // store the creds in the shared context — the badge updates and later
  // downloads reuse them without re-verifying. The already-signed-in path
  // (onPrimary) calls `start` directly: those creds were already verified
  // at SignInCenter, so re-checking them would waste a request.
  const signInAndStart = async (username: string, password: string): Promise<boolean> => {
    setStartError(null);
    const res = await verifyMpiCredentials('mamma', username, password);
    if (!res.valid) {
      setStartError(res.detail);
      return false;
    }
    ctx.signIn('mamma', { username, password });
    await start(username, password);
    return true;
  };

  const cancel = async () => {
    if (!job) return;
    try {
      await fetch(`/api/datasets/job/${job.id}/cancel`, { method: 'POST' });
    } catch {}
  };

  if (!catalog) return <PanelSkeleton />;

  const active = catalog.families.find(f => f.id === activeFamily);
  const sel = active ? selections[active.id] : undefined;
  const planCount = planCounts[activeFamily] ?? null;
  const isCollapsed = collapsed === true;
  const mamma = ctx.creds.mamma;
  const sessionSignedIn = !!mamma;

  // Primary CTA action: if the user is signed in to MAMMA via the
  // top-of-Home SignInCenter, fire the download immediately with those
  // creds. Otherwise expand the inline sign-in form for this family.
  const onPrimary = () => {
    if (mamma) {
      start(mamma.username, mamma.password);
      return;
    }
    setSignInOpen(true);
  };

  return (
    <div className="bg-surface-1 border border-border-subtle rounded-xl shadow-sm shadow-black/30 ring-1 ring-inset ring-white/[0.02] overflow-hidden">
      {/* Header */}
      <button
        type="button"
        onClick={() => setCollapsed(c => !c)}
        aria-expanded={!isCollapsed}
        aria-controls="dataset-library-body"
        className={
          'w-full px-5 py-3 flex items-center justify-between gap-4 ' +
          'text-left hover:bg-surface-2/40 transition-colors ' +
          (isCollapsed ? '' : 'border-b border-border-subtle')
        }
      >
        <div className="flex items-baseline gap-3 min-w-0">
          <ChevronDown
            className={
              'w-3.5 h-3.5 text-foreground-faint transition-transform flex-shrink-0 ' +
              (isCollapsed ? '-rotate-90' : '')
            }
            aria-hidden
          />
          <h2 className="text-foreground text-[11px] uppercase tracking-[0.18em] font-medium">
            MAMMA Datasets
          </h2>
          <span className="text-foreground-faint text-[11px] hidden sm:inline truncate">
            paper data
          </span>
        </div>
        <div className="flex items-center gap-3 flex-shrink-0">
          {job && job.state === 'downloading' && (
            <span className="inline-flex items-center gap-1.5 text-[11px] text-status-running">
              <Loader2 className="w-3 h-3 animate-spin" />
              <span className="tabular-nums">{job.files_done} / {job.files_total}</span>
            </span>
          )}
          {job && job.state === 'done' && (
            <span className="inline-flex items-center gap-1.5 text-[11px] text-status-completed">
              <CheckCircle2 className="w-3 h-3" />
              <span className="tabular-nums">{job.files_done} of {job.files_total} ready</span>
            </span>
          )}
          <span className="text-foreground-faint text-[11px] tabular-nums">
            {catalog.families.length} families · MAMMA login
          </span>
        </div>
      </button>

      {!isCollapsed && active && sel && (
        <div id="dataset-library-body">
          {job ? (
            // A job in flight owns the whole body — no tabs, no form.
            // The user can dismiss back to the form view from inside
            // the ProgressPanel once the job ends.
            <ProgressPanel
              job={job}
              familyLabel={(catalog.families.find(f => f.id === job.family)?.label) || job.family}
              onCancel={cancel}
              onDismiss={() => { setJob(null); setStartError(null); }}
            />
          ) : (
            <>
              {/* Tab bar */}
              <div className="px-5 pt-3 flex gap-0 overflow-x-auto border-b border-border-subtle">
                {catalog.families.map(f => (
                  <button
                    key={f.id}
                    onClick={() => setActiveFamily(f.id)}
                    aria-selected={f.id === activeFamily}
                    role="tab"
                    className={
                      'relative px-3 py-2 text-[12px] font-medium whitespace-nowrap transition-colors ' +
                      (f.id === activeFamily
                        ? 'text-primary'
                        : 'text-foreground-muted hover:text-foreground')
                    }
                  >
                    {f.label}
                    {f.id === activeFamily && (
                      <span className="absolute left-2 right-2 -bottom-px h-[2px] bg-primary rounded-t" />
                    )}
                  </button>
                ))}
              </div>

              {/* Description */}
              <div className="px-5 pt-3 pb-2">
                <p className="text-foreground-muted text-[12.5px] leading-relaxed">
                  {active.description}
                </p>
                {active.notes && (
                  <p className="text-foreground-faint text-[11.5px] mt-1 leading-relaxed">
                    {active.notes}
                  </p>
                )}
              </div>

              <FamilyForm
                family={active}
                selection={sel}
                planCount={planCount}
                onChange={mut => setSelection(active.id, mut)}
                signInOpen={signInOpen && !sessionSignedIn}
                sessionSignedIn={sessionSignedIn}
                onOpenSignIn={onPrimary}
                onCancelSignIn={() => { setSignInOpen(false); setStartError(null); }}
                onSubmit={signInAndStart}
                startError={startError}
              />
            </>
          )}
        </div>
      )}
    </div>
  );
}

// ───────────────────────── family form ─────────────────────────────────

function FamilyForm({
  family, selection, planCount, onChange,
  signInOpen, sessionSignedIn, onOpenSignIn, onCancelSignIn, onSubmit,
  startError,
}: {
  family: FamilyInfo;
  selection: Selection;
  planCount: number | null;
  onChange: (mut: (s: Selection) => Selection) => void;
  signInOpen: boolean;
  sessionSignedIn: boolean;
  onOpenSignIn: () => void;
  onCancelSignIn: () => void;
  onSubmit: (u: string, p: string) => Promise<boolean | void> | boolean | void;
  startError: string | null;
}) {
  const toggleIn = (key: keyof Selection, id: string) => {
    onChange(s => {
      const arr = (s[key] as string[]).slice();
      const i = arr.indexOf(id);
      if (i >= 0) arr.splice(i, 1); else arr.push(id);
      return { ...s, [key]: arr };
    });
  };

  const setVideo = (id: string) => onChange(s => ({ ...s, video_variant: id }));

  const allCams = family.default_cameras;
  const camsSelected = selection.cameras;

  return (
    <div className="px-5 py-3 space-y-4">
      {/* Content groups */}
      <FieldGroup title="Content">
        <div className="flex flex-wrap gap-2">
          {family.content_groups.map(g => (
            <CheckboxPill
              key={g.id}
              checked={selection.groups.includes(g.id)}
              onChange={() => toggleIn('groups', g.id)}
              label={g.label}
              meta={g.count > 0 ? `${g.count}` : undefined}
            />
          ))}
        </div>
      </FieldGroup>

      {/* Asset types */}
      {family.asset_types.length > 0 && (
        <FieldGroup title="Asset types">
          <div className="flex flex-wrap gap-2">
            {family.asset_types.map(a => (
              <CheckboxPill
                key={a.id}
                checked={selection.asset_types.includes(a.id)}
                onChange={() => toggleIn('asset_types', a.id)}
                label={a.label}
                title={a.description}
              />
            ))}
          </div>
        </FieldGroup>
      )}

      {/* Video variants — radio */}
      {family.video_variants.length > 0 && (
        <FieldGroup title="Videos">
          <div className="flex flex-wrap gap-2">
            {family.video_variants.map(v => (
              <RadioPill
                key={v.id}
                checked={selection.video_variant === v.id}
                onChange={() => setVideo(v.id)}
                label={v.label}
                title={v.description}
              />
            ))}
          </div>
        </FieldGroup>
      )}

      {/* Cameras */}
      {family.camera_kind !== 'none' && (
        <CamerasField
          family={family}
          selected={camsSelected}
          allCams={allCams}
          onToggle={(cam) => toggleIn('cameras', cam)}
          onSetAll={() => onChange(s => ({ ...s, cameras: allCams.slice() }))}
          onClear={() => onChange(s => ({ ...s, cameras: [] }))}
        />
      )}

      {/* Sequences subset (optional) */}
      <SequencesField
        family={family}
        selectedSeqs={selection.sequences}
        activeGroups={selection.groups}
        onChange={(next) => onChange(s => ({ ...s, sequences: next }))}
      />

      {/* Footer: plan count + CTA. The file count is the load-bearing
          signal for "you're about to download this much" — render it
          large enough that nobody clicks Download without seeing it. */}
      <div className="pt-3 border-t border-border-subtle flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-baseline gap-2 tabular-nums">
          {planCount === null ? (
            <span className="text-[12px] text-foreground-faint">computing plan…</span>
          ) : planCount === 0 ? (
            <span className="text-[12px] text-status-pending">
              empty plan — pick at least one content group and one asset
            </span>
          ) : (
            <>
              <span className="text-foreground text-[20px] font-mono leading-none">
                {fmtFiles(planCount)}
              </span>
              <span className="text-foreground-muted text-[12px]">
                file{planCount === 1 ? '' : 's'} queued
              </span>
            </>
          )}
        </div>
        {!signInOpen && (
          <div className="flex items-center gap-3">
            {sessionSignedIn && (
              <span
                className="hidden sm:inline-flex items-center gap-1 text-[10.5px] text-foreground-faint"
                title="Will use the credentials you signed in with at the top of Home"
              >
                <KeyRound className="w-3 h-3" aria-hidden />
                uses MAMMA credentials
              </span>
            )}
            <button
              type="button"
              onClick={onOpenSignIn}
              disabled={planCount === 0 || planCount === null}
              className="inline-flex items-center gap-2 px-3 py-2 bg-primary text-primary-foreground text-[12px] font-medium rounded-md disabled:opacity-50 disabled:cursor-not-allowed shadow-sm shadow-black/30"
            >
              {sessionSignedIn ? (
                <>
                  <Download className="w-3.5 h-3.5" />
                  Download
                </>
              ) : (
                <>
                  <LogIn className="w-3.5 h-3.5" />
                  Sign in &amp; download
                </>
              )}
            </button>
          </div>
        )}
      </div>

      {/* Inline sign-in form — fallback path for users who haven't
          signed in at the top of Home. Shown only when no session
          credentials are available. */}
      {signInOpen && (
        <SignInForm
          accountLabel="MAMMA"
          registerUrl="https://mamma.is.tue.mpg.de/register.php"
          onCancel={onCancelSignIn}
          onSubmit={onSubmit}
          error={startError}
        />
      )}
    </div>
  );
}

// ───────────────────────── controls ────────────────────────────────────

function FieldGroup({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="text-[10.5px] uppercase tracking-[0.16em] text-foreground-faint mb-1.5">
        {title}
      </div>
      {children}
    </div>
  );
}

function CheckboxPill({
  checked, onChange, label, meta, title,
}: {
  checked: boolean; onChange: () => void;
  label: string; meta?: string; title?: string;
}) {
  return (
    <label
      title={title}
      className={
        'cursor-pointer inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border text-[12px] transition-colors ' +
        (checked
          ? 'border-primary/60 text-primary bg-primary-muted'
          : 'border-border text-foreground-muted hover:border-border-strong hover:text-foreground')
      }
    >
      <input
        type="checkbox"
        checked={checked}
        onChange={onChange}
        className="sr-only"
      />
      <span
        className={
          'inline-flex w-3 h-3 rounded-[3px] items-center justify-center transition-colors ' +
          (checked
            ? 'bg-primary text-primary-foreground'
            : 'border border-border-strong')
        }
        aria-hidden
      >
        {checked && <Check className="w-2.5 h-2.5" strokeWidth={3} />}
      </span>
      <span>{label}</span>
      {meta && <span className="text-[10.5px] text-foreground-faint tabular-nums">·&nbsp;{meta}</span>}
    </label>
  );
}

function RadioPill({
  checked, onChange, label, title,
}: { checked: boolean; onChange: () => void; label: string; title?: string }) {
  return (
    <label
      title={title}
      className={
        'cursor-pointer inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border text-[12px] transition-colors ' +
        (checked
          ? 'border-primary/60 text-primary bg-primary-muted'
          : 'border-border text-foreground-muted hover:border-border-strong hover:text-foreground')
      }
    >
      <input type="radio" checked={checked} onChange={onChange} className="sr-only" />
      <span
        className={
          'inline-flex w-3 h-3 rounded-full items-center justify-center transition-colors ' +
          (checked ? 'bg-primary' : 'border border-border-strong')
        }
        aria-hidden
      >
        {checked && <span className="w-1.5 h-1.5 rounded-full bg-primary-foreground" />}
      </span>
      <span>{label}</span>
    </label>
  );
}

// ───────────────────────── cameras ─────────────────────────────────────

function CamerasField({
  family, selected, allCams, onToggle, onSetAll, onClear,
}: {
  family: FamilyInfo;
  selected: string[];
  allCams: string[];
  onToggle: (cam: string) => void;
  onSetAll: () => void;
  onClear: () => void;
}) {
  const [open, setOpen] = useState(false);
  const selSet = new Set(selected);
  const isIPhone = family.camera_kind === 'iphone4';

  return (
    <FieldGroup title="Cameras">
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={() => setOpen(o => !o)}
          aria-expanded={open}
          className={
            'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border text-[12px] transition-colors ' +
            (open
              ? 'border-primary/60 text-primary bg-primary-muted'
              : 'border-border text-foreground-muted hover:border-border-strong hover:text-foreground')
          }
        >
          <span className="tabular-nums">{selected.length}</span>
          <span>/</span>
          <span className="tabular-nums">{allCams.length}</span>
          <span>selected</span>
          <ChevronDown
            className={'w-3 h-3 transition-transform ' + (open ? 'rotate-180' : '')}
            aria-hidden
          />
        </button>
        <button
          type="button"
          onClick={onSetAll}
          className="text-[10.5px] uppercase tracking-[0.14em] text-foreground-faint hover:text-foreground transition-colors"
        >
          all
        </button>
        <button
          type="button"
          onClick={onClear}
          className="text-[10.5px] uppercase tracking-[0.14em] text-foreground-faint hover:text-foreground transition-colors"
        >
          none
        </button>
        {family.camera_kind === 'ioi16-or-32' && (
          <span className="text-[10.5px] text-foreground-faint">
            16-cam sub-families auto-clip
          </span>
        )}
      </div>

      {open && (
        <div className={
          'mt-2 rounded-md border border-border-subtle bg-surface-2/60 p-2 grid gap-1 ' +
          (isIPhone ? 'grid-cols-4' : 'grid-cols-8')
        }>
          {allCams.map(cam => {
            const checked = selSet.has(cam);
            return (
              <button
                type="button"
                key={cam}
                onClick={() => onToggle(cam)}
                className={
                  'px-1.5 py-1 rounded text-[11px] font-mono tabular-nums transition-colors ' +
                  (checked
                    ? 'bg-primary-muted text-primary border border-primary/40'
                    : 'border border-border text-foreground-muted hover:text-foreground hover:border-border-strong')
                }
              >
                {cam}
              </button>
            );
          })}
        </div>
      )}
    </FieldGroup>
  );
}

// ───────────────────────── sequences picker ─────────────────────────────

function SequencesField({
  family, selectedSeqs, activeGroups, onChange,
}: {
  family: FamilyInfo;
  selectedSeqs: string[];
  activeGroups: string[];
  onChange: (next: string[]) => void;
}) {
  const [open, setOpen] = useState(false);
  const [sequences, setSequences] = useState<{ path: string; group: string | null }[] | null>(null);
  const [filter, setFilter] = useState('');

  // Lazy-load the sequence list on first expand.
  useEffect(() => {
    if (!open || sequences !== null) return;
    let cancelled = false;
    fetch(`/api/datasets/${family.id}/sequences`)
      .then(r => r.json() as Promise<SequencesResponse>)
      .then(d => { if (!cancelled) setSequences(d.sequences); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [open, family.id, sequences]);

  const groupSet = useMemo(() => new Set(activeGroups), [activeGroups]);

  const filtered = useMemo(() => {
    if (!sequences) return [];
    const f = filter.trim().toLowerCase();
    return sequences.filter(s => {
      if (s.group && !groupSet.has(s.group)) return false;
      if (f && !s.path.toLowerCase().includes(f)) return false;
      return true;
    });
  }, [sequences, filter, groupSet]);

  const selSet = useMemo(() => new Set(selectedSeqs), [selectedSeqs]);
  const showFilter = sequences && sequences.length > 30;

  return (
    <FieldGroup title="Sequences">
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={() => setOpen(o => !o)}
          aria-expanded={open}
          className={
            'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border text-[12px] transition-colors ' +
            (open
              ? 'border-primary/60 text-primary bg-primary-muted'
              : 'border-border text-foreground-muted hover:border-border-strong hover:text-foreground')
          }
        >
          {selectedSeqs.length === 0 ? (
            <span>All sequences in selected groups</span>
          ) : (
            <>
              <span className="tabular-nums">{selectedSeqs.length}</span>
              <span>selected</span>
            </>
          )}
          <ChevronDown
            className={'w-3 h-3 transition-transform ' + (open ? 'rotate-180' : '')}
            aria-hidden
          />
        </button>
        {selectedSeqs.length > 0 && (
          <button
            type="button"
            onClick={() => onChange([])}
            className="text-[10.5px] uppercase tracking-[0.14em] text-foreground-faint hover:text-foreground transition-colors"
          >
            clear filter
          </button>
        )}
      </div>

      {open && (
        <div className="mt-2 rounded-md border border-border-subtle bg-surface-2/60 p-2 space-y-2">
          {showFilter && (
            <div className="relative">
              <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3 h-3 text-foreground-faint" aria-hidden />
              <input
                type="text"
                value={filter}
                onChange={e => setFilter(e.target.value)}
                placeholder="search sequences…"
                className="w-full bg-surface-1 border border-border rounded pl-7 pr-2 py-1 text-[11.5px] text-foreground placeholder:text-foreground-faint focus:outline-none focus:border-primary/60"
              />
            </div>
          )}
          {!sequences ? (
            <div className="text-[11px] text-foreground-faint py-2 px-1">loading…</div>
          ) : filtered.length === 0 ? (
            <div className="text-[11px] text-foreground-faint py-2 px-1">no matches</div>
          ) : (
            <div className="max-h-48 overflow-y-auto space-y-0.5 pr-1">
              {filtered.map(s => {
                const checked = selSet.has(s.path);
                return (
                  <label
                    key={s.path}
                    className={
                      'flex items-center gap-2 px-2 py-1 rounded cursor-pointer text-[11.5px] font-mono ' +
                      (checked ? 'bg-primary-muted/60' : 'hover:bg-surface-3/40')
                    }
                  >
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={() => {
                        const next = selSet.has(s.path)
                          ? selectedSeqs.filter(x => x !== s.path)
                          : [...selectedSeqs, s.path];
                        onChange(next);
                      }}
                      className="sr-only"
                    />
                    <span
                      className={
                        'inline-flex w-3 h-3 rounded-[3px] items-center justify-center flex-shrink-0 ' +
                        (checked
                          ? 'bg-primary text-primary-foreground'
                          : 'border border-border-strong')
                      }
                      aria-hidden
                    >
                      {checked && <Check className="w-2.5 h-2.5" strokeWidth={3} />}
                    </span>
                    <span className="truncate text-foreground-muted">{s.path}</span>
                  </label>
                );
              })}
            </div>
          )}
          <div className="flex justify-end gap-2 pt-1 border-t border-border-subtle">
            <button
              type="button"
              onClick={() => filtered && onChange(filtered.map(s => s.path))}
              className="text-[10.5px] uppercase tracking-[0.14em] text-foreground-faint hover:text-foreground transition-colors"
            >
              add all matching
            </button>
            <button
              type="button"
              onClick={() => onChange([])}
              className="text-[10.5px] uppercase tracking-[0.14em] text-foreground-faint hover:text-foreground transition-colors"
            >
              clear filter
            </button>
          </div>
        </div>
      )}
    </FieldGroup>
  );
}

// ───────────────────────── sign-in form ─────────────────────────────────

function SignInForm({
  accountLabel, registerUrl, onCancel, onSubmit, error,
}: {
  accountLabel: string;
  registerUrl: string;
  onCancel: () => void;
  onSubmit: (u: string, p: string) => Promise<boolean | void> | boolean | void;
  error: string | null;
}) {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [show, setShow] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const canSubmit = username.trim().length > 0 && password.length > 0 && !submitting;

  return (
    <form
      onSubmit={async e => {
        e.preventDefault();
        if (!canSubmit) return;
        setSubmitting(true);
        try {
          // Keep the typed credentials on a failed verify (onSubmit returns
          // false) so the user can fix them; only wipe on success.
          const ok = await onSubmit(username.trim(), password);
          if (ok !== false) {
            setUsername('');
            setPassword('');
          }
        } finally {
          setSubmitting(false);
        }
      }}
      className="mt-2 rounded-md border border-border-subtle bg-surface-2/60 p-3"
    >
      <div className="flex items-center justify-between mb-2.5">
        <span className="text-[10.5px] uppercase tracking-[0.16em] text-foreground-muted">
          {accountLabel} account
        </span>
        <span className="text-[10.5px] text-foreground-faint flex items-center gap-1">
          <span className="w-1 h-1 rounded-full bg-status-completed inline-block" />
          not stored
        </span>
      </div>
      <div className="flex flex-col sm:flex-row gap-2">
        <input
          type="text"
          autoComplete="username"
          value={username}
          onChange={e => setUsername(e.target.value)}
          placeholder="username or email"
          className="flex-1 min-w-0 bg-surface-1 border border-border rounded px-2.5 py-1.5 text-[12px] text-foreground placeholder:text-foreground-faint focus:outline-none focus:border-primary/60"
        />
        <div className="flex-1 min-w-0 relative">
          <input
            type={show ? 'text' : 'password'}
            autoComplete="current-password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            placeholder="password"
            className="w-full bg-surface-1 border border-border rounded px-2.5 py-1.5 pr-8 text-[12px] text-foreground placeholder:text-foreground-faint focus:outline-none focus:border-primary/60"
          />
          <button
            type="button"
            onClick={() => setShow(s => !s)}
            tabIndex={-1}
            title={show ? 'Hide password' : 'Show password'}
            className="absolute right-1.5 top-1/2 -translate-y-1/2 text-foreground-faint hover:text-foreground p-1"
          >
            {show ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
          </button>
        </div>
        <div className="flex gap-2 flex-shrink-0">
          <button
            type="button"
            onClick={onCancel}
            className="px-2.5 py-1.5 text-[11px] rounded border border-border text-foreground-muted hover:text-foreground transition-colors"
          >
            cancel
          </button>
          <button
            type="submit"
            disabled={!canSubmit}
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 text-[11px] rounded bg-primary text-primary-foreground font-medium disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {submitting ? (
              <>
                <Loader2 className="w-3 h-3 animate-spin" />
                submitting…
              </>
            ) : (
              <>
                <Download className="w-3 h-3" />
                download
              </>
            )}
          </button>
        </div>
      </div>
      {error && (
        <div className="mt-2 flex items-start gap-1.5 text-[11px] text-status-failed">
          <AlertCircle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}
      <div className="mt-2 text-[10.5px] text-foreground-faint leading-relaxed">
        Sent once over HTTPS to <span className="font-mono">download.is.tue.mpg.de</span>.
        The backend uses them to compose a single POST request, then drops the
        values from memory — no log line, no disk write.
        {' '}
        Need an account?{' '}
        <a
          href={registerUrl}
          target="_blank"
          rel="noreferrer"
          className="text-primary hover:underline inline-flex items-center gap-0.5"
        >
          Register
          <ExternalLink className="w-3 h-3" />
        </a>.
        {' '}
        Tip: sign in to <span className="text-foreground-muted">{accountLabel}</span> at the top
        of Home to skip this step on every dataset.
      </div>
    </form>
  );
}

// ───────────────────────── progress panel ───────────────────────────────

function ProgressPanel({
  job, familyLabel, onCancel, onDismiss,
}: {
  job: Job;
  familyLabel: string;
  onCancel: () => void;
  onDismiss: () => void;
}) {
  const overall = job.files_total > 0
    ? Math.min(100, (job.files_done / job.files_total) * 100)
    : 0;
  const currentPct = job.current_total > 0
    ? Math.min(100, (job.current_bytes / job.current_total) * 100)
    : 0;
  const running = job.state === 'downloading';
  const [cancelling, setCancelling] = useState(false);
  const onCancelClick = () => {
    if (cancelling) return;
    setCancelling(true);
    onCancel();
  };

  return (
    <div className="px-5 py-3 space-y-3">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 min-w-0">
          {job.state === 'downloading' && <Loader2 className="w-4 h-4 text-status-running animate-spin flex-shrink-0" />}
          {job.state === 'done' && job.files_failed === 0 && <CheckCircle2 className="w-4 h-4 text-status-completed flex-shrink-0" />}
          {job.state === 'done' && job.files_failed > 0 && <AlertCircle className="w-4 h-4 text-status-pending flex-shrink-0" />}
          {job.state === 'error' && <AlertCircle className="w-4 h-4 text-status-failed flex-shrink-0" />}
          {job.state === 'cancelled' && <X className="w-4 h-4 text-foreground-muted flex-shrink-0" />}
          <span className="text-[13px] font-medium text-foreground truncate">
            {job.state === 'downloading' && `Downloading ${familyLabel}`}
            {job.state === 'done' && job.files_failed === 0 && `${familyLabel} — download complete`}
            {job.state === 'done' && job.files_failed > 0  && `${familyLabel} — finished with errors`}
            {job.state === 'error'       && `${familyLabel} — download failed`}
            {job.state === 'cancelled'   && `${familyLabel} — cancelled`}
          </span>
        </div>
        <div className="flex items-center gap-2 flex-shrink-0">
          {running ? (
            <button
              type="button"
              onClick={onCancelClick}
              disabled={cancelling}
              className="inline-flex items-center gap-1.5 px-2.5 py-1 text-[11px] rounded border border-border text-foreground-muted hover:text-foreground hover:border-status-failed/40 disabled:opacity-60 disabled:cursor-not-allowed transition-colors"
            >
              {cancelling && <Loader2 className="w-3 h-3 animate-spin" />}
              {cancelling ? 'cancelling…' : 'Cancel'}
            </button>
          ) : (
            <button
              type="button"
              onClick={onDismiss}
              className="inline-flex items-center gap-1.5 px-2.5 py-1 text-[11px] rounded border border-border text-foreground-muted hover:text-foreground transition-colors"
            >
              <RotateCcw className="w-3 h-3" />
              new download
            </button>
          )}
        </div>
      </div>

      {/* Overall bar */}
      <div className="space-y-1.5">
        <div className="h-1.5 bg-surface-2 rounded-full overflow-hidden">
          <div
            className={
              'h-full transition-[width] duration-300 ' +
              (job.state === 'error' ? 'bg-status-failed'
                : job.state === 'cancelled' ? 'bg-foreground-faint'
                : job.state === 'done' && job.files_failed === 0 ? 'bg-status-completed'
                : job.state === 'done' ? 'bg-status-pending'
                : 'bg-status-running')
            }
            style={{ width: `${overall}%` }}
          />
        </div>
        <div className="flex items-center justify-between text-[11px] font-mono tabular-nums text-foreground-muted">
          <span>
            <span className="text-foreground">{job.files_done}</span>
            {' / '}
            <span>{job.files_total}</span>
            {' files'}
            {job.files_failed > 0 && (
              <span className="text-status-failed"> · {job.files_failed} failed</span>
            )}
          </span>
          <span>{fmtBytes(job.bytes_done_total)}</span>
        </div>
      </div>

      {/* Current file */}
      {running && job.current_label && (
        <div className="space-y-1">
          <div className="text-[11px] text-foreground-faint font-mono truncate" title={job.current_label}>
            {job.current_label}
          </div>
          <div className="h-[2px] bg-surface-2 rounded-full overflow-hidden">
            <div
              className="h-full bg-status-running/60 transition-[width] duration-300"
              style={{ width: job.current_total > 0 ? `${currentPct}%` : '20%' }}
            />
          </div>
          <div className="text-[10.5px] text-foreground-faint font-mono tabular-nums">
            {fmtBytes(job.current_bytes)}
            {job.current_total > 0 ? ` / ${fmtBytes(job.current_total)}` : ''}
          </div>
        </div>
      )}

      {job.error && (
        <div className="flex items-start gap-2 text-[11px] text-foreground-muted bg-status-failed-bg/40 border border-status-failed/20 rounded p-2">
          <AlertCircle className="w-3.5 h-3.5 text-status-failed flex-shrink-0 mt-0.5" />
          <span className="break-words">
            {job.error}
            {' '}
            <a
              href={REGISTER_URL.mamma.replace('register.php', '')}
              target="_blank"
              rel="noreferrer"
              className="text-primary hover:underline inline-flex items-center gap-0.5 whitespace-nowrap"
            >
              Open the {DOMAIN_LABEL.mamma} website
              <ExternalLink className="w-3 h-3" />
            </a>
          </span>
        </div>
      )}
    </div>
  );
}

// ───────────────────────── skeleton ─────────────────────────────────────

function PanelSkeleton() {
  return (
    <div className="bg-surface-1 border border-border-subtle rounded-xl p-5 shadow-sm shadow-black/30">
      <div className="h-3 w-40 rounded-sm animate-shimmer mb-3" />
      <div className="h-5 rounded-sm animate-shimmer" />
    </div>
  );
}

export default DatasetLibraryPanel;
