import { useCallback, useEffect, useRef, useState } from 'react';
import {
  AlertCircle,
  Check,
  ChevronDown,
  Download,
  ExternalLink,
  Eye,
  EyeOff,
  KeyRound,
  Loader2,
  LogIn,
  RotateCcw,
  ScrollText,
  X,
} from 'lucide-react';
import { domainFromAccountLabel, useCredentials, verifyMpiCredentials } from './CredentialsContext';

/*
 * Data readiness panel — companion to ExampleDataPanel on the Home page.
 *
 * Renders a compact "pre-flight" board over the assets the MAMMA
 * pipeline expects under data/. Each row shows the on-disk state and
 * the appropriate acquisition affordance:
 *
 *   public  → one-click download (no credentials)
 *   mpi     → expand an inline sign-in form, POST creds once, stream
 *             the download into place. Credentials are never persisted
 *             — see gui/backend/data_readiness.py for the security
 *             contract — and the form's helper text makes that visible
 *             to the user too.
 *   manual  → expand a short inline docs block (the file is not
 *             automatable to fetch).
 *
 * Polling is scoped: the panel only polls /api/data/readiness/job/<id>
 * while at least one job is in flight, and re-probes /status when a
 * job lands so the row flips to "ready" without a refresh.
 */

interface AssetSourcePublic   { kind: 'public' }
interface AssetSourceGDrive   { kind: 'gdrive'; link: string }
interface AssetSourceMpi      {
  kind: 'mpi';
  account_label: string;
  register_url: string;
  extract: boolean;
}
interface AssetSourceManual   {
  kind: 'manual';
  steps: string[];
  link: string | null;
  link_label: string | null;
}
interface AssetSourceHfHub    {
  kind: 'hf_hub';
  model_id: string;
  account_label: string;
  register_url: string;
  gated: boolean;
  steps: string[];
}
type AssetSource =
  | AssetSourcePublic
  | AssetSourceGDrive
  | AssetSourceMpi
  | AssetSourceManual
  | AssetSourceHfHub;

interface Asset {
  id: string;
  label: string;
  rel_path: string;
  fs_kind: 'file' | 'dir';
  purpose: string;
  size_hint_mb: number;
  optional: boolean;
  /** Visual group key from the backend. Empty string = default (top)
   *  section, no header. See SECTION_DEFS for the known groups. */
  group?: string;
  present: boolean;
  size_bytes: number;
  source: AssetSource;
}

/** Section ordering + per-group title for the readiness panel. Sections
 *  whose key isn't returned by the backend simply don't render. Keep
 *  the order here; the backend doesn't dictate it. */
const SECTION_DEFS: { group: string; title: string }[] = [
  { group: '' /* default — catches any unclassified asset */, title: '' },
  { group: 'detectors', title: 'Bounding box detectors' },
  { group: 'segmenters', title: 'Mask segmenters' },
  { group: 'body_models', title: 'Body models' },
  { group: 'mamma_assets', title: 'MAMMA assets' },
  { group: 'training', title: 'Training only' },
];

interface StatusResponse {
  items: Asset[];
  ready: number;
  total: number;
}

interface JobResponse {
  id: string;
  asset_id: string;
  state: 'downloading' | 'extracting' | 'ready' | 'error';
  bytes_downloaded: number;
  bytes_total: number;
  error: string | null;
  started_at: number;
}

const POLL_MS = 1500;

// ───────────────────────── helpers ─────────────────────────────────────

function formatBytes(b: number): string {
  if (!b) return '–';
  const units = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  let n = b;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  const fixed = n >= 100 || i === 0 ? n.toFixed(0) : n.toFixed(1);
  return `${fixed} ${units[i]}`;
}

function formatHint(mb: number): string {
  if (mb >= 1024) return `~${(mb / 1024).toFixed(1)} GB`;
  return `~${mb} MB`;
}

function shortPath(p: string): string {
  // Drop a leading "data/" since the whole panel is about data/* assets.
  return p.startsWith('data/') ? p.slice(5) : p;
}

function makeRunningJob(jobId: string, assetId: string): JobResponse {
  return {
    id: jobId, asset_id: assetId, state: 'downloading',
    bytes_downloaded: 0, bytes_total: 0,
    error: null, started_at: Date.now() / 1000,
  };
}

function makeErrorJob(assetId: string, error: string): JobResponse {
  return {
    id: 'inline-error', asset_id: assetId, state: 'error',
    bytes_downloaded: 0, bytes_total: 0,
    error, started_at: Date.now() / 1000,
  };
}

// ───────────────────────── component ───────────────────────────────────

export function DataReadinessPanel() {
  const ctx = useCredentials();
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [jobs, setJobs] = useState<Record<string, JobResponse>>({});
  const [expanded, setExpanded] = useState<string | null>(null);
  // null = not initialised yet; once status arrives we set true/false based
  // on whether every required asset is on disk (collapsed if all-ready,
  // expanded if anything's missing). After that, only the chevron toggles it.
  const [collapsed, setCollapsed] = useState<boolean | null>(null);
  const initialisedRef = useRef(false);
  const pollRef = useRef<number | null>(null);
  const inFlightRef = useRef<Set<string>>(new Set());

  const fetchStatus = useCallback(async () => {
    try {
      const r = await fetch('/api/data/readiness/status');
      if (r.ok) setStatus(await r.json());
    } catch {
      // Network blip — keep the previous snapshot.
    }
  }, []);

  useEffect(() => { fetchStatus(); }, [fetchStatus]);

  // First time status arrives, choose an initial collapsed state.
  useEffect(() => {
    if (status && !initialisedRef.current) {
      initialisedRef.current = true;
      setCollapsed(status.ready === status.total);
    }
  }, [status]);

  // One interval shared across all live jobs.
  useEffect(() => {
    const anyLive = Object.values(jobs).some(
      j => j.state === 'downloading' || j.state === 'extracting',
    );
    if (!anyLive) {
      if (pollRef.current !== null) {
        window.clearInterval(pollRef.current);
        pollRef.current = null;
      }
      return;
    }
    if (pollRef.current !== null) return;
    pollRef.current = window.setInterval(async () => {
      const ids = Array.from(inFlightRef.current);
      const results = await Promise.all(
        ids.map(id => fetch(`/api/data/readiness/job/${id}`)
          .then(r => (r.ok ? (r.json() as Promise<JobResponse>) : null))
          .catch(() => null)),
      );
      const arrived = results.filter((r): r is JobResponse => !!r);
      arrived.forEach(r => {
        if (r.state === 'ready' || r.state === 'error') {
          inFlightRef.current.delete(r.id);
        }
      });
      if (arrived.length > 0) {
        setJobs(prev => {
          const next = { ...prev };
          arrived.forEach(r => { next[r.asset_id] = r; });
          return next;
        });
      }
      if (arrived.some(r => r.state === 'ready')) fetchStatus();
    }, POLL_MS);
  }, [jobs, fetchStatus]);

  useEffect(() => () => {
    if (pollRef.current !== null) window.clearInterval(pollRef.current);
  }, []);

  const startPublic = async (asset: Asset) => {
    setCollapsed(false);
    const r = await fetch('/api/data/readiness/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: asset.id }),
    });
    if (!r.ok) {
      const detail = await r.json().catch(() => ({} as { error?: string }));
      setJobs(prev => ({
        ...prev,
        [asset.id]: makeErrorJob(asset.id, detail.error || `HTTP ${r.status}`),
      }));
      return;
    }
    const { job_id } = await r.json();
    inFlightRef.current.add(job_id);
    setJobs(prev => ({
      ...prev,
      [asset.id]: makeRunningJob(job_id, asset.id),
    }));
  };

  const startMpi = async (asset: Asset, username: string, password: string) => {
    setCollapsed(false);
    const r = await fetch('/api/data/readiness/start-mpi', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: asset.id, username, password }),
    });
    // `username` and `password` are about to fall out of scope; the request
    // body has already been sent on the wire.
    if (!r.ok) {
      const detail = await r.json().catch(() => ({} as { error?: string }));
      setJobs(prev => ({
        ...prev,
        [asset.id]: makeErrorJob(asset.id, detail.error || `HTTP ${r.status}`),
      }));
      return;
    }
    const { job_id } = await r.json();
    inFlightRef.current.add(job_id);
    setExpanded(null);
    setJobs(prev => ({
      ...prev,
      [asset.id]: makeRunningJob(job_id, asset.id),
    }));
  };

  // True iff the asset is gated by an MPI account that the user has
  // already signed in to at the top-of-Home SignInCenter. When true,
  // clicking the action chip fires the download directly instead of
  // expanding the inline form.
  const isAssetSignedIn = (asset: Asset): boolean => {
    if (asset.source.kind !== 'mpi') return false;
    const d = domainFromAccountLabel(asset.source.account_label);
    return !!(d && ctx.creds[d]);
  };

  // Primary action for an MPI asset's chip: auto-fire when signed in,
  // otherwise fall through to the inline-form toggle.
  const onMpiAction = (asset: Asset) => {
    if (asset.source.kind !== 'mpi') return;
    const d = domainFromAccountLabel(asset.source.account_label);
    if (d && ctx.creds[d]) {
      const c = ctx.creds[d]!;
      startMpi(asset, c.username, c.password);
      return;
    }
    onToggleExpand(asset.id);
  };

  // When the user clicks a sign-in / steps chip we also clear any
  // previous error job for that asset, so the error block doesn't
  // hang around above the freshly-opened form.
  const onToggleExpand = (assetId: string) => {
    const isOpen = expanded === assetId;
    setExpanded(isOpen ? null : assetId);
    if (!isOpen) {
      setCollapsed(false);
      setJobs(prev => {
        if (!prev[assetId] || prev[assetId].state !== 'error') return prev;
        const next = { ...prev };
        delete next[assetId];
        return next;
      });
    }
  };

  if (!status) return <PanelSkeleton />;

  // Group items by their `group` key, preserving the order in SECTION_DEFS.
  // Unknown groups (forward-compat with future backend additions) fall
  // through to the default top section so nothing silently disappears.
  const knownGroups = new Set(SECTION_DEFS.map(s => s.group));
  const sections = SECTION_DEFS
    .map(def => ({
      ...def,
      items: status.items.filter(a => {
        const g = a.group ?? '';
        return def.group === '' ? !knownGroups.has(g) || g === '' : g === def.group;
      }),
    }))
    .filter(s => s.items.length > 0);

  const isCollapsed = collapsed === true;
  const allReady = status.ready === status.total;

  return (
    <div className="bg-surface-1 border border-border-subtle rounded-xl shadow-sm shadow-black/30 ring-1 ring-inset ring-white/[0.02] overflow-hidden">
      {/* Header — always visible. Doubles as the collapse toggle. */}
      <button
        type="button"
        onClick={() => setCollapsed(c => !c)}
        aria-expanded={!isCollapsed}
        aria-controls="data-readiness-list"
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
            Body Models &amp; Weights
          </h2>
          <span className="text-foreground-faint text-[12px] hidden sm:inline truncate">
            pipeline assets
          </span>
        </div>
        <div className="flex items-center gap-3 flex-shrink-0">
          <span className="text-[11.5px] font-mono tabular-nums">
            <span className={allReady ? 'text-status-completed' : 'text-foreground'}>
              {status.ready}
            </span>
            <span className="text-foreground-faint"> / {status.total} ready</span>
          </span>
          <SegmentedBar items={status.items} />
          <span
            role="button"
            tabIndex={0}
            onClick={e => { e.stopPropagation(); fetchStatus(); }}
            onKeyDown={e => {
              if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fetchStatus(); }
            }}
            title="Re-probe disk"
            className="text-foreground-faint hover:text-foreground transition-colors p-0.5 cursor-pointer"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </span>
        </div>
      </button>

      {/* Rows — hidden when collapsed. Items render in named groups
          (SECTION_DEFS) so a "Mask segmenters" / "Training only" header
          appears above the relevant subset instead of a flat "Optional"
          catch-all. */}
      {!isCollapsed && (
        <>
          <ul id="data-readiness-list" className="divide-y divide-border-subtle">
            {sections.map(section => (
              <li key={section.group || '__default'} className="contents">
                {section.title && (
                  <div className="px-5 py-2 bg-surface-1/40 border-t border-border-subtle">
                    <div className="text-foreground-subtle text-[10.5px] uppercase tracking-[0.16em] font-medium">
                      {section.title}
                    </div>
                  </div>
                )}
                <ul className="divide-y divide-border-subtle">
                  {section.items.map(asset => (
                    <AssetRow
                      key={asset.id}
                      asset={asset}
                      job={jobs[asset.id] || null}
                      isExpanded={expanded === asset.id}
                      signedIn={isAssetSignedIn(asset)}
                      onToggleExpand={() => onToggleExpand(asset.id)}
                      onStartPublic={() => startPublic(asset)}
                      onSubmitMpi={(u, p) => startMpi(asset, u, p)}
                      onMpiAction={() => onMpiAction(asset)}
                    />
                  ))}
                </ul>
              </li>
            ))}
          </ul>
          {/* Override hint — points users at the existing .env.local
              mechanism without duplicating it in the GUI. See
              docs/INSTALL.md §"Customising paths". */}
          <div className="px-5 py-2.5 border-t border-border-subtle bg-surface-1/30">
            <div className="text-foreground-faint text-[11px] leading-relaxed">
              Keep weights elsewhere? Override these defaults by setting{' '}
              <span className="font-mono text-foreground-subtle">MAMMA_*</span> paths in{' '}
              <span className="font-mono text-foreground-subtle">.env.local</span> at the repo root.
              See{' '}
              <span className="font-mono text-foreground-subtle">docs/INSTALL.md</span>
              {' '}§ Customising paths.
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ───────────────────────── header bar ──────────────────────────────────

function SegmentedBar({ items }: { items: Asset[] }) {
  return (
    <div className="flex items-center gap-[3px]" aria-hidden>
      {items.map(item => {
        const present = item.present;
        const required = !item.optional;
        let cls = 'bg-surface-3';
        if (present) cls = 'bg-status-completed';
        else if (required) cls = 'bg-status-pending ring-1 ring-inset ring-status-pending/40';
        else cls = 'bg-surface-3 ring-1 ring-inset ring-border';
        return <span key={item.id} className={`h-2.5 w-2.5 rounded-[1.5px] ${cls}`} />;
      })}
    </div>
  );
}

// ───────────────────────── rows ────────────────────────────────────────

function AssetRow({
  asset, job, isExpanded, signedIn, onToggleExpand, onStartPublic, onSubmitMpi, onMpiAction,
}: {
  asset: Asset;
  job: JobResponse | null;
  isExpanded: boolean;
  signedIn: boolean;
  onToggleExpand: () => void;
  onStartPublic: () => void;
  onSubmitMpi: (username: string, password: string) => void;
  onMpiAction: () => void;
}) {
  const downloading = !!job && (job.state === 'downloading' || job.state === 'extracting');
  const errored = !!job && job.state === 'error';
  const showsTrustBadge =
    asset.source.kind === 'mpi' && signedIn && !asset.present && !downloading;

  return (
    <li className="px-5 py-2.5">
      <div className="flex items-center gap-3 min-h-[28px]">
        <StatusDot
          present={asset.present}
          required={!asset.optional}
          downloading={downloading}
          errored={errored}
        />
        <div className="min-w-0 flex-1 flex items-baseline gap-3">
          <span
            className="text-foreground text-[13px] font-medium flex-shrink-0"
            title={asset.purpose}
          >
            {asset.label}
          </span>
          <span
            className="text-foreground-subtle text-[11px] font-mono truncate flex-1 min-w-0"
            title={asset.rel_path}
          >
            {shortPath(asset.rel_path)}
          </span>
        </div>
        {showsTrustBadge && asset.source.kind === 'mpi' && (
          <span
            className="hidden md:inline-flex items-center gap-1 text-[10.5px] text-foreground-faint flex-shrink-0"
            title="Will use the credentials you signed in with at the top of Home"
          >
            <KeyRound className="w-3 h-3" aria-hidden />
            uses {asset.source.account_label} credentials
          </span>
        )}
        <span className="text-foreground-faint text-[11px] font-mono tabular-nums flex-shrink-0 w-[60px] text-right">
          {asset.present
            ? formatBytes(asset.size_bytes)
            : formatHint(asset.size_hint_mb)}
        </span>
        <div className="w-[96px] flex justify-end flex-shrink-0">
          <ActionChip
            asset={asset}
            job={job}
            isExpanded={isExpanded}
            signedIn={signedIn}
            onToggleExpand={onToggleExpand}
            onStartPublic={onStartPublic}
            onMpiAction={onMpiAction}
          />
        </div>
      </div>

      {downloading && (
        <ProgressInline state={job!.state} downloaded={job!.bytes_downloaded} total={job!.bytes_total} />
      )}

      {errored && (
        <ErrorInline
          error={job!.error || 'Unknown error.'}
          siteUrl={
            asset.source.kind === 'mpi'
              ? asset.source.register_url.replace('register.php', '')
              : undefined
          }
          siteLabel={asset.source.kind === 'mpi' ? asset.source.account_label : undefined}
        />
      )}

      {isExpanded && asset.source.kind === 'mpi' && (
        <SignInForm
          source={asset.source}
          onCancel={onToggleExpand}
          onSubmit={onSubmitMpi}
        />
      )}

      {isExpanded && asset.source.kind === 'manual' && (
        <DocsInline
          source={asset.source}
          onClose={onToggleExpand}
        />
      )}

      {isExpanded && asset.source.kind === 'hf_hub' && (
        <HFHubInline
          source={asset.source}
          onClose={onToggleExpand}
        />
      )}
    </li>
  );
}

function StatusDot({
  present, required, downloading, errored,
}: { present: boolean; required: boolean; downloading: boolean; errored: boolean }) {
  if (downloading) {
    return (
      <span className="inline-flex w-3 h-3 items-center justify-center flex-shrink-0" aria-label="Downloading">
        <Loader2 className="w-3 h-3 text-status-running animate-spin" />
      </span>
    );
  }
  if (errored) {
    return (
      <span className="inline-flex w-3 h-3 items-center justify-center flex-shrink-0" aria-label="Error">
        <AlertCircle className="w-3 h-3 text-status-failed" />
      </span>
    );
  }
  if (present) {
    return (
      <span
        className="inline-flex w-3 h-3 items-center justify-center flex-shrink-0"
        aria-label="Ready"
      >
        <span className="w-2 h-2 rounded-full bg-status-completed" />
      </span>
    );
  }
  if (required) {
    return (
      <span
        className="inline-flex w-3 h-3 items-center justify-center flex-shrink-0 relative"
        aria-label="Missing (required)"
      >
        <span className="w-2 h-2 rounded-full bg-status-pending" />
        <span className="absolute inset-0 rounded-full bg-status-pending/40 animate-ping" />
      </span>
    );
  }
  return (
    <span
      className="inline-flex w-3 h-3 items-center justify-center flex-shrink-0"
      aria-label="Missing (optional)"
    >
      <span className="w-2 h-2 rounded-full border border-border-strong" />
    </span>
  );
}

// ───────────────────────── action chip ─────────────────────────────────

function ActionChip({
  asset, job, isExpanded, signedIn, onToggleExpand, onStartPublic, onMpiAction,
}: {
  asset: Asset;
  job: JobResponse | null;
  isExpanded: boolean;
  signedIn: boolean;
  onToggleExpand: () => void;
  onStartPublic: () => void;
  onMpiAction: () => void;
}) {
  if (asset.present) {
    return (
      <span
        className="inline-flex items-center gap-1 text-status-completed/80 text-[10px] uppercase tracking-[0.14em]"
      >
        <Check className="w-3 h-3" />
        ready
      </span>
    );
  }
  if (job && (job.state === 'downloading' || job.state === 'extracting')) {
    const pct = job.bytes_total > 0
      ? Math.min(100, Math.round((job.bytes_downloaded / job.bytes_total) * 100))
      : null;
    return (
      <span className="text-foreground-muted text-[10.5px] font-mono tabular-nums">
        {job.state === 'extracting' ? 'extracting…' : pct === null ? '…' : `${pct}%`}
      </span>
    );
  }
  if (asset.source.kind === 'public' || asset.source.kind === 'gdrive') {
    return (
      <button
        onClick={onStartPublic}
        className="inline-flex items-center gap-1 px-2 py-1 text-[10px] uppercase tracking-[0.14em] rounded border border-border hover:border-primary/60 hover:text-primary text-foreground-muted transition-colors"
      >
        <Download className="w-3 h-3" />
        download
      </button>
    );
  }
  const retry = job?.state === 'error';
  if (asset.source.kind === 'mpi') {
    if (signedIn && !isExpanded) {
      return (
        <button
          onClick={onMpiAction}
          className="inline-flex items-center gap-1 px-2 py-1 text-[10px] uppercase tracking-[0.14em] rounded border border-border hover:border-primary/60 hover:text-primary text-foreground-muted transition-colors"
          title={`Download using your signed-in ${
            asset.source.kind === 'mpi' ? asset.source.account_label : ''
          } credentials`}
        >
          <Download className="w-3 h-3" />
          {retry ? 'retry' : 'download'}
        </button>
      );
    }
    return (
      <button
        onClick={onMpiAction}
        aria-pressed={isExpanded}
        className={
          'inline-flex items-center gap-1 px-2 py-1 text-[10px] uppercase tracking-[0.14em] rounded border transition-colors ' +
          (isExpanded
            ? 'border-primary/60 text-primary bg-primary-muted'
            : 'border-border hover:border-primary/60 hover:text-primary text-foreground-muted')
        }
      >
        {isExpanded ? <X className="w-3 h-3" /> : <LogIn className="w-3 h-3" />}
        {isExpanded ? 'close' : retry ? 'retry' : 'sign in'}
      </button>
    );
  }
  if (asset.source.kind === 'hf_hub') {
    // Lazy-download: the run will fetch it on first use. The chip
    // opens prep steps (request access + hf-login) but there's no
    // local download to kick off here.
    return (
      <button
        onClick={onToggleExpand}
        aria-pressed={isExpanded}
        className={
          'inline-flex items-center gap-1 px-2 py-1 text-[10px] uppercase tracking-[0.14em] rounded border transition-colors ' +
          (isExpanded
            ? 'border-primary/60 text-primary bg-primary-muted'
            : 'border-border hover:border-primary/60 hover:text-primary text-foreground-muted')
        }
      >
        {isExpanded ? <X className="w-3 h-3" /> : <ScrollText className="w-3 h-3" />}
        {isExpanded ? 'close' : 'setup'}
      </button>
    );
  }
  // manual
  return (
    <button
      onClick={onToggleExpand}
      aria-pressed={isExpanded}
      className={
        'inline-flex items-center gap-1 px-2 py-1 text-[10px] uppercase tracking-[0.14em] rounded border transition-colors ' +
        (isExpanded
          ? 'border-primary/60 text-primary bg-primary-muted'
          : 'border-border hover:border-primary/60 hover:text-primary text-foreground-muted')
      }
    >
      {isExpanded ? <X className="w-3 h-3" /> : <ScrollText className="w-3 h-3" />}
      {isExpanded ? 'close' : 'steps'}
    </button>
  );
}

// ───────────────────────── inline expansions ───────────────────────────

function ProgressInline({
  state, downloaded, total,
}: { state: 'downloading' | 'extracting' | 'ready' | 'error'; downloaded: number; total: number }) {
  const pct = total > 0 ? Math.min(100, (downloaded / total) * 100) : 0;
  return (
    <div className="mt-2 ml-6 flex items-center gap-2.5 text-[10.5px] font-mono tabular-nums text-foreground-faint">
      <div className="h-[2px] flex-1 bg-surface-2 rounded-full overflow-hidden">
        <div
          className="h-full bg-status-running transition-[width] duration-500"
          style={{ width: total > 0 ? `${pct}%` : '40%' }}
        />
      </div>
      <span>{formatBytes(downloaded)}{total > 0 ? ` / ${formatBytes(total)}` : ''}</span>
      {state === 'extracting' && <span className="text-status-running">unpacking…</span>}
    </div>
  );
}

function ErrorInline({
  error, siteUrl, siteLabel,
}: { error: string; siteUrl?: string; siteLabel?: string }) {
  return (
    <div className="mt-2 ml-6 flex items-start gap-2 text-[11px] text-foreground-muted">
      <AlertCircle className="w-3.5 h-3.5 text-status-failed flex-shrink-0 mt-0.5" />
      <span className="break-words">
        {error}
        {siteUrl && (
          <>
            {' '}
            <a
              href={siteUrl}
              target="_blank"
              rel="noreferrer"
              className="text-primary hover:underline inline-flex items-center gap-0.5 whitespace-nowrap"
            >
              Open the {siteLabel} website
              <ExternalLink className="w-3 h-3" />
            </a>
          </>
        )}
      </span>
    </div>
  );
}

function SignInForm({
  source, onCancel, onSubmit,
}: {
  source: AssetSourceMpi;
  onCancel: () => void;
  onSubmit: (username: string, password: string) => Promise<void> | void;
}) {
  const ctx = useCredentials();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [show, setShow] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const canSubmit = username.trim().length > 0 && password.length > 0 && !submitting;
  return (
    <form
      onSubmit={async e => {
        e.preventDefault();
        if (!canSubmit) return;
        setSubmitting(true);
        setError(null);
        const user = username.trim();
        try {
          // Verify before downloading: a wrong-credential submit should
          // fail fast here (login.php check) with a clear message instead of
          // kicking off a download that fails. On success, store the creds
          // in the shared context (so the top-of-Home badge reflects it and
          // later downloads skip this form), then start the download.
          const domain = domainFromAccountLabel(source.account_label);
          const res = await verifyMpiCredentials(domain ?? 'mamma', user, password);
          if (!res.valid) {
            setError(res.detail);
            return;
          }
          if (domain) ctx.signIn(domain, { username: user, password });
          await onSubmit(user, password);
          // Best-effort wipe of the in-component copy of the password.
          // The browser may retain it in form-autofill caches, which is
          // out of our reach — the helper text on the form is explicit
          // about the backend not storing anything.
          setUsername('');
          setPassword('');
        } finally {
          setSubmitting(false);
        }
      }}
      className="mt-2.5 ml-6 rounded-md border border-border-subtle bg-surface-2/60 p-3"
    >
      <div className="flex items-center justify-between mb-2.5">
        <span className="text-[10.5px] uppercase tracking-[0.16em] text-foreground-muted">
          {source.account_label} account
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
          <span className="break-words">{error}</span>
        </div>
      )}
      <div className="mt-2 text-[10.5px] text-foreground-faint leading-relaxed">
        Sent once over HTTPS to <span className="font-mono">download.is.tue.mpg.de</span>.
        The backend uses them to compose a single POST request, then drops the
        values from memory — no log line, no disk write.
        {' '}
        Need an account?{' '}
        <a
          href={source.register_url}
          target="_blank"
          rel="noreferrer"
          className="text-primary hover:underline inline-flex items-center gap-0.5"
        >
          Register
          <ExternalLink className="w-3 h-3" />
        </a>.
        {' '}
        Tip: sign in to <span className="text-foreground-muted">{source.account_label}</span> at
        the top of Home to skip this step on every download.
      </div>
    </form>
  );
}

function DocsInline({
  source, onClose,
}: { source: AssetSourceManual; onClose: () => void }) {
  return (
    <div className="mt-2.5 ml-6 rounded-md border border-border-subtle bg-surface-2/60 p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10.5px] uppercase tracking-[0.16em] text-foreground-muted">
          Manual steps
        </span>
        <button
          onClick={onClose}
          className="text-foreground-faint hover:text-foreground p-0.5"
          title="Close"
        >
          <X className="w-3 h-3" />
        </button>
      </div>
      <ol className="space-y-1.5 list-decimal list-inside marker:text-foreground-faint marker:text-[11px]">
        {source.steps.map((step, idx) => (
          <li key={idx} className="text-[12px] text-foreground-muted leading-relaxed">
            {step}
          </li>
        ))}
      </ol>
      {source.link && (
        <a
          href={source.link}
          target="_blank"
          rel="noreferrer"
          className="mt-2.5 inline-flex items-center gap-1.5 text-[11px] text-primary hover:underline"
        >
          {source.link_label || source.link}
          <ExternalLink className="w-3 h-3" />
        </a>
      )}
    </div>
  );
}

function HFHubInline({
  source, onClose,
}: { source: AssetSourceHfHub; onClose: () => void }) {
  return (
    <div className="mt-2.5 ml-6 rounded-md border border-border-subtle bg-surface-2/60 p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[10.5px] uppercase tracking-[0.16em] text-foreground-muted">
          {source.account_label} · auto-downloads on first run
        </span>
        <button
          onClick={onClose}
          className="text-foreground-faint hover:text-foreground p-0.5"
          title="Close"
        >
          <X className="w-3 h-3" />
        </button>
      </div>
      {source.steps.length > 0 && (
        <ol className="space-y-1.5 list-decimal list-inside marker:text-foreground-faint marker:text-[11px]">
          {source.steps.map((step, idx) => (
            <li key={idx} className="text-[12px] text-foreground-muted leading-relaxed">
              {step}
            </li>
          ))}
        </ol>
      )}
      <div className="mt-2.5 flex items-center gap-3">
        <a
          href={source.register_url}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1.5 text-[11px] text-primary hover:underline"
        >
          {source.gated ? `Request access · ${source.model_id}` : `Open ${source.model_id}`}
          <ExternalLink className="w-3 h-3" />
        </a>
        <span className="text-[10.5px] text-foreground-faint font-mono truncate">
          {source.model_id}
        </span>
      </div>
    </div>
  );
}

// ───────────────────────── skeleton ────────────────────────────────────

function PanelSkeleton() {
  return (
    <div className="bg-surface-1 border border-border-subtle rounded-xl p-5 shadow-sm shadow-black/30">
      <div className="h-3 w-32 rounded-sm animate-shimmer mb-3" />
      <div className="space-y-2">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="h-5 rounded-sm animate-shimmer" />
        ))}
      </div>
    </div>
  );
}

export default DataReadinessPanel;
