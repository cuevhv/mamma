import { useState, useEffect } from 'react';
import { toast } from 'sonner';
import { Cpu, Box, Container, FileCode, Plus, X, RotateCcw, ChevronDown, ChevronRight, AlertTriangle, Pencil, Save, Trash2, Globe } from 'lucide-react';
import { stepLabel } from './shared/stepLabels';
import { FlagCatalogPopover } from './FlagCatalogPopover';
import { StepSettings } from './StepSettings';
import { ClusterProfileForm } from './ClusterProfileForm';
import type { StepSetting, StepSettingsSchema } from '../lib/flagModel';

export interface PresetStep {
  name: string;
  enabled: boolean;
  engine: 'conda' | 'apptainer' | 'docker' | string;
  condaEnv: string;
  sifPath: string;
  dockerImage: string;
  script: string;
  repoPath: string;
  flags: string[];
  dependencies: string[];
  extras: Record<string, unknown>;
}

export interface PresetDigest {
  name: string;
  path: string;
  displayName: string;
  description: string;
  global: {
    datasetName: string;
    condaEnv: string;
    /** Cameras the run used — only set for a frozen run config (Re-run pre-fill). */
    camNames?: string[];
    bind: string[];
    /** Run frame window. null = unset = all frames. Becomes ma_cap's
     *  --start/--end at runtime; downstream steps inherit it via the NPZ. */
    startFrame: number | null;
    endFrame: number | null;
    /** Signed world up axis (consumed by ma_3d + ma_vis). null = preset default (z). */
    upAxis?: string | null;
    // Whole-run execution choice stored in the preset's global block
    // ('' = none stored: per-step engines apply). The Engine dropdown
    // pre-selects from these so a saved choice survives a reload.
    engine?: string;
    sifPath?: string;
    dockerImage?: string;
    clusterProfile?: string;
  };
  steps: PresetStep[];
}

/**
 * Deep-partial of the preset structure. Mirrors what the backend's
 * /api/tasks expects under `taskOverrides`. Only fields the user has
 * touched are present.
 */
export interface PresetOverrides {
  global?: {
    dataset_name?: string;
    conda_env?: string;
    // Whole-run execution settings, saved into the preset's global block.
    // engine '' / undefined = use the preset's per-step engines; a value sets
    // global.engine (StepBuilder falls back to it). The matching field per
    // engine: conda->conda_env, apptainer->sif_path, docker->docker_image.
    engine?: string;
    sif_path?: string;
    docker_image?: string;
    // Where it runs: undefined = this machine; a profile name = that cluster.
    cluster_profile?: string;
    bind?: string[];
    // Run frame window. Explicit null clears the preset's range (= all frames);
    // deep-merged into global.start_frame/end_frame on the backend.
    start_frame?: number | null;
    end_frame?: number | null;
    up_axis?: string;
  };
  // Per-step overrides keyed by step name (ma_cap, ma_masks, ...).
  [stepName: string]: any;
}

interface Props {
  digest: PresetDigest | null;
  /** When provided, the card becomes editable. Edits flow up via onChange. */
  overrides?: PresetOverrides;
  onOverridesChange?: (next: PresetOverrides) => void;
  /** Called after a successful "Save as new preset" with the new preset's name
   *  (e.g. "user/my_horses") so the parent can refresh the list and select it. */
  onPresetSaved?: (newPresetName: string) => void;
  /** Called after a successful delete of the currently-shown user preset.
   *  Parent should refresh the preset list and pick a new selection. */
  onPresetDeleted?: (deletedName: string) => void;
  /** When set, the frame-range editor is locked and this note explains why.
   *  Used by the Re-run form: the range is baked into the reused ma_cap NPZs,
   *  so editing it is inert unless ma_cap itself is among the re-run steps. */
  frameRangeLockedNote?: string | null;
  /** Per-step "common settings" schema (GET /api/steps/settings). When
   *  provided, each step card shows friendly widgets above the raw flags. */
  stepSettings?: StepSettingsSchema;
}

/**
 * Read-only summary by default; an inline editor when `overrides` and
 * `onOverridesChange` are provided. Shows what a preset will run (engine,
 * conda env, sif paths, flags) and lets the user tweak any of it for this
 * one submission. Edits are tracked as a deep-partial — only touched
 * fields are sent to the backend, which deep-merges them on top of the
 * preset before the form-level fields (seq_ids, cam_names, ...) land.
 */
export function PresetDigestCard({ digest, overrides, onOverridesChange, onPresetSaved, onPresetDeleted, stepSettings, frameRangeLockedNote }: Props) {
  const [saveOpen, setSaveOpen] = useState(false);
  const [saveName, setSaveName] = useState('');
  const [saveDisplayName, setSaveDisplayName] = useState('');
  const [saveDescription, setSaveDescription] = useState('');
  const [saving, setSaving] = useState(false);
  const [savingInPlace, setSavingInPlace] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  // Body (global + steps) collapses by default to keep the form short;
  // expand to inspect or edit the preset for this submission.
  const [bodyOpen, setBodyOpen] = useState(false);
  // Cluster profiles for the Execution "Run on" control (fetched once).
  const [clusters, setClusters] = useState<{ name: string; host: string; sifPath?: string }[]>([]);
  const [showAddCluster, setShowAddCluster] = useState(false);
  // Edit mode: the full profile dict fetched from GET /api/clusters/<name>,
  // handed to ClusterProfileForm as `initial`. Delete uses a two-click
  // confirm (no native dialogs).
  const [editCluster, setEditCluster] = useState<Record<string, any> | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  useEffect(() => {
    let alive = true;
    fetch('/api/clusters')
      .then(r => (r.ok ? r.json() : []))
      .then(rows => { if (alive) setClusters(Array.isArray(rows) ? rows : []); })
      .catch(() => {});
    return () => { alive = false; };
  }, []);
  // Authentication status of the selected cluster profile: a read-only probe
  // of GET /api/clusters/<name>/connection (keys / terminal session / none).
  // Computed pre-early-return because hooks must run unconditionally.
  const connProfile = ((overrides?.global?.cluster_profile !== undefined
    ? overrides.global.cluster_profile
    : digest?.global.clusterProfile) ?? '') as string;
  const connEngine = ((overrides?.global?.engine !== undefined
    ? overrides.global.engine
    : digest?.global.engine) ?? '') as string;
  const [conn, setConn] = useState<{ connected: boolean; method: string | null; connectCommand: string | null } | null>(null);
  const [connBusy, setConnBusy] = useState(false);
  const [connNonce, setConnNonce] = useState(0); // bump = "check again"
  useEffect(() => {
    if (connEngine !== 'cluster' || !connProfile) { setConn(null); return; }
    let alive = true;
    setConnBusy(true);
    fetch(`/api/clusters/${encodeURIComponent(connProfile)}/connection`)
      .then(r => (r.ok ? r.json() : null))
      .then(j => { if (alive) setConn(j); })
      .catch(() => { if (alive) setConn(null); })
      .finally(() => { if (alive) setConnBusy(false); });
    return () => { alive = false; };
  }, [connEngine, connProfile, connNonce]);

  if (!digest) {
    return (
      <div className="bg-surface-2/40 border border-border-subtle rounded-lg p-4 text-sm text-foreground-subtle italic">
        Pick a preset to see what it will run.
      </div>
    );
  }

  const editable = !!onOverridesChange;
  const ov = overrides ?? {};
  const isDirty = Object.keys(ov).length > 0;
  const isUserPreset = digest.name.startsWith('user/');

  // `bind` paths only affect apptainer/docker steps — hide the global block
  // entirely when no enabled step uses a container engine. (Dataset name and
  // global conda env used to live here too, but: dataset name follows the
  // capture, not the preset, and step-level Conda env fields make a global
  // default redundant. Both have been removed from this UI.)
  // Effective global engine: the override when touched (an explicit '' means
  // "cleared: per-step engines apply"), else what the preset stored.
  const globalEngineEff =
    ov.global?.engine !== undefined ? ov.global.engine : (digest.global.engine ?? '');
  const usedEngines = new Set<string>();
  for (const step of digest.steps) {
    if (!step.enabled) continue;
    // Per-step engine → effective global engine → preset's per-step engine.
    const engine = (ov[step.name]?.engine ?? (globalEngineEff || undefined) ?? step.engine) || 'conda';
    usedEngines.add(engine);
  }
  // 'cluster' executes via the profile's container (apptainer under the hood).
  const showBind = usedEngines.has('apptainer') || usedEngines.has('docker') || usedEngines.has('cluster');

  // -----------------------------------------------------------------
  // Override helpers — keep mutations to a single setOv() call.
  // -----------------------------------------------------------------
  const setOv = (next: PresetOverrides) => onOverridesChange?.(next);
  const reset = () => setOv({});

  const setGlobalField = (
    key: 'dataset_name' | 'conda_env' | 'engine' | 'sif_path' | 'docker_image' | 'cluster_profile' | 'bind' | 'up_axis',
    value: any,
  ) => {
    const nextGlobal = { ...(ov.global ?? {}), [key]: value };
    setOv({ ...ov, global: nextGlobal });
  };
  // Atomic multi-field global update (e.g. selecting a cluster sets
  // cluster_profile + engine + sif_path at once). Sequential setGlobalField
  // calls would clobber each other since each spreads the same stale ov.global.
  const setGlobalFields = (patch: Record<string, any>) => {
    setOv({ ...ov, global: { ...(ov.global ?? {}), ...patch } });
  };
  const reloadClusters = async (): Promise<{ name: string; host: string; sifPath?: string }[]> => {
    try {
      const r = await fetch('/api/clusters');
      const rows = r.ok ? await r.json() : [];
      const list = Array.isArray(rows) ? rows : [];
      setClusters(list);
      return list;
    } catch {
      return [];
    }
  };
  // Picking a cluster only records the profile name; the backend resolves the
  // execution (engine 'cluster' -> apptainer + the profile's container) at
  // preview/submit time, so the profile stays the single source of truth.
  const chooseCluster = (name: string) => {
    setConfirmDelete(false);
    setGlobalFields({ cluster_profile: name });
  };
  const openEditCluster = async (name: string) => {
    try {
      const r = await fetch(`/api/clusters/${encodeURIComponent(name)}`);
      if (!r.ok) return;
      setShowAddCluster(false);
      setEditCluster(await r.json());
    } catch { /* leave the form closed */ }
  };
  const deleteCluster = async (name: string) => {
    try {
      const r = await fetch(`/api/clusters/${encodeURIComponent(name)}`, { method: 'DELETE' });
      if (!r.ok) return;
    } catch { return; }
    setConfirmDelete(false);
    setEditCluster(null);
    const list = await reloadClusters();
    chooseCluster(list[0]?.name ?? '');
  };
  const setStepField = (stepName: string, key: string, value: any) => {
    const nextStep = { ...(ov[stepName] ?? {}), [key]: value };
    setOv({ ...ov, [stepName]: nextStep });
  };
  // Set both frame bounds in one override write. null = "all frames from/through".
  const setFrameRange = (start: number | null, end: number | null) => {
    setOv({ ...ov, global: { ...(ov.global ?? {}), start_frame: start, end_frame: end } });
  };

  // Effective values = preset value, overridden if the user touched it.
  // For the frame bounds, `!== undefined` so an explicit null override (the
  // user cleared the range = "all frames") wins over the preset's value.
  const eff = {
    bind: ov.global?.bind ?? digest.global.bind,
    startFrame: ov.global?.start_frame !== undefined ? ov.global.start_frame : digest.global.startFrame,
    endFrame: ov.global?.end_frame !== undefined ? ov.global.end_frame : digest.global.endFrame,
    upAxis: ov.global?.up_axis ?? digest.global.upAxis ?? 'z',
  };
  // Execution: the Engine dropdown answers "how does this task run" in one
  // place — conda / apptainer / docker locally, or 'cluster' (submit to the
  // selected Cluster Profile; steps there execute in the profile's env).
  // Falls back to the preset's stored values so a saved choice is visible.
  const effEngine = globalEngineEff;
  const isCluster = effEngine === 'cluster';
  const effCluster = ov.global?.cluster_profile !== undefined
    ? (ov.global.cluster_profile ?? '')
    : (digest.global.clusterProfile ?? '');
  const GINPUT = 'w-full bg-surface-2 border border-border rounded-md px-2.5 py-1.5 text-foreground text-sm focus:outline-none focus:border-primary/60 disabled:opacity-60';
  const frameRangeDirty = ov.global?.start_frame !== undefined || ov.global?.end_frame !== undefined;
  const frameRangeLabel = eff.startFrame == null && eff.endFrame == null
    ? 'All frames'
    : `Frames ${eff.startFrame ?? 0}–${eff.endFrame ?? 'end'}`;

  return (
    <div className="bg-surface-2/40 border border-border-subtle rounded-lg p-4 space-y-3">
      {/* Header — clicking the title row toggles the body. Action buttons
          (Save / Reset / Delete) live on the right and stopPropagation so
          they don't accidentally collapse the section. */}
      <div className="flex items-start justify-between gap-3">
        <button
          type="button"
          onClick={() => setBodyOpen(o => !o)}
          aria-expanded={bodyOpen}
          aria-controls="preset-digest-body"
          className="min-w-0 text-left flex-1 group"
        >
          <div className="text-foreground text-sm font-semibold flex items-center gap-2 flex-wrap">
            {bodyOpen
              ? <ChevronDown className="w-3.5 h-3.5 text-foreground-subtle group-hover:text-foreground transition-colors" />
              : <ChevronRight className="w-3.5 h-3.5 text-foreground-subtle group-hover:text-foreground transition-colors" />}
            {digest.displayName}
            {isUserPreset && (
              <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] uppercase tracking-wider bg-status-completed-bg border border-status-completed/35 text-status-completed">
                User
              </span>
            )}
            {editable && (
              <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] uppercase tracking-wider bg-primary-muted border border-primary/30 text-primary">
                <Pencil className="w-2.5 h-2.5" /> Editable
              </span>
            )}
            {isDirty && (
              <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] uppercase tracking-wider bg-status-pending-bg border border-status-pending/35 text-status-pending">
                Modified
              </span>
            )}
            {/* Frame window at a glance — the preset's slice is otherwise
                invisible (e.g. `quick` runs only 60–120). Expand the body to
                change it. */}
            <span
              className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] tracking-wide border font-mono ${
                frameRangeDirty
                  ? 'bg-status-pending-bg border-status-pending/35 text-status-pending'
                  : 'bg-surface-2 border-border-subtle text-foreground-muted'
              }`}
              title={frameRangeLabel === 'All frames'
                ? 'This run processes every frame.'
                : `This run is sliced to ${frameRangeLabel.toLowerCase()} (ma_cap --start/--end). Expand to change.`}
            >
              {frameRangeLabel}
            </span>
          </div>
          {digest.description && (
            <div className="text-foreground-subtle text-xs mt-1">{digest.description}</div>
          )}
        </button>
        <div className="flex items-center gap-2 flex-shrink-0 flex-wrap justify-end" onClick={e => e.stopPropagation()}>
          {editable && isDirty && isUserPreset && onPresetSaved && (
            <button
              type="button"
              disabled={savingInPlace}
              onClick={async () => {
                setSavingInPlace(true);
                try {
                  const res = await fetch(`/api/task-presets/${encodeURIComponent(digest.name)}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ overrides: ov }),
                  });
                  if (res.ok) {
                    toast.success(`Saved “${digest.displayName}”`);
                    onPresetSaved(digest.name);
                  } else {
                    const err = await res.json().catch(() => ({}));
                    toast.error(err.error || `Failed to save preset (${res.status})`);
                  }
                } catch (e) {
                  toast.error('Failed to save preset. See console.');
                  console.error(e);
                } finally {
                  setSavingInPlace(false);
                }
              }}
              className="inline-flex items-center gap-1.5 text-xs text-primary-foreground bg-primary hover:opacity-90 disabled:opacity-40 px-2 py-1 rounded-md transition-opacity"
              title="Save these changes into this user preset"
            >
              <Save className="w-3 h-3" />
              {savingInPlace ? 'Saving…' : 'Save'}
            </button>
          )}
          {editable && isDirty && onPresetSaved && (
            <button
              type="button"
              onClick={() => { setSaveOpen(true); setDeleteOpen(false); setSaveName(''); setSaveDisplayName(''); setSaveDescription(''); setBodyOpen(true); }}
              className="inline-flex items-center gap-1.5 text-xs text-primary hover:opacity-80 px-2 py-1 rounded-md bg-primary-muted border border-primary/30 hover:border-primary/50 transition-colors"
              title="Save these changes as a new reusable preset"
            >
              <Save className="w-3 h-3" />
              Save as preset…
            </button>
          )}
          {editable && isDirty && (
            <button
              type="button"
              onClick={reset}
              className="inline-flex items-center gap-1.5 text-xs text-foreground-muted hover:text-foreground px-2 py-1 rounded-md border border-border hover:border-border-strong transition-colors"
              title="Discard all overrides and revert to preset defaults"
            >
              <RotateCcw className="w-3 h-3" />
              Reset
            </button>
          )}
          {isUserPreset && onPresetDeleted && (
            <button
              type="button"
              onClick={() => { setDeleteOpen(true); setSaveOpen(false); }}
              className="inline-flex items-center gap-1.5 text-xs text-status-failed hover:opacity-80 px-2 py-1 rounded-md border border-status-failed/30 hover:border-status-failed/50 transition-colors"
              title="Delete this user-saved preset"
            >
              <Trash2 className="w-3 h-3" />
              Delete
            </button>
          )}
        </div>
      </div>

      {/* Delete confirmation */}
      {deleteOpen && onPresetDeleted && (
        <div className="bg-surface-1 border border-status-failed/40 rounded-md p-3 space-y-2">
          <div className="text-foreground text-sm font-medium flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-status-failed" />
            Delete this preset?
          </div>
          <div className="text-foreground-subtle text-xs">
            This will permanently remove <code className="font-mono text-foreground-muted">{digest.name}.json</code> from
            your user presets. Runs already submitted with it are unaffected — their frozen configs live in <code className="font-mono">run_configs/</code>.
          </div>
          <div className="flex items-center justify-end gap-2 pt-1">
            <button
              type="button"
              onClick={() => setDeleteOpen(false)}
              className="px-3 py-1.5 text-xs text-foreground-muted hover:text-foreground rounded-md border border-border hover:border-border-strong transition-colors"
            >
              Cancel
            </button>
            <button
              type="button"
              disabled={deleting}
              onClick={async () => {
                setDeleting(true);
                try {
                  const res = await fetch(`/api/task-presets/${encodeURIComponent(digest.name)}`, { method: 'DELETE' });
                  if (res.ok) {
                    toast.success(`Deleted preset “${digest.displayName}”`);
                    setDeleteOpen(false);
                    onPresetDeleted(digest.name);
                  } else {
                    const err = await res.json().catch(() => ({}));
                    toast.error(err.error || `Failed to delete preset (${res.status})`);
                  }
                } catch (e) {
                  toast.error('Failed to delete preset. See console.');
                  console.error(e);
                } finally {
                  setDeleting(false);
                }
              }}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-status-failed/90 hover:bg-status-failed text-white disabled:opacity-40 disabled:cursor-not-allowed text-xs font-medium rounded-md transition-colors"
            >
              <Trash2 className="w-3 h-3" />
              {deleting ? 'Deleting…' : 'Delete preset'}
            </button>
          </div>
        </div>
      )}

      {/* Save-as-preset inline form */}
      {saveOpen && onPresetSaved && (
        <div className="bg-surface-1 border border-primary/30 rounded-md p-3 space-y-2">
          <div className="text-foreground text-sm font-medium flex items-center gap-2">
            <Save className="w-4 h-4 text-primary" />
            Save as new preset
          </div>
          <div className="text-foreground-subtle text-xs">
            Will be written to <code className="font-mono text-foreground-muted">samples/presets/user/&lt;name&gt;.json</code> with your edits applied.
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            <div>
              <FieldLabel label="Name" />
              <input
                type="text"
                value={saveName}
                onChange={e => setSaveName(e.target.value.replace(/[^A-Za-z0-9_-]/g, ''))}
                placeholder="fast_draft"
                autoFocus
                className="w-full bg-surface-2 border border-border rounded-md px-2.5 py-1.5 text-foreground text-xs font-mono focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition-colors placeholder:text-foreground-faint"
              />
            </div>
            <div>
              <FieldLabel label="Display name (optional)" />
              <input
                type="text"
                value={saveDisplayName}
                onChange={e => setSaveDisplayName(e.target.value)}
                placeholder="Fast draft"
                className="w-full bg-surface-2 border border-border rounded-md px-2.5 py-1.5 text-foreground text-xs focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition-colors placeholder:text-foreground-faint"
              />
            </div>
          </div>
          <div>
            <FieldLabel label="Description (optional)" />
            <input
              type="text"
              value={saveDescription}
              onChange={e => setSaveDescription(e.target.value)}
              placeholder="What makes this variant different…"
              className="w-full bg-surface-2 border border-border rounded-md px-2.5 py-1.5 text-foreground text-xs focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition-colors placeholder:text-foreground-faint"
            />
          </div>
          <div className="flex items-center justify-end gap-2 pt-1">
            <button
              type="button"
              onClick={() => setSaveOpen(false)}
              className="px-3 py-1.5 text-xs text-foreground-muted hover:text-foreground rounded-md border border-border hover:border-border-strong transition-colors"
            >
              Cancel
            </button>
            <button
              type="button"
              disabled={!saveName || saving}
              onClick={async () => {
                setSaving(true);
                try {
                  const res = await fetch('/api/task-presets', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                      sourceName: digest.name,
                      newName: saveName,
                      overrides: ov,
                      displayName: saveDisplayName,
                      description: saveDescription,
                    }),
                  });
                  if (res.ok) {
                    const data = await res.json();
                    toast.success(`Saved preset “${data.displayName}”`);
                    setSaveOpen(false);
                    onPresetSaved(data.name);
                  } else {
                    const err = await res.json().catch(() => ({}));
                    toast.error(err.error || `Failed to save preset (${res.status})`);
                  }
                } catch (e) {
                  toast.error('Failed to save preset. See console.');
                  console.error(e);
                } finally {
                  setSaving(false);
                }
              }}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-primary text-primary-foreground hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed text-xs font-medium rounded-md transition-opacity"
            >
              <Save className="w-3 h-3" />
              {saving ? 'Saving…' : 'Save preset'}
            </button>
          </div>
        </div>
      )}

      {/* Collapsible body — global container settings + per-step cards.
          Default collapsed: the header above is the at-a-glance view; the
          dropdown handles preset selection. Expand to inspect or edit. */}
      {bodyOpen && (
        <div id="preset-digest-body" className="space-y-3">
          {/* Global block — bind paths only (when container steps are in
              play). Dataset name and global conda env were removed: the
              former is a property of the capture, not the preset, and
              per-step Conda env fields already cover the latter. */}
          {(showBind || editable) && (
            <div className="space-y-3 bg-surface-1/50 border border-border-subtle rounded-md p-3.5">
              <div className="flex items-center gap-2 text-[11px] font-semibold tracking-wide text-foreground-muted">
                <Globe className="w-3.5 h-3.5 text-foreground-faint" />
                Global
              </div>

              {/* Execution — Engine and its companion field side by side:
                  the dropdown answers "how does this task run" (a local
                  conda / apptainer / docker engine, or 'cluster' to submit
                  to a Cluster Profile), the field next to it pins where
                  that environment lives. Cluster management (edit / delete /
                  add a profile) sits below, full width. */}
              <div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div>
                    <FieldLabel label="Engine" dirty={ov.global?.engine !== undefined} />
                    <select
                      value={effEngine}
                      disabled={!editable}
                      onChange={e => {
                        const v = e.target.value;
                        if (v === 'cluster') {
                          setGlobalFields({ engine: 'cluster', cluster_profile: effCluster || clusters[0]?.name || '' });
                        } else {
                          // '' clears: when the preset STORED an engine, write an
                          // explicit '' override so per-step engines apply again;
                          // otherwise just drop the override.
                          const cleared = digest.global.engine ? '' : undefined;
                          setGlobalFields({ engine: v || cleared, cluster_profile: undefined });
                        }
                      }}
                      className={GINPUT}
                    >
                      <option value="">From preset (per-step)</option>
                      <option value="conda">conda</option>
                      <option value="apptainer">apptainer</option>
                      <option value="docker">docker</option>
                      <option value="cluster">cluster (submit to a cluster)</option>
                    </select>
                  </div>
                  {effEngine === 'conda' && (
                    <div>
                      <FieldLabel label="Conda env" dirty={ov.global?.conda_env !== undefined} />
                      <input type="text" disabled={!editable}
                        value={ov.global?.conda_env ?? digest.global.condaEnv ?? ''}
                        onChange={e => setGlobalField('conda_env', e.target.value || undefined)}
                        placeholder="conda env (default: mamma)" className={GINPUT} />
                    </div>
                  )}
                  {effEngine === 'apptainer' && (
                    <div>
                      <FieldLabel label="Container (.sif)" dirty={ov.global?.sif_path !== undefined} />
                      <input type="text" disabled={!editable}
                        value={ov.global?.sif_path ?? digest.global.sifPath ?? ''}
                        onChange={e => setGlobalField('sif_path', e.target.value || undefined)}
                        placeholder="/path/to/mamma.sif" className={GINPUT} />
                    </div>
                  )}
                  {effEngine === 'docker' && (
                    <div>
                      <FieldLabel label="Docker image" dirty={ov.global?.docker_image !== undefined} />
                      <input type="text" disabled={!editable}
                        value={ov.global?.docker_image ?? digest.global.dockerImage ?? ''}
                        onChange={e => setGlobalField('docker_image', e.target.value || undefined)}
                        placeholder="docker image, e.g. mamma:v1.1.0" className={GINPUT} />
                    </div>
                  )}
                  {isCluster && (
                    <div>
                      <FieldLabel label="Cluster profile" dirty={ov.global?.cluster_profile !== undefined} />
                      {clusters.length > 0 ? (
                        <select value={effCluster} disabled={!editable}
                          onChange={e => chooseCluster(e.target.value)} className={GINPUT}>
                          <option value="" disabled>Select a cluster…</option>
                          {clusters.map(c => <option key={c.name} value={c.name}>{c.name} — {c.host}</option>)}
                        </select>
                      ) : (
                        <div className="px-2.5 py-1.5 text-foreground-faint text-[12px]">
                          No clusters yet; add one below.
                        </div>
                      )}
                    </div>
                  )}
                </div>
                {isCluster && (
                  <div className="mt-2 space-y-2">
                    {effCluster && (
                      <div className="flex items-center gap-2 text-[12px]">
                        <span className={`inline-block w-1.5 h-1.5 rounded-full ${
                          connBusy ? 'bg-foreground-faint animate-pulse'
                            : conn?.connected ? 'bg-status-completed' : 'bg-status-failed'}`} />
                        {connBusy ? (
                          <span className="text-foreground-faint">checking connection…</span>
                        ) : conn?.connected ? (
                          <span className="text-foreground-muted">
                            Connected{conn.method === 'session' ? ' (terminal session)'
                              : conn.method === 'keys' ? ' (SSH keys)' : ''}
                          </span>
                        ) : conn ? (
                          <span className="text-foreground-muted">Not connected</span>
                        ) : (
                          <span className="text-foreground-faint">connection status unavailable</span>
                        )}
                        <button type="button" onClick={() => setConnNonce(n => n + 1)}
                          className="text-foreground-faint hover:text-foreground text-[11px] hover:underline">
                          check again
                        </button>
                      </div>
                    )}
                    {effCluster && conn && !conn.connected && conn.connectCommand && (
                      <div className="text-[11px] text-foreground-muted leading-snug rounded-md border border-border-subtle bg-surface-2/40 px-2.5 py-2 space-y-1.5">
                        <div>
                          This cluster needs an interactive login (password / 2FA). Run this
                          once in a terminal; ssh itself prompts there, and the GUI then
                          reuses that session. No password is entered into or stored by MAMMA.
                        </div>
                        <div className="flex items-center gap-2">
                          <code className="font-mono text-foreground bg-surface-2 border border-border-subtle rounded px-1.5 py-0.5">
                            {conn.connectCommand}
                          </code>
                          <button type="button"
                            onClick={() => { navigator.clipboard?.writeText(conn.connectCommand!); toast.success('Copied'); }}
                            className="text-primary text-[11px] hover:underline">
                            Copy
                          </button>
                        </div>
                      </div>
                    )}
                    {digest.steps.some(s => s.name === 'ma_masks') && (
                      <label className="flex items-start gap-2 text-[12px] text-foreground-muted cursor-pointer select-none">
                        <input type="checkbox" disabled={!editable}
                          className="mt-0.5 accent-primary"
                          checked={!!ov.ma_masks?.fanout_cameras}
                          onChange={e => setStepField('ma_masks', 'fanout_cameras',
                            e.target.checked ? true : undefined)} />
                        <span>
                          <span className="text-foreground">Parallelize ma_masks across cameras</span>
                          {' '}— one cluster job per camera after the init view
                          (capped by the profile's max concurrent jobs).
                          Cross-camera identity references reduce to the init
                          camera; wall time drops by roughly the concurrency factor.
                        </span>
                      </label>
                    )}
                    {editable && !showAddCluster && !editCluster && (
                      <div className="flex items-center gap-3">
                        {effCluster && (
                          <>
                            <button type="button" onClick={() => openEditCluster(effCluster)}
                              className="text-primary text-[12px] hover:underline">
                              Edit '{effCluster}'
                            </button>
                            <button type="button"
                              onClick={() => (confirmDelete ? deleteCluster(effCluster) : setConfirmDelete(true))}
                              className={`text-[12px] hover:underline ${confirmDelete ? 'text-status-failed font-medium' : 'text-foreground-faint'}`}>
                              {confirmDelete ? `Really delete '${effCluster}'?` : 'Delete'}
                            </button>
                            {confirmDelete && (
                              <button type="button" onClick={() => setConfirmDelete(false)}
                                className="text-foreground-faint text-[12px] hover:underline">
                                keep it
                              </button>
                            )}
                          </>
                        )}
                        <button type="button" onClick={() => setShowAddCluster(true)}
                          className="text-primary text-[12px] hover:underline">
                          {clusters.length === 0 ? '+ Add a cluster' : '+ Add another cluster'}
                        </button>
                      </div>
                    )}
                    {(showAddCluster || editCluster) && (
                      <ClusterProfileForm
                        key={editCluster?.name ?? '__new__'}
                        initial={editCluster ?? undefined}
                        onSaved={async (nm) => {
                          setShowAddCluster(false); setEditCluster(null);
                          await reloadClusters(); chooseCluster(nm);
                        }}
                        onCancel={() => { setShowAddCluster(false); setEditCluster(null); }}
                      />
                    )}
                    <div className="text-[11px] text-foreground-faint leading-snug rounded-md border border-border-subtle bg-surface-2/40 px-2.5 py-2">
                      Steps run in the cluster profile's Environment (its .sif container,
                      conda env, or docker image); per-step engine choices don't apply.
                      The capture's footage and calibration must be on a path the cluster
                      can read (a shared filesystem / your workspace), not a local-only
                      path. Outputs are written under the profile's workspace folder.
                    </div>
                  </div>
                )}
              </div>

              <FrameRangeField
                start={eff.startFrame}
                end={eff.endFrame}
                dirty={frameRangeDirty}
                editable={editable && !frameRangeLockedNote}
                onChange={setFrameRange}
              />
              {frameRangeLockedNote && (
                <p className="text-foreground-faint text-[11px] leading-snug">{frameRangeLockedNote}</p>
              )}
              {/* Up axis widget hidden for now (plumbing kept: eff.upAxis +
                  setGlobalField('up_axis') + digest.global.upAxis). */}
              {showBind && (
                <ArrayField
                  label="Bind paths"
                  values={eff.bind ?? []}
                  dirty={ov.global?.bind !== undefined}
                  editable={editable}
                  onChange={list => setGlobalField('bind', list)}
                  placeholder="/path/to/mount"
                  hint="Mounted into apptainer (--bind) and docker (-v) containers."
                />
              )}
            </div>
          )}

          {/* Steps — two columns on wide viewports so the form doesn't
              stretch taller than the screen. Numbering is derived from
              the canonical position in `digest.steps` (which the backend
              returns in pipeline order) so e.g. ma_2d stays "2." even if
              masks is disabled in this preset. */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
            {digest.steps.filter(s => s.enabled).map(step => (
              <StepCard
                key={step.name}
                step={step}
                stepIndex={digest.steps.findIndex(s => s.name === step.name)}
                stepOverride={ov[step.name] ?? {}}
                editable={editable}
                settings={stepSettings?.[step.name] ?? []}
                clusterProfile={isCluster ? (effCluster || '(select a cluster)') : undefined}
                onChange={(key, value) => setStepField(step.name, key, value)}
              />
            ))}
            {digest.steps.filter(s => !s.enabled).length > 0 && (
              <div className="text-xs text-foreground-subtle italic pt-1 lg:col-span-2">
                (disabled in preset: {digest.steps.filter(s => !s.enabled).map(s => stepLabel(s.name)).join(', ')})
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// -------------------------------------------------------------------
// Step card — one per enabled preset step.
// -------------------------------------------------------------------

function StepCard({
  step,
  stepIndex,
  stepOverride,
  editable,
  settings,
  clusterProfile,
  onChange,
}: {
  step: PresetStep;
  /** Canonical position in the pipeline (0 = ma_cap, 1 = ma_masks, …).
   *  Stays stable across presets so users learn "ma_3d is step 3". */
  stepIndex: number;
  stepOverride: Record<string, any>;
  editable: boolean;
  /** Curated "common settings" for this step (may be empty). */
  settings: StepSetting[];
  /** Set when the task's global engine is 'cluster': per-step engine choices
   *  don't apply (steps run in the profile's container on the cluster). */
  clusterProfile?: string;
  onChange: (key: string, value: any) => void;
}) {
  const [advancedOpen, setAdvancedOpen] = useState(false);
  // Steps with no common-settings widgets show the raw flag editor open
  // (it's the only flag UI); steps that have widgets keep it collapsed.
  const [flagsOpen, setFlagsOpen] = useState(settings.length === 0);

  const eff = {
    engine: stepOverride.engine ?? step.engine,
    condaEnv: stepOverride.conda_env ?? step.condaEnv,
    sifPath: stepOverride.sif_path ?? step.sifPath,
    dockerImage: stepOverride.docker_image ?? step.dockerImage,
    script: stepOverride.script ?? step.script,
    repoPath: stepOverride.repo_path ?? step.repoPath,
    flags: stepOverride.flags ?? step.flags,
  };
  const dirty = (k: string) => stepOverride[k] !== undefined;
  const stepIsDirty = Object.keys(stepOverride).length > 0;

  // Flags owned by a Common-settings widget are edited there; split them out
  // so the raw "Advanced flags" list shows only the *extra* flags — the same
  // flag never appears in both places. Widget edits still preserve these.
  const managedFlagNames = new Set<string>();
  for (const s of settings) {
    if (s.target.kind === 'flag') managedFlagNames.add(s.target.flag);
    else if (s.target.kind === 'flag_pair') { managedFlagNames.add(s.target.on); managedFlagNames.add(s.target.off); }
    else if (s.target.kind === 'flags') s.target.flags.forEach(f => managedFlagNames.add(f));
  }
  const flagTokenName = (tok: string) => tok.trim().split(/\s+/)[0] ?? '';
  const effFlags: string[] = eff.flags ?? [];
  const managedFlags = effFlags.filter(f => managedFlagNames.has(flagTokenName(f)));
  const unmanagedFlags = effFlags.filter(f => !managedFlagNames.has(flagTokenName(f)));

  const engineIcon = eff.engine === 'apptainer' ? <Container className="w-3.5 h-3.5" />
                    : eff.engine === 'docker'   ? <Box className="w-3.5 h-3.5" />
                    : <Cpu className="w-3.5 h-3.5" />;

  return (
    <div className="bg-surface-2 border border-border rounded-lg overflow-hidden shadow-sm shadow-black/30 ring-1 ring-inset ring-white/[0.02] flex flex-col">
      {/* Numbered header — distinct background + bottom border so the
          eye groups the body fields under the right step header. */}
      <div className="flex items-center flex-wrap gap-2 px-3 py-2 bg-surface-3 border-b border-border-subtle">
        <span
          className="inline-flex items-center justify-center w-6 h-6 rounded-md bg-primary-muted text-primary text-xs font-mono font-medium tabular-nums"
          aria-label={`Pipeline step ${stepIndex}`}
        >
          {stepIndex}
        </span>
        <span className="text-foreground text-sm font-medium">{stepLabel(step.name)}</span>
        <span className="text-foreground-subtle font-mono text-[11px]">{step.name}</span>
        {stepIsDirty && (
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-status-pending" title="This step has overrides" />
        )}
      </div>
      <div className="p-3.5 space-y-3 flex-1">

      {/* Common settings — friendly widgets for this step's important flags
          (from GET /api/steps/settings). Reads/writes the same flags/extras
          as the raw editor below, so the two stay in sync. */}
      <StepSettings
        settings={settings}
        stepName={step.name}
        flags={step.flags}
        extras={step.extras}
        override={stepOverride}
        editable={editable}
        onChange={onChange}
      />

      {/* Advanced flags (raw) — the full free-text list + "View available
          flags" catalog. Collapsed when the common-settings widgets above
          already cover the usual knobs; both views edit the same flag list. */}
      <div className="border-t border-border-subtle pt-2">
        <button
          type="button"
          onClick={() => setFlagsOpen(o => !o)}
          aria-expanded={flagsOpen}
          className="inline-flex items-center gap-2 text-[11px] font-semibold tracking-wide text-foreground-muted hover:text-foreground transition-colors"
        >
          {flagsOpen ? <ChevronDown className="w-3.5 h-3.5 text-foreground-faint" /> : <ChevronRight className="w-3.5 h-3.5 text-foreground-faint" />}
          Advanced flags
          {unmanagedFlags.length > 0 && (
            <span className="text-foreground-faint">({unmanagedFlags.length})</span>
          )}
          {dirty('flags') && <span className="inline-block w-1 h-1 rounded-full bg-status-pending" title="Modified from preset" />}
        </button>
        {flagsOpen && (
          <div className="mt-2">
            <ArrayField
              label="Flags"
              values={unmanagedFlags}
              dirty={dirty('flags')}
              editable={editable}
              onChange={list => onChange('flags', [...managedFlags, ...list])}
              placeholder="--flag value"
              mono
              hint={settings.length > 0 ? 'Flags shown as Common settings above are edited there and hidden from this list.' : undefined}
            />
            {editable && (
              <div className="mt-1.5">
                <FlagCatalogPopover
                  stepName={step.name}
                  onInsert={(snippet) => onChange('flags', [...(eff.flags ?? []), snippet])}
                  alreadyPresent={new Set(
                    (eff.flags ?? [])
                      .map(f => /^--?(\S+)/.exec(f)?.[1] || '')
                      .filter(Boolean),
                  )}
                />
              </div>
            )}
          </div>
        )}
      </div>

      {/* Advanced (script + repo path) — risky, collapsed by default */}
      {editable && (
        <div className="border-t border-border-subtle pt-2">
          <button
            type="button"
            onClick={() => setAdvancedOpen(o => !o)}
            className="inline-flex items-center gap-2 text-[11px] font-semibold tracking-wide text-foreground-muted hover:text-foreground transition-colors"
          >
            {advancedOpen ? <ChevronDown className="w-3.5 h-3.5 text-foreground-faint" /> : <ChevronRight className="w-3.5 h-3.5 text-foreground-faint" />}
            Engine
          </button>
          {advancedOpen && clusterProfile !== undefined && (
            <div className="mt-2 text-[11px] text-foreground-faint leading-snug rounded-md border border-border-subtle bg-surface-2/40 px-2.5 py-2">
              Runs on cluster <span className="text-foreground-muted font-mono">{clusterProfile}</span> in
              the profile's Environment; per-step engine choices don't apply here.
              Pick a local engine in the Global block to edit this again.
            </div>
          )}
          {advancedOpen && clusterProfile === undefined && (
            <div className="mt-2 space-y-2">
              {/* Engine + engine-specific fields (conda env / SIF / docker image). */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                {editable ? (
                  <SelectField
                    label="Engine"
                    value={eff.engine}
                    dirty={dirty('engine')}
                    onChange={v => onChange('engine', v)}
                    options={['conda', 'apptainer', 'docker']}
                    icon={engineIcon}
                  />
                ) : (
                  <ReadOnlyField label="Engine" value={eff.engine} icon={engineIcon} />
                )}
                {eff.engine === 'conda' && (
                  <Field
                    label="Conda env"
                    value={eff.condaEnv ?? ''}
                    dirty={dirty('conda_env')}
                    placeholder="(falls back to global)"
                    editable={editable}
                    onChange={v => onChange('conda_env', v)}
                    mono
                  />
                )}
                {eff.engine === 'apptainer' && (
                  <Field
                    label="SIF path"
                    value={eff.sifPath ?? ''}
                    dirty={dirty('sif_path')}
                    placeholder="/path/to/image.sif"
                    editable={editable}
                    onChange={v => onChange('sif_path', v)}
                    mono
                  />
                )}
                {eff.engine === 'docker' && (
                  <Field
                    label="Docker image"
                    value={eff.dockerImage ?? ''}
                    dirty={dirty('docker_image')}
                    placeholder="org/image:tag"
                    editable={editable}
                    onChange={v => onChange('docker_image', v)}
                    mono
                  />
                )}
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                <Field
                  label="Script"
                  value={eff.script ?? ''}
                  dirty={dirty('script')}
                  placeholder="path/to/script.py"
                  editable={editable}
                  onChange={v => onChange('script', v)}
                  mono
                />
                <Field
                  label="Module dir path"
                  value={eff.repoPath ?? ''}
                  dirty={dirty('repo_path')}
                  placeholder="/path/to/module"
                  editable={editable}
                  onChange={v => onChange('repo_path', v)}
                  mono
                />
              </div>
            </div>
          )}
        </div>
      )}

      {/* In read-only mode, render the original FileCode summary so the layout
          still resembles the legacy preview. */}
      {!editable && (
        <div className="text-xs text-foreground-subtle flex items-start gap-2">
          <FileCode className="w-3 h-3 mt-0.5 flex-shrink-0" />
          <code className="font-mono text-foreground-muted break-all">{step.script || '(no script)'}</code>
        </div>
      )}
      </div>
    </div>
  );
}

// -------------------------------------------------------------------
// Field primitives — text input, select, array editor.
// -------------------------------------------------------------------

function Field({
  label, value, onChange, placeholder, dirty, editable, mono, hint,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  dirty?: boolean;
  editable: boolean;
  mono?: boolean;
  hint?: string;
}) {
  return (
    <div>
      <FieldLabel label={label} dirty={dirty} />
      {editable ? (
        <input
          type="text"
          value={value}
          onChange={e => onChange(e.target.value)}
          placeholder={placeholder}
          className={`w-full bg-surface-2 border ${dirty ? 'border-status-pending/60' : 'border-border'} rounded-md px-2.5 py-1.5 text-foreground text-xs focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition-colors placeholder:text-foreground-faint ${mono ? 'font-mono' : ''}`}
        />
      ) : (
        <div className={`px-2.5 py-1.5 text-foreground-muted text-xs ${mono ? 'font-mono' : ''}`}>
          {value || <span className="text-foreground-faint italic">{placeholder}</span>}
        </div>
      )}
      {hint && <div className="text-foreground-faint text-[10px] mt-1">{hint}</div>}
    </div>
  );
}

function ReadOnlyField({ label, value, icon }: { label: string; value: string; icon?: React.ReactNode }) {
  return (
    <div>
      <FieldLabel label={label} />
      <div className="inline-flex items-center gap-1 px-2 py-1 bg-surface-2 text-foreground-muted text-xs rounded-md border border-border-subtle">
        {icon}
        {value}
      </div>
    </div>
  );
}

function SelectField({
  label, value, options, onChange, dirty, icon,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (v: string) => void;
  dirty?: boolean;
  icon?: React.ReactNode;
}) {
  return (
    <div>
      <FieldLabel label={label} dirty={dirty} />
      <div className="relative">
        {icon && <span className="absolute left-2 top-1/2 -translate-y-1/2 text-foreground-subtle pointer-events-none">{icon}</span>}
        <select
          value={value}
          onChange={e => onChange(e.target.value)}
          className={`w-full bg-surface-2 border ${dirty ? 'border-status-pending/60' : 'border-border'} rounded-md ${icon ? 'pl-7' : 'pl-2.5'} pr-2.5 py-1.5 text-foreground text-xs focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition-colors`}
        >
          {options.map(o => (
            <option key={o} value={o} className="bg-surface-2">{o}</option>
          ))}
        </select>
      </div>
    </div>
  );
}

/** Run frame window editor. Two optional integer bounds; both empty = all
 *  frames. Writes through to global.start_frame/end_frame (→ ma_cap --start/--end). */
function FrameRangeField({
  start, end, dirty, editable, onChange,
}: {
  start: number | null;
  end: number | null;
  dirty?: boolean;
  editable: boolean;
  onChange: (start: number | null, end: number | null) => void;
}) {
  const parse = (v: string): number | null => {
    const t = v.trim();
    if (t === '') return null;
    const n = parseInt(t, 10);
    return Number.isFinite(n) && n >= 0 ? n : null;
  };
  const label = 'Frame range';
  if (!editable) {
    const txt = start == null && end == null ? 'All frames' : `${start ?? 0} – ${end ?? 'end'}`;
    return (
      <div>
        <FieldLabel label={label} />
        <div className="px-2.5 py-1.5 text-foreground-muted text-xs font-mono">{txt}</div>
      </div>
    );
  }
  const inputCls = `w-24 bg-surface-2 border ${dirty ? 'border-status-pending/60' : 'border-border'} rounded-md px-2.5 py-1.5 text-foreground text-xs font-mono tabular-nums focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition-colors placeholder:text-foreground-faint`;
  return (
    <div>
      <FieldLabel label={label} dirty={dirty} />
      <div className="flex items-center gap-2 flex-wrap">
        <input
          type="number" min={0} value={start ?? ''} placeholder="start"
          onChange={e => onChange(parse(e.target.value), end)}
          className={inputCls}
        />
        <span className="text-foreground-faint text-xs">→</span>
        <input
          type="number" min={0} value={end ?? ''} placeholder="end"
          onChange={e => onChange(start, parse(e.target.value))}
          className={inputCls}
        />
        {(start != null || end != null) && (
          <button
            type="button"
            onClick={() => onChange(null, null)}
            className="inline-flex items-center gap-1 px-2 py-1 text-[11px] text-foreground-muted hover:text-foreground border border-border rounded-md hover:border-border-strong transition-colors"
            title="Clear the slice — process every frame"
          >
            <X className="w-3 h-3" /> All frames
          </button>
        )}
      </div>
      <div className="text-foreground-faint text-[10px] mt-1">
        Passed to <code className="font-mono">ma_cap</code> as <code className="font-mono">--start/--end</code>; downstream steps inherit it. Leave both empty to process every frame.
      </div>
    </div>
  );
}

function ArrayField({
  label, values, onChange, placeholder, dirty, editable, mono, hint,
}: {
  label: string;
  values: string[];
  onChange: (v: string[]) => void;
  placeholder?: string;
  dirty?: boolean;
  editable: boolean;
  mono?: boolean;
  hint?: string;
}) {
  if (!editable) {
    if (values.length === 0) {
      return (
        <div>
          <FieldLabel label={label} />
          <span className="text-foreground-faint text-xs italic">none</span>
        </div>
      );
    }
    return (
      <div>
        <FieldLabel label={label} />
        <div className="flex flex-wrap gap-1">
          {values.map((v, i) => (
            <code key={i} className="bg-surface-2 border border-border-subtle text-foreground-muted px-2 py-0.5 rounded text-[11px] font-mono break-all">{v}</code>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div>
      <FieldLabel label={label} dirty={dirty} />
      <div className="space-y-1">
        {values.map((v, i) => (
          <div key={i} className="flex gap-1">
            <input
              type="text"
              value={v}
              onChange={e => {
                const next = [...values];
                next[i] = e.target.value;
                onChange(next);
              }}
              placeholder={placeholder}
              className={`flex-1 bg-surface-2 border ${dirty ? 'border-status-pending/60' : 'border-border'} rounded-md px-2.5 py-1.5 text-foreground text-xs focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition-colors placeholder:text-foreground-faint ${mono ? 'font-mono' : ''}`}
            />
            <button
              type="button"
              onClick={() => {
                const next = values.filter((_, idx) => idx !== i);
                onChange(next);
              }}
              className="px-2 text-foreground-subtle hover:text-status-failed border border-border rounded-md hover:border-status-failed/50 transition-colors"
              title="Remove"
            >
              <X className="w-3 h-3" />
            </button>
          </div>
        ))}
        <button
          type="button"
          onClick={() => onChange([...values, ''])}
          className="inline-flex items-center gap-1 px-2 py-1 text-[11px] text-foreground-muted hover:text-foreground border border-border rounded-md hover:border-border-strong transition-colors"
        >
          <Plus className="w-3 h-3" />
          Add
        </button>
        {hint && <div className="text-foreground-faint text-[10px]">{hint}</div>}
      </div>
    </div>
  );
}

function FieldLabel({ label, dirty }: { label: string; dirty?: boolean }) {
  return (
    <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-foreground-subtle font-medium mb-1">
      {label}
      {dirty && <span className="inline-block w-1 h-1 rounded-full bg-status-pending" title="Modified from preset" />}
    </div>
  );
}
