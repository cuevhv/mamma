import { useState } from 'react';
import { toast } from 'sonner';
import { Cpu, Box, Container, FileCode, Plus, X, RotateCcw, ChevronDown, ChevronRight, AlertTriangle, Pencil, Save, Trash2 } from 'lucide-react';
import { stepLabel } from './shared/stepLabels';
import { FlagCatalogPopover } from './FlagCatalogPopover';

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
    bind: string[];
    /** Run frame window. null = unset = all frames. Becomes ma_cap's
     *  --start/--end at runtime; downstream steps inherit it via the NPZ. */
    startFrame: number | null;
    endFrame: number | null;
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
    bind?: string[];
    // Run frame window. Explicit null clears the preset's range (= all frames);
    // deep-merged into global.start_frame/end_frame on the backend.
    start_frame?: number | null;
    end_frame?: number | null;
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
}

/**
 * Read-only summary by default; an inline editor when `overrides` and
 * `onOverridesChange` are provided. Shows what a preset will run (engine,
 * conda env, sif paths, flags) and lets the user tweak any of it for this
 * one submission. Edits are tracked as a deep-partial — only touched
 * fields are sent to the backend, which deep-merges them on top of the
 * preset before the form-level fields (seq_ids, cam_names, ...) land.
 */
export function PresetDigestCard({ digest, overrides, onOverridesChange, onPresetSaved, onPresetDeleted }: Props) {
  const [saveOpen, setSaveOpen] = useState(false);
  const [saveName, setSaveName] = useState('');
  const [saveDisplayName, setSaveDisplayName] = useState('');
  const [saveDescription, setSaveDescription] = useState('');
  const [saving, setSaving] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  // Body (global + steps) collapses by default to keep the form short;
  // expand to inspect or edit the preset for this submission.
  const [bodyOpen, setBodyOpen] = useState(false);

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
  const usedEngines = new Set<string>();
  for (const step of digest.steps) {
    if (!step.enabled) continue;
    const engine = (ov[step.name]?.engine ?? step.engine) || 'conda';
    usedEngines.add(engine);
  }
  const showBind = usedEngines.has('apptainer') || usedEngines.has('docker');

  // -----------------------------------------------------------------
  // Override helpers — keep mutations to a single setOv() call.
  // -----------------------------------------------------------------
  const setOv = (next: PresetOverrides) => onOverridesChange?.(next);
  const reset = () => setOv({});

  const setGlobalField = (key: 'dataset_name' | 'conda_env' | 'bind', value: any) => {
    const nextGlobal = { ...(ov.global ?? {}), [key]: value };
    setOv({ ...ov, global: nextGlobal });
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
  };
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
          <div className="text-foreground text-sm font-medium flex items-center gap-2 flex-wrap">
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
                placeholder="my_horses_v2"
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
                placeholder="My horses, v2"
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
            <div className="space-y-2 bg-surface-1/50 border border-border-subtle rounded-md p-3">
              <div className="text-foreground-subtle text-[11px] uppercase tracking-wider font-medium">Global</div>
              <FrameRangeField
                start={eff.startFrame}
                end={eff.endFrame}
                dirty={frameRangeDirty}
                editable={editable}
                onChange={setFrameRange}
              />
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
  onChange,
}: {
  step: PresetStep;
  /** Canonical position in the pipeline (0 = ma_cap, 1 = ma_masks, …).
   *  Stays stable across presets so users learn "ma_3d is step 3". */
  stepIndex: number;
  stepOverride: Record<string, any>;
  editable: boolean;
  onChange: (key: string, value: any) => void;
}) {
  const [advancedOpen, setAdvancedOpen] = useState(false);

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
      <div className="p-3 space-y-2 flex-1">

      {/* Engine + engine-specific fields */}
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

      {/* Flags — with a "View available flags" popover when editing, so
          the user can discover what the step's script actually accepts
          instead of guessing or grepping run_ma_*.py. */}
      <div>
        <ArrayField
          label="Flags"
          values={eff.flags ?? []}
          dirty={dirty('flags')}
          editable={editable}
          onChange={list => onChange('flags', list)}
          placeholder="--flag value"
          mono
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

      {/* Advanced (script + repo path) — risky, collapsed by default */}
      {editable && (
        <div className="border-t border-border-subtle pt-2">
          <button
            type="button"
            onClick={() => setAdvancedOpen(o => !o)}
            className="inline-flex items-center gap-1 text-[11px] text-foreground-subtle hover:text-foreground transition-colors"
          >
            {advancedOpen ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
            Advanced
            <AlertTriangle className="w-3 h-3 text-status-pending ml-1" />
          </button>
          {advancedOpen && (
            <div className="mt-2 space-y-2">
              <div className="text-[11px] text-status-pending bg-status-pending-bg border border-status-pending/30 rounded p-2 flex items-start gap-1.5">
                <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                <span>Editing <code className="font-mono">script</code> or <code className="font-mono">repo_path</code> can silently break the run if the path doesn't exist.</span>
              </div>
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
