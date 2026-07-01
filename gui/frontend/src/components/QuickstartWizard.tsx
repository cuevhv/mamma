import { useEffect, useMemo, useState } from 'react';
import { X, ChevronLeft, ChevronRight, ChevronDown, Check, AlertCircle, AlertTriangle, Download, Play, ArrowUpRight } from 'lucide-react';
import { toast } from 'sonner';
import { PipelineOverview } from './PipelineOverview';
import { formatTaskId } from './shared/formatTaskId';

/**
 * Asset IDs (from gui/backend/data_readiness.py:ASSETS) that the demo
 * pipeline genuinely requires *on disk* before submission. The demo
 * runs the SAM2 preset (configs/examples/presets/quick_sam2.yaml),
 * which requires the local SAM2 hiera-large checkpoint — so `sam2` is
 * in this list. If we ever switch the demo to a SAM3 preset, drop
 * `sam2` (SAM 3 self-resolves through the HF cache and the warning
 * doesn't apply). Keep this list in sync with the chosen preset.
 */
const DEMO_REQUIRED_ASSET_IDS = [
  'yolo',
  'sam2',
  'smplx_locked_head',
  'downsampled_verts',
  'ma_2d_checkpoint',
] as const;

interface ReadinessAsset {
  id: string;
  label: string;
  present: boolean;
}

interface ReadinessStatus {
  items: ReadinessAsset[];
  ready: number;
  total: number;
}

/**
 * Inspect a readiness snapshot and report which demo prerequisites are
 * missing. Returns labels rather than ids so the warning copy reads
 * naturally ("MammaNet landmark net (ma_2d) is missing").
 */
function computeDemoMissing(status: ReadinessStatus | null): string[] {
  if (!status) return [];
  const byId = new Map(status.items.map(a => [a.id, a]));
  const missing: string[] = [];
  for (const id of DEMO_REQUIRED_ASSET_IDS) {
    const a = byId.get(id);
    if (!a || !a.present) missing.push(a?.label ?? id);
  }
  return missing;
}

const DATASET_URL = 'https://mamma.is.tue.mpg.de/download.php';
/** Capture pre-selected in step 1 — the lightweight 4-cam demo that
 *  lives at data/mamma_example/ (a symlink curtain over an iphones
 *  outdoors sequence). Smaller than the Breakdance smoke fixture, so
 *  the wizard's "Try the example" path runs end-to-end on a modest
 *  GPU box in a few minutes. */
const DEFAULT_CAPTURE_NAME = 'mamma_example';
/** Preset pre-selected in step 2. Shipped presets are "quick", "full",
 *  "debug", and "full_tensorrt"; all use SAM2 by default to avoid the
 *  Hugging Face login / gated-access step SAM 3 requires. "quick" is the
 *  ~5min smoke variant and the right default for a first-time demo. */
const DEFAULT_PRESET_DISPLAY = 'quick';
/** For the `quick` preset, mirror smoke_test.py's 4-camera cap. Combined
 *  with the preset's short frame slice (start_frame:60 / end_frame:120,
 *  ~2 s), this keeps the demo bounded to ~5 min on a GPU. `full` runs the
 *  capture's full camera list. */
const QUICK_CAMERA_CAP = 4;

interface CaptureSummary {
  id: string;
  captureName: string;
  jsonPath: string;
  /** Resolved on-disk data root, displayed for orientation. Repo-root
   *  relative when the data lives under the repo (e.g.
   *  `./data/mamma_markerless_iphones/indoors`), absolute otherwise.
   *  Empty string if the capture has no resolvable root. */
  dataPath?: string;
  seqNames: string[];
  cams: string[];
  source?: 'user' | 'example' | 'db';
  releasedDataPresent?: boolean;
}

interface PresetSummary {
  name: string;
  displayName: string;
  description: string;
  path: string;
  source?: 'user' | 'example';
}

interface PresetDigestStep {
  name: string;
  enabled: boolean;
}
interface PresetDigest {
  name: string;
  path: string;
  steps: PresetDigestStep[];
}

interface Props {
  open: boolean;
  onClose: () => void;
  /** Called with the new task's id after a successful submit. The host
   *  page is expected to navigate to the Tasks tab so the user can watch
   *  the matrix light up. */
  onSubmitted: (taskId: number) => void;
}

/**
 * Three-step modal for first-time users: pick an example capture, pick
 * an example preset, review and submit. Teaches the vocabulary one term
 * at a time and short-circuits the full NewTaskForm.
 *
 * Self-contained: no edits to NewTaskForm.tsx. Reuses the same backend
 * endpoints (GET /api/captures, GET /api/task-presets,
 * GET /api/task-presets/<name>/digest, POST /api/tasks) so a wizard-
 * submitted task is indistinguishable from a form-submitted one.
 */
export function QuickstartWizard({ open, onClose, onSubmitted }: Props) {
  const [step, setStep] = useState<1 | 2 | 3>(1);
  const [captures, setCaptures] = useState<CaptureSummary[]>([]);
  const [presets, setPresets] = useState<PresetSummary[]>([]);
  const [captureName, setCaptureName] = useState<string>('');
  const [presetName, setPresetName] = useState<string>('');
  const [digest, setDigest] = useState<PresetDigest | null>(null);
  const [loadError, setLoadError] = useState<string>('');
  const [submitting, setSubmitting] = useState(false);
  /** Pipeline-asset readiness snapshot. Null while loading; the warning
   *  banner stays hidden until we know — better than flashing a "missing"
   *  state for a frame and then clearing it. */
  const [readiness, setReadiness] = useState<ReadinessStatus | null>(null);

  // Reset on open so a closed-and-reopened wizard always starts at step 1.
  useEffect(() => {
    if (!open) return;
    setStep(1);
    setLoadError('');
    setSubmitting(false);
    setReadiness(null);

    // Asset readiness is fetched in parallel with the captures/presets
    // load. We don't gate the wizard's main load on this — a slow asset
    // probe shouldn't block the user from selecting their capture.
    fetch('/api/data/readiness/status')
      .then(r => (r.ok ? r.json() : null))
      .then((s: ReadinessStatus | null) => setReadiness(s))
      .catch(() => setReadiness(null));

    Promise.all([
      fetch('/api/captures').then(r => r.ok ? r.json() : []),
      fetch('/api/task-presets').then(r => r.ok ? r.json() : []),
    ])
      .then(([caps, prs]: [CaptureSummary[], PresetSummary[]]) => {
        const examples = caps.filter(c => c.source === 'example');
        const examplePresets = prs.filter(p => p.source === 'example');
        if (examples.length === 0 || examplePresets.length === 0) {
          setLoadError("Couldn't load the shipped examples. Use the regular Submit form instead.");
          return;
        }
        // Pin the canonical demo capture to the top of the list so
        // first-time users see it before scrolling through alternatives.
        const sortedExamples = [...examples].sort((a, b) => {
          if (a.captureName === DEFAULT_CAPTURE_NAME) return -1;
          if (b.captureName === DEFAULT_CAPTURE_NAME) return 1;
          return 0;
        });
        setCaptures(sortedExamples);
        setPresets(examplePresets);
        // Prefer Breakdance + quick when present; otherwise fall back to
        // the first item in each list so the wizard still works on a
        // future repo that ships different examples.
        const breakdance = examples.find(c => c.captureName === DEFAULT_CAPTURE_NAME);
        setCaptureName((breakdance ?? examples[0]).captureName);
        const quick = examplePresets.find(p => p.displayName === DEFAULT_PRESET_DISPLAY);
        setPresetName((quick ?? examplePresets[0]).name);
      })
      .catch(() => setLoadError("Couldn't reach the backend. Try again or use the regular Submit form."));
  }, [open]);

  // Fetch the digest for the picked preset — we need the step list to
  // build the POST payload's `processes` array, matching what
  // NewTaskForm sends.
  useEffect(() => {
    if (!open || !presetName) { setDigest(null); return; }
    let cancelled = false;
    fetch(`/api/task-presets/${encodeURIComponent(presetName)}/digest`)
      .then(r => r.ok ? r.json() : null)
      .then((d: PresetDigest | null) => { if (!cancelled) setDigest(d); })
      .catch(() => { if (!cancelled) setDigest(null); });
    return () => { cancelled = true; };
  }, [open, presetName]);

  // Hook must run on every render — even when the modal is closed —
  // so the hook count stays stable. Keep this above the `!open` early
  // return.
  const missingDemoAssets = useMemo(() => computeDemoMissing(readiness), [readiness]);

  if (!open) return null;

  const selectedCapture = captures.find(c => c.captureName === captureName) ?? null;
  const selectedPreset = presets.find(p => p.name === presetName) ?? null;
  const dataReady = !!selectedCapture?.releasedDataPresent;
  const isQuick = selectedPreset?.displayName === DEFAULT_PRESET_DISPLAY;
  const demoAssetsReady = readiness !== null && missingDemoAssets.length === 0;
  /** The wizard's overall "can submit" gate. The capture data must be on
   *  disk AND every demo-required pipeline asset must be present.
   *  Asset state of `null` (still loading) keeps Submit disabled — we
   *  refuse to submit before we know. */
  const canSubmit = dataReady && demoAssetsReady;

  // Selection rules — wizard short-circuits the multi-select UX from
  // NewTaskForm with sensible defaults per preset.
  const wizardSeqNames = selectedCapture
    ? (isQuick ? selectedCapture.seqNames.slice(0, 1) : selectedCapture.seqNames)
    : [];
  const wizardCameras = selectedCapture
    ? (isQuick ? selectedCapture.cams.slice(0, QUICK_CAMERA_CAP) : selectedCapture.cams)
    : [];

  const submit = async () => {
    if (submitting) return;
    if (!digest || !selectedCapture || !selectedPreset) return;
    if (!canSubmit) return; // step 3's Submit button is already disabled when not ready
    setSubmitting(true);
    const processes = digest.steps.filter(s => s.enabled).map(s => s.name);
    try {
      const res = await fetch('/api/tasks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          captureJsonPath: selectedCapture.jsonPath,
          taskJsonPath: digest.path,
          seqNames: wizardSeqNames,
          cameras: wizardCameras,
          outputDir: '',
          outputId: '',
          processes,
          taskOverrides: {},
        }),
      });
      if (res.ok) {
        const d = await res.json();
        toast.success(`Task ${formatTaskId(d.taskId)} started`);
        onSubmitted(d.taskId);
      } else {
        const err = await res.json().catch(() => ({}));
        toast.error(`Failed: ${err.error || res.statusText}`);
      }
    } catch (e) {
      toast.error('Error starting task. See console for details.');
      console.error(e);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/65 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="w-full max-w-2xl max-h-[90vh] overflow-y-auto bg-surface-1 border border-border rounded-xl shadow-2xl shadow-black/50 ring-1 ring-inset ring-white/[0.03]"
      >
        {/* Header */}
        <div className="flex items-start justify-between px-6 py-4 border-b border-border-subtle">
          <div>
            <div className="text-foreground-subtle text-[11px] uppercase tracking-[0.18em]">
              Quickstart · Step {step} of 3
            </div>
            <h2 className="text-foreground text-xl tracking-tight font-medium mt-1">
              {step === 1 && 'Pick a capture'}
              {step === 2 && 'Pick a pipeline configuration'}
              {step === 3 && 'Review and submit the task'}
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 -mt-1 -mr-1 rounded-md text-foreground-muted hover:text-foreground hover:bg-surface-2 transition-colors"
            aria-label="Close"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Progress strip */}
        <div className="flex gap-1 px-6 pt-4">
          {[1, 2, 3].map(n => (
            <div
              key={n}
              className={`h-1 flex-1 rounded-full transition-colors ${
                n <= step ? 'bg-primary' : 'bg-surface-3'
              }`}
            />
          ))}
        </div>

        {/* Asset readiness warning — surfaces above all step bodies so
            the user can't miss why the wizard's Submit might be blocked. */}
        {!loadError && readiness !== null && missingDemoAssets.length > 0 && (
          <div className="px-6 pt-4">
            <DemoAssetWarning missing={missingDemoAssets} />
          </div>
        )}

        {/* Body */}
        <div className="px-6 py-5 min-h-[280px]">
          {loadError ? (
            <div className="text-status-failed text-sm flex items-start gap-2">
              <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
              <span>{loadError}</span>
            </div>
          ) : (
            <>
              {step === 1 && (
                <Step1Captures
                  captures={captures}
                  selected={captureName}
                  onSelect={setCaptureName}
                />
              )}
              {step === 2 && (
                <Step2Presets
                  presets={presets}
                  selected={presetName}
                  onSelect={setPresetName}
                />
              )}
              {step === 3 && selectedCapture && selectedPreset && (
                <Step3Review
                  capture={selectedCapture}
                  preset={selectedPreset}
                  seqNames={wizardSeqNames}
                  cameras={wizardCameras}
                  isQuick={isQuick}
                  dataReady={dataReady}
                />
              )}
            </>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between px-6 py-4 border-t border-border-subtle bg-surface-1/80">
          <button
            onClick={() => step > 1 ? setStep((step - 1) as 1 | 2 | 3) : onClose()}
            className="inline-flex items-center gap-1.5 px-3 py-2 text-sm text-foreground-muted hover:text-foreground transition-colors"
          >
            <ChevronLeft className="w-4 h-4" />
            {step === 1 ? 'Cancel' : 'Back'}
          </button>

          {step < 3 ? (
            <button
              onClick={() => setStep((step + 1) as 1 | 2 | 3)}
              disabled={!!loadError || !captureName || !presetName}
              className="inline-flex items-center gap-1.5 px-4 py-2 bg-primary text-primary-foreground hover:opacity-90 rounded-md text-sm font-medium transition-opacity shadow-sm shadow-black/30 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              Next
              <ChevronRight className="w-4 h-4" />
            </button>
          ) : (
            <button
              onClick={submit}
              disabled={!digest || !canSubmit || submitting}
              title={
                !demoAssetsReady && readiness !== null
                  ? 'Pipeline assets are missing. Download them from the Home page before running the demo.'
                  : undefined
              }
              className="inline-flex items-center gap-1.5 px-4 py-2 bg-primary text-primary-foreground hover:opacity-90 rounded-md text-sm font-medium transition-opacity shadow-sm shadow-black/30 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <Play className="w-3.5 h-3.5" />
              {submitting ? 'Starting…' : 'Submit'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------

function Step1Captures({
  captures, selected, onSelect,
}: {
  captures: CaptureSummary[];
  selected: string;
  onSelect: (name: string) => void;
}) {
  const COLLAPSED_COUNT = 4;
  // Auto-expand when the user's pick lives below the fold — they should
  // be able to see what's selected without hunting through the toggle.
  const selectedIdx = captures.findIndex(c => c.captureName === selected);
  const initialExpanded = selectedIdx >= COLLAPSED_COUNT;
  const [expanded, setExpanded] = useState(initialExpanded);
  const hidden = Math.max(0, captures.length - COLLAPSED_COUNT);
  const displayed = expanded || hidden === 0
    ? captures
    : captures.slice(0, COLLAPSED_COUNT);

  return (
    <div className="space-y-4">
      <p className="text-foreground-muted text-sm leading-relaxed">
        A <span className="text-foreground font-medium">capture</span> is one multi-view recording session containing multiple sequences.
      </p>
      <div className="space-y-1.5">
        {displayed.map(c => (
          <CaptureRow
            key={c.captureName}
            capture={c}
            selected={selected === c.captureName}
            onSelect={() => onSelect(c.captureName)}
          />
        ))}
      </div>
      {hidden > 0 && (
        <button
          type="button"
          onClick={() => setExpanded(v => !v)}
          className="inline-flex items-center gap-1.5 text-foreground-muted hover:text-foreground text-xs transition-colors"
        >
          <ChevronDown
            className={`w-3.5 h-3.5 transition-transform duration-150 ${expanded ? 'rotate-180' : ''}`}
          />
          {expanded ? 'Show fewer' : `Show ${hidden} more capture${hidden === 1 ? '' : 's'}`}
        </button>
      )}
    </div>
  );
}

function CaptureRow({
  capture, selected, onSelect,
}: {
  capture: CaptureSummary;
  selected: boolean;
  onSelect: () => void;
}) {
  const ready = !!capture.releasedDataPresent;
  return (
    <button
      type="button"
      onClick={onSelect}
      className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-md border text-left transition-colors ${
        selected
          ? 'bg-primary-muted border-primary/55 ring-1 ring-inset ring-white/10'
          : 'bg-surface-2 border-border hover:border-border-strong hover:bg-surface-3'
      }`}
    >
      <span
        className={`w-3.5 h-3.5 rounded-full border flex items-center justify-center flex-shrink-0 ${
          selected ? 'border-primary bg-primary' : 'border-border-strong bg-transparent'
        }`}
      >
        {selected && <span className="w-1.5 h-1.5 rounded-full bg-primary-foreground" />}
      </span>
      <div className="flex-1 min-w-0">
        <div className="flex items-baseline gap-2 min-w-0">
          <span className="text-foreground text-sm font-medium truncate">{capture.captureName}</span>
          {capture.dataPath && (
            <span
              className="text-foreground-subtle text-[11px] font-mono truncate"
              title={capture.dataPath}
            >
              {capture.dataPath}
            </span>
          )}
        </div>
        <div className="text-foreground-faint text-[11px] mt-0.5">
          {capture.cams.length} cam{capture.cams.length === 1 ? '' : 's'} ·{' '}
          {capture.seqNames.length} seq{capture.seqNames.length === 1 ? '' : 's'}
        </div>
      </div>
      <div className="flex items-center gap-1.5 text-[11px]">
        <span
          className={`w-1.5 h-1.5 rounded-full ${
            ready ? 'bg-status-completed' : 'bg-status-pending'
          }`}
        />
        <span className={ready ? 'text-status-completed' : 'text-status-pending'}>
          {ready ? 'data ready' : 'data not downloaded'}
        </span>
      </div>
    </button>
  );
}

// ---------------------------------------------------------------------------

function Step2Presets({
  presets, selected, onSelect,
}: {
  presets: PresetSummary[];
  selected: string;
  onSelect: (name: string) => void;
}) {
  const presetBlurb = (displayName: string): string => {
    if (displayName === 'quick') return '~5 min · 4 cams · ~2 s slice · SAM2';
    if (displayName === 'full') return 'slower · all cams · all frames · SAM2';
    return '';
  };
  return (
    <div className="space-y-4">
      <p className="text-foreground-muted text-sm leading-relaxed">
        Start with{' '}
        <span className="font-mono text-foreground">quick</span> for a 5 to 10-minute run.
      </p>
      <div className="space-y-1.5">
        {presets.map(p => (
          <button
            key={p.name}
            type="button"
            onClick={() => onSelect(p.name)}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-md border text-left transition-colors ${
              selected === p.name
                ? 'bg-primary-muted border-primary/55 ring-1 ring-inset ring-white/10'
                : 'bg-surface-2 border-border hover:border-border-strong hover:bg-surface-3'
            }`}
          >
            <span
              className={`w-3.5 h-3.5 rounded-full border flex items-center justify-center flex-shrink-0 ${
                selected === p.name ? 'border-primary bg-primary' : 'border-border-strong bg-transparent'
              }`}
            >
              {selected === p.name && <span className="w-1.5 h-1.5 rounded-full bg-primary-foreground" />}
            </span>
            <div className="flex-1 min-w-0 flex items-baseline gap-2">
              <span className="text-foreground text-sm font-medium font-mono flex-shrink-0">{p.displayName}</span>
              <span className="text-foreground-faint text-[11px] truncate" title={presetBlurb(p.displayName)}>
                {presetBlurb(p.displayName)}
              </span>
            </div>
          </button>
        ))}
      </div>
      <PipelineOverview compact />
    </div>
  );
}

// ---------------------------------------------------------------------------

function Step3Review({
  capture, preset, seqNames, cameras, isQuick, dataReady,
}: {
  capture: CaptureSummary;
  preset: PresetSummary;
  seqNames: string[];
  cameras: string[];
  isQuick: boolean;
  dataReady: boolean;
}) {
  return (
    <div className="space-y-4">
      <p className="text-foreground-muted text-sm leading-relaxed">
        You're about to submit a <span className="text-foreground font-medium">task</span> to run the full MAMMA pipeline.
      </p>
      <div className="bg-surface-2/60 border border-border-subtle rounded-lg divide-y divide-border-subtle text-sm">
        <ReviewRow label="Capture" value={capture.captureName} />
        <ReviewRow label="Preset" value={preset.displayName} mono />
        <ReviewRow
          label="Sequences"
          value={
            seqNames.length === capture.seqNames.length
              ? `all ${seqNames.length}`
              : `${seqNames.length} of ${capture.seqNames.length} (${seqNames[0]}${seqNames.length > 1 ? ', …' : ''})`
          }
        />
        <ReviewRow
          label="Cameras"
          value={
            cameras.length === capture.cams.length
              ? `all ${cameras.length}`
              : `${cameras.length} of ${capture.cams.length}`
          }
        />
        <ReviewRow label="Estimated time" value={isQuick ? '~5-10 minutes on a GPU' : 'slower - full run'} />
      </div>
      {dataReady ? (
        <div className="flex items-center gap-2 text-status-completed text-xs">
          <Check className="w-3.5 h-3.5" />
          <span>Example data is on disk — ready to run.</span>
        </div>
      ) : (
        <div className="bg-surface-2/60 border border-status-pending/40 rounded-lg p-3 text-sm">
          <div className="flex items-start gap-2 text-status-pending">
            <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <div className="flex-1">
              <div className="font-medium">Download the dataset first</div>
              <div className="text-foreground-muted text-xs mt-1 leading-relaxed">
                This capture's video frames are not on disk yet. Grab them from the project's
                dataset page, then re-open this wizard.
              </div>
              <a
                href={DATASET_URL}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1.5 mt-2 text-xs text-primary hover:underline"
              >
                <Download className="w-3.5 h-3.5" />
                Open dataset download page
              </a>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function ReviewRow({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-center gap-4 px-3 py-2">
      <div className="text-foreground-subtle text-[11px] uppercase tracking-wider w-28 flex-shrink-0">
        {label}
      </div>
      <div className={`text-foreground text-sm ${mono ? 'font-mono' : ''}`}>{value}</div>
    </div>
  );
}

/**
 * Banner shown above the wizard steps when one or more pipeline assets
 * the demo needs are missing on disk. The wizard's Submit button stays
 * disabled while this is visible. The user is pointed back at Home's
 * "Pipeline assets" panel, which is where every listed asset can be
 * obtained.
 */
function DemoAssetWarning({ missing }: { missing: string[] }) {
  return (
    <div className="rounded-md border border-status-pending/45 bg-status-pending-bg/40 p-3 flex items-start gap-2.5">
      <AlertTriangle className="w-4 h-4 text-status-pending mt-0.5 flex-shrink-0" aria-hidden />
      <div className="flex-1 min-w-0">
        <div className="text-foreground text-sm font-medium">
          The demo can't run yet. Pipeline assets are missing
        </div>
        <div className="text-foreground-muted text-xs mt-1 leading-relaxed">
          Download the following before submitting.
        </div>
        <ul className="mt-2 space-y-0.5">
          {missing.map(label => (
            <li key={label} className="text-foreground-subtle text-xs flex items-center gap-1.5">
              <span className="w-1 h-1 rounded-full bg-status-pending inline-block" aria-hidden />
              {label}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
