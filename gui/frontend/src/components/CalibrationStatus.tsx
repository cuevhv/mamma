import { useEffect, useState } from 'react';
import { Loader2, Check, AlertTriangle, Video } from 'lucide-react';
import { toast } from 'sonner';
import { RerunWebViewer, openRrdNative } from './RerunWebViewer';
import { UpAxis, UpAxisValue, UpAxisSelect, upAxisLabel } from './UpAxisToggle';

/** Result of `/api/calib/inspect` (also the shape of preflight's `calibration`). */
export interface CalibInfo {
  ok: boolean;
  error?: string | null;
  cameraCount?: number;
  cameraNames?: string[];
  distortionModels?: string[];
  sourceFormat?: string;
  upAxis?: UpAxis;            // auto-detected world up-axis
  upAxisConfidence?: number;
}

/** Friendly labels for the calibration file format (`Calibration.source_format`). */
const FORMAT_LABELS: Record<string, string> = {
  yaml: 'MAMMA YAML',
  opencv: 'OpenCV FileStorage',
  easymocap: 'EasyMocap',
  xcp: 'Vicon XCP',
  json: 'OpenCV JSON',
};

/**
 * Reusable live calibration status shown under a calibration-path input.
 *
 * One component drives every calib entry point: it validates the file (camera
 * count + distortion) via `/api/calib/inspect`, offers a "Preview camera rig"
 * button, and owns the Rerun viewer overlay + native-open wiring.
 *
 * Resolution mirrors the pipeline: pass `baseDir` (capture.json dir) or
 * `captureJsonPath` so relative `../calib/foo.yaml` paths work.
 */
export function CalibrationStatus({
  calibPath, baseDir, captureJsonPath, upAxis: controlledUpAxis, onUpAxisChange,
}: {
  calibPath?: string;
  baseDir?: string;
  captureJsonPath?: string;
  /** Controlled up-axis choice (e.g. bound to capture.json `up_axis`). When
   *  `onUpAxisChange` is given the host owns the value; otherwise it's internal. */
  upAxis?: UpAxisValue;
  onUpAxisChange?: (v: UpAxisValue) => void;
}) {
  const [info, setInfo] = useState<CalibInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [rrd, setRrd] = useState<string | null>(null);
  const [rrdAxis, setRrdAxis] = useState<UpAxis | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [internalUpAxis, setInternalUpAxis] = useState<UpAxisValue>('auto');

  const upAxis = controlledUpAxis ?? internalUpAxis;
  const setUpAxis = (v: UpAxisValue) => (onUpAxisChange ? onUpAxisChange(v) : setInternalUpAxis(v));

  const hasInput = !!(calibPath?.trim() || captureJsonPath);

  // Debounced validation.
  useEffect(() => {
    if (!hasInput) { setInfo(null); return; }
    setLoading(true);
    const t = setTimeout(async () => {
      try {
        const res = await fetch('/api/calib/inspect', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ calibPath, baseDir, captureJsonPath }),
        });
        const data = await res.json();
        setInfo(data);
      } catch {
        setInfo({ ok: false, error: 'Could not reach the backend.' });
      } finally {
        setLoading(false);
      }
    }, 400);
    return () => clearTimeout(t);
  }, [hasInput, calibPath, baseDir, captureJsonPath]);

  const preview = async (axis: UpAxisValue = upAxis) => {
    if (previewBusy) return;
    setPreviewBusy(true);
    try {
      const res = await fetch('/api/calib/preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ calibPath, baseDir, captureJsonPath, upAxis: axis }),
      });
      const data = await res.json();
      if (!res.ok) {
        toast.error(data.error || `Could not build camera-rig preview (${res.status})`);
        return;
      }
      setRrd(data.rrdPath);
      setRrdAxis(data.upAxis ?? null);   // the resolved signed axis (for the title)
    } catch {
      toast.error('Failed to reach the backend for the camera-rig preview.');
    } finally {
      setPreviewBusy(false);
    }
  };

  if (!hasInput) return null;

  return (
    <div className="space-y-1 text-xs">
      {loading && (
        <span className="inline-flex items-center gap-1.5 text-foreground-muted">
          <Loader2 className="w-3.5 h-3.5 animate-spin" /> Checking calibration…
        </span>
      )}
      {!loading && info && !info.ok && (
        <span className="inline-flex items-start gap-1.5 text-status-failed">
          <AlertTriangle className="w-3.5 h-3.5 mt-px shrink-0" /> {info.error || 'Invalid calibration'}
        </span>
      )}
      {!loading && info?.ok && (
        <>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <span className="inline-flex items-center gap-1.5 text-status-completed">
              <Check className="w-3.5 h-3.5" />
              {info.cameraCount} camera{info.cameraCount === 1 ? '' : 's'}
              {info.sourceFormat ? ` · ${FORMAT_LABELS[info.sourceFormat] ?? info.sourceFormat}` : ''}
            </span>
            <button
              type="button"
              onClick={() => preview()}
              disabled={previewBusy}
              className="inline-flex items-center gap-1.5 text-primary hover:underline disabled:opacity-50 disabled:no-underline"
              title="Open a 3D view of the camera rig to confirm the convention — a correct rig shows cameras around the scene, looking inward."
            >
              <Video className="w-3.5 h-3.5" /> {previewBusy ? 'Building preview…' : 'Preview camera rig'}
            </button>
            <UpAxisSelect
              value={upAxis}
              detected={info.upAxis ?? null}
              // Switching axis re-renders an open preview with that up-axis.
              onChange={(a) => { setUpAxis(a); if (rrd) preview(a); }}
            />
          </div>
          <div className="text-foreground-faint">
            {onUpAxisChange
              ? 'The Up-axis is saved with this capture. It is by default auto-detected. Open the rig preview to confirm it matches your real setup.'
              : "We can't verify the camera poses automatically — preview the rig and check it matches your real setup."}
          </div>
        </>
      )}

      {rrd && (
        <RerunWebViewer
          key={rrd}
          rrdPath={rrd}
          fileName={`camera rig${rrdAxis ? ` · ${upAxisLabel(rrdAxis)} up` : ''}`}
          onClose={() => setRrd(null)}
          onOpenNative={(fresh) => openRrdNative(rrd, fresh)}
        />
      )}
    </div>
  );
}
