import { useState } from 'react';
import { Info, FolderOpen, FileJson } from 'lucide-react';
import { toast } from 'sonner';
import { CalibrationStatus } from './CalibrationStatus';
import { UpAxisValue } from './UpAxisToggle';

interface Props {
  /** Called with the new capture-json's relative path on success. */
  onCreated: (relativePath: string) => void;
}

/**
 * Form that mints a new capture.json from an images-root + calibration
 * pair via `POST /api/captures/generate-json`. Mounted in the Captures
 * tab beneath the "+ New capture" button.
 *
 * The backend auto-detects sequences from the images-root's immediate
 * subdirectories (excluding `logs`) and cameras from the first
 * sequence's subdirectories. `cam_fps` defaults to 30 and can be
 * tuned later in the manage page.
 */
export function InlineCaptureJsonForm({ onCreated }: Props) {
  const [ioiRoot, setIoiRoot] = useState('');
  const [calib, setCalib] = useState('');
  const [outputName, setOutputName] = useState('');
  const [upAxis, setUpAxis] = useState<UpAxisValue>('auto');   // saved into capture.json
  const [submitting, setSubmitting] = useState(false);

  const handleCreate = async () => {
    if (!ioiRoot || !calib) {
      toast.error('Footage root and calibration are required');
      return;
    }
    setSubmitting(true);
    try {
      const res = await fetch('/api/captures/generate-json', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ioiRoot, calib, outputName, upAxis }),
      });
      const data = await res.json();
      if (res.ok) {
        toast.success(`Created ${data.outputName}.json (${data.sequenceCount} sequences)`);
        onCreated(data.path);
        // Reset the form so the user can immediately add another if they want.
        setIoiRoot('');
        setCalib('');
        setOutputName('');
        setUpAxis('auto');
      } else {
        toast.error(`Failed: ${data.error || 'Unknown error'}`);
      }
    } catch (e) {
      toast.error('Error creating capture JSON. See console for details.');
      console.error(e);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* What this does — kept terse but explicit so users know what to expect. */}
      <div className="bg-surface-2/40 border border-border-subtle rounded-md p-3 flex items-start gap-2.5">
        <Info className="w-4 h-4 text-primary flex-shrink-0 mt-0.5" />
        <div className="text-foreground-muted text-xs leading-relaxed">
          Provide a <span className="text-foreground">footage root</span> (images or videos) and a{' '}
          <span className="text-foreground">calibration file</span>. The backend will:
          <ul className="list-disc ml-4 mt-1 space-y-0.5">
            <li>Detect <span className="text-foreground">sequences</span> from each subfolder of the footage root (excluding <code className="font-mono">logs</code>).</li>
            <li>Detect <span className="text-foreground">cameras</span> from the first sequence's subfolders (any naming — <code className="font-mono">cam01</code>, <code className="font-mono">A001</code>, <code className="font-mono">IOI_01</code>, …).</li>
            <li>Set <code className="font-mono">cam_fps=30</code> (override later in the manage page).</li>
          </ul>
          You can edit any field afterwards via <span className="text-foreground">Edit</span> on the new capture's row.
        </div>
      </div>

      <Field
        label="Footage root"
        value={ioiRoot}
        onChange={setIoiRoot}
        placeholder="/path/to/footage_root"
        icon={<FolderOpen className="w-3.5 h-3.5" />}
        required
      />
      <pre className="text-foreground-faint text-[11px] font-mono leading-relaxed bg-surface-2/40 border border-border-subtle rounded-md p-2.5 -mt-2 whitespace-pre overflow-x-auto">
{`footage_root/
├── sequence_01/        ← one folder per sequence
│   ├── cam01/          ← per camera: either a folder of frames…
│   │   ├── 0001.jpg     (.jpg / .png / …)
│   │   └── …
│   └── cam02.mp4       …or a single video file (.mp4 / .mov / …)
└── sequence_02/`}
      </pre>
      <Field
        label="Calibration"
        value={calib}
        onChange={setCalib}
        placeholder="/path/to/calib.yaml  (or an OpenCV / EasyMocap folder)"
        icon={<FileJson className="w-3.5 h-3.5" />}
        required
      />
      <div className="flex items-start gap-3">
        <span className="w-32 flex-shrink-0" aria-hidden />
        <CalibrationStatus calibPath={calib} upAxis={upAxis} onUpAxisChange={setUpAxis} />
      </div>
      <Field
        label="Capture name"
        value={outputName}
        onChange={setOutputName}
        placeholder="defaults to the last folder of the footage root"
      />

      <div className="flex justify-end">
        <button
          onClick={handleCreate}
          disabled={!ioiRoot || !calib || submitting}
          className="inline-flex items-center px-4 py-1.5 bg-primary hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed text-primary-foreground text-sm font-medium rounded-md transition"
        >
          {submitting ? 'Creating…' : 'Create capture'}
        </button>
      </div>
    </div>
  );
}

function Field({
  label, value, onChange, placeholder, icon, required,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  icon?: React.ReactNode;
  required?: boolean;
}) {
  return (
    <div className="flex items-center gap-3">
      <label className="text-foreground-muted text-xs w-32 text-right flex-shrink-0 inline-flex items-center justify-end gap-1">
        {icon}
        {label}
        {required && <span className="text-status-failed">*</span>}
      </label>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="flex-1 bg-surface-2 border border-border rounded-md px-3 py-1.5 text-foreground text-sm font-mono focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition-colors placeholder:text-foreground-faint"
      />
    </div>
  );
}
