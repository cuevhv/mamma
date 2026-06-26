/** World up-axis for the rig preview / pipeline. Signed cardinal axis, since
 *  rigs are not all Z-up (e.g. an OpenCV rig can be −Y / "Y down"). */
export type UpAxis = 'x' | 'y' | 'z' | '-x' | '-y' | '-z';
/** A user choice: a concrete signed axis, or "auto" (detect from geometry). */
export type UpAxisValue = UpAxis | 'auto';

const OPTIONS: { value: UpAxisValue; label: string }[] = [
  { value: 'auto', label: 'Auto (detect)' },
  { value: 'x',  label: '+X  (X up)' },
  { value: '-x', label: '−X  (X down)' },
  { value: 'y',  label: '+Y  (Y up)' },
  { value: '-y', label: '−Y  (Y down)' },
  { value: 'z',  label: '+Z  (Z up)' },
  { value: '-z', label: '−Z  (Z down)' },
];

/** Pretty signed-axis label, e.g. "-y" → "−Y". */
export function upAxisLabel(a: string): string {
  const s = a.startsWith('-') ? '−' : '+';
  return `${s}${a.replace('-', '').toUpperCase()}`;
}

/**
 * Dropdown for the world up-axis (6 signed options + Auto). Auto-detect is the
 * default; this control is the override. When `detected` is given, the Auto
 * option shows what was detected, e.g. "Auto (detected −Y)".
 */
export function UpAxisSelect({
  value, onChange, detected, className,
}: {
  value: UpAxisValue;
  onChange: (v: UpAxisValue) => void;
  detected?: UpAxis | null;
  className?: string;
}) {
  return (
    <label className={`inline-flex items-center gap-1.5 text-xs text-foreground-faint ${className ?? ''}`}>
      up-axis
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as UpAxisValue)}
        title="Which world axis points up. Auto-detects from the camera geometry; override if it looks wrong."
        className="bg-surface-2 border border-border rounded px-1.5 py-0.5 text-foreground text-xs focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition-colors"
      >
        {OPTIONS.map((o) => (
          <option key={o.value} value={o.value}>
            {o.value === 'auto' && detected ? `Auto (detected ${upAxisLabel(detected)})` : o.label}
          </option>
        ))}
      </select>
    </label>
  );
}
