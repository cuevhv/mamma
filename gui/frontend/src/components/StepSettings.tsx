/*
 * "Common settings" — friendly, type-appropriate widgets for the important
 * flags of one pipeline step, rendered from the backend schema
 * (GET /api/steps/settings). Reads/writes the SAME `flags`/`extras` the raw
 * "Advanced flags" editor uses (via ../lib/flagModel), so the two stay in sync.
 *
 * Decoupled from PresetDigest by design: it takes the preset step's `flags`
 * and `extras` plus the step's override object, and reports edits through a
 * single `onChange(key, value)` (the same setter the raw editor uses —
 * `'flags'` for the array, or a dedicated extra key like `config_file`).
 */
import { ChevronDown, ChevronRight, SlidersHorizontal, FileText, X, Loader2, AlertCircle, Check, List } from 'lucide-react';
import { Fragment, useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';

import {
  applySetting,
  isEnabled,
  readSetting,
  type SettingValue,
  type StepSetting,
} from '../lib/flagModel';
import { CodeBlock } from './shared/CodeBlock';

interface Props {
  settings: StepSetting[];
  stepName: string;                         // e.g. "ma_3d" — for the recipe peek endpoint
  flags: string[];                          // preset step.flags
  extras: Record<string, unknown>;          // preset step.extras
  override: Record<string, any>;            // this step's overrides
  editable: boolean;
  onChange: (key: string, value: any) => void;
}

export function StepSettings({ settings, stepName, flags, extras, override, editable, onChange }: Props) {
  const [open, setOpen] = useState(true);
  if (!settings || settings.length === 0) return null;

  // Effective state = preset value unless the user overrode it.
  const effFlags: string[] = (override.flags ?? flags ?? []) as string[];
  const effExtras = { ...(extras ?? {}), ...override };
  const presetCtx = { flags: flags ?? [], extras: extras ?? {} };
  const effCtx = { flags: effFlags, extras: effExtras };

  // Resolve every value once so `dependsOn` can read siblings.
  const values: Record<string, SettingValue> = {};
  for (const s of settings) values[s.id] = readSetting(s, effCtx);

  const commit = (s: StepSetting, raw: SettingValue) => {
    if (s.target.kind === 'extra') onChange(s.target.key, raw);
    else onChange('flags', applySetting(s, raw, effFlags));
  };

  const renderRow = (s: StepSetting) => (
    <SettingRow
      setting={s}
      stepName={stepName}
      value={values[s.id]}
      dirty={!sameValue(values[s.id], readSetting(s, presetCtx))}
      disabled={!editable || !isEnabled(s, values)}
      onChange={(v) => commit(s, v)}
    />
  );

  // Layout: value controls (select / number / slider / text) sit in a 2-col
  // grid — selects and any control carrying a note take the full width; short
  // ones pair two-up. Toggles are gathered into one pill cluster that lives in
  // the same grid: a single toggle slots into a free half-cell beside a lone
  // field (one tidy line), while several toggles span their own full row so
  // the pills don't wrap raggedly.
  const fieldSettings = settings.filter((s) => s.widget !== 'toggle');
  const toggleSettings = settings.filter((s) => s.widget === 'toggle');
  const fullWidth = (s: StepSetting) => s.widget === 'select' || !!s.note;

  return (
    <div className="@container rounded-md border border-border-subtle bg-surface-1/40">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className={`w-full flex items-center gap-2 px-3 py-2 text-[11px] font-semibold tracking-wide text-foreground-muted hover:text-foreground transition-colors ${open ? 'border-b border-border-subtle' : ''}`}
      >
        {open ? <ChevronDown className="w-3.5 h-3.5 text-foreground-faint" /> : <ChevronRight className="w-3.5 h-3.5 text-foreground-faint" />}
        <SlidersHorizontal className="w-3.5 h-3.5 text-foreground-faint" />
        Common settings
      </button>
      {open && (
        <div className="px-3 py-3">
          <div className="grid grid-cols-1 @[22rem]:grid-cols-2 gap-x-5 gap-y-3">
            {fieldSettings.map((s) => (
              <div key={s.id} className={fullWidth(s) ? '@[22rem]:col-span-2' : ''}>
                {renderRow(s)}
              </div>
            ))}
            {toggleSettings.length > 0 && (
              <div
                className={`flex flex-wrap items-center gap-2 self-end ${
                  toggleSettings.length > 1 ? '@[22rem]:col-span-2' : ''
                }`}
              >
                {toggleSettings.map((s) => <Fragment key={s.id}>{renderRow(s)}</Fragment>)}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function sameValue(a: SettingValue, b: SettingValue): boolean {
  return a === b || (a == null && b == null);
}

// ── One setting → the right control ─────────────────────────────────────────
function SettingRow({
  setting, stepName, value, dirty, disabled, onChange,
}: {
  setting: StepSetting;
  stepName: string;
  value: SettingValue;
  dirty: boolean;
  disabled: boolean;
  onChange: (v: SettingValue) => void;
}) {
  const label = <SettingLabel setting={setting} dirty={dirty} />;

  // Toggles render as a compact pill that hugs the label + switch together,
  // so they stay visually paired instead of pinned to opposite edges.
  if (setting.widget === 'toggle') {
    const on = value === true;
    return (
      <button
        type="button"
        role="switch"
        aria-checked={on}
        disabled={disabled}
        onClick={() => onChange(!on)}
        title={settingTip(setting)}
        className={`inline-flex items-center gap-2 pl-3 pr-1.5 py-1 rounded-full border transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
          dirty ? 'border-status-pending/60' : on ? 'border-primary/40' : 'border-border hover:border-border-strong'
        } ${on ? 'bg-primary-muted text-foreground' : 'bg-surface-2 text-foreground-muted hover:text-foreground'}`}
      >
        <span className="text-[11px] tracking-wide">{setting.label}</span>
        <SwitchVisual on={on} />
      </button>
    );
  }

  // Everything else stacks: label above the control.
  return (
    <div>
      {label}
      {setting.note && (
        <p className="mt-0.5 text-[10px] leading-snug text-foreground-faint">{setting.note}</p>
      )}
      <div className="mt-1">
        {setting.widget === 'select' && (
          <SelectControl setting={setting} value={value} disabled={disabled} dirty={dirty} stepName={stepName} onChange={onChange} />
        )}
        {setting.widget === 'slider' && (
          <SliderControl setting={setting} value={value} disabled={disabled} onChange={onChange} />
        )}
        {setting.widget === 'number' && (
          <NumberControl setting={setting} value={value} disabled={disabled} dirty={dirty} onChange={onChange} />
        )}
        {setting.widget === 'text' && (
          <TextControl setting={setting} value={value} disabled={disabled} dirty={dirty} onChange={onChange} />
        )}
      </div>
    </div>
  );
}

/** The underlying CLI flag(s) / preset key a setting maps to — appended to the
 *  tooltip so users can see what each control actually sets. */
function targetFlag(t: StepSetting['target']): string {
  if (t.kind === 'flag') return t.flag;
  if (t.kind === 'flag_pair') return `${t.on} / ${t.off}`;
  if (t.kind === 'flags') return t.flags.join(' + ');
  return `--${t.key}`; // extra (config_file / config_path / undistort) → its CLI flag
}

function settingTip(setting: StepSetting): string | undefined {
  const flag = targetFlag(setting.target);
  return [setting.help, flag ? `Flag: ${flag}` : null].filter(Boolean).join('\n\n') || undefined;
}

function SettingLabel({ setting, dirty }: { setting: StepSetting; dirty: boolean }) {
  const tip = settingTip(setting);
  return (
    <div className="flex items-center gap-1.5">
      <span
        className={`text-[10.5px] uppercase tracking-wide text-foreground-subtle ${
          tip ? 'cursor-help decoration-dotted decoration-foreground-faint/60 underline-offset-2 hover:underline' : ''
        }`}
        title={tip}
      >
        {setting.label}
      </span>
      {dirty && <span className="inline-block w-1 h-1 rounded-full bg-status-pending" title="Modified from preset" />}
    </div>
  );
}

// ── Controls (native elements, app styling) ─────────────────────────────────
const inputCls = (dirty?: boolean) =>
  `w-full bg-surface-2 border ${dirty ? 'border-status-pending/60' : 'border-border'} rounded-md px-2.5 py-1.5 text-foreground text-xs focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition-colors placeholder:text-foreground-faint disabled:opacity-50`;

/** Visual-only switch (spans, not a button) so it can live inside the toggle
 *  pill button without nesting interactive elements. */
function SwitchVisual({ on }: { on: boolean }) {
  return (
    <span
      className={`relative inline-flex h-4 w-7 flex-shrink-0 items-center rounded-full transition-colors ${
        on ? 'bg-primary' : 'bg-surface-3 border border-border'
      }`}
    >
      <span
        className={`inline-block h-3 w-3 rounded-full bg-white transition-transform ${
          on ? 'translate-x-3.5' : 'translate-x-0.5'
        }`}
      />
    </span>
  );
}

// Sentinel choice value: selecting it switches a file-path select to a free-text
// input so the user can point at any .yaml outside the curated list.
const CUSTOM_PATH = '__custom_path__';

function SelectControl({ setting, value, disabled, dirty, stepName, onChange }: {
  setting: StepSetting; value: SettingValue; disabled: boolean; dirty: boolean; stepName: string; onChange: (v: SettingValue) => void;
}) {
  const current = value == null ? '' : String(value);
  const choices = setting.choices ?? [];
  const allowCustom = !!setting.peek;                 // file-path selects (config_path / config_file)
  const isListed = choices.some((c) => c.value === current);
  // Start in custom mode when the preset already holds a path that isn't one of
  // the listed files (e.g. a hand-edited config_path).
  const [customMode, setCustomMode] = useState(allowCustom && current !== '' && !isListed);

  // Free-text path entry (custom mode) — for an arbitrary .yaml outside the list.
  if (allowCustom && customMode) {
    return (
      <div className="flex items-center gap-1.5">
        <input
          type="text"
          value={current}
          placeholder="path/to/config.yaml"
          disabled={disabled}
          onChange={(e) => onChange(e.target.value)}
          className={`flex-1 min-w-0 font-mono ${inputCls(dirty)}`}
        />
        <button
          type="button"
          onClick={() => setCustomMode(false)}
          disabled={disabled}
          title="Choose from the list instead"
          className="flex-shrink-0 p-1.5 rounded-md text-foreground-faint hover:text-primary hover:bg-surface-2 transition-colors disabled:opacity-50"
        >
          <List className="w-3.5 h-3.5" />
        </button>
      </div>
    );
  }

  const richChoices = allowCustom
    ? [...choices, { value: CUSTOM_PATH, label: 'Custom path…', description: 'Point at any .yaml outside this list' }]
    : choices;

  return (
    <div className="flex items-center gap-1.5">
      <div className="flex-1 min-w-0">
        <RichSelect
          value={current}
          choices={richChoices}
          disabled={disabled}
          dirty={dirty}
          onChange={(v) => (v === CUSTOM_PATH ? setCustomMode(true) : onChange(v))}
        />
      </div>
      {/* Peek only resolves listed (known) recipes; hide it for custom paths. */}
      {setting.peek && current && isListed && <RecipePeek stepName={stepName} path={current} />}
    </div>
  );
}

/** A select that can show a one-line description under each option. The option
 *  list renders in a body-level portal (positioned at the trigger) so the
 *  card's `overflow-hidden` can't clip it; closes on outside-click / Esc /
 *  scroll / resize. Falls back to label-only when a choice has no description. */
type RichChoice = { value: string; label: string; description?: string; advanced?: boolean };

function RichSelect({ value, choices, disabled, dirty, onChange }: {
  value: string;
  choices: RichChoice[];
  disabled: boolean;
  dirty: boolean;
  onChange: (v: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLUListElement>(null);
  const [rect, setRect] = useState<DOMRect | null>(null);

  // The "Custom path…" sentinel always sits last; `advanced` choices hide
  // behind a "show other…" expander (auto-open when the value is one of them).
  const customChoice = choices.find((c) => c.value === CUSTOM_PATH);
  const real = choices.filter((c) => c.value !== CUSTOM_PATH);
  const primary = real.filter((c) => !c.advanced);
  const advanced = real.filter((c) => c.advanced);
  const [showAdvanced, setShowAdvanced] = useState(advanced.some((c) => c.value === value));

  const openMenu = () => {
    if (triggerRef.current) setRect(triggerRef.current.getBoundingClientRect());
    setOpen(true);
  };

  useEffect(() => {
    if (!open) return;
    const close = () => setOpen(false);
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false); };
    // Close on BACKGROUND scroll/resize (the menu is fixed-positioned at the
    // trigger, so it would otherwise detach) — but ignore scrolls that happen
    // INSIDE the menu's own option list (e.g. after expanding "other …").
    const onScroll = (e: Event) => {
      if (menuRef.current && e.target instanceof Node && menuRef.current.contains(e.target)) return;
      setOpen(false);
    };
    window.addEventListener('scroll', onScroll, true);
    window.addEventListener('resize', close);
    document.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('scroll', onScroll, true);
      window.removeEventListener('resize', close);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  const selected = choices.find((c) => c.value === value);
  const triggerText = selected?.label ?? (value ? `${value} (custom)` : 'Select…');

  const renderOption = (c: RichChoice) => {
    const active = c.value === value;
    return (
      <li key={c.value}>
        <button
          type="button"
          role="option"
          aria-selected={active}
          onClick={() => { onChange(c.value); setOpen(false); }}
          className={`w-full text-left px-2.5 py-1.5 transition-colors ${active ? 'bg-primary-muted' : 'hover:bg-surface-2'}`}
        >
          <div className="flex items-center gap-2">
            <span className={`text-xs ${active ? 'text-primary font-medium' : 'text-foreground'}`}>{c.label}</span>
            {active && <Check className="w-3 h-3 text-primary ml-auto flex-shrink-0" />}
          </div>
          {c.description && (
            <div className="text-[10.5px] text-foreground-faint leading-snug mt-0.5">{c.description}</div>
          )}
        </button>
      </li>
    );
  };

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => (open ? setOpen(false) : openMenu())}
        className={`w-full flex items-center justify-between gap-2 bg-surface-2 border ${dirty ? 'border-status-pending/60' : 'border-border'} rounded-md px-2.5 py-1.5 text-foreground text-xs hover:border-border-strong focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition-colors disabled:opacity-50 disabled:cursor-not-allowed`}
      >
        <span className="truncate text-left">{triggerText}</span>
        <ChevronDown className={`w-3.5 h-3.5 flex-shrink-0 text-foreground-faint transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && rect && createPortal(
        <>
          {/* transparent click-catcher for outside-click close */}
          <div className="fixed inset-0 z-40" onMouseDown={() => setOpen(false)} />
          <ul
            ref={menuRef}
            role="listbox"
            style={{ position: 'fixed', top: rect.bottom + 4, left: rect.left, width: rect.width }}
            className="z-50 max-h-72 overflow-auto rounded-md border border-border bg-surface-1 shadow-xl shadow-black/40 ring-1 ring-inset ring-white/[0.04] py-1"
          >
            {primary.map(renderOption)}
            {advanced.length > 0 && (showAdvanced ? (
              <>
                <li className="px-2.5 pt-2 pb-0.5 text-[10px] uppercase tracking-wide text-foreground-faint">Other architectures</li>
                {advanced.map(renderOption)}
              </>
            ) : (
              <li>
                <button
                  type="button"
                  onClick={() => setShowAdvanced(true)}
                  className="w-full flex items-center gap-1.5 px-2.5 py-1.5 text-[11px] text-foreground-subtle hover:text-foreground hover:bg-surface-2 transition-colors"
                >
                  <ChevronRight className="w-3 h-3" />
                  Show other architectures ({advanced.length})
                </button>
              </li>
            ))}
            {customChoice && (
              <>
                {(primary.length > 0 || advanced.length > 0) && (
                  <li className="my-1 border-t border-border-subtle" aria-hidden />
                )}
                {renderOption(customChoice)}
              </>
            )}
          </ul>
        </>,
        document.body,
      )}
    </>
  );
}

/** Eye/file button next to a recipe select; opens a read-only modal showing the
 *  YAML contents (GET /api/steps/<step>/recipe). Self-contained like the flag
 *  catalog popover. */
function RecipePeek({ stepName, path }: { stepName: string; path: string }) {
  const [open, setOpen] = useState(false);
  const [text, setText] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setLoading(true); setError(null); setText(null);
    fetch(`/api/steps/${encodeURIComponent(stepName)}/recipe?path=${encodeURIComponent(path)}`)
      .then(async (r) => {
        if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.error || `HTTP ${r.status}`); }
        return r.json();
      })
      .then((d) => { if (!cancelled) setText(d.text); })
      .catch((e) => { if (!cancelled) setError(String(e.message ?? e)); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [open, stepName, path]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open]);

  const fileName = path.split('/').pop() || path;
  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        title={`View ${fileName}`}
        className="flex-shrink-0 p-1.5 rounded-md text-foreground-faint hover:text-primary hover:bg-surface-2 transition-colors"
      >
        <FileText className="w-3.5 h-3.5" />
      </button>
      {open && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/65 backdrop-blur-sm"
          onClick={() => setOpen(false)}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="w-full max-w-3xl max-h-[85vh] bg-surface-1 border border-border rounded-xl shadow-2xl shadow-black/50 ring-1 ring-inset ring-white/[0.03] overflow-hidden flex flex-col"
          >
            <div className="flex items-center justify-between gap-3 px-5 py-3.5 border-b border-border-subtle">
              <div className="min-w-0">
                <div className="text-foreground text-base font-medium truncate">{fileName}</div>
                <div className="text-foreground-subtle text-[11px] mt-0.5 font-mono truncate">{path}</div>
              </div>
              <button
                type="button"
                onClick={() => setOpen(false)}
                aria-label="Close"
                className="flex-shrink-0 p-1.5 rounded-md text-foreground-muted hover:text-foreground hover:bg-surface-2 transition-colors"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
            <div className="flex-1 overflow-auto">
              {loading && (
                <div className="text-foreground-subtle text-xs px-5 py-4 flex items-center gap-2">
                  <Loader2 className="w-3.5 h-3.5 animate-spin" /> Loading…
                </div>
              )}
              {error && (
                <div className="m-4 p-2.5 text-status-failed text-xs bg-status-failed-bg border border-status-failed/35 rounded flex items-start gap-2">
                  <AlertCircle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
                  <div>
                    <div className="font-medium mb-0.5">Couldn't load recipe</div>
                    <div className="text-foreground-muted">{error}</div>
                  </div>
                </div>
              )}
              {text != null && (
                <CodeBlock
                  code={text}
                  language="yaml"
                  className="text-[11px] text-foreground whitespace-pre font-mono px-5 py-3 leading-snug"
                />
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}

function SliderControl({ setting, value, disabled, onChange }: {
  setting: StepSetting; value: SettingValue; disabled: boolean; onChange: (v: SettingValue) => void;
}) {
  const num = typeof value === 'number' ? value : Number(setting.default ?? setting.min ?? 0);
  return (
    <div className="flex items-center gap-3">
      <input
        type="range"
        min={setting.min}
        max={setting.max}
        step={setting.step ?? 1}
        value={num}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ accentColor: 'var(--primary)' }}
        className="flex-1 h-1.5 cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
      />
      <span className="w-12 text-right text-xs font-mono tabular-nums text-foreground font-medium">
        {num}
        {setting.unit ? <span className="text-foreground-faint font-normal"> {setting.unit}</span> : null}
      </span>
    </div>
  );
}

function NumberControl({ setting, value, disabled, dirty, onChange }: {
  setting: StepSetting; value: SettingValue; disabled: boolean; dirty: boolean; onChange: (v: SettingValue) => void;
}) {
  const parse = (raw: string): SettingValue => {
    const t = raw.trim();
    if (t === '') return null;
    const n = Number(t);
    return Number.isFinite(n) ? n : null;
  };
  return (
    <div className="flex items-center gap-2">
      <input
        type="number"
        min={setting.min}
        max={setting.max}
        step={setting.step ?? 1}
        value={typeof value === 'number' ? value : ''}
        placeholder={setting.default == null ? 'auto' : String(setting.default)}
        disabled={disabled}
        onChange={(e) => onChange(parse(e.target.value))}
        className={`w-28 font-mono tabular-nums ${inputCls(dirty)}`}
      />
      {setting.unit && <span className="text-foreground-faint text-[11px]">{setting.unit}</span>}
    </div>
  );
}

function TextControl({ setting, value, disabled, dirty, onChange }: {
  setting: StepSetting; value: SettingValue; disabled: boolean; dirty: boolean; onChange: (v: SettingValue) => void;
}) {
  return (
    <input
      type="text"
      value={value == null ? '' : String(value)}
      placeholder={setting.placeholder}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value)}
      className={`font-mono ${inputCls(dirty)}`}
    />
  );
}
