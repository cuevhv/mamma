/*
 * Pure mapping between the backend "common settings" schema
 * (gui/backend/step_settings.py) and a preset step's raw `flags: string[]`
 * + dedicated `extras` keys.
 *
 * The friendly widgets in Step 2 read/write the SAME flags/extras the raw
 * "Advanced flags" editor shows, so the two views stay in sync. This module is
 * the single place that knows how a setting's value is encoded as flag tokens.
 *
 * A flag entry is a shell fragment (the runner `shlex.split`s each one), e.g.
 * "--sam_version sam2" or "--lazy-frames". We treat the first whitespace-
 * delimited word as the flag name and the remainder as its value.
 */

// ── Schema types (mirror step_settings.py) ──────────────────────────────────
export type FlagTarget = { kind: 'flag'; flag: string; valued: boolean; invert?: boolean };
export type FlagPairTarget = { kind: 'flag_pair'; on: string; off: string; defaultOn: boolean };
export type ExtraTarget = { kind: 'extra'; key: string };
/** One toggle backing several store_true flags, all emitted/removed together. */
export type FlagsTarget = { kind: 'flags'; flags: string[] };
export type SettingTarget = FlagTarget | FlagPairTarget | ExtraTarget | FlagsTarget;

export type SettingWidget = 'toggle' | 'select' | 'slider' | 'number' | 'text';

export interface SettingChoice { value: string; label: string; description?: string; advanced?: boolean }

export interface StepSetting {
  id: string;
  label: string;
  widget: SettingWidget;
  target: SettingTarget;
  help?: string;
  default?: unknown;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  choices?: SettingChoice[];
  omitWhenDefault?: boolean;
  dependsOn?: { id: string; equals: unknown };
  /** Value is a recipe file viewable via GET /api/steps/<step>/recipe. */
  peek?: boolean;
  /** Small help line rendered above the control. */
  note?: string;
  /** In-field hint for text / number inputs. */
  placeholder?: string;
}

/** `{ step_name: StepSetting[] }` — the payload of GET /api/steps/settings. */
export type StepSettingsSchema = Record<string, StepSetting[]>;

export type SettingValue = boolean | number | string | null;

export interface SettingContext {
  flags: string[];
  extras: Record<string, unknown>;
}

// ── Token helpers ───────────────────────────────────────────────────────────
const flagName = (token: string): string => token.trim().split(/\s+/)[0] ?? '';

const flagValue = (token: string): string => {
  const t = token.trim();
  const i = t.search(/\s/);
  return i === -1 ? '' : t.slice(i + 1).trim();
};

// ── Read: current value of a setting from the preset state ──────────────────
export function readSetting(setting: StepSetting, ctx: SettingContext): SettingValue {
  const { target } = setting;
  const fallback = (setting.default ?? null) as SettingValue;

  if (target.kind === 'extra') {
    const v = ctx.extras?.[target.key];
    return v === undefined || v === null ? fallback : (v as SettingValue);
  }

  if (target.kind === 'flags') {
    // On when ANY managed flag is present, so a managed flag can never be "set
    // but the toggle reads off" (which would strand it, hidden, in the digest).
    return target.flags.some((name) => ctx.flags.some((f) => flagName(f) === name));
  }

  if (target.kind === 'flag_pair') {
    if (ctx.flags.some((f) => flagName(f) === target.on)) return true;
    if (ctx.flags.some((f) => flagName(f) === target.off)) return false;
    return target.defaultOn;
  }

  // kind === 'flag'
  const tok = ctx.flags.find((f) => flagName(f) === target.flag);
  if (!target.valued) {
    const present = tok !== undefined;
    return target.invert ? !present : present; // store_true (optionally inverted)
  }
  if (tok === undefined) return fallback;
  const raw = flagValue(tok);
  if (setting.widget === 'number' || setting.widget === 'slider') {
    const n = Number(raw);
    return raw === '' || Number.isNaN(n) ? fallback : n;
  }
  return raw; // select / text
}

// ── Write: recompute the flags array for a flag/flag_pair setting ───────────
// Extra-targeted settings are written by the caller (onChange(key, value));
// this returns the flags unchanged for them.
export function applySetting(setting: StepSetting, value: SettingValue, flags: string[]): string[] {
  const { target } = setting;
  if (target.kind === 'extra') return flags;

  if (target.kind === 'flags') {
    // Multi-flag toggle: drop all managed flags, re-add them all when on.
    const out = flags.filter((f) => !target.flags.includes(flagName(f)));
    if (value === true) out.push(...target.flags);
    return out;
  }

  // Drop existing tokens this setting manages, keep everything else untouched.
  const managed = target.kind === 'flag_pair' ? [target.on, target.off] : [target.flag];
  const out = flags.filter((f) => !managed.includes(flagName(f)));

  if (target.kind === 'flag_pair') {
    // Emit only the explicit (non-default) side; rely on the default otherwise.
    if (value === !target.defaultOn) out.push(value ? target.on : target.off);
    return out;
  }

  if (!target.valued) {
    // store_true: presence = on. An inverted toggle is "on when ABSENT", so the
    // flag is emitted when the value is off.
    const present = target.invert ? value === false : value === true;
    if (present) out.push(target.flag);
    return out;
  }

  // Valued flag (select / number / slider / text).
  const empty = value === null || value === undefined || value === '';
  const isDefault =
    !!setting.omitWhenDefault && !empty && String(value) === String(setting.default ?? '');
  if (empty || isDefault) return out; // omit the token
  out.push(`${target.flag} ${value}`);
  return out;
}

/** Whether a setting's `dependsOn` guard is satisfied by the current values. */
export function isEnabled(setting: StepSetting, values: Record<string, SettingValue>): boolean {
  if (!setting.dependsOn) return true;
  return values[setting.dependsOn.id] === setting.dependsOn.equals;
}
