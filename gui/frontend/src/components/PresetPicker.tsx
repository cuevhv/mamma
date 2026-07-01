/*
 * Step 2 preset picker: selectable cards (name + description + source badge)
 * in place of a bare full-width <select>, so each preset explains itself and
 * picking one feels deliberate rather than ambiguous. Selecting a card sets the
 * active preset; the PresetDigestCard below still reviews / edits the full
 * configuration. Descriptions come from each preset's global.description.
 */

export interface PresetOption {
  name: string;
  displayName: string;
  description: string;
  source?: 'user' | 'example';
}

export function PresetPicker({ presets, value, onPick }: {
  presets: PresetOption[];
  value: string;                 // selected preset name
  onPick: (name: string) => void;
}) {
  if (presets.length === 0) {
    return (
      <div className="rounded-md border border-border-subtle bg-surface-2 px-3 py-4 text-sm text-foreground-subtle">
        No presets found in <code className="text-foreground-faint">$MAMMA_INTERFACE_DIR/samples/presets/</code>.
      </div>
    );
  }

  return (
    // Cap the height so a long list of presets scrolls instead of pushing the
    // rest of Step 2 down. pr-1 keeps the scrollbar off the card focus rings.
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-h-80 overflow-y-auto pr-1">
      {presets.map((p) => {
        const selected = p.name === value;
        return (
          <button
            key={p.name}
            type="button"
            onClick={() => onPick(p.name)}
            aria-pressed={selected}
            className={`group text-left rounded-lg border p-3 transition-colors focus:outline-none focus:ring-2 focus:ring-primary/30 ${
              selected
                ? 'border-primary/60 bg-primary-muted'
                : 'border-border bg-surface-2 hover:border-border-strong hover:bg-surface-3'
            }`}
          >
            <div className="flex items-center gap-2">
              {/* radio affordance — telegraphs "pick one" */}
              <span className={`w-3.5 h-3.5 rounded-full border flex-shrink-0 flex items-center justify-center transition-colors ${
                selected ? 'border-primary bg-primary' : 'border-border-strong group-hover:border-primary/50'
              }`}>
                {selected && <span className="w-1.5 h-1.5 rounded-full bg-white" />}
              </span>
              <span className={`flex-1 min-w-0 truncate text-sm font-medium ${selected ? 'text-primary' : 'text-foreground'}`}>
                {p.displayName}
              </span>
              <span className={`flex-shrink-0 text-[9px] uppercase tracking-wide px-1.5 py-0.5 rounded border ${
                p.source === 'user'
                  ? 'border-primary/30 text-primary/80'
                  : 'border-border-subtle text-foreground-faint'
              }`}>
                {p.source === 'user' ? 'User' : 'Example'}
              </span>
            </div>
            {p.description && (
              <p className="mt-1.5 text-[11px] leading-snug text-foreground-subtle">{p.description}</p>
            )}
          </button>
        );
      })}
    </div>
  );
}
