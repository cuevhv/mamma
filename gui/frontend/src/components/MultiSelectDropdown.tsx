import { useState, useRef, useEffect } from 'react';
import { ChevronDown, Search, X } from 'lucide-react';

interface MultiSelectDropdownProps {
  label: string;
  options: string[];
  selected: string[];
  onToggle: (value: string) => void;
  onSelectAll: () => void;
  onRemove: (value: string) => void;
  columns?: number;
  placeholder?: string;
  /** Show a text filter above the option list. For long lists (sequence
   *  names); with a filter active, the header row selects/deselects the
   *  matches only. */
  searchable?: boolean;
}

export function MultiSelectDropdown({
  label, options, selected, onToggle, onSelectAll, onRemove, columns = 1,
  placeholder = 'No items selected', searchable = false,
}: MultiSelectDropdownProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState('');
  const dropdownRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) setIsOpen(false);
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Focus the filter on open; drop any stale filter on close so reopening
  // always shows the full list.
  useEffect(() => {
    if (isOpen) searchRef.current?.focus();
    else setQuery('');
  }, [isOpen]);

  const allSelected = options.length > 0 && selected.length === options.length;

  const q = query.trim().toLowerCase();
  const visible = q ? options.filter((o) => o.toLowerCase().includes(q)) : options;
  const visibleAllSelected = visible.length > 0 && visible.every((o) => selected.includes(o));

  /** Header row: whole list when no filter; just the matches when filtering.
   *  The per-match path calls onToggle per item — safe because the parents
   *  update state functionally, so batched toggles compose. */
  const handleHeaderClick = () => {
    if (!q) { onSelectAll(); return; }
    const targets = visibleAllSelected ? visible : visible.filter((o) => !selected.includes(o));
    targets.forEach(onToggle);
  };
  const colsCls = columns === 2 ? 'grid grid-cols-2 gap-1 p-2'
                : columns === 3 ? 'grid grid-cols-3 gap-1 p-2'
                : columns === 4 ? 'grid grid-cols-4 gap-1 p-2'
                : '';

  return (
    <div className="flex items-start gap-4">
      <label className="text-foreground-muted text-xs uppercase tracking-wider font-medium w-32 text-right flex-shrink-0 pt-2.5">
        {label}
      </label>
      <div className="flex-1 min-w-0 relative" ref={dropdownRef}>
        {/* Selected chips */}
        <div className="flex flex-wrap gap-1.5 mb-2 min-h-[28px]">
          {selected.length > 0 ? (
            selected.map((item) => (
              <span
                key={item}
                className="inline-flex items-center gap-1.5 bg-primary-muted border border-primary/30 text-primary px-2 py-0.5 rounded-md text-xs"
              >
                <span className="font-mono">{item}</span>
                <button
                  onClick={(e) => { e.stopPropagation(); onRemove(item); }}
                  className="hover:text-foreground transition-colors"
                  aria-label={`Remove ${item}`}
                >
                  <X className="w-3 h-3" />
                </button>
              </span>
            ))
          ) : (
            <div className="text-foreground-faint text-xs py-1">{placeholder}</div>
          )}
        </div>

        {/* Trigger */}
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="w-full bg-surface-2 border border-border rounded-md px-3 py-2 text-foreground text-sm focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 transition-colors hover:border-border-strong flex items-center justify-between"
        >
          <span className="text-foreground-muted">
            {selected.length === 0 ? placeholder.replace('No ', 'Select ').replace(' selected', '…')
                                   : `${selected.length} selected`}
          </span>
          <ChevronDown className={`w-4 h-4 text-foreground-subtle transition-transform ${isOpen ? 'rotate-180' : ''}`} />
        </button>

        {/* Menu */}
        {isOpen && (
          <div className="absolute z-30 mt-1 w-full bg-surface-2 border border-border rounded-md shadow-2xl shadow-black/60 max-h-72 overflow-y-auto">
            <div className="sticky top-0 z-10 bg-surface-2">
              {searchable && (
                <div className="px-2 pt-2 pb-1.5">
                  <div className="relative">
                    <Search className="w-3.5 h-3.5 absolute left-2 top-1/2 -translate-y-1/2 text-foreground-subtle pointer-events-none" />
                    <input
                      ref={searchRef}
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      onKeyDown={(e) => {
                        // Esc clears the filter first; a second Esc closes.
                        if (e.key === 'Escape') {
                          e.stopPropagation();
                          if (query) setQuery(''); else setIsOpen(false);
                        }
                      }}
                      placeholder="Type to filter…"
                      className="w-full bg-surface-3 border border-border rounded pl-7 pr-7 py-1.5 text-foreground text-sm placeholder:text-foreground-faint focus:outline-none focus:border-primary/60"
                    />
                    {query && (
                      <button
                        onClick={() => { setQuery(''); searchRef.current?.focus(); }}
                        className="absolute right-2 top-1/2 -translate-y-1/2 text-foreground-subtle hover:text-foreground transition-colors"
                        aria-label="Clear filter"
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </div>
                </div>
              )}
              <div
                onClick={handleHeaderClick}
                className="px-3 py-2.5 hover:bg-surface-3 cursor-pointer border-b border-border-subtle flex items-center gap-3"
              >
                <input
                  type="checkbox"
                  checked={q ? visibleAllSelected : allSelected}
                  onChange={() => {}}
                  className="w-3.5 h-3.5 accent-primary"
                />
                <span className="text-foreground text-sm font-medium">
                  {q
                    ? `${visibleAllSelected ? 'Deselect' : 'Select'} ${visible.length} match${visible.length === 1 ? '' : 'es'}`
                    : allSelected ? 'Deselect all' : 'Select all'}
                </span>
              </div>
            </div>

            {visible.length === 0 && (
              <div className="px-3 py-3 text-foreground-faint text-xs">No matches for “{query}”</div>
            )}

            <div className={colsCls}>
              {visible.map((option) => (
                <div
                  key={option}
                  onClick={() => onToggle(option)}
                  className={`cursor-pointer flex items-center gap-2 ${
                    columns > 1 ? 'px-2 py-1.5 hover:bg-surface-3 rounded' : 'px-3 py-2 hover:bg-surface-3'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={selected.includes(option)}
                    onChange={() => {}}
                    className="w-3.5 h-3.5 accent-primary flex-shrink-0"
                  />
                  <span className="text-foreground text-sm whitespace-nowrap overflow-hidden text-ellipsis">{option}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
