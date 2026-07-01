/*
 * Lightweight, dependency-free syntax highlighting for JSON / YAML previews.
 *
 * Tuned for the app's dark theme (VS Code "Dark+" token colours). Comments are
 * preserved (so a config's inline docs stay visible — a tree view would drop
 * them). Falls back to plain text for very large files to keep the DOM light.
 *
 * Usage: <CodeBlock code={text} language="yaml" className="…" />
 * The caller owns the <pre> styling via `className` (bg / border / padding /
 * whitespace); set a base text colour there for un-tokenised characters.
 */
import { Fragment, type ReactNode } from 'react';

const C = {
  key: '#9CDCFE',      // mapping keys / object properties
  string: '#CE9178',   // string + scalar values
  number: '#B5CEA8',
  keyword: '#569CD6',  // booleans / null
  comment: '#6A9955',
  punct: '#808080',    // : , { } [ ] and list markers
};

// Above this size, skip highlighting (thousands of spans hurt). Plain text.
const MAX_HIGHLIGHT = 120_000;

export function CodeBlock({ code, language, className }: {
  code: string;
  language: 'json' | 'yaml';
  className?: string;
}) {
  const cls = className ?? 'text-xs font-mono whitespace-pre text-foreground';
  if (code.length > MAX_HIGHLIGHT) {
    return <pre className={cls}>{code}</pre>;
  }
  const nodes = language === 'json' ? highlightJson(code) : highlightYaml(code);
  return <pre className={cls}><code>{nodes}</code></pre>;
}

// ── JSON ────────────────────────────────────────────────────────────────────
const JSON_RE =
  /("(?:\\.|[^"\\])*")(\s*:)?|\b(true|false|null)\b|(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)|([{}\[\],])/g;

function highlightJson(code: string): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0, k = 0;
  let m: RegExpExecArray | null;
  JSON_RE.lastIndex = 0;
  while ((m = JSON_RE.exec(code))) {
    if (m.index > last) out.push(code.slice(last, m.index));
    if (m[1] !== undefined) {
      if (m[2]) {
        out.push(<span key={k++} style={{ color: C.key }}>{m[1]}</span>);
        out.push(<span key={k++} style={{ color: C.punct }}>{m[2]}</span>);
      } else {
        out.push(<span key={k++} style={{ color: C.string }}>{m[1]}</span>);
      }
    } else if (m[3] !== undefined) {
      out.push(<span key={k++} style={{ color: C.keyword }}>{m[3]}</span>);
    } else if (m[4] !== undefined) {
      out.push(<span key={k++} style={{ color: C.number }}>{m[4]}</span>);
    } else if (m[5] !== undefined) {
      out.push(<span key={k++} style={{ color: C.punct }}>{m[5]}</span>);
    }
    last = JSON_RE.lastIndex;
  }
  if (last < code.length) out.push(code.slice(last));
  return out;
}

// ── YAML (line-based) ────────────────────────────────────────────────────────
function highlightYaml(code: string): ReactNode[] {
  return code.split('\n').map((line, i) => (
    <Fragment key={i}>{yamlLine(line)}{'\n'}</Fragment>
  ));
}

function yamlLine(line: string): ReactNode {
  const indent = (line.match(/^\s*/) || [''])[0];
  let rest = line.slice(indent.length);
  if (rest === '') return indent;
  if (rest.startsWith('#')) {
    return <>{indent}<span style={{ color: C.comment }}>{rest}</span></>;
  }
  // list marker ("- ")
  let marker = '';
  const lm = rest.match(/^- +/);
  if (lm) { marker = lm[0]; rest = rest.slice(marker.length); }
  // split off an inline comment (" #…", outside quotes)
  let comment = '';
  const ci = inlineCommentIndex(rest);
  if (ci >= 0) { comment = rest.slice(ci); rest = rest.slice(0, ci); }
  // key: value
  let body: ReactNode;
  const kv = rest.match(/^([^:#]+?)(:)(\s|$)([\s\S]*)$/);
  if (kv) {
    body = (
      <>
        <span style={{ color: C.key }}>{kv[1]}</span>
        <span style={{ color: C.punct }}>{kv[2]}</span>
        {kv[3]}
        {yamlValue(kv[4])}
      </>
    );
  } else {
    body = yamlValue(rest);
  }
  return (
    <>
      {indent}
      {marker && <span style={{ color: C.punct }}>{marker}</span>}
      {body}
      {comment && <span style={{ color: C.comment }}>{comment}</span>}
    </>
  );
}

function yamlValue(v: string): ReactNode {
  const t = v.trim();
  if (t === '') return v;
  if (/^(["']).*\1$/.test(t)) return <span style={{ color: C.string }}>{v}</span>;
  if (/^(true|false|null|yes|no|on|off|~)$/i.test(t)) return <span style={{ color: C.keyword }}>{v}</span>;
  if (/^-?\d+(\.\d+)?$/.test(t)) return <span style={{ color: C.number }}>{v}</span>;
  return <span style={{ color: C.string }}>{v}</span>;
}

function inlineCommentIndex(s: string): number {
  let inS = false, inD = false;
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (ch === "'" && !inD) inS = !inS;
    else if (ch === '"' && !inS) inD = !inD;
    else if (ch === '#' && !inS && !inD && (i === 0 || /\s/.test(s[i - 1]))) return i;
  }
  return -1;
}
