import { useEffect, useRef, useState } from 'react';
import { X, Sparkles, Loader2, Maximize2, Minimize2, AlertTriangle } from 'lucide-react';
import { WebViewer } from '@rerun-io/web-viewer';
import { isChromiumBased } from './shared/browser';
import { NativeOpenButton } from './NativeOpenButton';

const CHROME_HINT_KEY = 'mamma.rrdChromeVideoHintDismissed';

interface Props {
  /** Absolute on-disk path of the .rrd to open. Streamed via /api/rrd/file.rrd. */
  rrdPath: string;
  fileName: string;
  onClose: () => void;
  /** Called when the user picks "Open native instead" from the modal —
   *  parent triggers POST /api/rrd/open. `fresh` requests a reset of the native
   *  viewer's saved layout for this recording before launching. */
  onOpenNative?: (fresh: boolean) => void;
}

/**
 * Modal-embedded Rerun web viewer.
 *
 * Loading model:
 *   - We pass `/api/rrd/file.rrd?path=<absolute>` as the URL — same-origin
 *     when proxied via Vite, so no CORS / mixed-content traps. The `.rrd`
 *     in the route path is significant: the WASM viewer dispatches on the
 *     URL extension to figure out it's looking at a recording.
 *   - The Flask endpoint sets `conditional=True`, which makes `send_file`
 *     honour `Range` requests; the WebViewer pulls the file
 *     progressively.
 *
 * Memory caveat: the WebViewer still has to materialise log messages in
 * browser memory to render them. For >500MB recordings we surface a hint
 * pointing the user at the native viewer (which streams to GPU directly
 * via the Rust desktop app) — but we don't refuse to launch.
 */
export function RerunWebViewer({ rrdPath, fileName, onClose, onOpenNative }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const viewerRef = useRef<InstanceType<typeof WebViewer> | null>(null);
  const [phase, setPhase] = useState<'starting' | 'ready' | 'error'>('starting');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  /** Modal size mode. We don't use Rerun's `allow_fullscreen` — see comment
   *  on the start() call below — and instead just expand our own modal
   *  to fill the window. */
  const [expanded, setExpanded] = useState(false);
  // Per-session, dismissible heads-up shown only to Chromium users: their H.264
  // camera backdrops can render black on some Linux GPU setups (Firefox + the
  // native viewer are unaffected). Phrased conditionally so it never alarms.
  // Stored in sessionStorage so it reappears in a fresh session but doesn't
  // nag on every reload within the same one.
  const [chromeHintDismissed, setChromeHintDismissed] = useState(() => {
    try { return sessionStorage.getItem(CHROME_HINT_KEY) === '1'; } catch { return false; }
  });
  const dismissChromeHint = () => {
    setChromeHintDismissed(true);
    try { sessionStorage.setItem(CHROME_HINT_KEY, '1'); } catch { /* ignore */ }
  };
  const showChromeHint = phase === 'ready' && !chromeHintDismissed && isChromiumBased();

  useEffect(() => {
    let cancelled = false;
    const container = containerRef.current;
    if (!container) return;

    const viewer = new WebViewer();
    viewerRef.current = viewer;

    // Absolute URL is required: the WASM viewer's URL parser treats a
    // bare `/api/…` path as scheme-relative and ends up resolving it to
    // `http://api/…` (interpreting "api" as a hostname). Going through
    // `new URL(…, location.origin)` forces a proper `http(s)://host:port`
    // prefix. Path still ends in `.rrd` so Rerun's extension dispatch
    // recognises the response as a recording.
    const url = new URL(
      `/api/rrd/file.rrd?path=${encodeURIComponent(rrdPath)}`,
      window.location.origin,
    ).toString();
    viewer
      // `width`/`height: 100%` makes the canvas grow to its parent
      // (default is a fixed 640x360 stuck at top-left).
      //
      // We deliberately DO NOT enable `allow_fullscreen`. The package's
      // fullscreen-out path strips the canvas's inline styles and
      // toggles position:fixed/static, which mid-WGPU-rendering causes
      // "canvas.getContext() returned null; canvas already in use" —
      // the existing WebGL2 context is still bound when the WASM tries
      // to recreate the wgpu surface for the new size. The error is
      // permanent until the user reloads. It also surfaces *two*
      // fullscreen buttons (ours in the header + Rerun's inside the
      // canvas) which is its own UX problem.
      //
      // Instead we resize our own modal — see the `expanded` state
      // below. The canvas stays a 100% in-flow child the whole time,
      // which is the same code path the WASM handles cleanly on a
      // browser window resize.
      .start(url, container, { width: '100%', height: '100%' })
      .then(() => { if (!cancelled) setPhase('ready'); })
      .catch((e: unknown) => {
        if (cancelled) return;
        const msg = e instanceof Error ? e.message : String(e);
        console.error('[RerunWebViewer] start failed:', e);
        setErrorMsg(msg);
        setPhase('error');
      });

    return () => {
      cancelled = true;
      // Grab the canvas BEFORE stop() — viewer.stop() removes it from the
      // DOM, after which `container.querySelector('canvas')` returns null.
      const canvas = container.querySelector('canvas');
      try {
        viewer.stop();
      } catch (e) {
        console.warn('[RerunWebViewer] stop threw:', e);
      }
      // Force-release the WebGL2 context. The package's stop() removes the
      // canvas element but doesn't explicitly call `loseContext()`, so the
      // GPU resources hang around until the GC reclaims the canvas — which
      // is non-deterministic. If we remount before that happens (e.g.
      // user reopens the same .rrd), the new canvas's getContext('webgl2')
      // can fail with "canvas already in use" because the per-tab context
      // pool is still saturated.
      if (canvas) {
        const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
        gl?.getExtension('WEBGL_lose_context')?.loseContext();
      }
      viewerRef.current = null;
    };
  }, [rrdPath]);

  // Escape: minimize from expanded first, otherwise close the modal.
  // Two presses to fully exit, one press to "step out" of fullscreen —
  // matches the rest of the app and most desktop conventions.
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      if (expanded) { setExpanded(false); return; }
      onClose();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [onClose, expanded]);

  return (
    <div
      className={`fixed inset-0 bg-black/85 backdrop-blur-sm z-50 flex items-center justify-center ${expanded ? 'p-0' : 'p-4'}`}
      onClick={onClose}
    >
      <div
        className={`bg-surface-1 border border-border flex flex-col shadow-2xl shadow-black/60 ring-1 ring-inset ring-white/[0.03] overflow-hidden ${
          expanded
            ? 'w-screen h-screen rounded-none'
            : 'w-[95vw] h-[90vh] rounded-xl'
        }`}
        onClick={(e) => e.stopPropagation()}
      >
        <header className="flex items-center justify-between gap-3 p-3 border-b border-border-subtle bg-surface-2/40">
          <div className="min-w-0 flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-status-mixed flex-shrink-0" />
            <span className="text-foreground text-sm font-medium truncate" title={fileName}>{fileName}</span>
            <span className="text-foreground-faint text-[11px] font-mono truncate hidden md:inline" title={rrdPath}>{rrdPath}</span>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            <button
              onClick={() => setExpanded(e => !e)}
              className="inline-flex items-center gap-1.5 px-2.5 py-1.5 bg-surface-2 hover:bg-surface-3 border border-border hover:border-border-strong text-foreground-muted hover:text-foreground rounded-md text-xs transition-colors"
              title={expanded ? 'Shrink back to a windowed viewer.' : 'Expand the Rerun viewer to fill the entire browser window.'}
            >
              {expanded ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
              {expanded ? 'Restore' : 'Fullscreen'}
            </button>
            {onOpenNative && (
              <NativeOpenButton
                label="Open native instead"
                size="md"
                title="Open this .rrd in the native Rerun desktop viewer — better for very large files. Keeps your saved layout."
                onOpen={onOpenNative}
              />
            )}
            <button
              onClick={onClose}
              className="text-foreground-subtle hover:text-foreground p-1.5 -m-1.5 rounded-md hover:bg-surface-3 transition-colors"
              aria-label="Close"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </header>

        {showChromeHint && (
          <div className="flex items-start gap-2.5 px-3 py-2 border-b border-status-mixed/30 border-l-2 border-l-status-mixed bg-status-mixed-bg text-xs" role="alert">
            <AlertTriangle className="w-4 h-4 text-status-mixed mt-px flex-shrink-0" />
            <p className="flex-1 text-foreground-muted leading-relaxed">
              <span className="text-status-mixed font-semibold">Using Chrome?</span>{' '}
              Camera video backdrops can render black on some Chrome setups. Try{' '}
              <span className="text-foreground font-medium">Firefox</span>
              {onOpenNative ? (
                <>, or <button onClick={() => onOpenNative(false)} className="text-primary hover:underline font-medium">open the native viewer</button></>
              ) : null}
              {' '}— the 3D scene and overlays are unaffected.
            </p>
            <button
              onClick={dismissChromeHint}
              aria-label="Dismiss"
              className="text-foreground-subtle hover:text-foreground p-0.5 -m-0.5 rounded hover:bg-surface-3 transition-colors flex-shrink-0"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        )}

        <div className="relative flex-1 bg-background">
          {phase === 'starting' && (
            <div className="absolute inset-0 flex items-center justify-center text-foreground-muted text-sm gap-2 pointer-events-none">
              <Loader2 className="w-4 h-4 animate-spin" />
              Loading recording into the browser…
            </div>
          )}
          {phase === 'error' && (() => {
            // Detect the WebGL2-context / WGPU-surface failure mode. When
            // this happens, Chrome has typically disabled the GPU process
            // after repeated crashes — the package's "Clear caches and
            // reload" message is misleading; the only fix is a full
            // browser restart. We surface that explicitly here so the
            // user isn't sent on a wild goose chase clearing caches.
            const lower = (errorMsg || '').toLowerCase();
            const looksLikeGpuCrash =
              lower.includes('wgpu') ||
              lower.includes('webgl') ||
              lower.includes('canvas.getcontext') ||
              lower.includes('canvas already in use');
            return (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-center px-6 bg-background">
                <div className="text-status-failed text-sm mb-2">Failed to load this .rrd in the browser.</div>
                {errorMsg && (
                  <pre className="text-foreground-faint text-[11px] font-mono mb-3 max-w-2xl whitespace-pre-wrap break-all">
                    {errorMsg}
                  </pre>
                )}
                {looksLikeGpuCrash ? (
                  <div className="max-w-lg text-left bg-status-failed-bg border border-status-failed/35 rounded-md p-3 text-xs text-foreground-muted space-y-2">
                    <div className="text-status-failed font-medium">Looks like Chrome's GPU process is disabled.</div>
                    <p>
                      This usually happens after a few WebGL/WGPU crashes — Chrome stops handing out GPU contexts
                      until the browser is fully restarted. Clearing the cache or reloading the tab will <em>not</em> help.
                    </p>
                    <p className="text-foreground"><strong>To recover:</strong></p>
                    <ol className="list-decimal pl-4 space-y-1">
                      <li>Quit <strong>every</strong> Chrome window (every profile). On Linux, <code className="font-mono bg-surface-2 px-1 rounded">pkill chrome</code> if any process lingers.</li>
                      <li>Reopen Chrome and try again.</li>
                      <li>Verify recovery at <code className="font-mono bg-surface-2 px-1 rounded">chrome://gpu</code> — WebGL2 should read "Hardware accelerated".</li>
                    </ol>
                    <p className="text-foreground-subtle">
                      If the problem persists after a clean restart, use <strong>Open native instead</strong> — the desktop viewer doesn't depend on the browser's GPU stack.
                    </p>
                  </div>
                ) : (
                  <div className="text-foreground-muted text-xs max-w-md">
                    The browser viewer is best for files under a few hundred MB. For larger or interactive sessions, use the
                    native viewer.
                  </div>
                )}
              </div>
            );
          })()}
          {/* The WebViewer mounts a canvas inside this div. The class
              `w-full h-full` lets it size to the modal frame. */}
          <div ref={containerRef} className="w-full h-full" />
        </div>
      </div>
    </div>
  );
}
