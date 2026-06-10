import { useEffect, useRef, useState } from 'react';
import {
  CheckCircle2,
  ChevronDown,
  ExternalLink,
  Eye,
  EyeOff,
  Loader2,
  LogIn,
  LogOut,
  UserPlus,
} from 'lucide-react';
import {
  CredentialDomain,
  DOMAIN_LABEL,
  REGISTER_URL,
  useCredentials,
  verifyMpiCredentials,
} from './CredentialsContext';

/*
 * Top-of-Home sign-in surface. Replaces the old "Accounts you may need"
 * strip — same visual register (a single rounded card hugging the page
 * gutter), but two interactive sub-cards instead of two pills.
 *
 *   - Signed-out sub-card: a tiny username/password form. On submit the
 *     credentials are verified against the project site's login.php (via
 *     the backend's /api/data/readiness/verify-mpi route) and are written
 *     into CredentialsContext *only on a confirmed-valid response* — so the
 *     "signed in" state reflects a real login, not just a filled form.
 *     A rejected attempt shows an inline error and keeps the form open.
 *   - Signed-in sub-card: green-dot indicator, `Signed in as <user>`,
 *     and a Sign out button that wipes that domain's slot.
 *
 * See CredentialsContext.tsx for the "never persisted" invariant.
 */

const DOMAINS: CredentialDomain[] = ['mamma', 'smplx'];

const DOMAIN_BLURB: Record<CredentialDomain, string> = {
  mamma: 'MAMMA assets and MammaNet weights',
  smplx: 'SMPL-X body models',
};

export function SignInCenter() {
  const ctx = useCredentials();
  const anySignedIn = !!(ctx.creds.mamma || ctx.creds.smplx);
  const allSignedIn = !!(ctx.creds.mamma && ctx.creds.smplx);

  // Default collapsed state: expand until both domains are signed in,
  // then auto-collapse on the user's first arrival at that state (so a
  // returning user with full creds doesn't see a fat empty card).
  // After that, the chevron is the only thing that toggles it.
  const [collapsed, setCollapsed] = useState(false);
  const initRef = useRef(false);
  useEffect(() => {
    if (initRef.current) return;
    initRef.current = true;
    if (allSignedIn) setCollapsed(true);
  }, [allSignedIn]);

  return (
    <div className="rounded-lg border border-border-subtle bg-surface-1/60">
      <button
        type="button"
        onClick={() => setCollapsed(c => !c)}
        aria-expanded={!collapsed}
        aria-controls="signin-center-body"
        className={
          'w-full text-left px-4 py-3 flex items-start justify-between gap-3 ' +
          'hover:bg-surface-2/30 transition-colors rounded-lg ' +
          (collapsed ? '' : 'rounded-b-none')
        }
      >
        <div className="flex items-start gap-2 min-w-0">
          <ChevronDown
            className={
              'w-3.5 h-3.5 text-foreground-faint mt-1 shrink-0 transition-transform ' +
              (collapsed ? '-rotate-90' : '')
            }
            aria-hidden
          />
          <UserPlus className="w-4 h-4 text-primary mt-0.5 shrink-0" aria-hidden />
          <div className="min-w-0">
            <div className="text-foreground text-sm font-medium">Accounts you may need</div>
            <p
              className="text-foreground-muted text-sm leading-relaxed mt-0.5"
              onClick={e => e.stopPropagation()}
            >
              Some downloads below are gated by free research accounts at{' '}
              <a
                href={REGISTER_URL.mamma}
                target="_blank"
                rel="noreferrer"
                className="text-primary hover:underline"
              >
                MAMMA
              </a>
              {' '}and{' '}
              <a
                href={REGISTER_URL.smplx}
                target="_blank"
                rel="noreferrer"
                className="text-primary hover:underline"
              >
                SMPL-X
              </a>
              . Register once, sign in here.
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3 shrink-0">
          {allSignedIn && (
            <span className="inline-flex items-center gap-1 text-[10.5px] uppercase tracking-[0.14em] text-status-completed">
              <span className="w-1.5 h-1.5 rounded-full bg-status-completed inline-block" />
              all signed in
            </span>
          )}
          {anySignedIn && (
            <span
              role="button"
              tabIndex={0}
              onClick={e => { e.stopPropagation(); ctx.signOutAll(); }}
              onKeyDown={e => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  e.stopPropagation();
                  ctx.signOutAll();
                }
              }}
              className="text-[11px] text-foreground-faint hover:text-foreground transition-colors inline-flex items-center gap-1 cursor-pointer"
              title="Wipe all signed-in credentials from browser memory"
            >
              <LogOut className="w-3 h-3" />
              Sign out all
            </span>
          )}
        </div>
      </button>

      {!collapsed && (
        <div id="signin-center-body" className="px-4 pb-3">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {DOMAINS.map(d => (
              <DomainCard key={d} domain={d} />
            ))}
          </div>

          <div className="mt-3 text-[10.5px] text-foreground-faint leading-relaxed">
            Credentials live in browser memory only. A page refresh, a tab close, or
            a Sign out wipes them. Nothing is written to disk, cookies, or any
            browser storage layer.
          </div>
        </div>
      )}
    </div>
  );
}

function DomainCard({ domain }: { domain: CredentialDomain }) {
  const ctx = useCredentials();
  const cred = ctx.creds[domain];
  const label = DOMAIN_LABEL[domain];
  const blurb = DOMAIN_BLURB[domain];
  const registerUrl = REGISTER_URL[domain];

  return (
    <div className="rounded-md border border-border bg-surface-2/60 p-3">
      <div className="flex items-center justify-between gap-2 mb-2">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-foreground text-[12.5px] font-medium tracking-wide">
            {label}
          </span>
          <span className="text-foreground-faint text-[11px] truncate">
            {blurb}
          </span>
        </div>
        {cred ? (
          <span className="inline-flex items-center gap-1 text-[10.5px] uppercase tracking-[0.14em] text-status-completed">
            <span className="w-1.5 h-1.5 rounded-full bg-status-completed inline-block" />
            signed in
          </span>
        ) : (
          <a
            href={registerUrl}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1 text-[11px] text-primary hover:underline whitespace-nowrap"
            title={`Open the ${label} account registration page in a new tab`}
          >
            <UserPlus className="w-3 h-3" />
            Register
            <ExternalLink className="w-2.5 h-2.5" />
          </a>
        )}
      </div>

      {cred ? <SignedInBody domain={domain} cred={cred} /> : <SignInBody domain={domain} />}
    </div>
  );
}

function SignedInBody({ domain, cred }: { domain: CredentialDomain; cred: { username: string } }) {
  const ctx = useCredentials();
  return (
    <div className="flex items-center justify-between gap-2">
      <div className="min-w-0 flex items-center gap-2 text-[12px] text-foreground-muted">
        <CheckCircle2 className="w-3.5 h-3.5 text-status-completed shrink-0" aria-hidden />
        <span className="truncate">
          Signed in as <span className="font-mono text-foreground">{cred.username}</span>
        </span>
      </div>
      <button
        type="button"
        onClick={() => ctx.signOut(domain)}
        className="inline-flex items-center gap-1 px-2 py-1 text-[11px] rounded border border-border text-foreground-muted hover:text-foreground hover:border-border-strong transition-colors"
      >
        <LogOut className="w-3 h-3" />
        Sign out
      </button>
    </div>
  );
}

function SignInBody({ domain }: { domain: CredentialDomain }) {
  const ctx = useCredentials();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [show, setShow] = useState(false);
  const [verifying, setVerifying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const label = DOMAIN_LABEL[domain];
  const canSubmit =
    username.trim().length > 0 && password.length > 0 && !verifying;

  // Verify the credentials against the project site's login.php before
  // flipping this domain to "signed in" — the backend reports a clean
  // valid/invalid (login is not rate-limited, unlike the download server).
  // Only on a confirmed-valid response do we hand the creds to the
  // context — so "signed in" reflects a real login, not just a filled form.
  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!canSubmit) return;
    setVerifying(true);
    setError(null);
    const user = username.trim();
    try {
      const res = await verifyMpiCredentials(domain, user, password);
      if (!res.valid) {
        setError(res.detail);
        return;
      }
      // Confirmed valid — only now persist into context, then wipe locals.
      ctx.signIn(domain, { username: user, password });
      setUsername('');
      setPassword('');
    } finally {
      setVerifying(false);
    }
  };

  return (
    <>
    <form
      onSubmit={submit}
      className="flex flex-col sm:flex-row gap-2"
    >
      <input
        type="text"
        autoComplete="username"
        value={username}
        onChange={e => setUsername(e.target.value)}
        placeholder={`${label} user email`}
        className="flex-1 min-w-0 bg-surface-1 border border-border rounded px-2.5 py-1.5 text-[12px] text-foreground placeholder:text-foreground-faint focus:outline-none focus:border-primary/60"
      />
      <div className="flex-1 min-w-0 relative">
        <input
          type={show ? 'text' : 'password'}
          autoComplete="current-password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          placeholder={`${label} password`}
          className="w-full bg-surface-1 border border-border rounded px-2.5 py-1.5 pr-8 text-[12px] text-foreground placeholder:text-foreground-faint focus:outline-none focus:border-primary/60"
        />
        <button
          type="button"
          onClick={() => setShow(s => !s)}
          tabIndex={-1}
          title={show ? 'Hide password' : 'Show password'}
          className="absolute right-1.5 top-1/2 -translate-y-1/2 text-foreground-faint hover:text-foreground p-1"
        >
          {show ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
        </button>
      </div>
      <button
        type="submit"
        disabled={!canSubmit}
        className="inline-flex items-center justify-center gap-1.5 px-3 py-1.5 text-[11px] rounded bg-primary text-primary-foreground font-medium disabled:opacity-50 disabled:cursor-not-allowed shadow-sm shadow-black/30"
      >
        {verifying ? (
          <Loader2 className="w-3 h-3 animate-spin" />
        ) : (
          <LogIn className="w-3 h-3" />
        )}
        {verifying ? 'Verifying…' : 'Sign in'}
      </button>
    </form>
    {error && (
      <p className="mt-1.5 text-[11px] text-status-failed leading-relaxed">
        {error}
      </p>
    )}
    </>
  );
}
