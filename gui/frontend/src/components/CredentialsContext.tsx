import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';

/*
 * Session-scoped credential store for the two MPI-account-gated download
 * surfaces (the "Pipeline assets" panel and the "MAMMA datasets" panel).
 *
 *  INVARIANT — DO NOT BREAK:
 *
 *      Credentials live in React state only. Nothing in this file (or in
 *      any component reading from this context) is permitted to write the
 *      password to localStorage, sessionStorage, IndexedDB, cookies, the
 *      URL, the DOM as data-* attributes, or any out-of-process surface.
 *      A hard refresh must wipe them. The backend already enforces the
 *      same "never persisted" contract on its side
 *      (gui/backend/data_readiness.py:395,
 *       gui/backend/dataset_downloader.py:801).
 *
 *      If you find yourself wanting to "remember" the password across
 *      refreshes, talk to the rest of the team first — that's a contract
 *      change, not a UX improvement.
 *
 * Two domains are tracked separately because two distinct upstream
 * accounts gate the downloads: "mamma" (MAMMA datasets + MammaNet ckpt)
 * and "smplx" (SMPL-X locked head). The backend keys auth on the
 * `domain=` form field, not on a shared session, so the frontend must
 * mirror that separation.
 */

export type CredentialDomain = 'mamma' | 'smplx';

export interface Credential {
  username: string;
  password: string;
}

interface CredentialsCtx {
  creds: { mamma: Credential | null; smplx: Credential | null };
  signIn: (d: CredentialDomain, c: Credential) => void;
  signOut: (d: CredentialDomain) => void;
  signOutAll: () => void;
}

const Ctx = createContext<CredentialsCtx | null>(null);

/** Display labels for each domain (the same strings the backend returns
 *  as `account_label`). Kept here so the sign-in card can stay
 *  data-driven without an extra round-trip. */
export const DOMAIN_LABEL: Record<CredentialDomain, string> = {
  mamma: 'MAMMA',
  smplx: 'SMPL-X',
};

export const REGISTER_URL: Record<CredentialDomain, string> = {
  mamma: 'https://mamma.is.tue.mpg.de/register.php',
  smplx: 'https://smpl-x.is.tue.mpg.de/register.php',
};

/** Map the backend's `account_label` string ("MAMMA" / "SMPL-X") to the
 *  internal lower-case domain key the context uses. Returns null for
 *  unknown labels — callers should fall back to the inline form in
 *  that case rather than guess. */
export function domainFromAccountLabel(label: string): CredentialDomain | null {
  const norm = label.trim().toLowerCase();
  if (norm === 'mamma') return 'mamma';
  if (norm === 'smpl-x' || norm === 'smplx') return 'smplx';
  return null;
}

export interface VerifyResult {
  valid: boolean;
  detail: string;
}

/**
 * Verify MPI credentials for a domain against the backend before treating
 * the user as signed in. The backend (`/api/data/readiness/verify-mpi`)
 * authenticates against the project site's login.php — NOT the
 * rate-limited download.php — so the result is unambiguous: `valid: true`
 * means the login succeeded, `valid: false` means the username/password is
 * wrong (the `detail` says which). Because logging in is not rate-limited,
 * verifying can't trip the 24-hour download block. Single source of truth
 * shared by the sign-in card and the inline download forms so all entry
 * points apply the same check. Credentials are sent once over HTTPS and
 * never persisted.
 */
export async function verifyMpiCredentials(
  domain: CredentialDomain,
  username: string,
  password: string,
): Promise<VerifyResult> {
  try {
    const r = await fetch('/api/data/readiness/verify-mpi', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ domain, username, password }),
    });
    const data = (await r.json().catch(() => ({}))) as {
      valid?: boolean;
      detail?: string;
      error?: string;
    };
    if (!r.ok || !data.valid) {
      return {
        valid: false,
        detail: data.detail || data.error || `Verification failed (HTTP ${r.status}).`,
      };
    }
    return { valid: true, detail: data.detail || 'Credentials accepted.' };
  } catch {
    return { valid: false, detail: 'Could not reach the server to verify credentials.' };
  }
}

export function CredentialsProvider({ children }: { children: React.ReactNode }) {
  const [creds, setCreds] = useState<{ mamma: Credential | null; smplx: Credential | null }>({
    mamma: null,
    smplx: null,
  });

  const signIn = useCallback((d: CredentialDomain, c: Credential) => {
    setCreds(prev => ({ ...prev, [d]: { username: c.username, password: c.password } }));
  }, []);
  const signOut = useCallback((d: CredentialDomain) => {
    setCreds(prev => ({ ...prev, [d]: null }));
  }, []);
  const signOutAll = useCallback(() => {
    setCreds({ mamma: null, smplx: null });
  }, []);

  // Best-effort wipe on provider unmount so the credential strings drop
  // their last hard reference and become eligible for collection. The
  // browser still has whatever the JS engine retained internally — same
  // posture as the existing per-form `setPassword('')` calls.
  useEffect(() => {
    return () => setCreds({ mamma: null, smplx: null });
  }, []);

  const value = useMemo(
    () => ({ creds, signIn, signOut, signOutAll }),
    [creds, signIn, signOut, signOutAll],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useCredentials(): CredentialsCtx {
  const v = useContext(Ctx);
  if (!v) throw new Error('useCredentials must be used inside a CredentialsProvider');
  return v;
}
