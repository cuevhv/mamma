import { useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import { toast } from 'sonner';
import { FieldRow } from './shared/CreateCaptureFields';

/** Inline form to create/overwrite a cluster profile from the New Task form,
 *  so users don't have to drop to `python -m inference cluster add`. Talks to
 *  the /api/clusters endpoints (create + preflight). No secrets are collected:
 *  auth is delegated to the user's SSH agent / ~/.ssh/config. */

type Check = { label: string; ok: boolean; detail: string };

const INPUT =
  'w-full bg-surface-2 border border-border rounded-md px-3 py-2 text-foreground text-sm ' +
  'focus:outline-none focus:border-primary/60 focus:ring-2 focus:ring-primary/20 ' +
  'transition-colors placeholder:text-foreground-faint';

interface Props {
  onSaved: (name: string) => void;
  onCancel: () => void;
  /** Full profile dict (snake_case, from GET /api/clusters/<name>) to edit.
   *  When set, the form pre-fills, the name is fixed, and saving overwrites.
   *  Fields the form does not manage (resources, ssh_opts, sync_excludes,
   *  port, transport, repo_dir) are carried over unchanged. */
  initial?: Record<string, any>;
}

function Hint({ children }: { children: ReactNode }) {
  return (
    <div className="text-foreground-faint text-[11px] leading-relaxed">{children}</div>
  );
}

const STEP_NAMES = ['ma_cap', 'ma_masks', 'ma_2d', 'ma_3d', 'ma_vis'] as const;
// Built-in per-step MAXJOBS baselines (inference/cluster/resources.py),
// shown as placeholders so the effective value is visible when unset.
const MAX_JOBS_BUILTIN: Record<string, number> = {
  ma_cap: 20, ma_masks: 10, ma_2d: 20, ma_3d: 20, ma_vis: 10,
};

export function ClusterProfileForm({ onSaved, onCancel, initial }: Props) {
  const editMode = initial !== undefined;
  const [name, setName] = useState(initial?.name ?? '');
  const [host, setHost] = useState(initial?.host ?? '');
  const [username, setUsername] = useState(initial?.username ?? '');
  const [workspaceRoot, setWorkspaceRoot] = useState(initial?.workspace_root ?? '');
  const [python, setPython] = useState(initial?.python ?? '');
  const [engine, setEngine] = useState<'apptainer' | 'conda' | 'docker'>(
    (initial?.engine as any) || 'apptainer');
  const [sifPath, setSifPath] = useState(initial?.sif_path ?? '');
  const [condaEnv, setCondaEnv] = useState(initial?.conda_env ?? '');
  const [dockerImage, setDockerImage] = useState(initial?.docker_image ?? '');
  const [activate, setActivate] = useState(initial?.activate ?? '');
  const [maxJobs, setMaxJobs] = useState(
    String(initial?.resources?.default?.max_jobs ?? ''));
  const [maxJobsStep, setMaxJobsStep] = useState<Record<string, string>>(() => {
    const out: Record<string, string> = {};
    for (const s of STEP_NAMES) {
      const v = initial?.resources?.[s]?.max_jobs;
      out[s] = v !== undefined && v !== null ? String(v) : '';
    }
    return out;
  });
  // Scheduler adapters come from the backend registry (/api/schedulers), so a
  // newly added adapter shows up here with no frontend change.
  const [schedulers, setSchedulers] = useState<{ name: string; flavors: string[] }[]>([
    { name: 'htcondor', flavors: ['bid', 'standard'] },
  ]);
  const [scheduler, setScheduler] = useState(initial?.scheduler || 'htcondor');
  const [flavor, setFlavor] = useState(initial?.flavor || 'bid');
  const [bidPrice, setBidPrice] = useState(String(initial?.bid_price ?? '100'));
  const [retries, setRetries] = useState(String(initial?.retries ?? '1'));
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [busy, setBusy] = useState(false);
  const [checks, setChecks] = useState<Check[] | null>(null);

  useEffect(() => {
    fetch('/api/schedulers')
      .then(r => (r.ok ? r.json() : null))
      .then(rows => {
        if (Array.isArray(rows) && rows.length) setSchedulers(rows);
      })
      .catch(() => {});
  }, []);
  const schedFlavors = schedulers.find(s => s.name === scheduler)?.flavors ?? [];

  const envLocation =
    engine === 'apptainer' ? sifPath : engine === 'conda' ? condaEnv : dockerImage;

  // The `resources` map is otherwise YAML-only; this form manages exactly
  // the max_jobs keys (all-steps default + per-step overrides) and must
  // preserve everything else stored per bucket (nvccli, CUDA caps, ...).
  const buildResources = (): Record<string, any> => {
    const res: Record<string, any> = { ...(initial?.resources ?? {}) };
    const setKey = (bucket: string, raw: string) => {
      const entry: Record<string, any> = { ...(res[bucket] ?? {}) };
      const n = Number(raw.trim());
      if (raw.trim() && Number.isInteger(n) && n >= 1) entry.max_jobs = n;
      else delete entry.max_jobs;
      if (Object.keys(entry).length) res[bucket] = entry;
      else delete res[bucket];
    };
    setKey('default', maxJobs);
    for (const s of STEP_NAMES) setKey(s, maxJobsStep[s] ?? '');
    return res;
  };

  const save = async (): Promise<boolean> => {
    if (!name.trim() || !host.trim() || !workspaceRoot.trim()) {
      toast.error('Name, host, and workspace folder are required');
      return false;
    }
    if (!envLocation.trim()) {
      toast.error(
        engine === 'apptainer' ? 'Environment: the .sif path on the cluster is required'
        : engine === 'conda' ? 'Environment: the conda env name on the cluster is required'
        : 'Environment: the docker image is required');
      return false;
    }
    const res = await fetch('/api/clusters', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        // In edit mode, carry over the fields this form doesn't manage
        // (resources, ssh_opts, sync_excludes, port, transport, repo_dir);
        // the explicit fields below override the spread.
        ...(initial ?? {}),
        name: name.trim(),
        host: host.trim(),
        username: username.trim(),
        workspaceRoot: workspaceRoot.trim(),
        python: python.trim(),          // empty = auto-detect on the login node
        engine,
        sifPath: sifPath.trim(),
        condaEnv: condaEnv.trim(),
        dockerImage: dockerImage.trim(),
        activate,
        scheduler,
        flavor,
        bidPrice: Number(bidPrice) || 100,
        // blank / invalid -> keep the default of one automatic retry
        retries: (() => {
          const r = parseInt(retries.trim(), 10);
          return Number.isFinite(r) && r >= 0 ? r : 1;
        })(),
        resources: buildResources(),
      }),
    });
    if (!res.ok) {
      const e = await res.json().catch(() => ({}));
      toast.error(`Save failed: ${e.error || res.statusText}`);
      return false;
    }
    return true;
  };

  const onSave = async () => {
    setBusy(true);
    try {
      if (await save()) {
        toast.success(`Cluster '${name.trim()}' saved`);
        onSaved(name.trim());
      }
    } finally {
      setBusy(false);
    }
  };

  const onSaveAndTest = async () => {
    setBusy(true);
    setChecks(null);
    try {
      if (!(await save())) return;
      const res = await fetch(`/api/clusters/${encodeURIComponent(name.trim())}/test`, {
        method: 'POST',
      });
      const j = await res.json().catch(() => ({}));
      setChecks(j.checks || []);
      if (j.ok) {
        toast.success(`Cluster '${name.trim()}' connected`);
        onSaved(name.trim());
      } else {
        toast.error('Connection test found problems (see below)');
      }
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-2 rounded-md border border-border bg-surface-1 p-3">
      <div className="text-foreground-subtle text-[10px] uppercase tracking-wider font-medium">
        {editMode ? `Edit cluster: ${name}` : 'New cluster'}
      </div>
      <p className="text-[11px] text-foreground-muted leading-snug">
        For an HTCondor cluster you have an account on and can reach by SSH (keys, or a one-time
        terminal login for password or 2FA). MAMMA stores no password; the profile is saved on this
        machine only. See docs/CLUSTER.md.
      </p>

      <FieldRow label="Name">
        <input className={INPUT} value={name} onChange={e => setName(e.target.value)}
          placeholder="my-cluster" disabled={editMode} />
      </FieldRow>
      <FieldRow label="Host">
        <input className={INPUT} value={host} onChange={e => setHost(e.target.value)}
          placeholder="login.mycluster.edu" />
      </FieldRow>
      <FieldRow label="Username">
        <input className={INPUT} value={username} onChange={e => setUsername(e.target.value)}
          placeholder="your cluster username" />
      </FieldRow>
      <FieldRow label="Workspace">
        <div className="space-y-1">
          <input className={INPUT} value={workspaceRoot} onChange={e => setWorkspaceRoot(e.target.value)}
            placeholder="/shared/you/mamma-workspace" />
          <div className="text-foreground-faint text-[11px] leading-relaxed">
            MAMMA's folder on the cluster's <span className="text-foreground-muted">shared filesystem</span>:
            the synced repo, run configs, DAG files, logs, and all step outputs live here.
            Write it exactly as the cluster nodes see it (login and execute nodes).
            If that storage is also mounted on this machine, prefer the name valid on both
            (e.g. /shared/&lt;user&gt;/... instead of /home/&lt;user&gt;/...), so logs and
            results open directly in this GUI.
          </div>
        </div>
      </FieldRow>
      <FieldRow label="Environment">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <select className={INPUT + ' max-w-[9rem]'} value={engine}
              onChange={e => setEngine(e.target.value as 'apptainer' | 'conda' | 'docker')}>
              <option value="apptainer">apptainer (.sif)</option>
              <option value="conda">conda env</option>
              <option value="docker">docker image</option>
            </select>
            {engine === 'apptainer' && (
              <input className={INPUT} value={sifPath} onChange={e => setSifPath(e.target.value)}
                placeholder="/shared/containers/mamma.sif (path on the cluster)" />
            )}
            {engine === 'conda' && (
              <input className={INPUT} value={condaEnv} onChange={e => setCondaEnv(e.target.value)}
                placeholder="env name on the execute nodes, e.g. mamma" />
            )}
            {engine === 'docker' && (
              <input className={INPUT} value={dockerImage} onChange={e => setDockerImage(e.target.value)}
                placeholder="image resolvable on the nodes, e.g. mamma:v1.1.0" />
            )}
          </div>
          <Hint>
            Required: how pipeline steps execute on this cluster and where that
            environment lives, written as the cluster nodes see it. Independent of
            the scheduler. On HPC clusters apptainer is the usual choice; conda /
            docker need the env or daemon present on the execute nodes.
          </Hint>
        </div>
      </FieldRow>
      <FieldRow label="Scheduler">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <select className={INPUT + ' max-w-[9rem]'} value={scheduler}
              onChange={e => {
                const name = e.target.value;
                setScheduler(name);
                const f = schedulers.find(s => s.name === name)?.flavors ?? [];
                setFlavor(f[0] ?? '');
              }}>
              {schedulers.map(s => <option key={s.name} value={s.name}>{s.name}</option>)}
            </select>
            {schedFlavors.length > 0 && (
              <select className={INPUT + ' max-w-[8rem]'} value={flavor}
                onChange={e => setFlavor(e.target.value)}>
                {schedFlavors.map(f => <option key={f} value={f}>{f}</option>)}
              </select>
            )}
            {flavor === 'bid' && (
              <input className={INPUT + ' max-w-[6rem]'} value={bidPrice}
                onChange={e => setBidPrice(e.target.value)} placeholder="bid" />
            )}
          </div>
          <Hint>
            How jobs are scheduled on this cluster. For HTCondor, pick "bid" if
            your cluster uses a bidding system (jobs are submitted with a price),
            otherwise standard.
          </Hint>
        </div>
      </FieldRow>

      <button type="button"
        className="text-foreground-muted text-xs hover:text-foreground transition"
        onClick={() => setShowAdvanced(v => !v)}>
        {showAdvanced ? '▾ Advanced' : '▸ Advanced'}
      </button>

      {showAdvanced && (
        <div className="space-y-2 border-l-2 border-border pl-3">
          <FieldRow label="Runner python">
            <div className="space-y-1">
              <input className={INPUT} value={python} onChange={e => setPython(e.target.value)}
                placeholder="auto-detect (python3, then python)" />
              <Hint>
                Used only to run MAMMA's tiny job wrapper on cluster nodes, not the
                pipeline environment. Leave empty to auto-detect.
              </Hint>
            </div>
          </FieldRow>
          <FieldRow label="Env setup">
            <div className="space-y-1">
              <input className={INPUT} value={activate} onChange={e => setActivate(e.target.value)}
                placeholder="optional, e.g. module load python" />
              <Hint>
                Optional shell snippet run before the job wrapper on cluster nodes.
                Not for the pipeline environment (that is the Environment field above).
              </Hint>
            </div>
          </FieldRow>
          <FieldRow label="Max jobs">
            <div className="space-y-1.5">
              <div className="flex items-center gap-2">
                <span className="text-foreground-muted text-[11px] w-20">all steps</span>
                <input className={INPUT + ' max-w-[7rem]'} value={maxJobs}
                  onChange={e => setMaxJobs(e.target.value)}
                  placeholder="per-step" inputMode="numeric" />
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
                {STEP_NAMES.map(s => (
                  <div key={s} className="space-y-0.5">
                    <div className="text-foreground-faint text-[10px] font-mono">{s}</div>
                    <input className={INPUT} value={maxJobsStep[s] ?? ''}
                      onChange={e => setMaxJobsStep(prev => ({ ...prev, [s]: e.target.value }))}
                      placeholder={String(maxJobs.trim() || MAX_JOBS_BUILTIN[s])}
                      inputMode="numeric" />
                  </div>
                ))}
              </div>
              <Hint>
                How many sequences of the same pipeline step may run on the
                cluster at once (the scheduler's MAXJOBS throttle, per step).
                Precedence: per-step value, else "all steps", else the
                built-in defaults shown as placeholders. Lower these to be
                gentler on the shared filesystem and GPU pool.
              </Hint>
            </div>
          </FieldRow>
          <FieldRow label="Retries">
            <div className="space-y-1">
              <input className={INPUT + ' max-w-[7rem]'} value={retries}
                onChange={e => setRetries(e.target.value)}
                placeholder="1" inputMode="numeric" />
              <Hint>
                Automatic re-attempts the scheduler makes per failed job
                (DAGMan RETRY). Helps with transient node issues; a job that
                fails deterministically just fails again, so set 0 while
                debugging to stop attempts from burning GPU time.
              </Hint>
            </div>
          </FieldRow>
        </div>
      )}

      <div className="text-foreground-faint text-[11px] leading-relaxed">
        🔒 No password is stored or handled by MAMMA. The GUI connects with your
        SSH keys/agent (~/.ssh/config is honored). If your cluster asks for a
        password or 2FA instead, run{' '}
        <code className="font-mono">python -m inference cluster connect &lt;name&gt;</code>{' '}
        once in a terminal: ssh itself prompts there, and the GUI reuses that session.
      </div>

      {checks && (
        <div className="space-y-1 rounded-md border border-border bg-surface-2 p-2">
          {checks.map((c, i) => (
            <div key={i} className="flex items-start gap-2 text-[12px]">
              <span className={c.ok ? 'text-emerald-400' : 'text-red-400'}>
                {c.ok ? '✓' : '✕'}
              </span>
              <span className="text-foreground-muted">{c.label}</span>
              {c.detail && <span className="text-foreground-faint truncate">: {c.detail}</span>}
            </div>
          ))}
        </div>
      )}

      <div className="flex items-center gap-2 pt-1">
        <button type="button" disabled={busy} onClick={onSaveAndTest}
          className="px-3 py-1.5 text-sm rounded-md bg-primary-muted-strong text-primary hover:brightness-110 disabled:opacity-50 transition">
          {busy ? 'Testing…' : 'Save & test'}
        </button>
        <button type="button" disabled={busy} onClick={onSave}
          className="px-3 py-1.5 text-sm rounded-md bg-surface-2 border border-border text-foreground-muted hover:text-foreground hover:bg-surface-3 disabled:opacity-50 transition">
          Save
        </button>
        <button type="button" disabled={busy} onClick={onCancel}
          className="px-3 py-1.5 text-sm rounded-md text-foreground-faint hover:text-foreground transition">
          Cancel
        </button>
      </div>
    </div>
  );
}
