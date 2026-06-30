import { useEffect, useState } from 'react';
import { BookOpen, ExternalLink } from 'lucide-react';

interface Section {
  id: string;
  title: string;
}

const SECTIONS: Section[] = [
  { id: 'overview',      title: 'Overview' },
  { id: 'quick-start',   title: 'Quick start' },
  { id: 'dataset',       title: 'Dataset' },
  { id: 'pipeline',      title: 'Pipeline stages' },
  { id: 'input-modes',   title: 'Input modes' },
  { id: 'presets',       title: 'Presets and captures' },
  { id: 'commands',      title: 'Useful commands' },
  { id: 'gui',           title: 'Running the GUI' },
  { id: 'layout',        title: 'Repository layout' },
  { id: 'resources',     title: 'Resources' },
];

const STEP_COLOR: Record<string, string> = {
  ma_cap:   'text-[oklch(0.730_0.145_235)]',
  ma_masks: 'text-[oklch(0.760_0.160_158)]',
  ma_2d:    'text-[oklch(0.810_0.140_80)]',
  ma_3d:    'text-[oklch(0.745_0.170_305)]',
  ma_vis:   'text-[oklch(0.755_0.155_50)]',
};

export function Docs() {
  const [active, setActive] = useState<string>(SECTIONS[0].id);

  useEffect(() => {
    const els = SECTIONS
      .map(s => document.getElementById(s.id))
      .filter((el): el is HTMLElement => el !== null);
    if (els.length === 0) return;
    const obs = new IntersectionObserver(
      entries => {
        const visible = entries
          .filter(e => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
        if (visible) setActive(visible.target.id);
      },
      { rootMargin: '-80px 0px -70% 0px', threshold: 0 },
    );
    els.forEach(el => obs.observe(el));
    return () => obs.disconnect();
  }, []);

  return (
    <div className="w-full max-w-6xl mx-auto px-6 py-10">
      <header className="mb-10 flex items-center gap-3">
        <span className="inline-flex items-center justify-center w-10 h-10 rounded-lg bg-primary-muted ring-1 ring-inset ring-primary/30">
          <BookOpen className="w-5 h-5 text-primary" />
        </span>
        <div>
          <h1 className="text-3xl text-foreground tracking-tight font-medium leading-tight">Docs</h1>
          <p className="text-foreground-muted text-sm mt-1">
            Distilled from <InlineCode>README.md</InlineCode> and the <InlineCode>docs/</InlineCode> folder.
            For the canonical reference, open those files in the repo.
          </p>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-[14rem_minmax(0,1fr)] gap-8">
        <aside className="lg:sticky lg:top-20 lg:self-start">
          <nav className="space-y-1">
            <div className="text-foreground-subtle text-xs uppercase tracking-[0.16em] mb-2">On this page</div>
            {SECTIONS.map((s, i) => {
              const isActive = active === s.id;
              return (
                <a
                  key={s.id}
                  href={`#${s.id}`}
                  className={`group flex items-center gap-2 pl-2 pr-2 py-1 text-sm rounded-md border-l-2 transition-all ${
                    isActive
                      ? 'border-primary bg-primary-muted/40 text-foreground'
                      : 'border-transparent text-foreground-muted hover:text-foreground hover:bg-surface-2/60'
                  }`}
                >
                  <span className={`font-mono text-[10.5px] tabular-nums ${isActive ? 'text-primary' : 'text-foreground-faint'}`}>
                    {String(i + 1).padStart(2, '0')}
                  </span>
                  <span>{s.title}</span>
                </a>
              );
            })}
          </nav>
        </aside>

        <article className="space-y-14 min-w-0">
          <Section id="overview" index={1} title="Overview">
            <p>
              MAMMA takes synchronized multi-view video of one or more people and returns per-frame{' '}
              <Strong>SMPL-X meshes</Strong> plus overlay visualizations. The pipeline has five stages —
              <em className="not-italic text-foreground"> capture, segmentation, 2D landmarks, 3D optimization, visualization</em>
              {' '}— and runs end-to-end with a single command, or one step at a time.
            </p>
          </Section>

          <Section id="quick-start" index={2} title="Quick start">
            <p>
              One-time install lives in <InlineCode>docs/INSTALL.md</InlineCode> (either{' '}
              <InlineCode>micromamba</InlineCode> or <InlineCode>conda</InlineCode>). Once the env is
              built, verify with <InlineCode>doctor</InlineCode> then run the bundled{' '}
              <InlineCode>mamma_example</InlineCode> demo — ~56 MB, no login required:
            </p>
            <Code>{`# 1. Verify install
micromamba activate mamma          # or: conda activate mamma
python -m inference doctor

# 2. Fetch the example videos (~56 MB)
bash data/download_example.sh

# 3. Run end-to-end
python -m inference run \\
  --preset  configs/examples/presets/quick.yaml \\
  --capture configs/examples/captures/mamma_example.json \\
  --out-tag demo -v`}</Code>
            <p>
              Outputs land under{' '}
              <PathCode>output/ma_*/demo/mamma_example/…</PathCode>: SMPL-X parameters,
              joint and vertex trajectories, per-camera overlays, and a preview MP4.
            </p>
            <p>
              For the full paper datasets (Markerless Dance, Multi-People, iPhone, Eval,
              and the synthetic training data), see <a href="#dataset" className="text-primary underline-offset-2 hover:underline">Dataset</a>.
            </p>
          </Section>

          <Section id="dataset" index={3} title="Dataset">
            <p>
              The MAMMA project page hosts the multi-view captures used in the paper,
              the evaluation set, and the synthetic training data. Access requires a
              free account:
            </p>
            <ol className="list-decimal pl-5 space-y-1.5">
              <li>
                Register at{' '}
                <a
                  href="https://mamma.is.tue.mpg.de/download.php"
                  target="_blank"
                  rel="noreferrer"
                  className="text-primary hover:underline underline-offset-2"
                >
                  mamma.is.tue.mpg.de/download.php
                </a>.
              </li>
              <li>Click the confirmation link in the email you receive.</li>
              <li>
                Open the <Strong>Pipeline assets</Strong> panel on the Home page,
                sign in once, and use the one-click download buttons. Or run a
                shell script from the table below — both flows write into{' '}
                <PathCode>&lt;repo&gt;/data/</PathCode>.
              </li>
            </ol>
            <Table
              head={['Script', 'Dataset']}
              rows={[
                ['data/download_mamma_weights.sh',       'MAMMA landmark .ckpt + downsampled SMPL-X verts (always needed for inference)'],
                ['data/download_mamma_dance.sh',         'Markerless Dance — WestCoastSwing, Bachata, Breakdance, Ballroom (32 cams)'],
                ['data/download_mamma_multi_people.sh',  'Markerless Multi-People — 34 sequences, 3–6 people (32 cams)'],
                ['data/download_mamma_iphone.sh',        'Markerless iPhone — 42 sequences indoors + outdoors (4 iPhones)'],
                ['data/download_mamma_eval.sh',          'Evaluation set — GT, masks, markers, videos'],
                ['data/download_mamma_syn_hf.sh',        'Synthetic training data (WebDataset) from Hugging Face — default; used by landmarks/train.py'],
                ['data/download_mamma_syn_wd.sh',        'Synthetic training data — MAMMA-account alternative to the Hugging Face download'],
              ]}
            />
            <p>
              Full catalog with sizes, asset types, and video-encoding tables is in{' '}
              <InlineCode>docs/DATASETS.md</InlineCode>. Each script also supports{' '}
              <InlineCode>--help</InlineCode>.
            </p>
            <p>
              <Strong>Optional pose prior?</Strong>{' '}
              <InlineCode>data/download_vposer.sh</InlineCode> fetches the VPoser v2.05 manifold
              prior used by <Step name="ma_3d" /> only when <InlineCode>vposer_recon_loss</InlineCode>{' '}
              is enabled (SMPL-X account; the default pipeline runs without it).
            </p>
            <p>
              <Strong>Running on your own footage?</Strong> See{' '}
              <InlineCode>docs/YOUR-DATA.md</InlineCode> for the calibration,
              capture-descriptor, and folder layout you need.
            </p>
          </Section>

          <Section id="pipeline" index={4} title="Pipeline stages">
            <p>
              The DAG is{' '}
              <span className="font-mono inline-flex items-center flex-wrap gap-1 align-middle">
                <Step name="ma_cap" /><Arrow /><Step name="ma_masks" /><Arrow /><Step name="ma_2d" /><Arrow /><Step name="ma_3d" /><Arrow /><Step name="ma_vis" />
              </span>
              , with both <Step name="ma_2d" /> and <Step name="ma_3d" /> also consuming{' '}
              <Step name="ma_cap" />. Each <InlineCode>(step, sequence)</InlineCode> pair writes a{' '}
              <span className="font-mono text-[12.5px] px-1.5 py-[1px] rounded bg-status-completed-bg text-status-completed">DONE</span>
              {' '}sentinel; subsequent runs skip it unless <InlineCode>--force</InlineCode> is passed.
            </p>
            <Table
              head={['Stage', 'What it does', 'Writes to']}
              rows={[
                ['ma_cap',   'Loads multi-view capture, normalizes cameras and frames', 'output/ma_cap/<tag>/<dataset>/<seq>/'],
                ['ma_masks', 'Per-person segmentation (SAM + YOLO) with re-ID',         'output/ma_masks/<tag>/<dataset>/<seq>/'],
                ['ma_2d',    '2D landmark detection per camera (MammaNet)',             'output/ma_2d/<tag>/<dataset>/<seq>/'],
                ['ma_3d',    'Multi-view SMPL-X optimization',                          'output/ma_3d/<tag>/<dataset>/<seq>/'],
                ['ma_vis',   'Per-camera overlays + interactive scene',                 'output/ma_vis/<tag>/<dataset>/<seq>/'],
              ]}
              firstColColor="step"
              lastColColor="path"
            />
          </Section>

          <Section id="input-modes" index={5} title="Input modes">
            <p>
              Every step accepts the same three input modes. At the task-config
              level you set them on the <InlineCode>ma_cap</InlineCode> block;
              the runner translates them to the CLI flags shown below.
            </p>
            <Table
              head={['Mode', 'Task-config source', 'CLI flags (run_ma_cap.py)', 'Layout']}
              rows={[
                ['NPZ (default, chained)', 'capture.json (capture_root + calib)',                 '--json <capture.json>',                          '<capture_root>/<seq>/<cam>/*.jpg'],
                ['Videos',                 'ma_cap.videos_dir (preset; derived from capture)',    '--videos_dir <dir> --calibration <file>',        '<videos_dir>/<cam_name>.mp4'],
                ['Images root',            'ma_cap.images_root_dir (preset) + capture.calib',     '--images_root_dir <dir> --calibration <file>',   '<images_root_dir>/<cam_name>/*.{jpg,png}'],
              ]}
              lastColColor="path"
            />
            <p>
              Filename stems map directly to camera names (<InlineCode>cam01.mp4</InlineCode> →{' '}
              <InlineCode>cam01</InlineCode>). The three modes are mutually
              exclusive at the step level — the builder raises if more than one
              is set. Image extensions accepted:{' '}
              <InlineCode>.jpg .jpeg .png .ppm .pgm .tif .tiff .webp</InlineCode>.
            </p>

            <h3 className="text-foreground font-medium pt-3">Per-camera NPZ contract</h3>
            <p>
              <Step name="ma_cap" /> emits one <PathCode>&lt;cam&gt;.npz</PathCode> per camera plus
              a sibling <PathCode>global.npz</PathCode> under{' '}
              <PathCode>output/ma_cap/&lt;tag&gt;/&lt;dataset&gt;/&lt;seq&gt;/gt/</PathCode>.
              The per-camera file always carries the same keys; only one
              frame-source carrier (<InlineCode>video_path</InlineCode> or{' '}
              <InlineCode>img_abs_path</InlineCode>) is populated at a time.
            </p>
            <Table
              head={['Field', 'Videos mode', 'Images / JSON mode']}
              rows={[
                ['video_path',                          'absolute MP4 path',         '"" (empty string)'],
                ['img_abs_path / img_rel_path',         'empty array',                'one path per frame'],
                ['frame_start, frame_end',              'canonical range (inherited by every step)', 'canonical range'],
                ['cam_int, cam_ext',                    '3x3 K, 4x4 world→cam',       '3x3 K, 4x4 world→cam'],
                ['cam_img_w, cam_img_h, cam_portrait',  'frame dimensions',           'frame dimensions'],
                ['vicon_radial_2',                      '5-float (XCP source) or None','5-float (XCP source) or None'],
                ['is_body_in_img',                      'bool[N], all True',          'bool[N], all True'],
              ]}
            />
          </Section>

          <Section id="presets" index={6} title="Presets and captures">
            <p>A run is described by two files, or one frozen merge:</p>
            <ul className="space-y-3">
              <Bullet label="Preset" color="primary">
                Capture-independent template. Says which steps run, which engine, which flags, which weights.
                Shipped: <InlineCode>presets/full.yaml</InlineCode> (all frames, memory-efficient),{' '}
                <InlineCode>presets/quick.yaml</InlineCode> (~2 s smoke slice),{' '}
                <InlineCode>presets/debug.yaml</InlineCode> (overlays + visualizations on), and{' '}
                <InlineCode>presets/full_tensorrt.yaml</InlineCode> (full + TensorRT 2D, NVIDIA-only).
              </Bullet>
              <Bullet label="Capture" color="completed">
                JSON manifest: <InlineCode>capture_root</InlineCode>, <InlineCode>calib</InlineCode>,{' '}
                <InlineCode>cams</InlineCode>, <InlineCode>sequences</InlineCode>, optional{' '}
                <InlineCode>videos_subdir</InlineCode>. 13 manifests under{' '}
                <PathCode>configs/examples/captures/</PathCode>.
              </Bullet>
              <Bullet label="Run config" color="pending">
                Generated, not hand-written. The in-memory merge of preset + capture + selection. The GUI
                persists one per submission at{' '}
                <PathCode>gui/var/interface/run_configs/run_&lt;id&gt;.json</PathCode>.
              </Bullet>
            </ul>
          </Section>

          <Section id="commands" index={7} title="Useful commands">
            <Code>{`python -m inference --help                                # all subcommands
python -m inference run --help                            # run flags
python -m inference run-step --help                       # run-step flags
python -m inference doctor                                # pre-flight env check
python -m inference doctor --task path/to/run.json        # validate a bound run config
python scripts/smoke_test.py                              # full regression (~5 min)`}</Code>
          </Section>

          <Section id="gui" index={8} title="Running the GUI">
            <p>
              A browser UI lives under <PathCode>gui/</PathCode>. Pipeline execution calls the same{' '}
              <InlineCode>inference</InlineCode> runner the CLI uses, so anything that runs from the CLI
              also runs from the UI. Everything stays in the single <InlineCode>mamma</InlineCode> conda env.
            </p>
            <Code>{`# Dev — auto-reload, two ports (Flask :8000 + Vite :3000)
gui/scripts/dev.sh

# Production — single bundle on :8000
gui/scripts/prod.sh
gui/scripts/prod.sh --skip-build   # reuse existing build/`}</Code>
            <p>
              GUI-only state (SQLite, task-config snapshots, runner logs) lives under{' '}
              <PathCode>gui/var/</PathCode>; pipeline artifacts land in the shared top-level{' '}
              <PathCode>output/</PathCode>.
            </p>
          </Section>

          <Section id="layout" index={9} title="Repository layout">
            <Code>{`.
README.md                       project overview
docs/INSTALL.md                 environment + dependency installation
docs/DATASETS.md                dataset catalog + per-script download usage
docs/YOUR-DATA.md               how to run on your own multi-view footage
docs/steps.md                   step → directory → builder mapping + DAG

inference/                      orchestration: env, runner, step builders, doctor CLI
capture/                        ma_cap   (capture / camera normalization)
segmentation/                   ma_masks (SAM + YOLO + re-ID)
landmarks/                      ma_2d    (2D landmark detection)
optimization/                   ma_3d    (multi-view SMPL-X fitting)
visualization/                  ma_vis   (overlays + preview)
configs/examples/               shipped presets, captures, calibration
data/                           body models + trained weights + datasets (gitignored)
output/                         run outputs (gitignored)
gui/                            Flask backend + Vite/React frontend
scripts/                        smoke tests + utilities`}</Code>
          </Section>

          <Section id="resources" index={10} title="Resources">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <ResourceCard accent="primary"   href="https://mamma.is.tue.mpg.de/"            title="Project page" desc="The official MAMMA landing page." />
              <ResourceCard accent="completed" href="https://arxiv.org/pdf/2506.13040"        title="Paper (PDF)"   desc="Read the paper (CVPR 2026 oral)." />
              <ResourceCard accent="pending"   href="https://mamma.is.tue.mpg.de/download.php" title="Dataset"      desc="Browse all datasets released with the paper." />
              <ResourceCard accent="mixed"     href="https://mamma.is.tue.mpg.de/license.html" title="License"      desc="Non-commercial research license terms." />
            </div>
          </Section>
        </article>
      </div>
    </div>
  );
}

function Section({ id, index, title, children }: { id: string; index: number; title: string; children: React.ReactNode }) {
  return (
    <section id={id} className="scroll-mt-20">
      <div className="flex items-center gap-3 mb-5">
        <span className="font-mono text-[11px] text-primary tabular-nums tracking-wider px-1.5 py-0.5 rounded bg-primary-muted/50 ring-1 ring-inset ring-primary/20">
          {String(index).padStart(2, '0')}
        </span>
        <h2 className="text-foreground text-xl font-medium tracking-tight">
          {title}
        </h2>
        <div className="flex-1 h-px bg-gradient-to-r from-border to-transparent" />
      </div>
      <div className="text-foreground-muted text-sm leading-relaxed space-y-4">
        {children}
      </div>
    </section>
  );
}

function InlineCode({ children }: { children: React.ReactNode }) {
  return (
    <code className="font-mono text-[12.5px] text-primary bg-primary-muted/35 px-1.5 py-[1px] rounded ring-1 ring-inset ring-primary/15">
      {children}
    </code>
  );
}

function PathCode({ children }: { children: React.ReactNode }) {
  return (
    <code className="font-mono text-[12.5px] text-status-completed bg-status-completed-bg/70 px-1.5 py-[1px] rounded ring-1 ring-inset ring-status-completed/20">
      {children}
    </code>
  );
}

function Strong({ children }: { children: React.ReactNode }) {
  return <strong className="font-medium text-foreground">{children}</strong>;
}

function Step({ name }: { name: keyof typeof STEP_COLOR | string }) {
  const cls = STEP_COLOR[name] ?? 'text-foreground';
  return <code className={`font-mono text-[12.5px] ${cls}`}>{name}</code>;
}

function Arrow() {
  return <span className="text-foreground-faint px-0.5">→</span>;
}

function Code({ children }: { children: string }) {
  const lines = children.split('\n');
  return (
    <pre className="relative bg-surface-1 border border-border-subtle rounded-md p-4 pl-5 overflow-x-auto text-[12.5px] leading-relaxed shadow-sm shadow-black/30 before:content-[''] before:absolute before:left-0 before:top-2 before:bottom-2 before:w-[3px] before:rounded-r before:bg-primary/70">
      <code className="!bg-transparent !p-0 font-mono whitespace-pre">
        {lines.map((line, i) => {
          const isComment = /^\s*#/.test(line);
          return (
            <span
              key={i}
              className={`block ${isComment ? 'text-foreground-subtle italic' : 'text-foreground'}`}
            >
              {line || ' '}
            </span>
          );
        })}
      </code>
    </pre>
  );
}

const ACCENT: Record<string, { text: string; bg: string; ring: string; stripe: string }> = {
  primary:   { text: 'text-primary',          bg: 'bg-primary-muted/40',        ring: 'ring-primary/25',          stripe: 'bg-primary' },
  completed: { text: 'text-status-completed', bg: 'bg-status-completed-bg/70',  ring: 'ring-status-completed/25', stripe: 'bg-status-completed' },
  pending:   { text: 'text-status-pending',   bg: 'bg-status-pending-bg/70',    ring: 'ring-status-pending/25',   stripe: 'bg-status-pending' },
  mixed:     { text: 'text-status-mixed',     bg: 'bg-status-mixed-bg/70',      ring: 'ring-status-mixed/25',     stripe: 'bg-status-mixed' },
};

function Bullet({ label, color, children }: { label: string; color: keyof typeof ACCENT; children: React.ReactNode }) {
  const a = ACCENT[color];
  return (
    <li className="flex gap-3 items-start">
      <span className={`shrink-0 mt-0.5 inline-flex items-center text-[11px] font-medium px-2 py-0.5 rounded ${a.bg} ${a.text} ring-1 ring-inset ${a.ring}`}>
        {label}
      </span>
      <div className="min-w-0 flex-1">{children}</div>
    </li>
  );
}

function Table({
  head,
  rows,
  firstColColor,
  lastColColor,
}: {
  head: string[];
  rows: string[][];
  firstColColor?: 'step';
  lastColColor?: 'path';
}) {
  return (
    <div className="overflow-x-auto rounded-md border border-border-subtle">
      <table className="w-full text-[12.5px]">
        <thead className="bg-primary-muted/25">
          <tr>
            {head.map(h => (
              <th
                key={h}
                className="text-left font-medium text-primary text-[11px] uppercase tracking-[0.08em] px-3 py-2 border-b border-primary/20"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr
              key={i}
              className={`border-b border-border-subtle/60 last:border-0 ${i % 2 === 1 ? 'bg-surface-1/50' : ''}`}
            >
              {r.map((c, j) => {
                const isFirst = j === 0;
                const isLast = j === r.length - 1;
                let cls = 'text-foreground-muted';
                if (isFirst && firstColColor === 'step') {
                  cls = `font-mono ${STEP_COLOR[c] ?? 'text-foreground'}`;
                } else if (isFirst) {
                  cls = 'text-foreground font-mono';
                }
                if (isLast && lastColColor === 'path') {
                  cls = 'font-mono text-status-completed';
                }
                return (
                  <td key={j} className={`px-3 py-2 align-top ${cls}`}>
                    {c}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ResourceCard({
  href, title, desc, accent,
}: { href: string; title: string; desc: string; accent: keyof typeof ACCENT }) {
  const a = ACCENT[accent];
  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className={`group relative block bg-surface-1 border border-border-subtle rounded-lg p-4 transition-colors hover:bg-surface-2 hover:border-border overflow-hidden`}
    >
      <span aria-hidden className={`absolute left-0 top-0 bottom-0 w-[3px] ${a.stripe}`} />
      <div className="flex items-center justify-between gap-2 mb-1 pl-2">
        <div className={`text-sm font-medium group-hover:underline ${a.text}`}>{title}</div>
        <ExternalLink className="w-3.5 h-3.5 text-foreground-subtle group-hover:text-primary transition-colors" />
      </div>
      <div className="text-foreground-muted text-xs leading-relaxed pl-2">{desc}</div>
    </a>
  );
}
