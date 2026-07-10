import { useEffect, useState } from 'react';
import { Box } from 'lucide-react';
import { ExportPanel, jget } from './ExportPanel';
import { SequenceSelect } from './SequenceSelect';

/** Inline SMPL-X export on a capture's results page. Reuses the shared ExportPanel;
 *  scopes the sequence list to this capture. Always expanded (no toggle); tool
 *  setup (Blender downloads) lives in the Exporter tab. */

interface Seq { tag: string; capture: string; seq: string; people: number; ma_3d_dir: string; ma_cap_dir: string; already_exported: boolean; mtime?: number; }
const key = (s: Seq) => `${s.tag}/${s.capture}/${s.seq}::${s.ma_3d_dir}`;

export function ResultExport({ captureName, initialSeq, onGoToExporter }: {
  captureName: string; initialSeq?: string; onGoToExporter?: () => void;
}) {
  const [seqs, setSeqs] = useState<Seq[] | null>(null);
  const [sels, setSels] = useState<string[]>([]);

  useEffect(() => {
    if (seqs !== null) return;
    jget<{ sequences: Seq[] }>('/api/exporter/sequences').then(r => {
      const mine = r.sequences.filter(s => s.capture === captureName);
      setSeqs(mine);
      const pre = mine.find(s => s.seq === initialSeq) ?? (mine.length === 1 ? mine[0] : undefined);
      if (pre) setSels([key(pre)]);
    });
  }, [seqs, captureName, initialSeq]);

  const targets = (seqs ?? []).filter(s => sels.includes(key(s)));

  return (
    <section className="h-full">
      <div className="h-full bg-surface-1 border border-border-subtle rounded-xl p-5 shadow-sm shadow-black/30 ring-1 ring-inset ring-white/[0.02]">
        <div className="flex items-center gap-2">
          <Box className="w-4 h-4 text-foreground-subtle" />
          <span className="text-foreground text-lg font-medium tracking-tight">Export animation</span>
          <span className="ml-auto text-foreground-muted text-xs">npz · FBX · Alembic · BVH · USD</span>
        </div>
        {seqs && seqs.length === 0 ? (
          <p className="text-foreground-muted text-sm mt-3 pl-6">
            No exportable <code className="font-mono">ma_3d</code> results for this capture yet. You can also export from a
            {' '}<button onClick={onGoToExporter} className="text-primary hover:underline">custom path in the Exporter tab</button>.
          </p>
        ) : (
          <div className="mt-4 space-y-3">
            <SequenceSelect seqs={seqs ?? []} values={sels} onChange={setSels} getKey={key} />
            <ExportPanel targets={targets} onNeedTools={onGoToExporter} setupLabel="Download them in the Exporter tab" />
          </div>
        )}
      </div>
    </section>
  );
}
