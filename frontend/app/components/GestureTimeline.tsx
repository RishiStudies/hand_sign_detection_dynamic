import type { GestureLogItem } from "../types";

type GestureTimelineProps = {
  items: GestureLogItem[];
};

export function GestureTimeline({ items }: GestureTimelineProps) {
  if (items.length === 0) {
    return (
      <div className="timeline-empty rounded-2xl border border-white/10 bg-black/20 px-3 py-3 text-xs text-[var(--text-dim)]">
        No gestures yet. Timeline updates when detection locks onto motion.
      </div>
    );
  }

  return (
    <div className="timeline-shell min-h-0 flex-1 overflow-auto pr-1">
      <div className="timeline-thread space-y-2">
        {items.map((item, index) => (
          <article
            key={`${item.label}-${item.time}-${index}`}
            className="hover-card timeline-item relative rounded-2xl border border-white/10 bg-black/20 px-3.5 py-3"
          >
            <span className="timeline-node absolute left-0 top-5 h-2.5 w-2.5 -translate-x-1/2 rounded-full bg-[var(--accent-blue)]" />
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="text-sm font-semibold text-white">{item.label}</p>
                <div className="mt-1 flex flex-wrap items-center gap-2 text-[10px] uppercase tracking-[0.18em] text-[var(--text-dim)]">
                  <span>{item.mode}</span>
                  <span>{item.confidence}% lock</span>
                  {item.combo ? <span className="text-[var(--accent-green)]">combo hit</span> : null}
                </div>
                {item.combo ? (
                  <p className="mt-2 text-[11px] text-[var(--accent-green)]">Combo event: {item.combo}</p>
                ) : null}
              </div>
              <span className="shrink-0 text-[11px] text-[var(--text-dim)]">{item.time}</span>
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}