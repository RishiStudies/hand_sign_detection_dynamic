import type { ReactNode } from "react";

type MetricTileProps = {
  icon: ReactNode;
  label: string;
  value: string;
  detail?: string;
  accent?: "blue" | "green" | "amber" | "neutral";
};

export function MetricTile({ icon, label, value, detail, accent = "neutral" }: MetricTileProps) {
  const accentClass =
    accent === "blue"
      ? "metric-tile-blue"
      : accent === "green"
        ? "metric-tile-green"
        : accent === "amber"
          ? "metric-tile-amber"
          : "metric-tile-neutral";

  return (
    <article
      className={`hover-card metric-tile rounded-2xl border border-[var(--line-soft)] bg-[var(--bg-panel)] px-3.5 py-3 ${accentClass}`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <p className="text-[10px] uppercase tracking-[0.24em] text-[var(--text-dim)]">{label}</p>
          <p className="mt-1 text-lg font-semibold leading-tight break-words text-white">{value}</p>
          {detail ? <p className="mt-1 text-[11px] leading-snug break-words text-[var(--text-dim)]">{detail}</p> : null}
        </div>
        <span className="metric-tile-icon mt-0.5 shrink-0 flex h-9 w-9 items-center justify-center rounded-xl border border-white/10 bg-black/20 text-white/90">
          {icon}
        </span>
      </div>
    </article>
  );
}