import type { ReactNode } from "react";

type ControlButtonProps = {
  icon: ReactNode;
  label: string;
  sublabel: string;
  onClick: () => void;
  emphasis?: "primary" | "secondary" | "ghost";
};

export function ControlButton({
  icon,
  label,
  sublabel,
  onClick,
  emphasis = "secondary",
}: ControlButtonProps) {
  const emphasisClass =
    emphasis === "primary"
      ? "control-button-primary"
      : emphasis === "ghost"
        ? "control-button-ghost"
        : "control-button-secondary";

  return (
    <button
      type="button"
      onClick={onClick}
      className={`glass hover-button control-button flex w-full items-center gap-3 rounded-[22px] px-4 py-3.5 text-left transition ${emphasisClass}`}
    >
      <span className="control-button-icon flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl border border-white/12 bg-white/6 text-white/90">
        {icon}
      </span>
      <span className="min-w-0 flex-1">
        <span className="block text-sm font-semibold tracking-[0.08em] text-white">{label}</span>
        <span className="mt-0.5 block text-[11px] uppercase tracking-[0.16em] text-[var(--text-dim)]">
          {sublabel}
        </span>
      </span>
    </button>
  );
}