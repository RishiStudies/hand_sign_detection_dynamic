import Link from "next/link";
import {
  Activity,
  Cpu,
  Gauge,
  Network,
  ShieldCheck,
  Sparkles,
} from "lucide-react";

export default function HomePage() {
  return (
    <main className="relative flex min-h-screen items-center justify-center overflow-hidden px-6 py-10 text-[var(--text-main)] md:px-10">
      <div className="landing-grid-pointer absolute inset-0 pointer-events-none" />

      <section className="glass reveal w-full max-w-[1180px] rounded-3xl border border-[var(--line)] p-7 md:p-12">
        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded-full border border-[var(--accent-amber)]/45 bg-[var(--accent-amber)]/12 px-3 py-1 text-[10px] uppercase tracking-[0.16em] text-amber-100">
            Dynamic Handsign Detection System
          </span>
          <span className="rounded-full border border-white/10 bg-black/25 px-3 py-1 text-[10px] uppercase tracking-[0.16em] text-[var(--text-dim)]">
            Real-time Inference Platform
          </span>
        </div>

        <h1 className="mt-5 max-w-[940px] text-3xl font-semibold leading-tight text-white md:text-5xl">
          A Full-Stack Real-Time Hand Sign Detection Console for Live Interaction, Model Validation, and System Observability
        </h1>

        <p className="mt-5 max-w-[980px] text-sm leading-relaxed text-[var(--text-dim)] md:text-base">
          This software project provides an end-to-end dynamic hand-sign detection workflow built for practical development,
          testing, and demonstration scenarios. The platform combines a browser-based live interface, a FastAPI backend for
          prediction and health telemetry, and shared model artifact contracts to ensure training and runtime consistency.
          The system supports low-latency single-frame inference, sequence-aware processing pathways, calibration capture,
          timeline logging, backend diagnostics, and controlled runtime operations from one integrated console.
        </p>

        <div className="mt-8 grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
          <article className="hover-card rounded-2xl border border-white/12 bg-black/25 p-4">
            <div className="flex items-center gap-2 text-[var(--accent-amber)]">
              <Activity size={16} />
              <h2 className="text-xs font-semibold uppercase tracking-[0.14em] text-white">Live Inference Workflow</h2>
            </div>
            <p className="mt-2 text-sm leading-relaxed text-[var(--text-dim)]">
              Streams webcam frames to backend prediction endpoints and returns gesture labels, confidence values, and mode-aware telemetry for live runtime interpretation.
            </p>
          </article>

          <article className="hover-card rounded-2xl border border-white/12 bg-black/25 p-4">
            <div className="flex items-center gap-2 text-[var(--accent-amber)]">
              <Cpu size={16} />
              <h2 className="text-xs font-semibold uppercase tracking-[0.14em] text-white">Model and Feature Discipline</h2>
            </div>
            <p className="mt-2 text-sm leading-relaxed text-[var(--text-dim)]">
              Uses explicit feature-schema contracts and artifact metadata so training outputs remain compatible with runtime loaders and inference request validation.
            </p>
          </article>

          <article className="hover-card rounded-2xl border border-white/12 bg-black/25 p-4">
            <div className="flex items-center gap-2 text-[var(--accent-amber)]">
              <Gauge size={16} />
              <h2 className="text-xs font-semibold uppercase tracking-[0.14em] text-white">Calibration and Diagnostics</h2>
            </div>
            <p className="mt-2 text-sm leading-relaxed text-[var(--text-dim)]">
              Includes guided calibration states, timeline logging, ping-based health visibility, and clear operational controls for robust software validation loops.
            </p>
          </article>

          <article className="hover-card rounded-2xl border border-white/12 bg-black/25 p-4">
            <div className="flex items-center gap-2 text-[var(--accent-amber)]">
              <Network size={16} />
              <h2 className="text-xs font-semibold uppercase tracking-[0.14em] text-white">Service-Oriented Architecture</h2>
            </div>
            <p className="mt-2 text-sm leading-relaxed text-[var(--text-dim)]">
              Frontend and backend are decoupled through API contracts, allowing independent iteration on interface design, model orchestration, and runtime resilience.
            </p>
          </article>

          <article className="hover-card rounded-2xl border border-white/12 bg-black/25 p-4">
            <div className="flex items-center gap-2 text-[var(--accent-amber)]">
              <ShieldCheck size={16} />
              <h2 className="text-xs font-semibold uppercase tracking-[0.14em] text-white">Operational Guardrails</h2>
            </div>
            <p className="mt-2 text-sm leading-relaxed text-[var(--text-dim)]">
              Supports endpoint protection, request-throttling controls, health probes, and deterministic runtime states to improve reliability under changing load patterns.
            </p>
          </article>

          <article className="hover-card rounded-2xl border border-white/12 bg-black/25 p-4">
            <div className="flex items-center gap-2 text-[var(--accent-amber)]">
              <Sparkles size={16} />
              <h2 className="text-xs font-semibold uppercase tracking-[0.14em] text-white">Development Purpose</h2>
            </div>
            <p className="mt-2 text-sm leading-relaxed text-[var(--text-dim)]">
              Designed as both a showcase and engineering workbench for experimenting with gesture UX, model performance, calibration behavior, and full-stack CV integration.
            </p>
          </article>
        </div>

        <div className="mt-9 flex flex-wrap items-center gap-4">
          <Link
            href="/console"
            className="inline-flex items-center rounded-2xl border border-[var(--accent-amber)]/55 bg-[var(--accent-amber)]/18 px-6 py-3 text-sm font-semibold uppercase tracking-[0.12em] text-amber-100 transition hover:bg-[var(--accent-amber)]/30 hover:border-[var(--accent-amber)]/75"
          >
            Get Started
          </Link>
          <p className="text-xs uppercase tracking-[0.14em] text-[var(--text-dim)]">
            Opens the live detection console at /console
          </p>
        </div>
      </section>
    </main>
  );
}
