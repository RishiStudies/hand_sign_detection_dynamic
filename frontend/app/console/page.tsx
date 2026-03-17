"use client";

import Image from "next/image";
import { useEffect, useMemo, useState } from "react";
import {
  Camera,
  Radar,
  Settings,
  SlidersHorizontal,
  Sparkles,
  X,
  Zap,
} from "lucide-react";

import { ControlButton } from "../components/ControlButton";
import { GestureTimeline } from "../components/GestureTimeline";
import { calibrationSlotBlueprint, useCalibrationFlow } from "../hooks/useCalibrationFlow";
import { useCameraCapture } from "../hooks/useCameraCapture";
import { usePredictionLoop } from "../hooks/usePredictionLoop";

function formatActiveTime(totalSeconds: number): string {
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;

  return [hours, minutes, seconds]
    .map((value) => value.toString().padStart(2, "0"))
    .join(":");
}

function clampPercent(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.min(100, Math.round(value * 100)));
}

type RightTabKey = "system" | "calibration" | "settings";

export default function ConsolePage() {
  const {
    videoRef,
    cameraReady,
    cameraError,
    startCamera,
    stopCamera,
    captureFrameBlob,
    captureCalibrationImage,
  } = useCameraCapture();

  const [detectionRunning, setDetectionRunning] = useState(false);

  const {
    predictionLabel,
    confidence,
    comboLabel,
    logItems,
    lastPredictionLatencyMs,
    lastPredictionMode,
    featureSchemaVersion,
    pingMs,
    pingOnline,
    predictionError,
    clearHistory,
  } = usePredictionLoop({ cameraReady, detectionRunning, captureFrameBlob });

  const {
    calibrationState,
    calibrationStep,
    calibrationSlots,
    showCalibration,
    calibrationError,
    runCalibration,
    abortCalibration,
  } = useCalibrationFlow({ cameraReady, captureCalibrationImage, clearHistory });

  const [activeSeconds, setActiveSeconds] = useState(0);
  const [activeRightTab, setActiveRightTab] = useState<RightTabKey>("settings");
  const [timelineOpen, setTimelineOpen] = useState(false);

  const displayError = cameraError ?? predictionError ?? calibrationError;
  const percent = useMemo(() => clampPercent(confidence), [confidence]);

  const runtimeStatus = useMemo(() => {
    if (!cameraReady) {
      return { label: "Offline", tone: "text-red-300" };
    }
    if (calibrationStep !== null) {
      return { label: "Calibrating", tone: "text-amber-300" };
    }
    if (detectionRunning) {
      return { label: "Tracking", tone: "text-[var(--accent-green)]" };
    }
    return { label: "Standby", tone: "text-white" };
  }, [calibrationStep, cameraReady, detectionRunning]);

  const signalQuality = useMemo(() => {
    if (!cameraReady) {
      return { label: "No signal", detail: "Start camera to begin" };
    }
    if (cameraError ?? predictionError ?? calibrationError) {
      return { label: "Interrupted", detail: "Backend or camera issue detected" };
    }
    if (!pingOnline) {
      return { label: "Unstable", detail: "Backend link unavailable" };
    }
    if (percent >= 82) {
      return { label: "Locked", detail: "High-confidence tracking" };
    }
    if (percent >= 56) {
      return { label: "Tracking", detail: "Minor drift" };
    }
    return { label: "Searching", detail: "Needs clearer pose" };
  }, [calibrationError, cameraError, cameraReady, percent, pingOnline, predictionError]);

  const spotlightToneClass = useMemo(() => {
    if (!cameraReady) {
      return "spotlight-idle";
    }
    if (comboLabel !== "None") {
      return "spotlight-combo";
    }
    if (percent >= 80) {
      return "spotlight-locked";
    }
    if (percent >= 50) {
      return "spotlight-searching";
    }
    return "spotlight-idle";
  }, [cameraReady, comboLabel, percent]);

  const backendOffline = !pingOnline;

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.code === "Space") {
        event.preventDefault();
        if (cameraReady) {
          setDetectionRunning(true);
        }
      }
      if (event.key.toLowerCase() === "s") {
        setDetectionRunning(false);
      }
      if (event.key.toLowerCase() === "c") {
        void clearHistory();
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [cameraReady, clearHistory]);

  useEffect(() => {
    const sessionStartedAt = Date.now();
    const updateActiveTime = () => {
      setActiveSeconds(Math.floor((Date.now() - sessionStartedAt) / 1000));
    };

    updateActiveTime();
    const intervalId = window.setInterval(updateActiveTime, 1000);

    return () => {
      window.clearInterval(intervalId);
    };
  }, []);

  if (backendOffline) {
    return (
      <main className="offline-shell flex h-screen items-center justify-center px-6 py-8 text-[var(--text-main)]">
        <section className="offline-card w-full max-w-[920px] rounded-3xl border border-amber-500/30 bg-black/45 p-8 backdrop-blur-md">
          <p className="hud-title text-xs uppercase tracking-[0.18em] text-amber-200/85">Dynamic Handsign Detection System</p>
          <h1 className="mt-3 text-3xl font-semibold text-white md:text-4xl">Backend link is currently unavailable</h1>
          <p className="mt-3 max-w-[760px] text-sm text-amber-100/90 md:text-base">
            Software scope: real-time hand-sign detection for live interaction, model validation, and system observability.
            The interface is paused until backend diagnostics recover.
          </p>

          <div className="mt-6 grid grid-cols-1 gap-3 md:grid-cols-3">
            <div className="rounded-2xl border border-white/12 bg-black/35 px-4 py-3">
              <p className="text-[10px] uppercase tracking-[0.16em] text-amber-100/75">Network</p>
              <p className="mt-1 text-lg font-semibold text-white">Offline</p>
            </div>
            <div className="rounded-2xl border border-white/12 bg-black/35 px-4 py-3">
              <p className="text-[10px] uppercase tracking-[0.16em] text-amber-100/75">Mode</p>
              <p className="mt-1 text-lg font-semibold text-white">{lastPredictionMode}</p>
            </div>
            <div className="rounded-2xl border border-white/12 bg-black/35 px-4 py-3">
              <p className="text-[10px] uppercase tracking-[0.16em] text-amber-100/75">Session</p>
              <p className="mt-1 text-lg font-semibold text-white">{formatActiveTime(activeSeconds)}</p>
            </div>
          </div>

          <div className="mt-6 rounded-2xl border border-amber-400/35 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
            <p className="font-semibold">Connection State</p>
            <p className="mt-1">{displayError ?? "Waiting for backend heartbeat on /health/live."}</p>
          </div>

          <div className="mt-6 flex flex-wrap items-center gap-3">
            <ControlButton
              icon={<Camera size={18} />}
              label={cameraReady ? "STOP CAMERA" : "START CAMERA"}
              sublabel={cameraReady ? "power down stage" : "stage readiness check"}
              onClick={() => {
                if (cameraReady) {
                  stopCamera();
                  setDetectionRunning(false);
                } else {
                  setDetectionRunning(true);
                  void startCamera();
                }
              }}
              emphasis="primary"
            />
          </div>
        </section>
      </main>
    );
  }

  return (
    <main className="relative flex h-screen flex-col overflow-hidden px-4 py-5 text-[var(--text-main)] md:px-8 md:py-7">
      <div className="console-grid mx-auto grid min-h-0 w-full flex-1 max-w-[1500px] grid-cols-1 gap-4 pb-3 lg:grid-cols-[280px_1fr_330px]">
        <section className="control-panel glass reveal rounded-3xl p-4">
          <div>
            <p className="hud-title text-[11px] uppercase tracking-[0.2em] text-white/85">Dynamic Handsign Detection System</p>
            <p className="mt-3 text-sm leading-relaxed text-[var(--text-dim)]">
              Scope: real-time hand-sign detection for live interaction, model validation, and system observability.
            </p>
          </div>

          <div className="mt-5 grid grid-cols-1 gap-3">
            <ControlButton
              icon={<Camera size={18} />}
              label={cameraReady ? "STOP CAMERA" : "START CAMERA"}
              sublabel={cameraReady ? "power down stage" : "bring stage online"}
              onClick={() => {
                if (cameraReady) {
                  stopCamera();
                  setDetectionRunning(false);
                } else {
                  setDetectionRunning(true);
                  void startCamera();
                }
              }}
              emphasis="primary"
            />
            <ControlButton
              icon={<SlidersHorizontal size={18} />}
              label={calibrationStep !== null ? `CALIBRATING ${calibrationStep + 1}/${calibrationSlotBlueprint.length}` : "CALIBRATE"}
              sublabel="alignment sequence"
              onClick={() => {
                setActiveRightTab("calibration");
                void runCalibration();
              }}
              emphasis="secondary"
            />
            <ControlButton
              icon={<Settings size={18} />}
              label={cameraReady ? (detectionRunning ? "PAUSE DETECTION" : "RESUME DETECTION") : "DETECTION IDLE"}
              sublabel={cameraReady ? "tracking cadence" : "camera required"}
              onClick={() => setDetectionRunning((prev) => !prev)}
              emphasis="ghost"
            />
          </div>

          <div className="spotlight-shell hover-card mt-4 rounded-2xl border border-white/12 bg-[var(--bg-panel)] px-4 py-4">
            <div className="flex items-center justify-between gap-3">
              <div className="min-w-0">
                <p className="hud-title text-[10px] uppercase tracking-[0.2em] text-white/70">Live Prediction</p>
                <p className="mt-1 truncate text-2xl font-semibold text-white">{predictionLabel}</p>
              </div>
              <span className={`rounded-full border border-white/15 bg-black/20 px-2.5 py-1 text-[10px] uppercase tracking-[0.15em] ${runtimeStatus.tone}`}>
                {runtimeStatus.label}
              </span>
            </div>
            <div className="mt-3 h-2 overflow-hidden rounded-full bg-black/35">
              <div
                className={`h-full rounded-full bg-gradient-to-r from-[var(--accent-amber)] to-[var(--accent-green)] transition-all duration-500 ${spotlightToneClass}`}
                style={{ width: `${percent}%` }}
              />
            </div>
            <div className="mt-2 flex items-center justify-between text-[11px] text-[var(--text-dim)]">
              <span>{percent}% confidence</span>
              <span>{lastPredictionMode}</span>
            </div>
            {comboLabel !== "None" ? (
              <div className="combo-banner mt-3 rounded-xl border border-[var(--accent-green)]/35 bg-[var(--accent-green)]/10 px-3 py-2 text-sm text-[var(--accent-green)]">
                <div className="flex items-center gap-2">
                  <Sparkles size={14} />
                  <span className="font-semibold">Combo:</span>
                  <span>{comboLabel}</span>
                </div>
              </div>
            ) : null}
          </div>
        </section>

        <section className="stage-panel stage-shell relative reveal rounded-[30px] border border-[var(--line)] bg-[var(--bg-panel-strong)] p-2">
          <div className="stage-status absolute left-5 top-5 z-30 flex items-center gap-2 rounded-full border border-white/20 bg-black/45 px-3 py-1 text-[11px] font-medium backdrop-blur-sm">
            <span className={`h-2 w-2 rounded-full ${cameraReady ? "bg-emerald-400 pulse-dot" : "bg-red-400"}`} />
            {cameraReady ? "Camera Live" : "Camera Offline"}
          </div>

          <div className="absolute right-5 top-5 z-30 flex items-center gap-2 rounded-full border border-white/20 bg-black/45 px-3 py-1 text-[10px] uppercase tracking-[0.18em] text-white/70 backdrop-blur-sm">
            <Radar size={12} />
            {signalQuality.label}
          </div>

          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            style={{ transform: "scaleX(-1)" }}
            className={`min-h-0 h-full w-full rounded-[26px] object-cover ${cameraReady ? "opacity-100" : "absolute inset-2 h-[calc(100%-16px)] w-[calc(100%-16px)] opacity-0"}`}
          />

          {!cameraReady && (
            <div className="stage-empty flex h-full min-h-0 w-full flex-col items-center justify-center rounded-[26px] px-6 text-center">
              <div className="max-w-[640px]">
                <p className="hud-title text-[11px] uppercase tracking-[0.2em] text-white/60">Start camera</p>
                <h1 className="mt-3 text-3xl font-semibold text-white md:text-4xl">Dynamic Handsign Detection System</h1>
                <p className="mt-3 text-sm text-[var(--text-dim)] md:text-base">
                  This console provides real-time gesture inference, confidence analysis, and calibration tooling for software
                  development and validation workflows.
                </p>
              </div>
            </div>
          )}

          <div className="stage-frame pointer-events-none absolute inset-2 rounded-[28px] border border-white/10" />

          <div className="spotlight-overlay pointer-events-none absolute bottom-5 left-1/2 w-[min(680px,calc(100%-48px))] -translate-x-1/2 rounded-[24px] border border-white/15 bg-black/55 px-5 py-4 backdrop-blur-md">
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0">
                <p className="hud-title text-[10px] uppercase tracking-[0.18em] text-white/70">Inference output</p>
                <p className="mt-1 truncate text-3xl font-semibold text-white">{predictionLabel}</p>
                <p className="mt-1 text-[11px] uppercase tracking-[0.15em] text-[var(--text-dim)]">{signalQuality.detail}</p>
              </div>
              <div className="text-right">
                <p className="text-[10px] uppercase tracking-[0.18em] text-[var(--text-dim)]">Latency</p>
                <p className="mt-1 text-base font-semibold text-white">{lastPredictionLatencyMs !== null ? `${lastPredictionLatencyMs}ms` : "pending"}</p>
              </div>
            </div>
            <div className="mt-3 h-2 overflow-hidden rounded-full bg-black/45">
              <div className="h-full rounded-full bg-gradient-to-r from-[var(--accent-amber)] to-[var(--accent-green)] transition-all duration-500" style={{ width: `${percent}%` }} />
            </div>
          </div>
        </section>

        <section className="system-panel glass reveal flex min-h-0 flex-col rounded-3xl p-4">
          <div className="panel-tabs grid grid-cols-3 gap-1.5 rounded-2xl border border-white/12 bg-black/25 p-1">
            <button
              type="button"
              className={`tab-button rounded-xl px-2 py-2 text-[11px] uppercase tracking-[0.12em] ${activeRightTab === "system" ? "is-active" : ""}`}
              onClick={() => setActiveRightTab("system")}
            >
              System
            </button>
            <button
              type="button"
              className={`tab-button rounded-xl px-2 py-2 text-[11px] uppercase tracking-[0.12em] ${activeRightTab === "calibration" ? "is-active" : ""}`}
              onClick={() => setActiveRightTab("calibration")}
            >
              Calibration
            </button>
            <button
              type="button"
              className={`tab-button rounded-xl px-2 py-2 text-[11px] uppercase tracking-[0.12em] ${activeRightTab === "settings" ? "is-active" : ""}`}
              onClick={() => setActiveRightTab("settings")}
            >
              Settings
            </button>
          </div>

          <div className="mt-3 min-h-0 flex-1 overflow-auto pr-1">
            {activeRightTab === "system" ? (
              <div className="space-y-2.5">
                <div className="rounded-2xl border border-white/10 bg-black/25 px-3 py-3">
                  <p className="text-[10px] uppercase tracking-[0.14em] text-[var(--text-dim)]">Ping latency</p>
                  <p className="mt-1 text-xl font-semibold text-white">{pingMs !== null ? `${pingMs}ms` : "offline"}</p>
                </div>
                <div className="rounded-2xl border border-white/10 bg-black/25 px-3 py-3">
                  <p className="text-[10px] uppercase tracking-[0.14em] text-[var(--text-dim)]">Model mode</p>
                  <p className="mt-1 text-xl font-semibold text-white">{lastPredictionMode}</p>
                </div>
                <div className="rounded-2xl border border-white/10 bg-black/25 px-3 py-3">
                  <p className="text-[10px] uppercase tracking-[0.14em] text-[var(--text-dim)]">Session duration</p>
                  <p className="mt-1 text-xl font-semibold text-white">{formatActiveTime(activeSeconds)}</p>
                </div>
                <div className="rounded-2xl border border-white/10 bg-black/25 px-3 py-3">
                  <p className="text-[10px] uppercase tracking-[0.14em] text-[var(--text-dim)]">Feature schema</p>
                  <p className="mt-1 text-base font-semibold text-white">{featureSchemaVersion ?? "pending"}</p>
                </div>
              </div>
            ) : null}

            {activeRightTab === "calibration" ? (
              <div className="space-y-3">
                <div className="rounded-2xl border border-white/10 bg-black/25 px-3 py-3">
                  <p className="text-[10px] uppercase tracking-[0.14em] text-[var(--text-dim)]">Calibration state</p>
                  <p className="mt-1 text-base font-semibold text-white">{calibrationState}</p>
                  <p className="mt-1 text-xs text-[var(--text-dim)]">
                    {calibrationStep !== null
                      ? `Capturing ${calibrationSlotBlueprint[calibrationStep]?.label} (${calibrationStep + 1}/${calibrationSlotBlueprint.length})`
                      : "Run alignment sequence to capture reference poses."}
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-1.5">
                  {calibrationSlots.map((slot, idx) => (
                    <div
                      key={slot.label}
                      className={`relative aspect-video overflow-hidden rounded-xl border ${
                        calibrationStep === idx ? "border-[var(--accent-green)]/70" : "border-white/12"
                      } bg-black/25`}
                    >
                      {slot.image ? (
                        <Image src={slot.image} alt={slot.label} fill unoptimized className="object-cover" />
                      ) : null}
                      <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/70 to-transparent px-2 py-1.5">
                        <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-white/90">{slot.label}</p>
                        <p className="text-[10px] text-[var(--text-dim)]">{slot.image ? "Captured" : slot.hint}</p>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    className="rounded-xl border border-[var(--accent-amber)]/45 bg-[var(--accent-amber)]/12 px-3 py-2 text-xs font-semibold uppercase tracking-[0.12em] text-amber-100"
                    onClick={() => {
                      void runCalibration();
                    }}
                  >
                    {showCalibration ? "Re-run calibration" : "Start calibration"}
                  </button>
                  {showCalibration ? (
                    <button
                      type="button"
                      className="rounded-xl border border-white/15 bg-black/25 px-3 py-2 text-xs uppercase tracking-[0.12em] text-[var(--text-dim)]"
                      onClick={abortCalibration}
                    >
                      Stop
                    </button>
                  ) : null}
                </div>
              </div>
            ) : null}

            {activeRightTab === "settings" ? (
              <div className="space-y-2.5">
                <div className="rounded-2xl border border-white/10 bg-black/25 px-3 py-3">
                  <p className="text-[10px] uppercase tracking-[0.14em] text-[var(--text-dim)]">Tracking mode</p>
                  <p className="mt-1 text-base font-semibold text-white">{detectionRunning ? "Active" : "Paused"}</p>
                </div>
                <button
                  type="button"
                  className="w-full rounded-2xl border border-white/12 bg-black/25 px-3 py-3 text-left transition hover:border-[var(--accent-amber)]/45"
                  onClick={() => setDetectionRunning((prev) => !prev)}
                >
                  <p className="text-[10px] uppercase tracking-[0.14em] text-[var(--text-dim)]">Action</p>
                  <p className="mt-1 text-sm font-semibold text-white">{detectionRunning ? "Pause detection" : "Resume detection"}</p>
                </button>
                <button
                  type="button"
                  className="w-full rounded-2xl border border-white/12 bg-black/25 px-3 py-3 text-left transition hover:border-[var(--accent-amber)]/45"
                  onClick={() => {
                    void clearHistory();
                  }}
                >
                  <p className="text-[10px] uppercase tracking-[0.14em] text-[var(--text-dim)]">Action</p>
                  <p className="mt-1 text-sm font-semibold text-white">Clear combo and timeline history</p>
                </button>
                {displayError ? (
                  <div className="rounded-2xl border border-[var(--accent-amber)]/45 bg-[var(--accent-amber)]/10 px-3 py-3">
                    <p className="text-[10px] uppercase tracking-[0.14em] text-amber-100/75">Error state</p>
                    <p className="mt-1 text-xs text-amber-100">{displayError}</p>
                  </div>
                ) : null}
              </div>
            ) : null}
          </div>
        </section>
      </div>

      <button
        type="button"
        className="timeline-fab fixed bottom-6 right-6 z-50 rounded-full border border-white/20 bg-black/50 p-3 text-white/80 backdrop-blur-md transition hover:border-[var(--accent-amber)]/55 hover:text-white"
        onClick={() => setTimelineOpen(true)}
        title="Open session timeline"
      >
        <Zap size={16} />
      </button>

      {timelineOpen ? (
        <div className="timeline-drawer-wrap fixed inset-0 z-[60] bg-black/45 backdrop-blur-sm" onClick={() => setTimelineOpen(false)}>
          <aside
            className="timeline-drawer absolute inset-y-0 right-0 w-[min(390px,100vw)] border-l border-white/12 bg-[rgba(8,14,24,0.95)] p-4"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="flex items-center justify-between gap-3">
              <div>
                <h2 className="hud-title text-xs uppercase tracking-[0.18em] text-white/85">Session Timeline</h2>
                <p className="mt-1 text-[11px] text-[var(--text-dim)]">Recent detections and combo events.</p>
              </div>
              <button
                type="button"
                className="rounded-full border border-white/15 bg-black/25 p-1.5 text-white/75 transition hover:bg-white/10"
                onClick={() => setTimelineOpen(false)}
                title="Close timeline"
              >
                <X size={14} />
              </button>
            </div>

            <div className="signal-strip mt-3 grid grid-cols-2 gap-2">
              <div className="rounded-xl border border-white/10 bg-black/20 px-3 py-2.5">
                <p className="text-[10px] uppercase tracking-[0.14em] text-[var(--text-dim)]">Latency</p>
                <p className="mt-1 text-sm font-semibold text-white">{lastPredictionLatencyMs !== null ? `${lastPredictionLatencyMs}ms` : "--"}</p>
              </div>
              <div className="rounded-xl border border-white/10 bg-black/20 px-3 py-2.5">
                <p className="text-[10px] uppercase tracking-[0.14em] text-[var(--text-dim)]">Signal</p>
                <p className="mt-1 text-sm font-semibold text-white">{signalQuality.label}</p>
              </div>
            </div>

            <div className="mt-3 flex min-h-0 h-[calc(100%-170px)] flex-col">
              <GestureTimeline items={logItems} />
            </div>
          </aside>
        </div>
      ) : null}

      <div className="keyboard-hints pointer-events-none fixed bottom-6 left-6 z-40 rounded-full border border-white/12 bg-black/35 px-3 py-1.5 text-[11px] text-[var(--text-dim)] backdrop-blur-sm">
        Space: start · S: stop · C: clear
      </div>
    </main>
  );
}
