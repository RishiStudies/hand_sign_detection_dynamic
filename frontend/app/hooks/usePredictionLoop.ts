"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { GestureLogItem, PredictionResponse } from "../types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
if (!API_BASE_URL) {
  throw new Error("NEXT_PUBLIC_API_BASE_URL is required for frontend runtime.");
}

const PREDICTION_INTERVAL_MS = 340;
const PREDICTION_JITTER_MS = 120;
const REQUEST_TIMEOUT_MS = 12000;
const MAX_BACKOFF_MULTIPLIER = 8;

function getClockLabel(): string {
  const now = new Date();
  return now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function clampPercent(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.min(100, Math.round(value * 100)));
}

type UsePredictionLoopParams = {
  cameraReady: boolean;
  detectionRunning: boolean;
  captureFrameBlob: () => Promise<Blob | null>;
};

export function usePredictionLoop({
  cameraReady,
  detectionRunning,
  captureFrameBlob,
}: UsePredictionLoopParams) {
  const predictionTimerRef = useRef<number | null>(null);
  const isRequestInFlightRef = useRef(false);
  const consecutivePredictionFailuresRef = useRef(0);
  const sessionIdRef = useRef(
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `session-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`,
  );

  const [predictionLabel, setPredictionLabel] = useState("Awaiting Input");
  const [confidence, setConfidence] = useState(0);
  const [comboLabel, setComboLabel] = useState("None");
  const [logItems, setLogItems] = useState<GestureLogItem[]>([]);
  const [lastPredictionLatencyMs, setLastPredictionLatencyMs] = useState<number | null>(null);
  const [lastPredictionMode, setLastPredictionMode] = useState("RF");
  const [featureSchemaVersion, setFeatureSchemaVersion] = useState<string | null>(null);
  const [pingMs, setPingMs] = useState<number | null>(null);
  const [pingOnline, setPingOnline] = useState(true);
  const [predictionError, setPredictionError] = useState<string | null>(null);

  const clearPredictionTimer = useCallback(() => {
    if (predictionTimerRef.current !== null) {
      window.clearTimeout(predictionTimerRef.current);
      predictionTimerRef.current = null;
    }
  }, []);

  const fetchWithTimeout = useCallback(
    async (input: RequestInfo | URL, init: RequestInit = {}, timeoutMs = REQUEST_TIMEOUT_MS) => {
      const controller = new AbortController();
      const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
      try {
        const headers = new Headers(init.headers);
        headers.set("X-Session-Id", sessionIdRef.current);
        return await fetch(input, {
          ...init,
          headers,
          signal: controller.signal,
        });
      } finally {
        window.clearTimeout(timeoutId);
      }
    },
    [],
  );

  const clearHistory = useCallback(async () => {
    setLogItems([]);
    setComboLabel("None");
    try {
      await fetchWithTimeout(`${API_BASE_URL}/clear_combos`, { method: "POST" });
      setPredictionError(null);
    } catch {
      setPredictionError("Could not clear backend combo history.");
    }
  }, [fetchWithTimeout]);

  const runPrediction = useCallback(async () => {
    if (!detectionRunning || !cameraReady || isRequestInFlightRef.current) {
      return;
    }

    isRequestInFlightRef.current = true;
    const startedAt = performance.now();

    try {
      const blob = await captureFrameBlob();
      if (!blob) {
        return;
      }

      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");

      const response = await fetchWithTimeout(`${API_BASE_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Prediction request failed with status ${response.status}`);
      }

      const payload = (await response.json()) as PredictionResponse;
      const nextLabel = payload.label ?? "Unknown";
      const nextConfidence = Number.isFinite(payload.prob) ? payload.prob : 0;
      const nextMode = String(payload.backend_mode ?? "random_forest")
        .replaceAll("_", " ")
        .toUpperCase();
      const nextCombo = payload.combo?.combo ?? null;

      consecutivePredictionFailuresRef.current = 0;
      setPredictionLabel(nextLabel);
      setConfidence(nextConfidence);
      setComboLabel(nextCombo ?? "None");
      setPredictionError(null);
      setLastPredictionLatencyMs(Math.max(1, Math.round(performance.now() - startedAt)));
      setLastPredictionMode(nextMode);
      setFeatureSchemaVersion(payload.feature_schema_version ?? null);

      setLogItems((prev) => {
        const nextItem: GestureLogItem = {
          label: nextLabel,
          time: getClockLabel(),
          confidence: clampPercent(nextConfidence),
          mode: nextMode,
          combo: nextCombo,
        };
        return [nextItem, ...prev].slice(0, 10);
      });
    } catch (error) {
      consecutivePredictionFailuresRef.current += 1;
      const message = error instanceof Error ? error.message : "Prediction failed";
      if (message.includes("status 429")) {
        setPredictionError("Backend is throttling requests. The console is slowing the scan rhythm automatically.");
        return;
      }
      if (message.includes("status 503")) {
        setPredictionError("Backend is temporarily busy. Hold your pose while the system re-locks.");
        return;
      }
      setPredictionError(message);
    } finally {
      isRequestInFlightRef.current = false;
    }
  }, [cameraReady, captureFrameBlob, detectionRunning, fetchWithTimeout]);

  // Completion-driven adaptive prediction loop
  useEffect(() => {
    if (!cameraReady || !detectionRunning) {
      clearPredictionTimer();
      return;
    }

    let cancelled = false;

    const loop = async () => {
      await runPrediction();
      if (cancelled || !cameraReady || !detectionRunning) {
        return;
      }

      const jitter = Math.floor(Math.random() * PREDICTION_JITTER_MS);
      const backoffMultiplier = Math.min(
        MAX_BACKOFF_MULTIPLIER,
        Math.max(1, 2 ** consecutivePredictionFailuresRef.current),
      );
      const adaptiveDelay = Math.max(
        PREDICTION_INTERVAL_MS,
        (lastPredictionLatencyMs ?? 0) + 80,
      );

      predictionTimerRef.current = window.setTimeout(() => {
        void loop();
      }, adaptiveDelay * backoffMultiplier + jitter);
    };

    void loop();

    return () => {
      cancelled = true;
      clearPredictionTimer();
    };
  }, [cameraReady, clearPredictionTimer, detectionRunning, lastPredictionLatencyMs, runPrediction]);

  // Backend health ping every 5 s
  useEffect(() => {
    let cancelled = false;

    const updatePing = async () => {
      const startedAt = performance.now();

      try {
        const response = await fetchWithTimeout(`${API_BASE_URL}/health/live`, {
          method: "GET",
          cache: "no-store",
        });

        if (cancelled) {
          return;
        }

        setPingMs(Math.max(1, Math.round(performance.now() - startedAt)));
        setPingOnline(response.ok);
      } catch (error) {
        if (cancelled) {
          return;
        }

        setPingMs(null);
        setPingOnline(false);
        if (error instanceof Error && error.name === "AbortError") {
          setPredictionError("Backend request timed out. Please retry.");
        }
      }
    };

    void updatePing();
    const intervalId = window.setInterval(() => {
      void updatePing();
    }, 5000);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [fetchWithTimeout]);

  return {
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
  };
}
