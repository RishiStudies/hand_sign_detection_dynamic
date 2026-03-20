"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { CalibrationSlot } from "../types";

export const calibrationSlotBlueprint: Array<Pick<CalibrationSlot, "label" | "hint">> = [
  { label: "Neutral", hint: "Center your hand" },
  { label: "Open Palm", hint: "Spread fingers fully" },
  { label: "Closed Fist", hint: "Curl fingers inward" },
  { label: "Point", hint: "Index finger forward" },
  { label: "Pinch", hint: "Bring thumb and index together" },
  { label: "Thumbs Up", hint: "Raise thumb clearly" },
];

type UseCalibrationFlowParams = {
  cameraReady: boolean;
  captureCalibrationImage: () => string | null;
  clearHistory: () => Promise<void>;
};

export function useCalibrationFlow({
  cameraReady,
  captureCalibrationImage,
  clearHistory,
}: UseCalibrationFlowParams) {
  const calibrationTimerRef = useRef<number | null>(null);
  const calibrationSessionRef = useRef(0);

  const [calibrationState, setCalibrationState] = useState("Ready");
  const [calibrationStep, setCalibrationStep] = useState<number | null>(null);
  const [calibrationSlots, setCalibrationSlots] = useState<CalibrationSlot[]>(() =>
    calibrationSlotBlueprint.map((slot) => ({ ...slot, image: null })),
  );
  const [showCalibration, setShowCalibration] = useState(false);
  const [calibrationError, setCalibrationError] = useState<string | null>(null);

  const clearCalibrationTimer = useCallback(() => {
    if (calibrationTimerRef.current !== null) {
      window.clearTimeout(calibrationTimerRef.current);
      calibrationTimerRef.current = null;
    }
  }, []);

  const abortCalibration = useCallback(() => {
    clearCalibrationTimer();
    calibrationSessionRef.current += 1;
    setCalibrationStep(null);
    setShowCalibration(false);
  }, [clearCalibrationTimer]);

  const runCalibration = useCallback(async () => {
    setShowCalibration(true);
    clearCalibrationTimer();
    calibrationSessionRef.current += 1;
    const sessionId = calibrationSessionRef.current;

    setCalibrationSlots(calibrationSlotBlueprint.map((slot) => ({ ...slot, image: null })));
    setCalibrationStep(null);

    if (!cameraReady) {
      setCalibrationState("Camera required");
      setCalibrationError("Start camera before calibration.");
      return;
    }

    setCalibrationState("Aligning capture ritual");
    await clearHistory();
    setCalibrationError(null);

    const captureNext = (index: number) => {
      if (calibrationSessionRef.current !== sessionId) {
        return;
      }

      if (index >= calibrationSlotBlueprint.length) {
        setCalibrationStep(null);
        setCalibrationState("Signal profile aligned");
        return;
      }

      setCalibrationStep(index);
      const image = captureCalibrationImage();

      setCalibrationSlots((prev) =>
        prev.map((slot, slotIndex) =>
          slotIndex === index ? { ...slot, image } : slot,
        ),
      );

      calibrationTimerRef.current = window.setTimeout(() => {
        captureNext(index + 1);
      }, 520);
    };

    captureNext(0);
  }, [cameraReady, captureCalibrationImage, clearCalibrationTimer, clearHistory]);

  // Cancel in-progress calibration and clean up timers on unmount
  useEffect(() => {
    return () => {
      clearCalibrationTimer();
      calibrationSessionRef.current += 1;
    };
  }, [clearCalibrationTimer]);

  return {
    calibrationState,
    calibrationStep,
    calibrationSlots,
    showCalibration,
    calibrationError,
    runCalibration,
    abortCalibration,
  };
}
