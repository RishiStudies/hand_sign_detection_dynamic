export type PredictionResponse = {
  label: string;
  prob: number;
  backend_mode?: string;
  feature_schema_version?: string;
  combo?: {
    combo: string;
  };
};

export type GestureLogItem = {
  label: string;
  time: string;
  confidence: number;
  mode: string;
  combo?: string | null;
};

export type CalibrationSlot = {
  label: string;
  hint: string;
  image: string | null;
};
