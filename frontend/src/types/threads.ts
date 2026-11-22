export interface ThreadSummary {
  id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
  status: string;
  latest_phase: string | null;
}

export interface HealthStatus {
  status: string;
  postgres: boolean;
  temporal: boolean;
  openai_mode: 'live' | 'demo';
}
