export interface ThreadSummary {
  id: string;
  title: string | null;
  created_at: string;
  updated_at: string;
  status?: string | null;
  item_count?: number;
  latest_phase: string | null;
}

export interface HealthStatus {
  status: string;
  postgres: boolean;
  temporal: boolean;
  openai_mode: 'live' | 'unconfigured';
}

export interface QueuedMessage {
  message_id: string;
  text: string;
  created_at: string;
  already_persisted?: boolean;
}

export interface ThreadWorkflowState {
  thread_id: string;
  openai_conversation_id: string | null;
  phase: string;
  active_turn_id: string | null;
  message: string;
  queued_messages: QueuedMessage[];
}

export type TurnMessageMode = 'queue' | 'steer';
