import type {
  HealthStatus,
  ThreadSummary,
  ThreadWorkflowState,
  TurnMessageMode,
} from '../types/threads';

const rawApiBaseUrl =
  import.meta.env.VITE_API_BASE_URL?.trim() || 'http://localhost:8080';

export const API_BASE_URL = rawApiBaseUrl.replace(/\/+$/, '');
export const CHATKIT_API_URL = `${API_BASE_URL}/chatkit`;
export const CHATKIT_DOMAIN_KEY =
  import.meta.env.VITE_CHATKIT_DOMAIN_KEY?.trim() || 'domain_pk_local_demo';

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      Accept: 'application/json',
      ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
      ...init?.headers,
    },
  });

  if (!response.ok) {
    const detail = await response.text().catch(() => '');
    throw new ApiError(
      response.status,
      detail || `Request failed with status ${response.status}`,
    );
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return (await response.json()) as T;
}

export async function listThreads(): Promise<ThreadSummary[]> {
  const data = await requestJson<{ threads: ThreadSummary[] } | ThreadSummary[]>(
    '/api/threads',
  );
  return Array.isArray(data) ? data : data.threads;
}

export async function createThread(title?: string): Promise<ThreadSummary> {
  return requestJson<ThreadSummary>('/api/threads', {
    method: 'POST',
    body: JSON.stringify(title ? { title } : {}),
  });
}

export async function deleteThread(threadId: string): Promise<void> {
  await requestJson<void>(`/api/threads/${encodeURIComponent(threadId)}`, {
    method: 'DELETE',
  });
}

export async function getHealth(): Promise<HealthStatus> {
  return requestJson<HealthStatus>('/health');
}

export async function getThreadState(
  threadId: string,
): Promise<ThreadWorkflowState> {
  return requestJson<ThreadWorkflowState>(
    `/api/threads/${encodeURIComponent(threadId)}/state`,
  );
}

export async function sendTurnMessage(
  threadId: string,
  text: string,
  mode: TurnMessageMode,
): Promise<{
  message_id: string;
  mode: TurnMessageMode;
  state: ThreadWorkflowState;
}> {
  return requestJson(`/api/threads/${encodeURIComponent(threadId)}/messages`, {
    method: 'POST',
    body: JSON.stringify({ text, mode }),
  });
}

export async function cancelQueuedMessage(
  threadId: string,
  messageId: string,
): Promise<ThreadWorkflowState> {
  return requestJson<ThreadWorkflowState>(
    `/api/threads/${encodeURIComponent(threadId)}/queue/${encodeURIComponent(messageId)}`,
    { method: 'DELETE' },
  );
}
