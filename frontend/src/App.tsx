import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  createThread,
  deleteThread,
  getHealth,
  listThreads,
} from './api/threads';
import { ResearchChatPane } from './components/ResearchChatPane';
import { ThreadSidebar } from './components/ThreadSidebar';
import type { HealthStatus, ThreadSummary } from './types/threads';

const ACTIVE_PHASES = new Set([
  'queued',
  'planning',
  'running',
  'in_progress',
  'researching',
  'synthesizing',
]);

function isThreadActive(thread: ThreadSummary | undefined): boolean {
  if (!thread) return false;
  const phase = (thread.latest_phase || thread.status || '').toLowerCase();
  return ACTIVE_PHASES.has(phase);
}

export function App() {
  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [isLoadingThreads, setIsLoadingThreads] = useState(true);
  const [isCreatingThread, setIsCreatingThread] = useState(false);
  const [isStreamingLocally, setIsStreamingLocally] = useState(false);
  const [chatRefreshNonce, setChatRefreshNonce] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const prevActiveSnapshotRef = useRef<{
    id: string | null;
    phase: string | null;
    itemCount: number | undefined;
  }>({ id: null, phase: null, itemCount: undefined });

  const refreshThreads = useCallback(async (options?: { silent?: boolean }) => {
    const silent = options?.silent ?? false;
    if (!silent) {
      setIsLoadingThreads(true);
    }
    try {
      const nextThreads = await listThreads();
      setThreads(nextThreads);
      setError(null);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to load threads';
      setError(message);
    } finally {
      if (!silent) {
        setIsLoadingThreads(false);
      }
    }
  }, []);

  const refreshHealth = useCallback(async () => {
    try {
      const nextHealth = await getHealth();
      setHealth(nextHealth);
    } catch {
      setHealth(null);
    }
  }, []);

  useEffect(() => {
    void refreshThreads();
    void refreshHealth();

    const intervalId = window.setInterval(() => {
      void refreshHealth();
    }, 15_000);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [refreshHealth, refreshThreads]);

  const hasAnyActiveThread = useMemo(
    () => threads.some((thread) => isThreadActive(thread)),
    [threads],
  );

  useEffect(() => {
    if (!hasAnyActiveThread) {
      return;
    }
    const pollId = window.setInterval(() => {
      void refreshThreads({ silent: true });
    }, 2_000);

    return () => {
      window.clearInterval(pollId);
    };
  }, [hasAnyActiveThread, refreshThreads]);

  const activeThread = useMemo(
    () => threads.find((thread) => thread.id === activeThreadId),
    [threads, activeThreadId],
  );
  const activeThreadPhase = activeThread
    ? (activeThread.latest_phase ?? activeThread.status ?? null)
    : null;
  const activeThreadItemCount = activeThread?.item_count;

  useEffect(() => {
    const prev = prevActiveSnapshotRef.current;
    if (prev.id === activeThreadId && activeThreadId !== null && !isStreamingLocally) {
      const wasActive = prev.phase ? ACTIVE_PHASES.has(prev.phase.toLowerCase()) : false;
      const isNowActive = activeThreadPhase
        ? ACTIVE_PHASES.has(activeThreadPhase.toLowerCase())
        : false;
      const itemsGrew =
        typeof prev.itemCount === 'number' &&
        typeof activeThreadItemCount === 'number' &&
        activeThreadItemCount > prev.itemCount;

      if ((wasActive && !isNowActive) || itemsGrew) {
        setChatRefreshNonce((n) => n + 1);
      }
    }
    prevActiveSnapshotRef.current = {
      id: activeThreadId,
      phase: activeThreadPhase,
      itemCount: activeThreadItemCount,
    };
  }, [activeThreadId, activeThreadPhase, activeThreadItemCount, isStreamingLocally]);

  const handleSelectThread = useCallback((threadId: string) => {
    setIsStreamingLocally(false);
    setActiveThreadId(threadId);
  }, []);

  const handleCreateThread = useCallback(async () => {
    setIsCreatingThread(true);
    try {
      const created = await createThread();
      setThreads((prev) => [
        created,
        ...prev.filter((item) => item.id !== created.id),
      ]);
      setIsStreamingLocally(false);
      setActiveThreadId(created.id);
      setError(null);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to create thread';
      setError(message);
    } finally {
      setIsCreatingThread(false);
    }
  }, []);

  const handleDeleteThread = useCallback(
    async (threadId: string) => {
      try {
        await deleteThread(threadId);
        setThreads((prev) => {
          const remaining = prev.filter((item) => item.id !== threadId);
          setActiveThreadId((current) =>
            current === threadId ? (remaining[0]?.id ?? null) : current,
          );
          return remaining;
        });
        setError(null);
      } catch (err) {
        const message =
          err instanceof Error ? err.message : 'Failed to delete thread';
        setError(message);
      }
    },
    [],
  );

  const handleChatKitThreadChange = useCallback(
    (threadId: string | null) => {
      setActiveThreadId(threadId);
      void refreshThreads({ silent: true });
    },
    [refreshThreads],
  );

  const handleResponseStart = useCallback(() => {
    setIsStreamingLocally(true);
    void refreshThreads({ silent: true });
  }, [refreshThreads]);

  const handleResponseEnd = useCallback(() => {
    setIsStreamingLocally(false);
    void refreshThreads({ silent: true });
    void refreshHealth();
  }, [refreshHealth, refreshThreads]);

  return (
    <div className="app-shell">
      <ThreadSidebar
        threads={threads}
        activeThreadId={activeThreadId}
        health={health}
        isLoading={isLoadingThreads}
        isCreating={isCreatingThread}
        error={error}
        onSelectThread={handleSelectThread}
        onCreateThread={handleCreateThread}
        onDeleteThread={handleDeleteThread}
        onRefresh={() => {
          void refreshThreads();
          void refreshHealth();
        }}
      />

      <ResearchChatPane
        activeThreadId={activeThreadId}
        activeThreadPhase={activeThreadPhase}
        isStreamingLocally={isStreamingLocally}
        refreshNonce={chatRefreshNonce}
        onThreadChange={handleChatKitThreadChange}
        onResponseStart={handleResponseStart}
        onResponseEnd={handleResponseEnd}
      />
    </div>
  );
}

export default App;
