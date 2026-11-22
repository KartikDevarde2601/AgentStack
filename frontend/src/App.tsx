import { useCallback, useEffect, useState } from 'react';
import {
  createThread,
  deleteThread,
  getHealth,
  listThreads,
} from './api/threads';
import { ResearchChatPane } from './components/ResearchChatPane';
import { ThreadSidebar } from './components/ThreadSidebar';
import type { HealthStatus, ThreadSummary } from './types/threads';

export function App() {
  const [threads, setThreads] = useState<ThreadSummary[]>([]);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [isLoadingThreads, setIsLoadingThreads] = useState(true);
  const [isCreatingThread, setIsCreatingThread] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refreshThreads = useCallback(async () => {
    setIsLoadingThreads(true);
    try {
      const nextThreads = await listThreads();
      setThreads(nextThreads);
      setError(null);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to load threads';
      setError(message);
    } finally {
      setIsLoadingThreads(false);
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

  const handleSelectThread = useCallback((threadId: string) => {
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
      void refreshThreads();
    },
    [refreshThreads],
  );

  const handleResponseEnd = useCallback(() => {
    void refreshThreads();
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
        onThreadChange={handleChatKitThreadChange}
        onResponseEnd={handleResponseEnd}
      />
    </div>
  );
}

export default App;
