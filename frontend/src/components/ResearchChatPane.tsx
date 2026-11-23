import { useCallback, useEffect, useRef, useState } from 'react';
import { ChatKit, useChatKit } from '@openai/chatkit-react';
import {
  cancelQueuedMessage,
  CHATKIT_API_URL,
  CHATKIT_DOMAIN_KEY,
  getThreadState,
  sendTurnMessage,
} from '../api/threads';
import { CHATKIT_FOODWORKS_THEME } from '../theme/foodworks';
import type { QueuedMessage, TurnMessageMode } from '../types/threads';
import { MidTurnControlBar } from './MidTurnControlBar';

const RESUME_RESEARCH_ACTION_TYPE = 'research.resume';
const FETCH_UPDATES_CACHE_TTL_MS = 5_500;

export interface ResearchChatPaneProps {
  activeThreadId: string | null;
  activeThreadPhase?: string | null;
  isStreamingLocally?: boolean;
  refreshNonce?: number;
  onThreadChange: (threadId: string | null) => void;
  onResponseStart?: () => void;
  onResponseEnd: () => void;
}

const STARTER_PROMPTS = [
  {
    label: 'Compare Postgres vs SQLite for agent state',
    prompt:
      'Compare Postgres and SQLite for durable AI agent state storage, including concurrency, transactions, and operational tradeoffs.',
    icon: 'search' as const,
  },
  {
    label: 'How Temporal workflows improve agent reliability',
    prompt:
      'Explain how Temporal durable execution improves reliability for multi-step deep research agents.',
    icon: 'sparkle' as const,
  },
  {
    label: 'Survey modern headless UI libraries',
    prompt:
      'Research the design tradeoffs of headless UI component primitives for accessible React applications.',
    icon: 'compass' as const,
  },
];

const ACTIVE_PHASES = new Set([
  'queued',
  'planning',
  'running',
  'in_progress',
  'researching',
  'synthesizing',
]);

const CONTROL_BAR_PHASES = new Set([
  ...ACTIVE_PHASES,
  'awaiting_approval',
]);

export function ResearchChatPane({
  activeThreadId,
  activeThreadPhase,
  isStreamingLocally = false,
  refreshNonce = 0,
  onThreadChange,
  onResponseStart,
  onResponseEnd,
}: ResearchChatPaneProps) {
  const [isReady, setIsReady] = useState(false);
  const [loadedThreadId, setLoadedThreadId] = useState<string | null>(null);
  const [isResponding, setIsResponding] = useState(false);
  const [queuedMessages, setQueuedMessages] = useState<QueuedMessage[]>([]);
  const [isSubmittingMidTurn, setIsSubmittingMidTurn] = useState(false);
  const [midTurnError, setMidTurnError] = useState<string | null>(null);

  const isReadyRef = useRef(false);
  const isRespondingRef = useRef(false);
  const activeThreadIdRef = useRef<string | null>(activeThreadId);
  const syncedThreadIdRef = useRef<string | null>(activeThreadId);
  const lastRefreshNonceRef = useRef(refreshNonce);
  const resumedThreadIdRef = useRef<string | null>(null);
  const lastThreadLoadAtRef = useRef<number>(0);

  activeThreadIdRef.current = activeThreadId;

  const { control, setThreadId, fetchUpdates, sendCustomAction } = useChatKit({
    api: {
      url: CHATKIT_API_URL,
      domainKey: CHATKIT_DOMAIN_KEY,
    },
    initialThread: activeThreadId,
    theme: CHATKIT_FOODWORKS_THEME,
    header: {
      enabled: true,
      title: {
        enabled: true,
      },
    },
    history: {
      enabled: false,
    },
    composer: {
      placeholder: 'Ask a deep research question…',
    },
    startScreen: {
      greeting: 'What would you like to research?',
      prompts: STARTER_PROMPTS,
    },
    thread: {
      autoScroll: true,
    },
    onReady: () => {
      isReadyRef.current = true;
      setIsReady(true);
      const desiredThreadId = activeThreadIdRef.current;
      if (syncedThreadIdRef.current !== desiredThreadId) {
        syncedThreadIdRef.current = desiredThreadId;
        setLoadedThreadId(null);
        void setThreadId(desiredThreadId).catch(() => {});
      }
    },
    onThreadChange: ({ threadId }) => {
      syncedThreadIdRef.current = threadId;
      if (threadId === null) {
        setLoadedThreadId(null);
      }
      onThreadChange(threadId);
    },
    onThreadLoadStart: () => {
      setLoadedThreadId(null);
    },
    onThreadLoadEnd: ({ threadId }) => {
      lastThreadLoadAtRef.current = Date.now();
      setLoadedThreadId(threadId);
    },
    onResponseStart: () => {
      isRespondingRef.current = true;
      setIsResponding(true);
      onResponseStart?.();
    },
    onResponseEnd: () => {
      isRespondingRef.current = false;
      setIsResponding(false);
      onResponseEnd();
    },
    onError: ({ error }) => {
      isRespondingRef.current = false;
      setIsResponding(false);
      console.error('[ChatKit error]', error);
    },
  });

  useEffect(() => {
    if (!isReadyRef.current) {
      return;
    }
    if (syncedThreadIdRef.current === activeThreadId) {
      return;
    }
    syncedThreadIdRef.current = activeThreadId;
    setLoadedThreadId(null);
    resumedThreadIdRef.current = null;
    setQueuedMessages([]);
    setMidTurnError(null);
    void setThreadId(activeThreadId).catch(() => {});
  }, [activeThreadId, setThreadId]);

  const normalizedPhase = (activeThreadPhase ?? '').toLowerCase();
  const isActivePhase = ACTIVE_PHASES.has(normalizedPhase);
  const showControlBar =
    Boolean(activeThreadId) &&
    (isResponding ||
      isStreamingLocally ||
      CONTROL_BAR_PHASES.has(normalizedPhase) ||
      queuedMessages.length > 0);

  useEffect(() => {
    if (!activeThreadId || !showControlBar) {
      setQueuedMessages([]);
      return;
    }
    let cancelled = false;
    const syncState = async () => {
      try {
        const state = await getThreadState(activeThreadId);
        if (!cancelled) {
          setQueuedMessages(state.queued_messages ?? []);
        }
      } catch {
        // Ignore transient polling failures
      }
    };
    void syncState();
    const timer = window.setInterval(() => {
      void syncState();
    }, 2_000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [activeThreadId, showControlBar]);

  useEffect(() => {
    if (!activeThreadId || !isActivePhase) {
      resumedThreadIdRef.current = null;
      return;
    }
    if (!isReady || loadedThreadId !== activeThreadId) {
      return;
    }
    if (isResponding || isStreamingLocally) {
      return;
    }
    if (resumedThreadIdRef.current === activeThreadId) {
      return;
    }

    resumedThreadIdRef.current = activeThreadId;
    void sendCustomAction({
      type: RESUME_RESEARCH_ACTION_TYPE,
      payload: { thread_id: activeThreadId },
    }).catch((err) => {
      console.warn('[ChatKit resume action failed]', err);
      if (resumedThreadIdRef.current === activeThreadId) {
        resumedThreadIdRef.current = null;
      }
    });
  }, [
    activeThreadId,
    isActivePhase,
    isReady,
    loadedThreadId,
    isResponding,
    isStreamingLocally,
    sendCustomAction,
  ]);

  useEffect(() => {
    if (refreshNonce === lastRefreshNonceRef.current) {
      return;
    }
    lastRefreshNonceRef.current = refreshNonce;
    const currentId = activeThreadIdRef.current;
    if (!isReadyRef.current || !currentId) {
      return;
    }

    const runRefresh = () => {
      void fetchUpdates().catch(() => {
        void setThreadId(currentId).catch(() => {});
      });
    };

    runRefresh();

    const elapsedSinceLoad = Date.now() - lastThreadLoadAtRef.current;
    if (elapsedSinceLoad < FETCH_UPDATES_CACHE_TTL_MS) {
      const delayMs = FETCH_UPDATES_CACHE_TTL_MS - elapsedSinceLoad;
      const timerId = window.setTimeout(() => {
        if (activeThreadIdRef.current === currentId && !isRespondingRef.current) {
          runRefresh();
        }
      }, delayMs);
      return () => {
        window.clearTimeout(timerId);
      };
    }
  }, [refreshNonce, fetchUpdates, setThreadId]);

  const handleMidTurnSend = useCallback(
    async (text: string, mode: TurnMessageMode) => {
      if (!activeThreadId) return;
      setIsSubmittingMidTurn(true);
      setMidTurnError(null);
      try {
        const res = await sendTurnMessage(activeThreadId, text, mode);
        setQueuedMessages(res.state.queued_messages ?? []);
        if (mode === 'steer' && !isRespondingRef.current && isReadyRef.current) {
          void sendCustomAction({
            type: RESUME_RESEARCH_ACTION_TYPE,
            payload: { thread_id: activeThreadId },
          }).catch(() => {});
        }
      } catch (err) {
        setMidTurnError(
          err instanceof Error ? err.message : 'Failed to send message',
        );
      } finally {
        setIsSubmittingMidTurn(false);
      }
    },
    [activeThreadId, sendCustomAction],
  );

  const handleCancelQueued = useCallback(
    async (messageId: string) => {
      if (!activeThreadId) return;
      setMidTurnError(null);
      try {
        const nextState = await cancelQueuedMessage(activeThreadId, messageId);
        setQueuedMessages(nextState.queued_messages ?? []);
      } catch (err) {
        setMidTurnError(
          err instanceof Error ? err.message : 'Failed to cancel queued message',
        );
      }
    },
    [activeThreadId],
  );

  return (
    <main
      className="chat-pane"
      data-chat-state={activeThreadId ? 'active' : 'empty'}
      aria-label="Research conversation"
    >
      {showControlBar && (
        <MidTurnControlBar
          phaseLabel={normalizedPhase || (isResponding ? 'streaming' : 'running')}
          queuedMessages={queuedMessages}
          isSubmitting={isSubmittingMidTurn}
          error={midTurnError}
          onSend={handleMidTurnSend}
          onCancelQueued={handleCancelQueued}
        />
      )}
      <ChatKit control={control} className="chatkit-surface" />
    </main>
  );
}
