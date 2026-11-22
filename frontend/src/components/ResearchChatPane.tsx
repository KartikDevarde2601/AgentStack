import { useEffect, useRef } from 'react';
import { ChatKit, useChatKit } from '@openai/chatkit-react';
import { Loader2 } from 'lucide-react';
import { CHATKIT_API_URL, CHATKIT_DOMAIN_KEY } from '../api/threads';

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

export function ResearchChatPane({
  activeThreadId,
  activeThreadPhase,
  isStreamingLocally = false,
  refreshNonce = 0,
  onThreadChange,
  onResponseStart,
  onResponseEnd,
}: ResearchChatPaneProps) {
  const isReadyRef = useRef(false);
  const activeThreadIdRef = useRef<string | null>(activeThreadId);
  const syncedThreadIdRef = useRef<string | null>(activeThreadId);
  const lastRefreshNonceRef = useRef(refreshNonce);

  activeThreadIdRef.current = activeThreadId;

  const { control, setThreadId, fetchUpdates } = useChatKit({
    api: {
      url: CHATKIT_API_URL,
      domainKey: CHATKIT_DOMAIN_KEY,
    },
    initialThread: activeThreadId,
    theme: {
      colorScheme: 'light',
      radius: 'round',
      density: 'normal',
      color: {
        grayscale: {
          hue: 220,
          tint: 2,
        },
        accent: {
          primary: '#0f172a',
          level: 2,
        },
      },
    },
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
      const desiredThreadId = activeThreadIdRef.current;
      if (syncedThreadIdRef.current !== desiredThreadId) {
        syncedThreadIdRef.current = desiredThreadId;
        void setThreadId(desiredThreadId).catch(() => {});
      }
    },
    onThreadChange: ({ threadId }) => {
      syncedThreadIdRef.current = threadId;
      onThreadChange(threadId);
    },
    onResponseStart: () => {
      onResponseStart?.();
    },
    onResponseEnd: () => {
      onResponseEnd();
    },
    onError: ({ error }) => {
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
    void setThreadId(activeThreadId).catch(() => {});
  }, [activeThreadId, setThreadId]);

  useEffect(() => {
    if (refreshNonce === lastRefreshNonceRef.current) {
      return;
    }
    lastRefreshNonceRef.current = refreshNonce;
    const currentId = activeThreadIdRef.current;
    if (!isReadyRef.current || !currentId) {
      return;
    }
    void fetchUpdates().catch(() => {
      void setThreadId(currentId).catch(() => {});
    });
  }, [refreshNonce, fetchUpdates, setThreadId]);

  const normalizedPhase = (activeThreadPhase ?? '').toLowerCase();
  const showBackgroundBanner =
    Boolean(activeThreadId) &&
    !isStreamingLocally &&
    ACTIVE_PHASES.has(normalizedPhase);

  return (
    <main className="chat-pane" aria-label="Research conversation">
      {showBackgroundBanner && (
        <div className="background-run-banner" role="status" aria-live="polite">
          <Loader2 size={14} className="spin" aria-hidden="true" />
          <span>
            Research is running in the background ({normalizedPhase})… Results
            will appear automatically when ready.
          </span>
        </div>
      )}
      <ChatKit control={control} className="chatkit-surface" />
    </main>
  );
}
