import { useEffect, useRef } from 'react';
import { ChatKit, useChatKit } from '@openai/chatkit-react';
import { CHATKIT_API_URL, CHATKIT_DOMAIN_KEY } from '../api/threads';

export interface ResearchChatPaneProps {
  activeThreadId: string | null;
  onThreadChange: (threadId: string | null) => void;
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

export function ResearchChatPane({
  activeThreadId,
  onThreadChange,
  onResponseEnd,
}: ResearchChatPaneProps) {
  const isReadyRef = useRef(false);
  const syncedThreadIdRef = useRef<string | null>(activeThreadId);

  const { control, setThreadId } = useChatKit({
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
      if (syncedThreadIdRef.current !== activeThreadId) {
        syncedThreadIdRef.current = activeThreadId;
        void setThreadId(activeThreadId).catch(() => {});
      }
    },
    onThreadChange: ({ threadId }) => {
      syncedThreadIdRef.current = threadId;
      onThreadChange(threadId);
    },
    onResponseEnd: () => {
      onResponseEnd();
    },
    onError: ({ error }) => {
      console.error('[ChatKit error]', error);
    },
  });

  useEffect(() => {
    if (syncedThreadIdRef.current === activeThreadId) {
      return;
    }
    syncedThreadIdRef.current = activeThreadId;
    if (!isReadyRef.current) {
      return;
    }
    void setThreadId(activeThreadId).catch(() => {});
  }, [activeThreadId, setThreadId]);

  return (
    <main className="chat-pane" aria-label="Research conversation">
      <ChatKit control={control} className="chatkit-surface" />
    </main>
  );
}
