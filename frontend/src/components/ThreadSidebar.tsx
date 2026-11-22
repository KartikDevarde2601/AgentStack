import { useMemo } from 'react';
import {
  createListboxController,
  Listbox,
  ListboxItem,
  ListboxItemIndicator,
  type ListboxItemState,
} from '@ui-sdk/react/listbox';
import { Button } from '@ui-sdk/react/button';
import { Separator } from '@ui-sdk/react/separator';
import {
  Activity,
  Check,
  Database,
  MessageSquare,
  Plus,
  RefreshCw,
  Sparkles,
  Trash2,
} from 'lucide-react';
import type { HealthStatus, ThreadSummary } from '../types/threads';

export interface ThreadSidebarProps {
  threads: ThreadSummary[];
  activeThreadId: string | null;
  health: HealthStatus | null;
  isLoading: boolean;
  isCreating: boolean;
  error: string | null;
  onSelectThread: (threadId: string) => void;
  onCreateThread: () => void | Promise<void>;
  onDeleteThread: (threadId: string) => void | Promise<void>;
  onRefresh: () => void | Promise<void>;
}

function formatRelativeTimestamp(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return '';
  }

  const diffMs = Date.now() - date.getTime();
  const diffMin = Math.floor(diffMs / 60_000);
  if (diffMin < 1) return 'Just now';
  if (diffMin < 60) return `${diffMin}m ago`;

  const diffHours = Math.floor(diffMin / 60);
  if (diffHours < 24) return `${diffHours}h ago`;

  return date.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
  });
}

function getStatusVariant(status: string): 'active' | 'done' | 'error' | 'idle' {
  const normalized = status.toLowerCase();
  if (
    normalized === 'running' ||
    normalized === 'in_progress' ||
    normalized === 'researching'
  ) {
    return 'active';
  }
  if (normalized === 'completed' || normalized === 'done') {
    return 'done';
  }
  if (normalized === 'failed' || normalized === 'error') {
    return 'error';
  }
  return 'idle';
}

function getItemClassName(state: ListboxItemState): string {
  const classes = ['thread-item'];
  if (state.selected) classes.push('thread-item--selected');
  if (state.highlighted) classes.push('thread-item--highlighted');
  return classes.join(' ');
}

export function ThreadSidebar({
  threads,
  activeThreadId,
  health,
  isLoading,
  isCreating,
  error,
  onSelectThread,
  onCreateThread,
  onDeleteThread,
  onRefresh,
}: ThreadSidebarProps) {
  const listboxController = useMemo(() => createListboxController<string>(), []);

  return (
    <aside className="sidebar" aria-label="Research threads">
      <div className="sidebar-header">
        <div className="brand-row">
          <div className="brand-mark" aria-hidden="true">
            <Sparkles size={16} />
          </div>
          <div className="brand-copy">
            <h1 className="brand-title">AgentStack</h1>
            <p className="brand-subtitle">Deep Research Demo</p>
          </div>
          <Button
            type="button"
            className="icon-button"
            aria-label="Refresh threads"
            title="Refresh threads"
            onClick={() => onRefresh()}
          >
            <RefreshCw size={14} className={isLoading ? 'spin' : undefined} />
          </Button>
        </div>

        <Button
          type="button"
          className="new-thread-button"
          disabled={isCreating}
          onClick={() => onCreateThread()}
        >
          <Plus size={16} aria-hidden="true" />
          <span>{isCreating ? 'Creating…' : 'New research'}</span>
        </Button>
      </div>

      <Separator decorative className="sidebar-separator" />

      <div className="sidebar-body">
        <div className="section-label-row">
          <span className="section-label" id="research-threads-label">
            Threads
          </span>
          <span className="section-count">{threads.length}</span>
        </div>

        {error && (
          <div className="sidebar-error" role="alert">
            {error}
          </div>
        )}

        {threads.length === 0 ? (
          <div className="empty-threads">
            <MessageSquare size={18} className="empty-threads-icon" aria-hidden="true" />
            <p className="empty-threads-title">
              {isLoading ? 'Loading threads…' : 'No research threads yet'}
            </p>
            <p className="empty-threads-hint">
              Start a new research thread or send a prompt on the right.
            </p>
          </div>
        ) : (
          <Listbox<string>
            controller={listboxController}
            value={activeThreadId}
            onValueChange={(nextThreadId) => {
              if (nextThreadId) {
                onSelectThread(nextThreadId);
              }
            }}
            activation="automatic"
            orientation="vertical"
            loop
            aria-labelledby="research-threads-label"
            className="thread-listbox"
          >
            {threads.map((thread) => {
              const title = thread.title?.trim() || 'Untitled research';
              const statusVariant = getStatusVariant(thread.status);
              const relativeTime = formatRelativeTimestamp(
                thread.updated_at || thread.created_at,
              );

              return (
                <ListboxItem
                  key={thread.id}
                  value={thread.id}
                  label={title}
                  className={getItemClassName}
                >
                  <div className="thread-item-main">
                    <div className="thread-item-title-row">
                      <ListboxItemIndicator className="thread-item-indicator">
                        <Check size={12} aria-hidden="true" />
                      </ListboxItemIndicator>
                      <span className="thread-item-title" title={title}>
                        {title}
                      </span>
                    </div>

                    <div className="thread-item-meta">
                      <span
                        className={`status-badge status-badge--${statusVariant}`}
                      >
                        {thread.latest_phase || thread.status}
                      </span>
                      {relativeTime && (
                        <span className="thread-item-time">{relativeTime}</span>
                      )}
                    </div>
                  </div>

                  <button
                    type="button"
                    className="delete-thread-button"
                    aria-label={`Delete ${title}`}
                    title={`Delete ${title}`}
                    onPointerDown={(event) => {
                      event.stopPropagation();
                    }}
                    onClick={(event) => {
                      event.stopPropagation();
                      void onDeleteThread(thread.id);
                    }}
                  >
                    <Trash2 size={14} aria-hidden="true" />
                  </button>
                </ListboxItem>
              );
            })}
          </Listbox>
        )}
      </div>

      <Separator decorative className="sidebar-separator" />

      <div className="sidebar-footer" aria-label="Backend status">
        <div className="health-pill-row">
          <span
            className={`health-pill ${
              health?.postgres ? 'health-pill--ok' : 'health-pill--warn'
            }`}
          >
            <Database size={12} aria-hidden="true" />
            <span>Postgres</span>
          </span>

          <span
            className={`health-pill ${
              health?.temporal ? 'health-pill--ok' : 'health-pill--warn'
            }`}
          >
            <Activity size={12} aria-hidden="true" />
            <span>Temporal</span>
          </span>

          <span
            className={`health-pill ${
              health?.openai_mode === 'live'
                ? 'health-pill--live'
                : 'health-pill--demo'
            }`}
          >
            <Sparkles size={12} aria-hidden="true" />
            <span>{health ? health.openai_mode : 'offline'}</span>
          </span>
        </div>
      </div>
    </aside>
  );
}
