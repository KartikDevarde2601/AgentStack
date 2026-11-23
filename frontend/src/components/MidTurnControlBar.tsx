import { useState, type FormEvent } from 'react';
import { CornerDownRight, Loader2, X, Zap } from 'lucide-react';
import type { QueuedMessage, TurnMessageMode } from '../types/threads';

export interface MidTurnControlBarProps {
  phaseLabel: string;
  queuedMessages: QueuedMessage[];
  isSubmitting: boolean;
  error: string | null;
  onSend: (text: string, mode: TurnMessageMode) => Promise<void> | void;
  onCancelQueued: (messageId: string) => Promise<void> | void;
}

export function MidTurnControlBar({
  phaseLabel,
  queuedMessages,
  isSubmitting,
  error,
  onSend,
  onCancelQueued,
}: MidTurnControlBarProps) {
  const [draft, setDraft] = useState('');

  const trimmed = draft.trim();

  const submit = async (mode: TurnMessageMode) => {
    if (!trimmed || isSubmitting) {
      return;
    }
    const text = trimmed;
    setDraft('');
    await onSend(text, mode);
  };

  const handleFormSubmit = (event: FormEvent) => {
    event.preventDefault();
    void submit('queue');
  };

  return (
    <section
      className="mid-turn-control-bar"
      role="status"
      aria-live="polite"
      aria-label="Mid-turn controls"
    >
      <div className="mid-turn-status-row">
        <Loader2 size={14} className="spin" aria-hidden="true" />
        <span className="mid-turn-status-text">
          Turn active ({phaseLabel || 'running'}) — queue a follow-up or steer
          immediately
        </span>
      </div>

      {queuedMessages.length > 0 && (
        <div className="mid-turn-chips" aria-label="Queued messages">
          {queuedMessages.map((item) => (
            <span key={item.message_id} className="mid-turn-chip">
              <span className="mid-turn-chip-text" title={item.text}>
                {item.text}
              </span>
              <button
                type="button"
                className="mid-turn-chip-remove"
                aria-label={`Cancel queued message: ${item.text}`}
                onClick={() => void onCancelQueued(item.message_id)}
              >
                <X size={12} aria-hidden="true" />
              </button>
            </span>
          ))}
        </div>
      )}

      <form className="mid-turn-form" onSubmit={handleFormSubmit}>
        <input
          type="text"
          className="mid-turn-input"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          placeholder="Queue next message or steer the current turn…"
          aria-label="Mid-turn message"
          disabled={isSubmitting}
        />
        <button
          type="submit"
          className="mid-turn-btn mid-turn-btn--queue"
          disabled={!trimmed || isSubmitting}
        >
          <CornerDownRight size={13} aria-hidden="true" />
          <span>Queue</span>
        </button>
        <button
          type="button"
          className="mid-turn-btn mid-turn-btn--steer"
          disabled={!trimmed || isSubmitting}
          onClick={() => void submit('steer')}
        >
          <Zap size={13} aria-hidden="true" />
          <span>Steer</span>
        </button>
      </form>

      {error && (
        <div className="mid-turn-error" role="alert">
          {error}
        </div>
      )}
    </section>
  );
}
