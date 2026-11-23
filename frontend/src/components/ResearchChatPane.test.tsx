import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import * as threadsApi from '../api/threads';
import { ResearchChatPane } from './ResearchChatPane';

type CapturedChatKitOptions = {
  initialThread?: string | null;
  onReady?: () => void;
  onThreadChange?: (event: { threadId: string | null }) => void;
  onThreadLoadStart?: (event: { threadId: string }) => void;
  onThreadLoadEnd?: (event: { threadId: string }) => void;
  onResponseStart?: () => void;
  onResponseEnd?: () => void;
  onError?: (event: { error: Error }) => void;
};

let latestOptions: CapturedChatKitOptions = {};
const setThreadIdMock = vi.fn().mockResolvedValue(undefined);
const fetchUpdatesMock = vi.fn().mockResolvedValue(undefined);
const sendCustomActionMock = vi.fn().mockResolvedValue(undefined);

vi.mock('@openai/chatkit-react', () => ({
  ChatKit: ({ className }: { className?: string }) => (
    <div data-testid="mock-chatkit" className={className} />
  ),
  useChatKit: (options: CapturedChatKitOptions) => {
    latestOptions = options;
    return {
      control: {},
      setThreadId: setThreadIdMock,
      fetchUpdates: fetchUpdatesMock,
      sendCustomAction: sendCustomActionMock,
    };
  },
}));

describe('ResearchChatPane', () => {
  beforeEach(() => {
    latestOptions = {};
    vi.restoreAllMocks();
    setThreadIdMock.mockReset().mockResolvedValue(undefined);
    fetchUpdatesMock.mockReset().mockResolvedValue(undefined);
    sendCustomActionMock.mockReset().mockResolvedValue(undefined);
    vi.spyOn(threadsApi, 'getThreadState').mockResolvedValue({
      thread_id: 'thr_active',
      openai_conversation_id: 'conv_1',
      phase: 'researching',
      active_turn_id: 'turn_1',
      message: 'Researching...',
      queued_messages: [],
    });
  });

  it('calls sendCustomAction(research.resume) only after onReady and onThreadLoadEnd for an active thread', () => {
    render(
      <ResearchChatPane
        activeThreadId="thr_active"
        activeThreadPhase="researching"
        isStreamingLocally={false}
        onThreadChange={vi.fn()}
        onResponseStart={vi.fn()}
        onResponseEnd={vi.fn()}
      />,
    );

    expect(sendCustomActionMock).not.toHaveBeenCalled();

    act(() => {
      latestOptions.onReady?.();
    });
    expect(sendCustomActionMock).not.toHaveBeenCalled();

    act(() => {
      latestOptions.onThreadLoadStart?.({ threadId: 'thr_active' });
      latestOptions.onThreadLoadEnd?.({ threadId: 'thr_active' });
    });

    expect(sendCustomActionMock).toHaveBeenCalledTimes(1);
    expect(sendCustomActionMock).toHaveBeenCalledWith({
      type: 'research.resume',
      payload: { thread_id: 'thr_active' },
    });
  });

  it('does not call sendCustomAction when already streaming locally or when thread is completed', () => {
    const { rerender } = render(
      <ResearchChatPane
        activeThreadId="thr_active"
        activeThreadPhase="planning"
        isStreamingLocally={true}
        onThreadChange={vi.fn()}
        onResponseStart={vi.fn()}
        onResponseEnd={vi.fn()}
      />,
    );

    act(() => {
      latestOptions.onReady?.();
      latestOptions.onThreadLoadEnd?.({ threadId: 'thr_active' });
    });
    expect(sendCustomActionMock).not.toHaveBeenCalled();

    rerender(
      <ResearchChatPane
        activeThreadId="thr_done"
        activeThreadPhase="completed"
        isStreamingLocally={false}
        onThreadChange={vi.fn()}
        onResponseStart={vi.fn()}
        onResponseEnd={vi.fn()}
      />,
    );

    act(() => {
      latestOptions.onThreadLoadEnd?.({ threadId: 'thr_done' });
    });
    expect(sendCustomActionMock).not.toHaveBeenCalled();
  });

  it('shows MidTurnControlBar during active turns, queues messages as removable chips, and supports steering', async () => {
    const sendSpy = vi
      .spyOn(threadsApi, 'sendTurnMessage')
      .mockResolvedValueOnce({
        message_id: 'msg_q1',
        mode: 'queue',
        state: {
          thread_id: 'thr_active',
          openai_conversation_id: 'conv_1',
          phase: 'researching',
          active_turn_id: 'turn_1',
          message: 'Researching...',
          queued_messages: [
            {
              message_id: 'msg_q1',
              text: 'Also check cost',
              created_at: new Date().toISOString(),
            },
          ],
        },
      })
      .mockResolvedValueOnce({
        message_id: 'msg_s1',
        mode: 'steer',
        state: {
          thread_id: 'thr_active',
          openai_conversation_id: 'conv_1',
          phase: 'planning',
          active_turn_id: 'turn_2',
          message: 'Planning...',
          queued_messages: [],
        },
      });
    const cancelSpy = vi
      .spyOn(threadsApi, 'cancelQueuedMessage')
      .mockResolvedValue({
        thread_id: 'thr_active',
        openai_conversation_id: 'conv_1',
        phase: 'researching',
        active_turn_id: 'turn_1',
        message: 'Researching...',
        queued_messages: [],
      });

    render(
      <ResearchChatPane
        activeThreadId="thr_active"
        activeThreadPhase="researching"
        isStreamingLocally={true}
        onThreadChange={vi.fn()}
        onResponseStart={vi.fn()}
        onResponseEnd={vi.fn()}
      />,
    );

    const input = screen.getByLabelText('Mid-turn message');
    fireEvent.change(input, { target: { value: 'Also check cost' } });
    fireEvent.click(screen.getByRole('button', { name: 'Queue' }));

    await waitFor(() => {
      expect(sendSpy).toHaveBeenCalledWith(
        'thr_active',
        'Also check cost',
        'queue',
      );
      expect(screen.getByText('Also check cost')).toBeInTheDocument();
    });

    fireEvent.click(
      screen.getByRole('button', {
        name: 'Cancel queued message: Also check cost',
      }),
    );
    await waitFor(() => {
      expect(cancelSpy).toHaveBeenCalledWith('thr_active', 'msg_q1');
      expect(screen.queryByText('Also check cost')).not.toBeInTheDocument();
    });

    fireEvent.change(input, { target: { value: 'Pivot now' } });
    fireEvent.click(screen.getByRole('button', { name: 'Steer' }));

    await waitFor(() => {
      expect(sendSpy).toHaveBeenCalledWith('thr_active', 'Pivot now', 'steer');
    });
  });

  it('re-attaches via sendCustomAction when switching away and back to an active thread', () => {
    const { rerender } = render(
      <ResearchChatPane
        activeThreadId="thr_1"
        activeThreadPhase="researching"
        isStreamingLocally={false}
        onThreadChange={vi.fn()}
        onResponseStart={vi.fn()}
        onResponseEnd={vi.fn()}
      />,
    );

    act(() => {
      latestOptions.onReady?.();
      latestOptions.onThreadLoadEnd?.({ threadId: 'thr_1' });
    });
    expect(sendCustomActionMock).toHaveBeenCalledTimes(1);

    rerender(
      <ResearchChatPane
        activeThreadId="thr_2"
        activeThreadPhase="completed"
        isStreamingLocally={false}
        onThreadChange={vi.fn()}
        onResponseStart={vi.fn()}
        onResponseEnd={vi.fn()}
      />,
    );
    act(() => {
      latestOptions.onThreadLoadEnd?.({ threadId: 'thr_2' });
    });

    rerender(
      <ResearchChatPane
        activeThreadId="thr_1"
        activeThreadPhase="researching"
        isStreamingLocally={false}
        onThreadChange={vi.fn()}
        onResponseStart={vi.fn()}
        onResponseEnd={vi.fn()}
      />,
    );
    act(() => {
      latestOptions.onThreadLoadEnd?.({ threadId: 'thr_1' });
    });

    expect(sendCustomActionMock).toHaveBeenCalledTimes(2);
  });
});
