import { act, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { ThreadSidebar } from './ThreadSidebar';
import type { HealthStatus, ThreadSummary } from '../types/threads';

const SAMPLE_THREADS: ThreadSummary[] = [
  {
    id: 'thr_1',
    title: 'Postgres vs SQLite for agent state',
    created_at: '2026-11-22T01:00:00Z',
    updated_at: '2026-11-22T01:15:00Z',
    status: 'completed',
    latest_phase: 'synthesizing',
  },
  {
    id: 'thr_2',
    title: 'Temporal durable execution patterns',
    created_at: '2026-11-22T01:10:00Z',
    updated_at: '2026-11-22T01:20:00Z',
    status: 'running',
    latest_phase: 'searching',
  },
];

const SAMPLE_HEALTH: HealthStatus = {
  status: 'ok',
  postgres: true,
  temporal: true,
  openai_mode: 'demo',
};

describe('ThreadSidebar', () => {
  it('renders threads, status badges, and backend health indicators', () => {
    render(
      <ThreadSidebar
        threads={SAMPLE_THREADS}
        activeThreadId="thr_1"
        health={SAMPLE_HEALTH}
        isLoading={false}
        isCreating={false}
        error={null}
        onSelectThread={vi.fn()}
        onCreateThread={vi.fn()}
        onDeleteThread={vi.fn()}
        onRefresh={vi.fn()}
      />,
    );

    const listbox = screen.getByRole('listbox', { name: 'Threads' });
    expect(listbox).toBeInTheDocument();

    const options = screen.getAllByRole('option');
    expect(options).toHaveLength(2);
    expect(options[0]).toHaveAttribute('aria-selected', 'true');
    expect(options[1]).toHaveAttribute('aria-selected', 'false');

    expect(screen.getByText('Postgres vs SQLite for agent state')).toBeInTheDocument();
    expect(screen.getByText('Temporal durable execution patterns')).toBeInTheDocument();
    expect(screen.getByText('synthesizing')).toBeInTheDocument();
    expect(screen.getByText('searching')).toBeInTheDocument();

    expect(screen.getByText('Postgres')).toBeInTheDocument();
    expect(screen.getByText('Temporal')).toBeInTheDocument();
    expect(screen.getByText('demo')).toBeInTheDocument();
  });

  it('selects a thread on click and via keyboard navigation through @ui-sdk Listbox', async () => {
    const user = userEvent.setup();
    const onSelectThread = vi.fn();

    render(
      <ThreadSidebar
        threads={SAMPLE_THREADS}
        activeThreadId="thr_1"
        health={SAMPLE_HEALTH}
        isLoading={false}
        isCreating={false}
        error={null}
        onSelectThread={onSelectThread}
        onCreateThread={vi.fn()}
        onDeleteThread={vi.fn()}
        onRefresh={vi.fn()}
      />,
    );

    const options = screen.getAllByRole('option');

    await user.click(options[1]);
    expect(onSelectThread).toHaveBeenCalledWith('thr_2');

    onSelectThread.mockClear();
    act(() => {
      options[0].focus();
    });
    await user.keyboard('{ArrowDown}');
    expect(onSelectThread).toHaveBeenCalledWith('thr_2');
  });

  it('invokes create and delete handlers without selecting the deleted thread', async () => {
    const user = userEvent.setup();
    const onSelectThread = vi.fn();
    const onCreateThread = vi.fn();
    const onDeleteThread = vi.fn();

    render(
      <ThreadSidebar
        threads={SAMPLE_THREADS}
        activeThreadId="thr_1"
        health={SAMPLE_HEALTH}
        isLoading={false}
        isCreating={false}
        error={null}
        onSelectThread={onSelectThread}
        onCreateThread={onCreateThread}
        onDeleteThread={onDeleteThread}
        onRefresh={vi.fn()}
      />,
    );

    await user.click(screen.getByRole('button', { name: 'New research' }));
    expect(onCreateThread).toHaveBeenCalledTimes(1);

    await user.click(
      screen.getByRole('button', {
        name: 'Delete Temporal durable execution patterns',
      }),
    );
    expect(onDeleteThread).toHaveBeenCalledWith('thr_2');
    expect(onSelectThread).not.toHaveBeenCalled();
  });

  it('renders threads cleanly when status is omitted by the backend payload', () => {
    const threadsWithoutStatus: ThreadSummary[] = [
      {
        id: 'thr_3',
        title: 'Backend summary without status field',
        created_at: '2026-11-22T01:30:00Z',
        updated_at: '2026-11-22T01:31:00Z',
        item_count: 2,
        latest_phase: 'completed',
      },
    ];

    render(
      <ThreadSidebar
        threads={threadsWithoutStatus}
        activeThreadId="thr_3"
        health={SAMPLE_HEALTH}
        isLoading={false}
        isCreating={false}
        error={null}
        onSelectThread={vi.fn()}
        onCreateThread={vi.fn()}
        onDeleteThread={vi.fn()}
        onRefresh={vi.fn()}
      />,
    );

    expect(
      screen.getByText('Backend summary without status field'),
    ).toBeInTheDocument();
    expect(screen.getByText('completed')).toBeInTheDocument();
  });
});
