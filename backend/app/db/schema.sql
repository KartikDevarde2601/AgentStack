CREATE TABLE IF NOT EXISTS chatkit_threads (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    payload JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chatkit_threads_created_at
    ON chatkit_threads (created_at DESC, id DESC);

CREATE TABLE IF NOT EXISTS chatkit_items (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL REFERENCES chatkit_threads(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    seq BIGSERIAL,
    payload JSONB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chatkit_items_thread_order
    ON chatkit_items (thread_id, created_at ASC, seq ASC);

CREATE TABLE IF NOT EXISTS chatkit_attachments (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    payload JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS research_runs (
    id TEXT PRIMARY KEY,
    thread_id TEXT NOT NULL REFERENCES chatkit_threads(id) ON DELETE CASCADE,
    workflow_id TEXT NOT NULL UNIQUE,
    query TEXT NOT NULL,
    phase TEXT NOT NULL,
    message TEXT NOT NULL DEFAULT '',
    sub_questions JSONB NOT NULL DEFAULT '[]'::jsonb,
    report TEXT,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_research_runs_thread_created
    ON research_runs (thread_id, created_at DESC);
