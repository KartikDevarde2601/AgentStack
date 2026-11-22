# AgentStack — Deep Research Agent Demo

A runnable full-stack deep research agent demo combining:

- **Frontend:** Vite + React 19 + TypeScript
  - **Left sidebar:** Thread list built with [`@ui-sdk/core` & `@ui-sdk/react`](https://ui-sdk.dev/) (`Listbox`, `Button`, `Separator`) for keyboard-accessible navigation and selection.
  - **Right pane:** [`@openai/chatkit-react`](https://platform.openai.com/docs/guides/chatkit) composer and streaming conversation view synced to the selected thread.
- **Backend:** FastAPI + [`openai-chatkit`](https://github.com/openai/chatkit-python) (`ChatKitServer`) + OpenAI Python SDK (`AsyncOpenAI`, with a deterministic demo fallback when `OPENAI_API_KEY` is unset).
- **Workflow orchestration:** [Temporal](https://temporal.io/) (`DeepResearchWorkflow` → plan sub-questions → parallel fan-out research activities → synthesize markdown report, with live `@workflow.query` progress streaming).
- **Persistence:** Postgres 16 storing ChatKit threads, messages/items, and Temporal research run records in one shared `PostgresChatKitStore`.

## Architecture

```text
+-------------------------------------------------------------------+
|  Vite + React 19 UI (http://localhost:5173)                       |
|  +---------------------------+  +-------------------------------+ |
|  | Left: ThreadSidebar       |  | Right: ResearchChatPane       | |
|  | @ui-sdk/react Listbox     |  | @openai/chatkit-react         | |
|  | GET/POST/DELETE /api/...  |  | POST /chatkit (SSE)           | |
|  +------------+--------------+  +---------------+---------------+ |
+---------------|---------------------------------|-----------------+
                v                                 v
+-------------------------------------------------------------------+
|  FastAPI Backend (http://localhost:8000)                          |
|  - /api/threads, /api/threads/{id}/runs, /health                  |
|  - /chatkit -> DeepResearchChatKitServer                          |
+----------------------+--------------------------+-----------------+
                       |                          |
                       v                          v
+------------------------------+   +--------------------------------+
|  Postgres 16 (port 5439)     |   |  Temporal Server (7233 / 8233) |
|  - chatkit_threads           |   |  - DeepResearchWorkflow        |
|  - chatkit_items             |   |    1. plan_research_activity   |
|  - chatkit_attachments       |   |    2. gather_finding (parallel)|
|  - research_runs             |   |    3. synthesize_report        |
+------------------------------+   |  - OpenAI SDK (live or demo)   |
                                   +--------------------------------+
```

## Quick Start

### 1. Start Postgres and Temporal

```bash
cp .env.example .env
docker compose up -d
```

- Postgres listens on `localhost:5439` (avoids colliding with a local Postgres on `5432`).
- Temporal gRPC listens on `localhost:7233` and the Temporal Web UI is at `http://localhost:8233`.

### 2. Start the Temporal Worker

```bash
cd backend
uv sync
uv run python -m app.temporal.worker
```

### 3. Start the FastAPI Backend

```bash
cd backend
uv run uvicorn app.main:app --reload --port 8000
```

Set `OPENAI_API_KEY` in `.env` to run live OpenAI SDK calls. Leave it empty to run the deterministic demo researcher.

### 4. Start the Vite React Frontend

```bash
cd frontend
pnpm install
pnpm dev
```

Open `http://localhost:5173`.

## Running Tests

```bash
# Backend (Temporal time-skipping workflow tests + FastAPI/ChatKit tests)
cd backend && uv run pytest -v

# Frontend (TypeScript check, production build, and Vitest UI SDK sidebar tests)
cd frontend && pnpm exec tsc --noEmit && pnpm run build && pnpm test --run
```
