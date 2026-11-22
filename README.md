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
|  Vite + React 19 UI (http://localhost:5175)                       |
|  +---------------------------+  +-------------------------------+ |
|  | Left: ThreadSidebar       |  | Right: ResearchChatPane       | |
|  | @ui-sdk/react Listbox     |  | @openai/chatkit-react         | |
|  | GET/POST/DELETE /api/...  |  | POST /chatkit (SSE)           | |
|  +------------+--------------+  +---------------+---------------+ |
+---------------|---------------------------------|-----------------+
                v                                 v
+-------------------------------------------------------------------+
|  FastAPI Backend (http://localhost:8080)                          |
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

### 1. Install workspace dependencies

```bash
pnpm install
pnpm nx run backend:sync
```

### 2. Start Postgres and Temporal

```bash
cp .env.example .env
docker compose up -d
```

- Postgres listens on `localhost:5439` (avoids colliding with a local Postgres on `5432`).
- Temporal gRPC listens on `localhost:7233` and the Temporal Web UI is at `http://localhost:8233`.

### 3. Start the Temporal Worker

```bash
pnpm dev:worker
```

### 4. Start the FastAPI Backend

```bash
pnpm dev:backend
```

Set `OPENAI_API_KEY` in `.env` to run live OpenAI SDK calls. Leave it empty to run the deterministic demo researcher.

### 5. Start the Vite React Frontend

```bash
pnpm dev:frontend
```

Open `http://localhost:5175`.

## Running Checks with Nx

```bash
# Run all tests (backend pytest + frontend Vitest)
pnpm test

# Run typechecking and production builds across all projects
pnpm typecheck
pnpm build

# Target a single project
pnpm nx test backend
pnpm nx test frontend
pnpm nx build frontend

# Inspect the workspace project graph
pnpm graph
```
