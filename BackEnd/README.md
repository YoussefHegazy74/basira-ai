# Basira BackEnd (Node/Express)

Node API that serves the Basira frontend and calls the Python RAG service.

## Requirements

- Node.js >= 18
- A running Python RAG API (see below)

## Install & run

From `BackEnd/`:

```
npm ci
npm start
```

Dependencies use caret ranges and a committed `package-lock.json`; prefer
`npm ci` so installs follow the lockfile exactly.

Run the backend middleware/route tests with:

```
npm test
```

## Configuration

`src/server.js` loads the repository-root `.env` file (`../../.env`).
Start from the placeholder-only example:

```
cp .env.example .env
```

Required keys (names only; never commit real values):

- `DB_URI`, `DB_NAME` — MongoDB connection
- `PORT` — HTTP port (defaults to `3000`)
- `JWT_SECRET`, `JWT_EXPIRES_IN` — auth tokens
- `FRONTEND_URL` — frontend origin used in password-reset links
- `AI_URL` — base URL of the Python RAG API (for example `http://127.0.0.1:8000`)
- `SMTP_SERVICE`, `SMTP_USER`, `SMTP_PASSWORD` — SMTP/mail settings used by
  `src/utils/Nodemailer.js`. The legacy names `Service`, `USER`, and
  `APP_PASSWORD` are still accepted as fallbacks.

The frontend reads `VITE_API_URL` from `FrontEnd/.env` (default
`http://localhost:3000`); see `FrontEnd/.env.example`. Open the frontend at
`http://localhost:5173` rather than `http://127.0.0.1:5173` so the
`SameSite=Lax` guest and auth cookies are sent to `http://localhost:3000`.

## Python RAG API (`src/rag/model_api.py`)

The Node backend expects the Python service to expose:

- `POST /api/chat` — body `{query, chat_history?}`, returns `{answer, sources, query}`
- `POST /api/tafsir` — body `{ayah}`
- `POST /api/verify` — body `{text, explanation?}`

Start it from the repository root:

```
python src/rag/model_api.py
```

It listens on `127.0.0.1:8000` and loads `data/processed/faiss.index` and
`data/processed/chunks.pkl` (must exist). It reads the repository-root
`.env` (for example `OPENAI_API_KEY`). Install the pinned dependencies from
the repository root:

```
pip install -r src/rag/model_api_requirements.lock
```

`src/rag/model_api_requirements.lock` is the canonical, complete dependency
file (generated with Python 3.14); the unpinned
`src/rag/model_api_requirements.txt` is a partial overview and is not
sufficient on its own.
