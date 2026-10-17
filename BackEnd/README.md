# Basira BackEnd (Node/Express)

Node API that serves the Basira frontend and calls the Python RAG service.

## Requirements

- Node.js >= 18
- A running Python RAG API (see below)

## Install & run

From `BackEnd/`:

```
npm install
npm start
```

`package.json` declares the dependencies but leaves the versions unpinned
(`"*"`): the original versions could not be recovered from this repository.
Pin exact versions and commit a lockfile before any real install/deploy.

## Configuration

`src/server.js` loads the repository-root `.env` file (`../../.env`).
Required keys (names only; never commit real values):

- `DB_URI`, `DB_NAME` — MongoDB connection
- `PORT` — HTTP port
- `JWT_SECRET`, `JWT_EXPIRES_IN` — auth tokens
- `BASE_URL` — public base URL
- `AI_URL` — base URL of the Python RAG API (for example `http://127.0.0.1:8000`)
- `Service`, `USER`, `APP_PASSWORD` — SMTP/mail settings used by `src/utils/Nodemailer.js`

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
`.env` (for example `OPENAI_API_KEY`). Python dependencies are listed in
`src/rag/model_api_requirements.txt` (also unpinned; pin before deploy).
