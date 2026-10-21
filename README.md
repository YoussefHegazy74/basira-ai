<div align="center">

# 👁 بصيرة — Basira

### AI-Powered Islamic Content Verification Platform

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18+-blue.svg)](https://reactjs.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Basira** is an AI-powered web platform for detecting and analyzing misleading Islamic content using NLP, RAG, and Cybersecurity techniques.

</div>

---

## 🎯 Overview

Basira helps users:

- ✅ Verify Quranic verses and Hadiths against trusted sources
- ✂️ Detect truncated or out-of-context religious texts
- 📖 Provide correct interpretations from authenticated Islamic references
- 🤖 Answer religious questions via an intelligent RAG-powered chatbot

---

## 🏗️ Project Structure

```
basira-ai/
├── src/
│   ├── data_pipeline/     # Data loading & validation
│   ├── nlp/               # Arabic NLP preprocessing
│   ├── embeddings/        # Text embedding generation
│   ├── vector_db/         # FAISS / Chroma vector store
│   └── rag/               # RAG pipeline & chatbot logic
├── BackEnd/               # Node/Express API (calls the RAG service)
├── FrontEnd/              # React/Vite frontend
├── notebooks/             # Exploration & experimentation
├── docs/                  # Documentation & reports
├── data/                  # Gitignored — stored on Google Drive
├── requirements.txt
└── README.md
```

---

## 🗄️ Knowledge Base

| Source                      | Format | Size    | Status       |
| --------------------------- | ------ | ------- | ------------ |
| Quran (Uthmani)             | JSON   | —       | ✅ Collected |
| Hadith — 9 Books            | JSON   | ~60 MB  | ✅ Collected |
| Tafsir Ibn Kathir (10 vols) | PDF    | ~518 MB | ✅ Collected |

> ⚠️ Raw data is stored on Google Drive (too large for GitHub).
> 📁 [Tafsir Ibn Kathir PDFs — Google Drive](https://drive.google.com/drive/folders/1bfVkbQJrCVrQTZ21M5iQEPWuXrzBna_x?usp=sharing)

---

## ⚙️ Tech Stack

| Layer      | Technology                           |
| ---------- | ------------------------------------ |
| NLP        | CAMeL Tools, NLTK, PyArabic          |
| Embeddings | Sentence Transformers                |
| Vector DB  | FAISS / Chroma                       |
| RAG        | OpenAI models (direct API client)    |
| Backend    | FastAPI                              |
| Frontend   | React                                |
| Security   | Penetration Testing + SOC Monitoring |

---

## 🚀 Pipeline

```
Raw Data (Quran + Hadith + Tafsir)
        ↓
  Data Validation
        ↓
  Text Extraction (PDF → Text)
        ↓
  Cleaning & Normalization (Arabic NLP)
        ↓
  Unified Schema
        ↓
  Chunking
        ↓
  Embeddings
        ↓
  Vector Database
        ↓
  RAG System ✅
```

---

## 🧪 RAG Validation

Deterministic checks (no paid API calls):

```bash
source venv/bin/activate
python src/rag/offline_tests.py
python -m pytest -q tests/test_rag_hardening.py tests/test_live_eval_rubrics.py
(cd BackEnd && npm test)
```

Integration baseline: **100** focused tests, **251** offline checks, and
**4** backend middleware/route tests passed.
Grounding preserves valid `(مصدر N)` citations for traceability and rejects an
inverted claim when a cited source also asserts the bare affirmative for the
same token, because that source is contradictory. Before integration, the
Final 6 baseline was **141** offline checks passed. These checks cover
retrieval gates, grounding, citation validation, refusal behavior, and
evaluation-rubric integrity. A separate live evaluation uses
`gpt-4o-mini` (observed `gpt-4o-mini-2024-07-18`) through the OpenAI API; it
requires `OPENAI_API_KEY` and writes each run to a new timestamped directory.
The latest stable runs (`live_20261021T183711Z_final5` and
`live_20261021T184253Z_final6`) scored 17/17 auto-scored answerable questions,
21/21 task checks, and 9/9 heldout source-citation checks.

---

## 📅 Current Status

- [x] Data collection (Quran, Hadith, Tafsir)
- [x] UI/UX design
- [x] Data pipeline & NLP preprocessing
- [x] Embeddings & Vector DB
- [x] RAG system (offline-validated; live `gpt-4o-mini` evaluation)
- [x] Backend (FastAPI & Node.js)
- [ ] Frontend (React)
- [ ] Cybersecurity layer

---

## 👥 Team

| Role              | Responsibility                               |
| ----------------- | -------------------------------------------- |
| AI Engineer       | NLP · Embeddings · Vector DB · RAG · Chatbot |
| Data Engineer     | Data collection · Pipeline · Schema          |
| Backend Engineer  | FastAPI · APIs & Node.js                     |
| Frontend Engineer | React · UI                                   |
| Security Engineer | Penetration Testing · SOC                    |

---

## 🏫 Faculty of Electronic Engineering — Menoufia University

> Graduation Project — Computer Science Engineering Department
