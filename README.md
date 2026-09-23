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
├── notebooks/             # Exploration & experimentation
├── docs/                  # Documentation & reports
├── data/                  # Gitignored — stored on Google Drive
├── requirements.txt
└── README.md
```

---

## 🗄️ Knowledge Base

| Source | Format | Size | Status |
|---|---|---|---|
| Quran (Uthmani) | JSON | — | ✅ Collected |
| Hadith — 9 Books | JSON | ~60 MB | ✅ Collected |
| Tafsir Ibn Kathir (10 vols) | PDF | ~518 MB | ✅ Collected |

> ⚠️ Raw data is stored on Google Drive (too large for GitHub).

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| NLP | CAMeL Tools, NLTK, PyArabic |
| Embeddings | Sentence Transformers |
| Vector DB | FAISS / Chroma |
| RAG | LangChain + LLM |
| Backend | FastAPI |
| Frontend | React |
| Security | Penetration Testing + SOC Monitoring |

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

## 📅 Current Status

- [x] Data collection (Quran, Hadith, Tafsir)
- [x] UI/UX design
- [ ] Data pipeline & NLP preprocessing
- [ ] Embeddings & Vector DB
- [ ] RAG system
- [ ] Backend (FastAPI)
- [ ] Frontend (React)
- [ ] Cybersecurity layer

---

## 👥 Team

| Role | Responsibility |
|---|---|
| AI Engineer | NLP · Embeddings · Vector DB · RAG · Chatbot |
| Data Engineer | Data collection · Pipeline · Schema |
| Backend Engineer | FastAPI · APIs |
| Frontend Engineer | React · UI |
| Security Engineer | Penetration Testing · SOC |

---

## 🏫 Faculty of Electronic Engineering — Menoufia University
> Graduation Project — Computer & Communications Engineering Department
