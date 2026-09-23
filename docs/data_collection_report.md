# Basira — Data Collection Report

## Datasets

| Dataset | Format | Files | Size | Status |
|---|---|---|---|---|
| Quran Uthmani | JSON | 1 | — | ✅ Downloaded |
| Hadith (9 books) | JSON | 9 | ~60 MB | ✅ Downloaded |
| Tafsir Ibn Kathir | PDF | 10 | ~518 MB | ✅ Downloaded |

## Storage Location
All raw data stored on Google Drive — not committed to GitHub.

```
basira_data/
├── quran/
│   └── quran_uthmani.json
├── hadith/
│   └── the_9_books/
│       ├── bukhari.json
│       ├── muslim.json
│       ├── abudawud.json
│       ├── tirmidhi.json
│       ├── nasai.json
│       ├── ibnmajah.json
│       ├── malik.json
│       ├── ahmed.json
│       └── darimi.json
└── tafsir/
    └── ibn_kathir/
        └── Tafsir_Ibn_Kathir_Vol._[1-10]_text.pdf
```

## Next Steps
- [ ] Validate Quran JSON structure
- [ ] Validate Hadith JSON structure  
- [ ] Extract text from Tafsir PDFs
- [ ] Clean & normalize Arabic text
- [ ] Build unified schema
- [ ] Chunking strategy
