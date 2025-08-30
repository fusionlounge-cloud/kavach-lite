# Kavach Starter Kit (Streamlit)

This kit gives you a working local app to reproduce the table like your screenshot:
**Employee | SOP Text | Spoken Text | Blacklisted Words | Word Variance | Variance % | Tone**

## What it does (v1)
- Upload **audio** (mp3/wav) **or** a **transcript text file**.
- Choose **on-device transcription** (faster-whisper) **or** skip if you already have a transcript.
- Load your **SOP items** (weighted checklist) and **Blacklist** CSVs (templates included).
- It computes:
  - **Blacklist hits** (exact + fuzzy)
  - **SOP compliance** and **variance %**
  - **Tone** (text sentiment; simple rule-based RUDE/NEUTRAL/POLITE)
- Shows a **summary row** per employee like your sample. You can export a CSV of results.

> v1 keeps tone simple (text sentiment). You can extend to audio prosody later (librosa).

## Quickstart
1) Install Python 3.10+
2) Create a virtual env and install deps:
   ```bash
   python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
   pip install -r requirements.txt
   ```
3) First run only (for sentiment):
   ```bash
   python -c "import nltk; nltk.download('vader_lexicon')"
   ```
4) Start the app:
   ```bash
   streamlit run app.py
   ```
5) In the app:
   - Upload **SOP** and **Blacklist** CSVs (or use the templates in this folder).
   - Upload an **audio** file (mp3/wav) or **transcript.txt**.
   - Enter Employee name/ID.
   - Click **Analyze**.

## Files
- `app.py` — the Streamlit app.
- `sop_template.csv` — example SOP items with weights, counts, and examples.
- `blacklist_template.csv` — example blacklist phrases with categories and severity.
- `sample_transcript.txt` — the exact text used in your screenshot demo.
- `requirements.txt` — Python dependencies.

## Notes
- On-device transcription downloads a small Whisper model on first use (internet required once).
- If you prefer 100% offline, predownload models or run without audio (upload transcripts).
- You can skin the UI later (colors/branding). The logic is separated and easy to extend.
