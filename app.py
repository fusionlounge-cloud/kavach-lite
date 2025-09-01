# app.py
# Kavach Lite — One-Row SOP Checker (TXT or Audio ➜ Transcribe ➜ Compare ➜ Tone ➜ Blacklist)
# Safe, robust, and simple. Works with minimal CSVs and tolerates missing columns.

import io
import os
import re
import tempfile
import string
from typing import List, Tuple

import pandas as pd
import streamlit as st

# ---------- CONFIG ----------
st.set_page_config(page_title="Kavach Lite — One-Row SOP Checker", layout="wide")


# ========== UTILITIES ==========

_PUNCT_TBL = str.maketrans("", "", string.punctuation)

def normalize_text(s: str) -> str:
    """Lowercase, remove punctuation, collapse spaces."""
    if not isinstance(s, str):
        s = str(s or "")
    return re.sub(r"\s+", " ", s.lower().translate(_PUNCT_TBL)).strip()


def read_txt(file) -> str:
    """Read a .txt file object to string safely (utf-8 fallback)."""
    if file is None:
        return ""
    try:
        return file.read().decode("utf-8", errors="ignore")
    except AttributeError:
        # Already str
        return str(file)
    except Exception:
        # Last resort
        try:
            return file.read().decode(errors="ignore")
        except Exception:
            return ""


def try_import_faster_whisper():
    try:
        from faster_whisper import WhisperModel
        return WhisperModel
    except Exception as e:
        return None


def transcribe_audio(file) -> Tuple[str, str]:
    """
    Transcribe audio with faster-whisper if available.
    Returns (text, message).
    """
    WhisperModel = try_import_faster_whisper()
    if WhisperModel is None:
        return "", "Whisper not installed. Install with: pip install faster-whisper"

    # Write uploaded audio to a temp path
    suffix = os.path.splitext(file.name)[-1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    try:
        # Small/int8 is fast and good enough on CPU
        model = WhisperModel("small", device="cpu", compute_type="int8")
        segments, info = model.transcribe(tmp_path, beam_size=1)
        text = " ".join(seg.text.strip() for seg in segments if seg and seg.text)
        if not text.strip():
            return "", "No speech found."
        return text.strip(), "Audio transcribed successfully with Whisper."
    except FileNotFoundError as e:
        # ffmpeg/ffprobe missing
        return "", f"Transcription failed: {e}. Install ffmpeg: brew install ffmpeg"
    except Exception as e:
        return "", f"Transcription failed: {e}"
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ---------- BLACKLIST MATCHING (phrase + inflection aware) ----------

def detect_blacklist(spoken_text: str, blacklist_terms: List[str]) -> List[str]:
    """
    Returns blacklist terms detected in spoken_text.
    - Multi-word phrases: substring match on normalized text
    - Single-word: word-boundary regex with optional suffixes (er/ed/ing/s)
    - Very short terms (<=3 letters): exact word only
    """
    norm = normalize_text(spoken_text)
    hits = set()

    for raw in blacklist_terms:
        term = normalize_text(str(raw or ""))
        if not term:
            continue

        if " " in term:
            # phrase
            if term in norm:
                hits.add(raw)
            continue

        # single word
        if len(term) > 3:
            pattern = r"\b" + re.escape(term) + r"(er|ers|ed|ing|s)?\b"
        else:
            pattern = r"\b" + re.escape(term) + r"\b"

        if re.search(pattern, norm):
            hits.add(raw)

    return sorted(hits)


# ---------- SOP PROCESSING ----------

def load_sop_csv(upload) -> pd.DataFrame:
    """
    Accepts very flexible SOP CSVs.
    Supported columns (case-insensitive; optional except 'examples'):
      - category
      - required (0/1 or TRUE/FALSE)
      - min_needed (int)
      - weight (float)
      - examples  <-- the example phrases to look for; if missing, try 'example' or 'sop'
    If 'examples' is absent, we create it from any text-like column found; if none, empty.
    """
    if upload is None:
        return pd.DataFrame()

    try:
        df = pd.read_csv(upload)
    except Exception:
        upload.seek(0)
        df = pd.read_csv(io.StringIO(upload.read().decode("utf-8", errors="ignore")))

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Find an examples-like column
    examples_col = None
    for cand in ("examples", "example", "sop", "phrases", "text"):
        if cand in df.columns:
            examples_col = cand
            break

    if examples_col is None:
        # Create an empty examples column to avoid KeyError
        df["examples"] = ""
    else:
        if examples_col != "examples":
            df["examples"] = df[examples_col]
        # Ensure str
        df["examples"] = df["examples"].fillna("").astype(str)

    # Optional fields with safe defaults
    if "category" not in df.columns:
        df["category"] = [f"Step {i+1}" for i in range(len(df))]
    if "required" not in df.columns:
        df["required"] = 1
    if "min_needed" not in df.columns:
        df["min_needed"] = 1
    if "weight" not in df.columns:
        df["weight"] = 1.0

    # Clean types
    df["required"] = df["required"].apply(lambda x: int(str(x).strip().lower() in ("1", "true", "yes")))
    df["min_needed"] = pd.to_numeric(df["min_needed"], errors="coerce").fillna(1).astype(int)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0).astype(float)

    return df[["category", "required", "min_needed", "weight", "examples"]]


def count_phrase_hits(spoken: str, phrase_blob: str) -> int:
    """
    Given spoken text and a 'phrase blob' (comma/pipe/semicolon separated examples),
    count how many of those example phrases occur in the spoken text (normalized, substring).
    """
    norm_spoken = normalize_text(spoken)
    if not phrase_blob:
        return 0

    # Split on common delimiters
    parts = re.split(r"[|;,]\s*|\n+", str(phrase_blob))
    parts = [normalize_text(p) for p in parts if normalize_text(p)]
    if not parts:
        return 0

    hits = 0
    for p in parts:
        if p in norm_spoken:
            hits += 1
    return hits


def compute_compliance(spoken_text: str, sop_df: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
    """
    Compliance score = weighted coverage of SOP rows.
    For each row: if hits >= min_needed, it's 'covered'. Required rows weigh more implicitly
    because missing them means 0 for that row.
    """
    if sop_df.empty:
        return 0.0, sop_df

    rows = []
    total_weight = sop_df["weight"].sum() if sop_df["weight"].sum() > 0 else len(sop_df)
    covered_weight = 0.0

    for _, row in sop_df.iterrows():
        hits = count_phrase_hits(spoken_text, row["examples"])
        covered = int(hits >= max(1, row["min_needed"]))
        weight = float(row["weight"])
        if covered:
            covered_weight += weight
        rows.append({
            "category": row["category"],
            "required": row["required"],
            "min_needed": row["min_needed"],
            "weight": weight,
            "examples": row["examples"],
            "hits": hits,
            "covered": covered
        })

    detail_df = pd.DataFrame(rows)
    # If you want required rows to be mandatory, you can multiply their weight by 2 (optional)
    compliance = 0.0 if total_weight <= 0 else (covered_weight / total_weight) * 100.0
    return round(compliance, 1), detail_df


def word_variance(sop_text: str, spoken_text: str) -> Tuple[int, List[str], List[str]]:
    sop_tokens = set(normalize_text(sop_text).split())
    spoken_tokens = set(normalize_text(spoken_text).split())
    only_in_sop = sorted(list(sop_tokens - spoken_tokens))
    only_in_spoken = sorted(list(spoken_tokens - sop_tokens))
    uv = len(only_in_sop) + len(only_in_spoken)
    return uv, only_in_sop, only_in_spoken


# ---------- TONE (lightweight heuristic) ----------
POS = {"please", "thank", "thanks", "welcome", "happy", "kind", "help", "sorry"}
NEG = {"no", "not", "never", "donot", "dont", "shut", "leave", "complaint", "angry", "hate", "idiot", "stupid"}

def tone_score(spoken: str, blacklist_hits: List[str]) -> Tuple[int, str]:
    """
    Simple tone: + for polite markers, - for negatives & blacklist.
    Bounded 0..100; label bucketed.
    """
    norm = normalize_text(spoken)
    score = 50
    for w in POS:
        if w in norm:
            score += 5
    for w in NEG:
        if w in norm:
            score -= 8
    # Penalize blacklist hits strongly
    score -= 12 * len(blacklist_hits)

    score = max(0, min(100, score))
    if score >= 80:
        label = "POLITE"
    elif score >= 50:
        label = "NEUTRAL"
    else:
        label = "RUDE"
    return score, label


# ========== UI ==========

st.title("Kavach Lite — One-Row SOP Checker")

with st.expander("Quick instructions", expanded=False):
    st.markdown(
        """
**Flow:** Upload **Transcript (.txt)** or **Audio (.wav/.mp3/.m4a)** ➜ Upload **SOP CSV** & **Blacklist CSV** ➜ Click **Rerun** (auto) ➜ See Compliance • Variance • Tone.

**CSV formats:**
- **SOP CSV** (flexible): columns may include  
  `category, required, min_needed, weight, examples`  
  – Only `examples` (phrases) is really needed. Others default safely.
- **Blacklist CSV**: **header exactly `term`**, one term/phrase per row.

**Audio:** requires `ffmpeg` and `faster-whisper`.  
Install: `brew install ffmpeg`  and  `pip install faster-whisper`
        """
    )

# Basic identity (free text for UI)
colA, colB = st.columns(2)
with colA:
    employee_name = st.text_input("Employee name", "John Smith")
with colB:
    employee_id = st.text_input("Employee ID", "EMP001")

st.subheader("Transcript (.txt) **OR** Audio")

col1, col2 = st.columns(2)
with col1:
    txt_file = st.file_uploader("Drag and drop .txt here", type=["txt"], help="Upload transcript as plain text")

with col2:
    audio_file = st.file_uploader("Or upload audio (.wav, .mp3, .m4a)", type=["wav", "mp3", "m4a"])

# SOP + Blacklist uploaders
st.subheader("SOP & Blacklist")
c3, c4 = st.columns(2)
with c3:
    sop_upload = st.file_uploader("Upload SOP CSV", type=["csv"], key="sop_csv")
with c4:
    bl_upload = st.file_uploader("Upload Blacklist CSV", type=["csv"], key="bl_csv")

# ---------- LOAD INPUTS ----------
spoken_text = ""
msg = ""

if txt_file is not None and txt_file.size > 0:
    spoken_text = read_txt(txt_file).strip()
    msg = "Transcript loaded."
elif audio_file is not None and audio_file.size > 0:
    spoken_text, msg = transcribe_audio(audio_file)

if msg:
    st.info(msg)

# SOP
sop_df = load_sop_csv(sop_upload)

# Blacklist
black_terms: List[str] = []
if bl_upload is not None:
    try:
        bl_df = pd.read_csv(bl_upload)
    except Exception:
        bl_upload.seek(0)
        bl_df = pd.read_csv(io.StringIO(bl_upload.read().decode("utf-8", errors="ignore")))
    # require column named 'term'
    bl_df.columns = [c.strip().lower() for c in bl_df.columns]
    if "term" not in bl_df.columns:
        st.warning("Could not read Blacklist CSV: must have column **term**.")
        bl_df = pd.DataFrame({"term": []})
    black_terms = [t for t in bl_df["term"].astype(str).tolist() if str(t).strip()]

# ---------- BUILD SOP TEXT (for variance table only) ----------
# Compact SOP 'examples' into a readable single line for the table
if not sop_df.empty:
    sop_compact = " • ".join(
        [normalize_text(x) for x in sop_df["examples"].fillna("").astype(str).tolist() if normalize_text(x)]
    )
else:
    sop_compact = ""

# ---------- METRICS ----------
compliance = 0.0
variance_pct = 0.0
tone = 50
tone_label = "NEUTRAL"
bl_hits: List[str] = []

if spoken_text.strip():
    # Blacklist
    bl_hits = detect_blacklist(spoken_text, black_terms)
    # Compliance
    compliance, sop_detail = compute_compliance(spoken_text, sop_df) if not sop_df.empty else (0.0, pd.DataFrame())
    # Variance
    uv_count, only_in_sop, only_in_spoken = word_variance(sop_compact, spoken_text)
    # Variance% is a soft indicator; we can normalize by len(sop words)+len(spoken words)
    denom = max(1, len(set(normalize_text(sop_compact).split())) + len(set(normalize_text(spoken_text).split())))
    variance_pct = round((uv_count / denom) * 100.0, 1)
    # Tone
    tone, tone_label = tone_score(spoken_text, bl_hits)

# ---------- KPI STRIP ----------
k1, k2, k3 = st.columns(3)
with k1:
    st.metric("Compliance %", f"{compliance:.1f}%")
with k2:
    st.metric("Variance %", f"{variance_pct:.1f}%")
with k3:
    st.metric("Tone score", f"{tone} ({tone_label})")

# ---------- SUMMARY TABLE ----------
st.subheader("Summary")
summary_df = pd.DataFrame([{
    "EMPLOYEE": f"{employee_name} {employee_id}",
    "SOP TEXT": sop_compact,
    "SPOKEN TEXT": spoken_text if spoken_text else "—"
}])
st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ---------- DETAILS ----------
with st.expander("SOP Details (per row)", expanded=False):
    if spoken_text and not sop_df.empty:
        show = sop_detail[["category", "required", "min_needed", "weight", "examples", "hits", "covered"]].copy()
        show.rename(columns={
            "category": "Category",
            "required": "Required",
            "min_needed": "Min hits",
            "weight": "Weight",
            "examples": "Examples",
            "hits": "Hits",
            "covered": "Covered"
        }, inplace=True)
        st.dataframe(show, use_container_width=True, hide_index=True)
    else:
        st.info("Upload SOP CSV and transcript/audio to see per-row coverage.")

with st.expander("Word Variance Terms", expanded=False):
    if spoken_text:
        cL, cR = st.columns(2)
        with cL:
            st.caption("In SOP but not Spoken")
            st.write(", ".join(only_in_sop) if sop_compact else "—")
        with cR:
            st.caption("In Spoken but not SOP")
            st.write(", ".join(only_in_spoken) if spoken_text else "—")
        st.caption(f"Unique word variance count: {len(only_in_sop) + len(only_in_spoken)}")
    else:
        st.info("Upload transcript or audio to see variance.")

with st.expander("Blacklist Hits", expanded=False):
    if bl_hits:
        st.error(", ".join(bl_hits))
    else:
        st.success("No blacklisted words detected.")

# ---------- FOOTER TIP ----------
st.caption(
    "Tip: Improve SOP matching by writing examples as short phrases (comma, pipe, or semicolon separated). "
    "Audio quality and speaking style impact transcription accuracy."
)
