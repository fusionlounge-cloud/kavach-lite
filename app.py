# app.py — Kavach Lite (Multilingual One-Row SOP Checker)

import io
import re
from typing import List, Set, Tuple, Dict

import streamlit as st
import pandas as pd

# Use rapidfuzz (fast and no external system deps)
try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None

# Optional English sentiment (VADER)
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    _HAVE_NLTK = True
except Exception:
    _HAVE_NLTK = False


# -------------------------------
# UI / Page
# -------------------------------
st.set_page_config(page_title="Kavach Lite — One-Row SOP Checker", layout="wide")
st.title("Kavach Lite — One-Row SOP Checker")

st.caption(
    "Upload transcript + SOP + Blacklist → EMPLOYEE | SOP TEXT | SPOKEN TEXT | "
    "BLACKLISTED WORDS | WORD VARIANCE | VARIANCE % | TONE | COMPLIANCE %"
)

# -------------------------------
# Language Selection
# -------------------------------
LANGUAGES = [
    "English", "Kannada", "Telugu", "Tamil", "Malayalam", "Hindi",
    "Arabic", "French", "Spanish"
]
language = st.selectbox("Select Language", LANGUAGES, index=0)
st.caption("⚠️ Make sure your transcript, SOP, and blacklist are in the **same language** you select here.")

# -------------------------------
# Sidebar: Employee Inputs + Uploaders
# -------------------------------
with st.sidebar:
    st.header("Upload")
    emp_name = st.text_input("Employee name", value="jhony")
    emp_id = st.text_input("Employee ID", value="EMP001")

    st.subheader("Transcript (.txt)")
    txt_file = st.file_uploader(
        "Drag and drop file here",
        type=["txt"],
        key="upl_txt",
        label_visibility="collapsed"
    )

    st.subheader("SOP CSV")
    sop_file = st.file_uploader(
        "Drag and drop file here",
        type=["csv"],
        key="upl_sop",
        label_visibility="collapsed",
        help="CSV with columns: category/description, examples (split by |), required (0/1), min_needed, weight"
    )

    st.subheader("Blacklist CSV")
    bl_file = st.file_uploader(
        "Drag and drop file here",
        type=["csv"],
        key="upl_bl",
        label_visibility="collapsed",
        help="First column is treated as list of blacklisted words/phrases"
    )


# -------------------------------
# Helpers (Language-aware)
# -------------------------------
def normalize_text(s: str) -> str:
    """Normalize text depending on selected language (lowercase + keep script range)."""
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()

    if language in ["English", "French", "Spanish"]:
        # keep latin letters, digits, underscore, apostrophes, and whitespace
        s = re.sub(r"[^\w\s']", " ", s)
        s = re.sub(r"\s+", " ", s).strip()

    elif language in ["Kannada", "Telugu", "Tamil", "Malayalam", "Hindi"]:
        # Hindi (Devanagari \u0900) through Malayalam (\u0D7F)
        s = re.sub(r"[^\u0900-\u0D7F\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()

    elif language == "Arabic":
        s = re.sub(r"[^\u0600-\u06FF\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()

    else:
        # default: be permissive
        s = re.sub(r"\s+", " ", s).strip()

    return s


def tokenize_set(s: str) -> Set[str]:
    """Split into unique tokens (space-separated). Keeps language-specific script."""
    s = normalize_text(s)
    return set([tok for tok in s.split(" ") if tok])


def parse_examples_cell(cell: str) -> List[str]:
    """Parse examples string from SOP CSV. Expected as 'a|b|c'. Also handles lists and commas."""
    if not isinstance(cell, str):
        return []
    cell = cell.strip()

    # If like ["a","b"] or ['a','b']
    if (cell.startswith("[") and cell.endswith("]")) or (cell.startswith("(") and cell.endswith(")")):
        # strip brackets
        raw = cell[1:-1]
        parts = re.split(r"[|,]", raw)
        return [normalize_text(p.strip().strip("'").strip('"')) for p in parts if p.strip()]

    # default split by |
    parts = [p.strip() for p in cell.split("|")]
    return [normalize_text(p) for p in parts if p]


def tone_from_text(spoken: str) -> str:
    """Tone classification: English → VADER; others → NEUTRAL placeholder."""
    if language != "English":
        return "NEUTRAL"

    if not _HAVE_NLTK:
        return "NEUTRAL"

    # Ensure VADER is available (silent if already downloaded)
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except Exception:
        try:
            nltk.download("vader_lexicon", quiet=True)
        except Exception:
            return "NEUTRAL"

    try:
        sia = SentimentIntensityAnalyzer()
        ss = sia.polarity_scores(spoken or "")
        if ss["compound"] >= 0.35:
            return "POLITE"
        if ss["compound"] <= -0.35:
            return "RUDE"
        return "NEUTRAL"
    except Exception:
        return "NEUTRAL"


def best_fuzzy_match(spoken: str, examples: List[str]) -> Tuple[str, int]:
    """Return (best_example, best_score) using rapidfuzz partial_ratio."""
    if not examples:
        return ("", 0)
    if fuzz is None:
        # graceful fallback: simple containment check
        s = normalize_text(spoken)
        best = 0
        best_ex = ""
        for ex in examples:
            if ex and ex in s:
                score = 100
            else:
                score = 0
            if score > best:
                best = score
                best_ex = ex
        return (best_ex, best)

    s = normalize_text(spoken)
    best = 0
    best_ex = ""
    for ex in examples:
        score = fuzz.partial_ratio(s, ex)
        if score > best:
            best = score
            best_ex = ex
    return (best_ex, int(best))


def find_blacklisted(spoken: str, bl_words: List[str]) -> List[str]:
    """Return list of blacklisted words/phrases found in spoken text."""
    found = []
    text_norm = normalize_text(spoken)
    for w in bl_words:
        w_norm = normalize_text(str(w))
        if not w_norm:
            continue
        # Use approximate containment by words (regex); for non-latin, a simple 'in' is ok
        try:
            # For Latin alphabets, use word boundaries where available
            if language in ["English", "French", "Spanish"]:
                pattern = r"\b" + re.escape(w_norm) + r"\b"
                if re.search(pattern, text_norm):
                    found.append(w)
            else:
                if w_norm in text_norm:
                    found.append(w)
        except Exception:
            if w_norm in text_norm:
                found.append(w)
    # unique preserve order
    seen = set()
    uniq = []
    for x in found:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def highlight_blacklist(spoken: str, bl_words: List[str]) -> str:
    """Wrap blacklist words in a red badge."""
    marked = normalize_text(spoken)

    # Replace longer words first to avoid partial overlaps
    words_sorted = sorted([normalize_text(str(w)) for w in bl_words if str(w).strip()], key=len, reverse=True)

    for w in words_sorted:
        if not w:
            continue
        try:
            if language in ["English", "French", "Spanish"]:
                pattern = r"(?<!<span[^>]*>)\b" + re.escape(w) + r"\b"
            else:
                pattern = re.escape(w)
            marked = re.sub(
                pattern,
                rf"<span class='bl-badge'>{w}</span>",
                marked,
                flags=re.IGNORECASE,
            )
        except Exception:
            marked = marked.replace(w, f"<span class='bl-badge'>{w}</span>")

    return marked


# -------------------------------
# CSS for badges / table chips
# -------------------------------
st.markdown(
    """
    <style>
      .metric-chip {
        display:inline-block; padding:4px 10px; border-radius:16px; font-weight:600;
        font-size:12px; background:#111; border:1px solid #333;
      }
      .chip-green { background:#0b3d19; border-color:#2b7a3d; color:#c9f7d9; }
      .chip-amber { background:#3d2b0b; border-color:#7a5a2b; color:#ffe1a6; }
      .chip-red   { background:#3d0b0b; border-color:#7a2b2b; color:#ffc9c9; }

      .bl-badge {
        background:#3d0b0b; color:#ffc9c9; padding:2px 6px; border-radius:8px;
        border:1px solid #7a2b2b; margin:0 2px; display:inline-block;
      }
      .ok-badge {
        background:#0b3d19; color:#c9f7d9; padding:2px 6px; border-radius:8px;
        border:1px solid #2b7a3d; font-size:12px;
      }
      .no-badge {
        background:#3d0b0b; color:#ffc9c9; padding:2px 6px; border-radius:8px;
        border:1px solid #7a2b2b; font-size:12px;
      }
      .small-note { font-size:12px; opacity:0.8; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------
# Main Logic (after uploads)
# -------------------------------
if not (txt_file and sop_file and bl_file):
    st.info("Upload a **Transcript (.txt)**, an **SOP CSV**, and a **Blacklist CSV** to begin.")
    st.stop()

try:
    transcript = txt_file.read().decode("utf-8", errors="ignore")
except Exception:
    transcript = io.TextIOWrapper(txt_file, encoding="utf-8", errors="ignore").read()

# SOP CSV
sop_df = pd.read_csv(sop_file)
sop_df.columns = [c.strip().lower() for c in sop_df.columns]

# Try to be forgiving on column names
def first_col(options: List[str], df_cols: List[str]) -> str:
    for o in options:
        if o in df_cols:
            return o
    return ""

col_category = first_col(["category", "description", "item", "label"], sop_df.columns) or "category"
col_examples = first_col(["examples", "phrases", "samples"], sop_df.columns) or "examples"
col_required = first_col(["required", "is_required", "must"], sop_df.columns) or "required"
col_min = first_col(["min_needed", "min_count", "min"], sop_df.columns) or "min_needed"
col_weight = first_col(["weight", "score", "points"], sop_df.columns) or "weight"

# Fill defaults where missing
if col_required not in sop_df:
    sop_df[col_required] = 1
if col_min not in sop_df:
    sop_df[col_min] = 1
if col_weight not in sop_df:
    sop_df[col_weight] = 1
if col_category not in sop_df:
    sop_df[col_category] = [f"Step {i+1}" for i in range(len(sop_df))]
if col_examples not in sop_df:
    sop_df[col_examples] = ""

# Normalize numeric fields
for c in [col_required, col_min, col_weight]:
    sop_df[c] = pd.to_numeric(sop_df[c], errors="coerce").fillna(1).astype(int)

# Build SOP rows
sop_rows: List[Dict] = []
all_sop_words: Set[str] = set()

for _, row in sop_df.iterrows():
    cat = str(row[col_category]).strip()
    examples = parse_examples_cell(str(row[col_examples]))
    req = int(row[col_required]) if pd.notna(row[col_required]) else 1
    min_need = int(row[col_min]) if pd.notna(row[col_min]) else 1
    w = int(row[col_weight]) if pd.notna(row[col_weight]) else 1

    for ex in examples:
        all_sop_words |= tokenize_set(ex)

    sop_rows.append({
        "category": cat,
        "examples": examples,
        "required": 1 if req else 0,
        "min_needed": max(1, min_need),
        "weight": max(1, w),
    })

# Blacklist CSV: treat first column as blacklisted words/phrases
bl_df = pd.read_csv(bl_file)
bl_df.columns = [c.strip().lower() for c in bl_df.columns]
bl_words = bl_df.iloc[:, 0].astype(str).fillna("").tolist()

# Calculate metrics
spoken_text = transcript.strip()
spoken_norm = normalize_text(spoken_text)
spoken_tokens = tokenize_set(spoken_text)

# Per-row fuzzy
FUZZ_THRESHOLD = 85
details = []
req_weight_total = sum(r["weight"] for r in sop_rows if r["required"] == 1)
req_weight_hit = 0

for r in sop_rows:
    best_ex, best_score = best_fuzzy_match(spoken_text, r["examples"])
    # count hits above threshold as 1 (simple)
    hit_count = 1 if best_score >= FUZZ_THRESHOLD else 0
    satisfied = (hit_count >= r["min_needed"])

    if r["required"] == 1 and satisfied:
        req_weight_hit += r["weight"]

    details.append({
        "category": r["category"],
        "required": r["required"],
        "min_needed": r["min_needed"],
        "weight": r["weight"],
        "hit_count": hit_count,
        "best_example_match": best_ex,
        "best_fuzzy_score": best_score,
        "satisfied": "✅" if satisfied else "✖️",
    })

# Word variance = proportion of transcript words NOT seen anywhere in SOP examples
# (Set-based; language-aware tokenization)
if all_sop_words:
    diff = spoken_tokens - all_sop_words
    word_variance_pct = 100.0 * len(diff) / max(1, len(spoken_tokens))
else:
    word_variance_pct = 100.0  # no SOP words => everything varies

# Compliance % = weighted coverage of required rows
compliance_pct = 100.0 * req_weight_hit / max(1, req_weight_total)

# Tone
tone_label = tone_from_text(spoken_text)

# Blacklisted words found
bl_found = find_blacklisted(spoken_text, bl_words)

# Highlight transcript with blacklist
highlighted = highlight_blacklist(spoken_text, bl_found)


# -------------------------------
# Summary Header
# -------------------------------
st.header("Summary")

summary_df = pd.DataFrame([{
    "EMPLOYEE": f"{emp_name} {emp_id}",
    "SOP TEXT": " | ".join([r["category"] for r in sop_rows]),
    "SPOKEN TEXT": spoken_text[:1000],  # preview
}])

# Clean display
st.dataframe(summary_df, use_container_width=True, hide_index=True)

colA, colB, colC, colD = st.columns(4)
with colA:
    st.metric("WORD VARIANCE", f"{word_variance_pct:.1f}%")
with colB:
    tone_class = (
        "chip-green" if tone_label == "POLITE"
        else "chip-red" if tone_label == "RUDE"
        else "chip-amber"
    )
    st.markdown(f"<span class='metric-chip {tone_class}'>{tone_label}</span>", unsafe_allow_html=True)
with colC:
    st.metric("COMPLIANCE", f"{compliance_pct:.1f}%")
with colD:
    bl_text = "—" if not bl_found else ", ".join(bl_found)
    st.write("**BLACKLISTED WORDS**")
    st.write(bl_text)


# -------------------------------
# Transcript (highlighted)
# -------------------------------
st.subheader("Transcript (highlighted)")
st.markdown(highlighted, unsafe_allow_html=True)

# -------------------------------
# SOP Details (per row)
# -------------------------------
st.subheader("SOP Details (per row)")
details_df = pd.DataFrame(details)
details_df = details_df[[
    "category", "required", "min_needed", "weight",
    "hit_count", "best_example_match", "best_fuzzy_score", "satisfied"
]]
st.dataframe(details_df, use_container_width=True, hide_index=True)

st.markdown(
    "<div class='small-note'>"
    "MATCH threshold = 0.85  |  VARIANCE % = word set difference vs SOP words  |  "
    "COMPLIANCE % = weighted coverage of required SOP rows</div>",
    unsafe_allow_html=True,
)
