import streamlit as st
from text_cleaner import clean_text, is_garbage_input


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL LOADERS
#
#  @st.cache_resource  — called once per server lifetime, never on rerun.
#  Lazy placement      — _load_bart() is only called when BART is actually
#                        needed.  If the user only ever sends short texts
#                        (Auto → T5), BART never enters RAM at all.
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_t5():
    from transformers import pipeline
    return pipeline(
        "summarization",
        model="google-t5/t5-small",   # 242 MB
        device=-1,
        framework="pt",
    )


@st.cache_resource(show_spinner=False)
def _load_bart():
    from transformers import pipeline
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-6-6",  # 600 MB  (vs bart-large-cnn 1.6 GB)
        device=-1,
        framework="pt",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def summarize_text(text: str, detail: str = "medium", model: str = "auto"):
    """
    Parameters
    ----------
    text   : raw user input (cleaning happens here)
    detail : "short" | "medium" | "long"
    model  : "auto"  | "t5"    | "bart"

    Returns
    -------
    (summary: str, model_used: str)
      model_used is one of: "t5" | "bart" | "auto"
    """

    # ── Clean + validate ────────────────────────────────────────────────────
    text = clean_text(text)

    if is_garbage_input(text):
        return "Input text is too short for meaningful summarization.", "none"

    words = len(text.split())

    # ── Dynamic length control ───────────────────────────────────────────────
    if detail == "short":
        max_len = int(words * 0.35)
        min_len = int(words * 0.15)
    elif detail == "long":
        max_len = int(words * 0.75)
        min_len = int(words * 0.40)
    else:                                        # medium (default)
        max_len = int(words * 0.55)
        min_len = int(words * 0.25)

    max_len = max(20, min(max_len, 200))
    min_len = max(10, min(min_len, max_len - 5))

    # ── Model selection + lazy load ──────────────────────────────────────────
    if model == "bart":
        pipe       = _load_bart()
        model_used = "bart"
        input_text = text                        # BART needs no task prefix

    elif model == "t5":
        pipe       = _load_t5()
        model_used = "t5"
        input_text = "summarize: " + text        # T5 requires task prefix

    else:                                        # auto
        if words >= 120:
            pipe       = _load_bart()
            model_used = "auto"
            input_text = text
        else:
            pipe       = _load_t5()
            model_used = "auto"
            input_text = "summarize: " + text

    # ── Inference ────────────────────────────────────────────────────────────
    result = pipe(
        input_text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
        early_stopping=True,
        truncation=True,
    )[0]["summary_text"]

    summary = result.strip()
    if summary:
        summary = summary[0].upper() + summary[1:]

    return summary, model_used
