"""[4] Translation

Two backends, selected via PipelineConfig.translation_backend:
  - "nllb": facebook/nllb-200-distilled-600M, local, free, offline-capable.
  - "gpt":  OpenAI gpt-4o-mini, better on idioms/register, needs API key.

Segment boundaries are preserved 1:1 (we translate each Segment.text
independently rather than the full transcript at once) — this keeps the
mapping to original timing windows unambiguous for the alignment stage.
The tradeoff (documented in docs/evaluation.md) is loss of cross-segment
context, which can hurt pronoun resolution / idiom translation for short
segments. GPT backend partially mitigates this by including a short
rolling context window in the prompt.
"""
from __future__ import annotations

from loguru import logger

from src.types import Segment

# NLLB uses FLORES-200 language codes, not plain ISO-639-1. Small mapping
# for the languages this project is likely to be demoed with; extend as needed.
_NLLB_LANG_MAP = {
    "en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn",
    "it": "ita_Latn", "pt": "por_Latn", "zh": "zho_Hans", "ja": "jpn_Jpan",
    "ko": "kor_Hang", "ru": "rus_Cyrl", "ar": "arb_Arab", "hi": "hin_Deva",
    "nl": "nld_Latn", "pl": "pol_Latn", "tr": "tur_Latn",
}

_nllb_cache: dict = {}


def translate_segments(
    segments: list[Segment],
    source_lang: str,
    target_lang: str,
    backend: str = "nllb",
    nllb_model: str = "facebook/nllb-200-distilled-600M",
    gpt_model: str = "gpt-4o-mini",
    device: str = "cuda",
) -> list[Segment]:
    """Fill `Segment.translated_text` for every segment. Mutates in place."""
    if backend == "nllb":
        _translate_nllb(segments, source_lang, target_lang, nllb_model, device)
    elif backend == "gpt":
        _translate_gpt(segments, source_lang, target_lang, gpt_model)
    else:
        raise ValueError(f"Unknown translation backend: {backend}")
    return segments


# ---------------------------------------------------------------------- NLLB

def _load_nllb(model_name: str, device: str):
    key = f"{model_name}:{device}"
    if key in _nllb_cache:
        return _nllb_cache[key]

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    logger.info(f"Loading NLLB translation model '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    _nllb_cache[key] = (tokenizer, model)
    return tokenizer, model


def _translate_nllb(segments, source_lang, target_lang, model_name, device):
    if source_lang not in _NLLB_LANG_MAP or target_lang not in _NLLB_LANG_MAP:
        raise ValueError(
            f"Language pair {source_lang}->{target_lang} not in the NLLB code "
            f"map; add it to _NLLB_LANG_MAP in src/translate.py "
            f"(see FLORES-200 codes) or switch translation_backend to 'gpt'."
        )

    tokenizer, model = _load_nllb(model_name, device)
    src_code = _NLLB_LANG_MAP[source_lang]
    tgt_code = _NLLB_LANG_MAP[target_lang]
    tokenizer.src_lang = src_code

    logger.info(f"Translating {len(segments)} segment(s) via NLLB ({src_code} -> {tgt_code})...")
    for seg in segments:
        if not seg.text.strip():
            continue
        inputs = tokenizer(seg.text, return_tensors="pt").to(device)
        forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_code)
        out = model.generate(**inputs, forced_bos_token_id=forced_bos_id, max_new_tokens=256)
        seg.translated_text = tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()


# ----------------------------------------------------------------------- GPT

def _translate_gpt(segments, source_lang, target_lang, model_name):
    import os
    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set; required for translation_backend='gpt'.")

    client = OpenAI()
    logger.info(f"Translating {len(segments)} segment(s) via {model_name} ({source_lang} -> {target_lang})...")

    context_window = 2  # segments of preceding context to include, for pronoun/idiom coherence
    history: list[str] = []

    for seg in segments:
        if not seg.text.strip():
            continue
        context = "\n".join(history[-context_window:])
        context_block = (
            f"Preceding context (for reference only, do not translate):\n{context}\n"
            if context else ""
        )
        prompt = (
            f"Translate the following {source_lang} speech transcript segment into "
            f"natural, spoken {target_lang}. Preserve tone and register. "
            f"Only output the translation, nothing else.\n\n"
            f"{context_block}"
            f"Segment to translate:\n{seg.text}"
        )
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        seg.translated_text = resp.choices[0].message.content.strip()
        history.append(seg.text)
