# ai.py
import os
import re
from typing import Tuple, Optional, List

from transformers import pipeline
from PIL import Image
import pdfplumber
from docx import Document

# ---------- Load models once (first run downloads weights) ----------
# Text summarization
_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Image tagging (top labels)
_image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

# ---------- Utilities ----------
_STOPWORDS = {
    "the","and","for","that","with","from","this","have","will","would","there","their","about",
    "which","when","into","your","been","were","what","these","those","them","they","you","but",
    "not","are","was","shall","than","then","also","such","over","more","some","only","like","make",
    "made","can","could","may","might","within","each","other","because","between","being","both",
    "after","before","under","above","below","while","where","who","whom","whose","how","why"
}

def _chunk_text(text: str, max_chars: int = 2500) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)] or [""]

def _extract_keywords(text: str, top_k: int = 5) -> List[str]:
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    freq = {}
    for w in words:
        if w in _STOPWORDS:
            continue
        freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]

def _is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

def _read_txt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""

def _read_pdf(path: str) -> str:
    out = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages[:10]:  # limit pages for speed
                out.append(page.extract_text() or "")
    except Exception:
        pass
    return "\n".join(out).strip()

def _read_docx(path: str) -> str:
    try:
        doc = Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""

def extract_text(filepath: str, mime_type: Optional[str]) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    if mime_type and "text" in mime_type:
        return _read_txt(filepath)
    if ext == ".txt":
        return _read_txt(filepath)
    if ext == ".pdf":
        return _read_pdf(filepath)
    if ext == ".docx":
        return _read_docx(filepath)
    return ""  # unsupported for text extraction

def summarize_text(text: str) -> str:
    if not text:
        return ""
    chunks = _chunk_text(text, max_chars=2500)
    partials = []
    for ch in chunks[:3]:  # keep it light
        res = _summarizer(ch, max_length=150, min_length=40, do_sample=False)
        partials.append(res[0]["summary_text"])
    combined = " ".join(partials)
    if len(partials) > 1:
        final = _summarizer(combined, max_length=120, min_length=40, do_sample=False)
        return final[0]["summary_text"]
    return partials[0]

def tag_image(filepath: str) -> List[str]:
    try:
        img = Image.open(filepath).convert("RGB")
        preds = _image_classifier(img)
        return [p["label"] for p in preds[:5]]
    except Exception:
        return []

def analyze_file(filepath: str, mime_type: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (ai_summary, ai_tags_csv)
    - For text/pdf/docx: summarize + keywords
    - For images: tags
    """
    if _is_image(filepath):
        tags = tag_image(filepath)
        return (None, ", ".join(tags) if tags else None)







    # Try to extract text for docs
    text = extract_text(filepath, mime_type)
    if text and len(text) > 80:
        summary = summarize_text(text)
        kw = _extract_keywords(text, top_k=5)
        tags_csv = ", ".join(kw) if kw else None
        return (summary or None, tags_csv)
    return (None, None)
