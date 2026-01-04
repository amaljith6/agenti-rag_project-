import unicodedata
import re


def normalize_text(text: str) -> str:
    """
    Light normalization (safe for legal text).
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()
