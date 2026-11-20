
import re
import unicodedata


def normalize_text(text: str) -> str:
    """
    Normalize Spanish text for ASR evaluation.

    Rules:
    - Unicode NFC normalization
    - Convert to lowercase
    - Remove punctuation
    - Preserve Spanish accents and ñ
    - Preserve numbers
    - Remove extra whitespace
    """

    text = unicodedata.normalize("NFC", text)
    text = text.lower()

    text = "".join(
        " " if unicodedata.category(char).startswith("P") else char
        for char in text
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text
