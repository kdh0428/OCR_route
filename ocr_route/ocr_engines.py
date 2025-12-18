"""Adapters around third-party OCR engines used by the router."""
from __future__ import annotations

import io
import logging
import re
from functools import lru_cache
from typing import Callable, Dict

import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)


def _load_easyocr() -> "easyocr.Reader":  # type: ignore[name-defined]
    import easyocr
    import torch

    gpu = torch.cuda.is_available()  # pragma: no cover - hardware dependent
    LOGGER.info("Initializing EasyOCR (gpu=%s)", gpu)
    return easyocr.Reader(lang_list=["en"], gpu=gpu)


@lru_cache(maxsize=1)
def _easyocr_reader() -> "easyocr.Reader":  # type: ignore[name-defined]
    try:
        return _load_easyocr()
    except Exception as exc:  # pragma: no cover - external dependency
        raise RuntimeError("Failed to initialize EasyOCR") from exc


def _load_paddleocr() -> "PaddleOCR":  # type: ignore[name-defined]
    from paddleocr import PaddleOCR

    LOGGER.info("Initializing PaddleOCR (lang=en)")
    return PaddleOCR(lang="en", show_log=False)


@lru_cache(maxsize=1)
def _paddle_reader() -> "PaddleOCR":  # type: ignore[name-defined]
    try:
        return _load_paddleocr()
    except Exception as exc:  # pragma: no cover - external dependency
        raise RuntimeError("Failed to initialize PaddleOCR") from exc


def _clean_ocr_text(text: str) -> str:
    """Normalize OCR output by collapsing whitespace and dropping noise."""
    if not text:
        return ""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[\r\t]", " ", text)
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def easyocr_ocr(image: Image.Image) -> str:
    """Run EasyOCR on the provided PIL image."""
    reader = _easyocr_reader()
    try:
        results = reader.readtext(np.array(image))
    except Exception as exc:  # pragma: no cover - external dependency
        LOGGER.error("EasyOCR inference failed: %s", exc)
        return ""
    texts = [text for _, text, conf in results if conf > 0.1]
    raw = " \n".join(texts)
    return _clean_ocr_text(raw)


def tesseract_ocr(image: Image.Image) -> str:
    """Run pytesseract on the provided PIL image."""
    try:
        import pytesseract
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("pytesseract is not installed") from exc

    try:
        raw = pytesseract.image_to_string(image, lang="eng")
    except Exception as exc:  # pragma: no cover - external dependency
        LOGGER.error("Tesseract OCR failed: %s", exc)
        return ""
    return _clean_ocr_text(raw)


def paddleocr_ocr(image: Image.Image) -> str:
    """Run PaddleOCR on the provided PIL image."""
    reader = _paddle_reader()
    try:
        results = reader.ocr(np.array(image), cls=False)  # type: ignore[arg-type]
    except Exception as exc:  # pragma: no cover - external dependency
        LOGGER.error("PaddleOCR inference failed: %s", exc)
        return ""
    texts: list[str] = []
    for block in results or []:
        for _, info in block:
            if not info:
                continue
            text, conf = info
            if conf is None or conf < 0.1:
                continue
            texts.append(str(text))
    raw = " \n".join(texts)
    return _clean_ocr_text(raw)


@lru_cache(maxsize=1)
def _doctr_predictor():
    """Load docTR OCR predictor (torch backend)."""
    try:
        from doctr.models import ocr_predictor
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("docTR is not installed; install `python-doctr` and a backend (torch/tf).") from exc
    try:
        # Automatically selects available backend; relies on env for GPU/CPU
        predictor = ocr_predictor(pretrained=True)
        return predictor
    except Exception as exc:  # pragma: no cover - external dependency
        raise RuntimeError("Failed to initialize docTR OCR predictor") from exc


def doctr_ocr(image: Image.Image) -> str:
    """Run docTR OCR on the provided PIL image."""
    try:
        from doctr.io import DocumentFile
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("docTR is not installed; install `python-doctr` and a backend (torch/tf).") from exc

    predictor = _doctr_predictor()
    try:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        doc = DocumentFile.from_images([buffer.getvalue()])
        result = predictor(doc)
    except Exception as exc:  # pragma: no cover - external dependency
        LOGGER.error("docTR inference failed: %s", exc)
        return ""

    # Flatten into lines of text
    texts: list[str] = []
    for page in result.pages or []:
        for block in page.blocks or []:
            for line in block.lines or []:
                line_text = " ".join(word.value for word in (line.words or []) if word.value)
                if line_text:
                    texts.append(line_text)
    raw = " \n".join(texts)
    return _clean_ocr_text(raw)


# Optional future adapters:
# def trocr_ocr(image: Image.Image) -> str:
#     """Placeholder for Microsoft TrOCR integration."""
#     raise NotImplementedError("TrOCR integration is not yet implemented.")
#
# def paddle_ocr(image: Image.Image) -> str:
#     """Placeholder for PaddleOCR integration."""
#     raise NotImplementedError("PaddleOCR integration is not yet implemented.")


OCR_ENGINES: Dict[str, Callable[[Image.Image], str]] = {
    "easyocr": easyocr_ocr,
    "tesseract": tesseract_ocr,
    "paddleocr": paddleocr_ocr,
    "doctr": doctr_ocr,
}


def run_ocr(engine: str, image: Image.Image) -> str:
    """Execute the selected OCR engine and return plain text."""
    if engine not in OCR_ENGINES:
        raise ValueError(f"Unsupported OCR engine: {engine}")
    LOGGER.info("Running OCR engine: %s", engine)
    text = OCR_ENGINES[engine](image)
    return text.strip()
