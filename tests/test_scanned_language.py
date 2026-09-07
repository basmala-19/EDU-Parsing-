"""OSD fallback must ignore decorative pages with zero script confidence."""

from routing import router


def test_osd_uses_later_arabic_page_after_low_confidence_cover(monkeypatch, tmp_path):
    class _Pix:
        def tobytes(self, _): return b"png"
    class _Page:
        def get_pixmap(self, **_): return _Pix()
    class _Doc:
        def __getitem__(self, _): return [_Page(), _Page()]
        def close(self): pass
    monkeypatch.setattr(router.fitz, "open", lambda _: _Doc())
    monkeypatch.setattr("PIL.Image.open", lambda _: object())
    responses = iter(["Script: Katakana\nScript confidence: 0.00", "Script: Arabic\nScript confidence: 32.42"])
    monkeypatch.setattr("pytesseract.image_to_osd", lambda _: next(responses))

    assert router._detect_language_with_osd(str(tmp_path / "scan.pdf")) == "ar"
