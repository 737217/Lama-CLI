import pytest

from lama.i18n import set_language


@pytest.fixture(autouse=True)
def _reset_language(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LAMA_LANG", raising=False)
    set_language("en")
    yield
    set_language("en")
