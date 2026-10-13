from datetime import UTC, datetime

from lama.formatting import format_bytes, format_expiry, format_modified


def test_format_bytes_matches_table_style() -> None:
    assert format_bytes(None) == "—"
    assert format_bytes(8_697_040_896) == "8.1 GB"
    assert format_bytes(0) == "0 B"


def test_format_modified_today() -> None:
    now = datetime(2026, 10, 12, 12, tzinfo=UTC)
    assert format_modified(datetime(2026, 10, 12, 8, tzinfo=UTC), now=now) == "today"
    assert format_modified(datetime(2026, 10, 10, 8, tzinfo=UTC), now=now) == "2 days ago"
    assert format_modified(None) == "—"


def test_format_expiry_hides_missing_and_forever() -> None:
    assert format_expiry(None) == "—"
    assert format_expiry(datetime(1, 1, 1, tzinfo=UTC)) == "does not unload"
