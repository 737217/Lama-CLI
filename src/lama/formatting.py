"""Форматирование размеров и времени без обращения к терминалу."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from lama.i18n import t, tn


def format_percent(value: float | int | None) -> str:
    if value is None:
        return "—"
    return f"{round(value)}%"


def format_bytes(size: int | None) -> str:
    if size is None:
        return "—"
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def format_modified(moment: datetime | None, *, now: datetime | None = None) -> str:
    if moment is None:
        return "—"
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    delta = current - moment
    if delta < timedelta(minutes=1):
        return t("just now")
    local_moment = moment.astimezone(current.astimezone().tzinfo)
    local_now = current.astimezone(current.astimezone().tzinfo)
    if local_moment.date() == local_now.date():
        return t("today")
    if local_moment.date() == (local_now.date() - timedelta(days=1)):
        return t("yesterday")
    days = max(delta.days, 1)
    if days < 7:
        return tn(days, en=("{n} day ago", "{n} days ago"), ru=("{n} day ago", "{n} days ago", "{n} days ago"))
    if days < 30:
        weeks = max(days // 7, 1)
        return tn(weeks, en=("{n} week ago", "{n} weeks ago"), ru=("{n} week ago", "{n} weeks ago", "{n} weeks ago"))
    return local_moment.strftime("%Y-%m-%d")


def format_expiry(moment: datetime | None, *, now: datetime | None = None) -> str:
    if moment is None:
        return "—"
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    if moment.year <= 1 or moment.year >= 9000:
        return t("does not unload")
    current = now or datetime.now(UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    delta = moment - current
    if delta <= timedelta(0):
        return t("expiring")
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return t("in {n}s", n=seconds)
    if seconds < 3600:
        return t("in {n} min", n=seconds // 60)
    if seconds < 86400:
        return t("in {n} h", n=seconds // 3600)
    return moment.astimezone().strftime("%Y-%m-%d %H:%M")
