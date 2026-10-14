"""Отрисовка таблиц и панелей. Не вызывает Ollama."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from rich import box
from rich.cells import cell_len
from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text

from lama.formatting import (
    format_bytes,
    format_context,
    format_expiry,
    format_modified,
    format_options,
    format_percent,
    format_temperature,
)
from lama.i18n import t
from lama.monitor.resources import CpuInfo, GpuInfo, MemoryInfo, Snapshot
from lama.ollama.types import InstalledModel, ModelShow, RunningModel
from lama.server.manager import ControlMode, ServerStatus

console = Console()

_BAR_WIDTH = 20
_SLOT_WIDTH = 6
_TRACK = Style(bgcolor="grey42")


def _mode_label(mode: ControlMode) -> str:
    labels = {
        ControlMode.SYSTEMD_SYSTEM: "systemd (system)",
        ControlMode.SYSTEMD_USER: "systemd (user)",
        ControlMode.MANAGED: t("lama process"),
        ControlMode.LISTENER: t("ollama serve"),
        ControlMode.EXTERNAL: t("external process"),
        ControlMode.STOPPED: t("stopped"),
        ControlMode.REMOTE: t("remote API"),
        ControlMode.WILDCARD: t("wildcard address"),
        ControlMode.AMBIGUOUS: t("ambiguous"),
    }
    return labels.get(mode, str(mode))


def show_error(message: str) -> None:
    console.print(Panel(Text(message), title=t("Error"), border_style="red", expand=False))


def show_success(message: str) -> None:
    console.print(Panel(Text(message), title=t("Done"), border_style="green", expand=False))


def show_header(*, api_up: bool, version: str | None, model_count: int | None) -> None:
    if api_up:
        server = f"[green]● {t('Running')}[/green]"
        if version:
            server = f"{server}  [dim]{version}[/dim]"
    else:
        server = f"[red]● {t('Stopped')}[/red]"
    models = t("unavailable") if model_count is None else t("{count} installed", count=model_count)
    body = Group(
        Text("🦙  LAMA", style="bold cyan"),
        Text(t("Ollama model manager"), style="dim"),
        Text(""),
        Text.from_markup(f"{t('Server:')}  {server}"),
        Text(f"{t('Models:')}  {models}"),
    )
    console.print(Panel(body, box=box.ROUNDED, border_style="cyan", expand=False, padding=(1, 2), width=42))


def _known(value: str) -> str:
    text = value.strip()
    if not text or text.lower() == "unknown":
        return "—"
    return text


def _wrap_cell(text: str, width: int) -> str:
    """Перенос по ширине ячейки. Разрыв предпочтителен после / : - ."""
    if width < 1 or cell_len(text) <= width:
        return text
    lines: list[str] = []
    rest = text
    while rest:
        if cell_len(rest) <= width:
            lines.append(rest)
            break
        cut = 0
        size = 0
        last_break = 0
        for index, char in enumerate(rest):
            size += cell_len(char)
            if size > width:
                break
            cut = index + 1
            if char in "/:-":
                last_break = cut
        if 0 < last_break < len(rest):
            cut = last_break
        if cut <= 0:
            cut = 1
        lines.append(rest[:cut])
        rest = rest[cut:]
    return "\n".join(lines)


@dataclass
class _ModelRow:
    name: str
    size: str
    params: str
    quant: str
    context: str
    temp: str
    options: tuple[str, ...]
    modified: str

    @classmethod
    def from_model(cls, model: InstalledModel) -> _ModelRow:
        options = format_options(model.parameters)
        parts = tuple(part.strip() for part in options.split("·") if part.strip() and part.strip() != "—")
        return cls(
            name=model.name,
            size=format_bytes(model.size),
            params=_known(model.parameter_size),
            quant=_known(model.quantization),
            context=format_context(model.parameters, model.max_context),
            temp=format_temperature(model.parameters),
            options=parts,
            modified=format_modified(model.modified_at),
        )


@dataclass
class _Column:
    key: str
    header: str
    justify: str
    flexible: bool
    natural: int
    width: int = 0


def _chrome(count: int) -> int:
    """Рамка, разделители и отступы rounded-таблицы с padding (0, 1)."""
    return 2 + (count - 1) + count * 2


def _natural(header: str, values: list[str]) -> int:
    return max([cell_len(header), *(cell_len(value) for value in values), 1])


def _column(key: str, rows: list[_ModelRow], *, flexible: bool = False, justify: str = "left") -> _Column:
    headers = {
        "model": t("MODEL"),
        "size": t("SIZE"),
        "params": t("PARAMS"),
        "quant": t("QUANT"),
        "context": t("CONTEXT"),
        "temp": t("TEMP"),
        "options": t("OPTIONS"),
        "modified": t("MODIFIED"),
    }
    header = headers[key]
    if key == "options":
        values = [item for row in rows for item in row.options] or ["—"]
    elif key == "model":
        values = [row.name for row in rows]
    else:
        values = [getattr(row, key) for row in rows]
    return _Column(
        key=key,
        header=header,
        justify=justify,
        flexible=flexible,
        natural=_natural(header, [] if flexible and key == "model" else values),
    )


def _assign_widths(width: int, columns: list[_Column], *, force: bool = False) -> bool:
    """Раздать ширины так, чтобы таблица не вылезла за терминал. False — колонок слишком много."""
    for column in columns:
        if not column.flexible:
            column.width = column.natural
    leftover = width - _chrome(len(columns)) - sum(column.width for column in columns if not column.flexible)
    flexible = [column for column in columns if column.flexible]
    if len(flexible) == 1:
        if leftover < 18 and not force:
            return False
        flexible[0].width = max(leftover, 8)
        return True
    options = next(column for column in flexible if column.key == "options")
    model = next(column for column in flexible if column.key == "model")
    # Колонка опций не уже самой длинной строки, иначе слово рвётся посередине.
    if leftover - options.natural < 22 and not force:
        return False
    options.width = options.natural if not force else max(8, min(options.natural, leftover // 3))
    model.width = max(leftover - options.width, 8)
    return True


def _select_columns(width: int, rows: list[_ModelRow]) -> list[_Column]:
    keys = ["model", "size", "params", "quant", "context", "temp", "options", "modified"]
    while True:
        columns = [
            _column(
                key,
                rows,
                flexible=key in {"model", "options"},
                justify="right" if key in {"size", "params", "context", "temp"} else "left",
            )
            for key in keys
        ]
        if _assign_widths(width, columns):
            return columns
        for optional in ("options", "quant", "params"):
            if optional in keys:
                keys.remove(optional)
                break
        else:
            _assign_widths(width, columns, force=True)
            return columns


def _name_cell(row: _ModelRow, width: int, present: set[str]) -> Text:
    text = Text()
    for index, line in enumerate(_wrap_cell(row.name, width).split("\n")):
        if index:
            text.append("\n")
        text.append(line, style="bold")
    extra: list[str] = []
    if "params" not in present and row.params != "—":
        extra.append(row.params)
    if "quant" not in present and row.quant != "—":
        extra.append(row.quant)
    if "options" not in present:
        extra.extend(row.options)
    if extra:
        for item in extra:
            for line in _wrap_cell(item, width).split("\n"):
                text.append("\n")
                text.append(line, style="dim")
    return text


def _options_cell(row: _ModelRow, width: int) -> Text:
    if not row.options:
        return Text("—")
    lines: list[str] = []
    for item in row.options:
        lines.extend(_wrap_cell(item, width).split("\n"))
    return Text("\n".join(lines))


def _models_table(rows: list[_ModelRow], width: int) -> Table:
    columns = _select_columns(max(width, 60), rows)
    present = {column.key for column in columns}
    table = Table(
        box=box.ROUNDED,
        header_style="bold cyan",
        expand=False,
        show_lines=True,
        padding=(0, 1),
        highlight=False,
    )
    for column in columns:
        table.add_column(
            column.header,
            justify=column.justify,
            width=column.width,
            no_wrap=True,
            overflow="fold",
            vertical="top",
        )
    for row in rows:
        cells: list[Text] = []
        for column in columns:
            if column.key == "model":
                cells.append(_name_cell(row, column.width, present))
            elif column.key == "options":
                cells.append(_options_cell(row, column.width))
            else:
                cells.append(Text(str(getattr(row, column.key))))
        table.add_row(*cells)
    return table


def show_models(models: list[InstalledModel]) -> None:
    console.print(_models_table([_ModelRow.from_model(model) for model in models], console.width))
    if not models:
        console.print(f"[dim]{t('No models are installed.')}[/dim]")


def show_running(models: list[RunningModel]) -> None:
    table = Table(box=box.SIMPLE, header_style="bold cyan", expand=False)
    table.add_column(t("MODEL"))
    table.add_column(t("MEMORY"), justify="right")
    table.add_column("VRAM", justify="right")
    table.add_column("CPU", justify="right")
    table.add_column(t("EXPIRES"))
    table.add_column(t("CONTEXT"), justify="right")
    for model in models:
        table.add_row(
            Text(model.name),
            Text(format_bytes(model.size)),
            Text(format_bytes(model.size_vram)),
            Text(format_bytes(model.cpu_bytes)),
            Text(format_expiry(model.expires_at)),
            Text("—" if model.context_length is None else str(model.context_length)),
        )
    console.print(table)
    if not models:
        console.print(f"[dim]{t('No models are loaded in memory.')}[/dim]")


def show_model_info(name: str, info: ModelShow) -> None:
    table = Table(title=name, box=box.SIMPLE, show_header=False, expand=False)
    table.add_column("field", style="dim")
    table.add_column("value")
    for key, value in info.details.items():
        table.add_row(Text(str(key)), Text(str(value)))
    if info.capabilities:
        table.add_row(Text("capabilities"), Text(", ".join(info.capabilities)))
    console.print(table)
    if info.parameters:
        console.print(Panel(Text(info.parameters), title="Parameters", expand=False))
    if info.system:
        console.print(Panel(Text(info.system), title="System", expand=False))
    if info.template:
        console.print(Panel(Text(info.template), title="Template", expand=False))
    if info.modelfile:
        console.print(Panel(Text(info.modelfile), title="Modelfile", expand=False))
    if not any((info.parameters, info.system, info.template, info.modelfile, info.details)):
        console.print(f"[dim]{t('Ollama did not return model details.')}[/dim]")


def show_server(status: ServerStatus) -> None:
    state = f"[green]● {t('Running')}[/green]" if status.api_up else f"[red]● {t('Stopped')}[/red]"
    lines = [
        f"{t('Status')}: {state}",
        f"{t('Address')}: {status.host}",
        f"{t('Control')}: {_mode_label(status.mode)}",
    ]
    if status.version:
        lines.append(f"{t('Version')}: {status.version}")
    if status.pid:
        lines.append(f"PID: {status.pid}")
    if status.api_error and not status.api_up:
        lines.append(f"[dim]{status.api_error}[/dim]")
    if status.stop_reason:
        lines.append(f"[yellow]{status.stop_reason}[/yellow]")
    console.print(Panel("\n".join(lines), title="Ollama Server", expand=False))


def show_text_block(title: str, text: str) -> None:
    console.print(Panel(Text(text), title=title, expand=False))


def _load_style(percent: float) -> str:
    if percent >= 90:
        return "red"
    if percent >= 70:
        return "yellow"
    return "cyan"


def _usage_bar(percent: float | None, *, width: int = _BAR_WIDTH) -> Text:
    """Сплошная полоса из фона ячеек: один рост, без блоков разной высоты."""
    bar = Text()
    if percent is None:
        bar.append(" " * width, style=_TRACK)
        return bar
    value = max(0.0, min(100.0, float(percent)))
    filled = int(round(value / 100.0 * width))
    if value > 0 and filled == 0:
        filled = 1
    filled = min(filled, width)
    if filled:
        bar.append(" " * filled, style=Style(bgcolor=_load_style(value)))
    if filled < width:
        bar.append(" " * (width - filled), style=_TRACK)
    return bar


def _join(parts: list[Text], *, width: int, gap: int = 2, indent: int = 0) -> Text:
    result = Text()
    column = 0
    for part in parts:
        if not part.plain:
            continue
        needed = part.cell_len
        if column and column + gap + needed > width:
            result.append("\n")
            pad = indent if indent + needed <= width else 0
            if pad:
                result.append(" " * pad)
            column = pad
        elif column:
            result.append(" " * gap)
            column += gap
        result.append_text(part)
        column += needed
    return result


def _meter(percent: float | None, label: str, *, slot: str | None = None) -> Text:
    text = Text()
    if slot is not None:
        padded = f"{slot:<4}  "
        if slot:
            text.append(padded, style="dim")
        else:
            text.append(padded)
    text.append_text(_usage_bar(percent))
    if label:
        text.append("  ")
        text.append(label, style="bold")
    return text


def _gauge_indent(slotted: bool) -> int:
    return (_SLOT_WIDTH if slotted else 0) + _BAR_WIDTH + 2


def _capacity_parts(used: int, total: int, free: int, *, slot: str | None = None) -> list[Text]:
    percent = None if total <= 0 else 100.0 * used / total
    return [
        _meter(percent, "", slot=slot),
        Text(f"{format_bytes(used)} / {format_bytes(total)}", style="bold"),
        Text(t("(free {size})", size=format_bytes(free)), style="dim"),
    ]


def _with_gap(text: Text) -> Text:
    text.append("\n")
    return text


def _cpu_cell(cpu: CpuInfo, *, width: int, slot: str | None) -> Text:
    parts = [_meter(cpu.percent, format_percent(cpu.percent), slot=slot)]
    if cpu.load_1 is not None and cpu.load_5 is not None and cpu.load_15 is not None:
        parts.append(
            Text(
                t("load {a}  {b}  {c}", a=f"{cpu.load_1:.2f}", b=f"{cpu.load_5:.2f}", c=f"{cpu.load_15:.2f}"),
                style="dim",
            )
        )
    return _with_gap(_join(parts, width=width, indent=_gauge_indent(slot is not None)))


def _gpu_cell(gpu: GpuInfo, *, width: int) -> Text:
    utilization = None if gpu.utilization is None else float(gpu.utilization)
    parts = [_meter(utilization, format_percent(gpu.utilization), slot="GPU")]
    if gpu.memory_utilization is not None:
        controller = Text()
        controller.append("·  ", style="dim")
        controller.append(t("memory controller "), style="dim")
        controller.append(format_percent(gpu.memory_utilization), style="bold")
        parts.append(controller)
    indent = _gauge_indent(True)
    text = _join(parts, width=width, indent=indent)
    text.append("\n\n")
    text.append_text(
        _join(
            _capacity_parts(gpu.memory_used, gpu.memory_total, gpu.memory_free, slot="VRAM"),
            width=width,
            indent=indent,
        )
    )
    return _with_gap(text)


def _memory_cell(memory: MemoryInfo, *, width: int, slot: str | None) -> Text:
    return _with_gap(
        _join(
            _capacity_parts(memory.used, memory.total, memory.available, slot=slot),
            width=width,
            indent=_gauge_indent(slot is not None),
        )
    )


def _models_cell(snapshot: Snapshot, *, width: int) -> Text:
    if snapshot.running_note:
        return Text(snapshot.running_note, style="yellow")
    if not snapshot.running:
        return Text(t("No models loaded"), style="dim")
    text = Text()
    for index, model in enumerate(snapshot.running):
        if index:
            text.append("\n")
        context = "—" if model.context_length is None else str(model.context_length)
        text.append_text(
            _join(
                [
                    Text(model.name, style="bold"),
                    Text(format_bytes(model.size)),
                    Text(f"VRAM {format_bytes(model.size_vram)}", style="dim"),
                    Text(f"CPU {format_bytes(model.cpu_bytes)}", style="dim"),
                    Text(format_expiry(model.expires_at), style="dim"),
                    Text(f"ctx {context}", style="dim"),
                ],
                width=width,
                gap=3,
            )
        )
    return text


def _usage_width(snapshot: Snapshot, total: int) -> int:
    labels = [t("Resource"), "CPU", "RAM", "GPU", t("Models")]
    labels.extend(f"GPU {index}: {gpu.name}" for index, gpu in enumerate(snapshot.gpus))
    resource = max(len(label) for label in labels)
    return max(28, total - resource - 7)


class _SnapshotView:
    def __init__(self, snapshot: Snapshot, *, live: bool, updated_at: datetime) -> None:
        self.snapshot = snapshot
        self.live = live
        self.updated_at = updated_at

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        snapshot = self.snapshot
        width = _usage_width(snapshot, options.max_width)
        table = Table(
            title=t("System resources  ·  updated {time}", time=f"{self.updated_at:%H:%M:%S}"),
            title_style="italic cyan",
            caption=t("q or Esc — quit   ·   refreshes every 2 s") if self.live else None,
            caption_style="dim",
            box=box.SQUARE,
            header_style="bold cyan",
            expand=False,
            padding=(0, 1),
        )
        table.add_column(t("Resource"), style="bold", vertical="top", no_wrap=True)
        table.add_column(t("Usage"), vertical="top", overflow="fold")

        slot = "" if snapshot.gpus else None
        if snapshot.cpu is not None:
            table.add_row("CPU", _cpu_cell(snapshot.cpu, width=width, slot=slot))
        elif snapshot.cpu_note:
            table.add_row("CPU", Text(snapshot.cpu_note, style="yellow"))

        if snapshot.memory is not None:
            table.add_row("RAM", _memory_cell(snapshot.memory, width=width, slot=slot))
        elif snapshot.memory_note:
            table.add_row("RAM", Text(snapshot.memory_note, style="yellow"))

        if snapshot.gpus:
            for index, gpu in enumerate(snapshot.gpus):
                table.add_row(f"GPU {index}: {gpu.name}", _gpu_cell(gpu, width=width))
        elif snapshot.gpu_note:
            table.add_row("GPU", Text(snapshot.gpu_note, style="dim"))

        table.add_row(t("Models"), _models_cell(snapshot, width=width))
        yield table


def render_snapshot(snapshot: Snapshot, *, live: bool = False, updated_at: datetime | None = None) -> RenderableType:
    moment = updated_at or datetime.now().astimezone()
    return _SnapshotView(snapshot, live=live, updated_at=moment)
