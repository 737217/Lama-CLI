# lama

[Русский](README-RU.md)

An interactive console tool for a local [Ollama](https://ollama.com). Instead of remembering `ollama run`, `ollama pull`, and `ollama create`, start `lama`, pick an action with the arrow keys, and answer the questions.

`lama` does not run models itself. Chat goes through the regular `ollama run` command. Everything else goes through the Ollama CLI or its HTTP API.

Tested with the Ollama 0.32 client. The target system is Ubuntu.

## Installation

You need Python 3.12+, [uv](https://docs.astral.sh/uv/), and the `ollama` command installed.

From the root of this repository:

```bash
uv tool install .
```

After that, `lama` is available from any directory.

Update from the same directory:

```bash
uv tool install . --reinstall
```

If the tool is published and uv sees it as the `lama` tool:

```bash
uv tool upgrade lama
```

Removal:

```bash
uv tool uninstall lama
```

Check it during development without installing it system-wide:

```bash
uv sync
uv run lama --help
```

## Language

The interface is English by default. The Language menu item saves the choice in `~/.config/lama/config.toml`, and the next menu opens in Russian or English immediately. `LAMA_LANG=ru` or `LAMA_LANG=en` overrides the saved language for one launch and does not change the file.

## HOW TO USE

Open the menu and leave it open:

```bash
lama
```

↑↓ moves the selection, Enter confirms it, Esc goes back one step. In a model list, type part of the name to filter. The header shows whether Ollama is responding and how many models are installed. If Ollama is stopped, the menu still opens, but models and server control are unavailable.

A usual session:

1. Download a model: `lama pull qwen3.5:9b`.
2. Chat with it: `lama run qwen3.5:9b`. This opens the regular Ollama chat. `lama run` without a name first asks which model to use and how to start it: a chat, a load into memory without a chat, or custom settings.
3. Free memory when the chat is done: `lama stop`. This unloads the model. It does not stop the Ollama server.
4. Watch the machine: `lama monitor`. Choose a live view or a one-time snapshot. The live view refreshes CPU, RAM, GPU, and loaded models every 2 seconds. Quit with `q` or Esc.

The same steps are items in the `lama` menu. A direct command runs one step and exits. `lama` with no command stays in the menu until you choose Exit.

## Menu

```bash
lama
```

The ↑↓ arrows select an item, Enter confirms it, and Esc goes back one step. In model lists you can type part of a name to filter them. Ctrl+C in the menu exits the tool. Ctrl+C during a chat or a download interrupts only that operation and returns to the menu.

The header shows whether the server is responding and how many models are installed. If Ollama is not running, the menu still opens: models are unavailable, and server control is not.

## Direct commands

A command without a required argument opens the same wizard as the menu item, then exits when it finishes.

```bash
lama run
lama run gemma4:12b
lama stop
lama stop gemma4:12b
lama pull qwen3.5:9b
lama pull hf.co/user/repo:Q4_K_M
lama list
lama ps
lama create
lama show gemma4:12b
lama rm gemma4:12b
lama rm gemma4:12b --yes
lama cp qwen3.5:9b my-qwen
lama push username/my-qwen
lama server status
lama server start
lama server stop
lama server restart
lama server logs
lama monitor
lama info
lama --help
```

`lama run gemma4:12b` opens the interactive Ollama chat immediately. `lama run` without a name first asks for a model and how to start it: a chat, a load into memory without a chat, or custom settings.

`lama monitor` asks whether to refresh live or show a snapshot, then displays CPU, RAM, GPU, and loaded models. Live refresh updates the screen every 2 seconds; quit with `q` or Esc.

## What the menu can do

- Start a chat through `ollama run`, handing the terminal over as-is.
- Load a model into memory without a chat (`POST /api/generate` with an empty prompt and `keep_alive`) and unload it with `ollama stop`. If that command is unavailable, the API is used with `keep_alive: 0`. This is not stopping the server, and it does not depend on whether the selected address is local or remote: the model goes to the selected API, and the local service is left alone.
- Download with progress, a tag picked from the Ollama registry when it is available, and Hugging Face references such as `hf.co/user/repo:Q4_K_M`. The model list is not hardcoded. You can open the catalog at <https://ollama.com/library>.
- Create a model from an installed one, a local GGUF file, or an existing Modelfile. Only changed parameters are written: `num_ctx`, `num_batch`, `num_gpu`, `temperature`, `top_p`, `top_k`, `repeat_penalty`, `num_predict`, `seed`, and `SYSTEM`. `num_gpu` is the number of layers on the GPU (`-1` auto, `0` CPU only), not the number of graphics cards.
- `keep_alive` is not written into the Modelfile: the installed Ollama does not accept it there. The value is stored in the lama config and passed at launch (`ollama run --keepalive` or the API).
- A table of installed models and actions for the selected one: run, details, copy, configure, delete.
- Loaded models: name, memory, VRAM, a CPU estimate (only when the API sent both the total size and `size_vram`), unload time, and context. Missing fields are shown as "—".
- Configure creates a new model. An existing model is overwritten only if you chose the same name and explicitly confirmed it.
- Delete asks for confirmation. `--yes` skips the question on the direct command.
- Publish with `ollama push`. The name must be `username/model`. If `~/.ollama/id_ed25519.pub` is missing, lama offers `ollama signin` and does not push the model blindly.
- Server: check the API and version. Process and systemd control is available only for loopback on this machine (`localhost`, `127.0.0.0/8`, `::1`). A remote address and a wildcard (`0.0.0.0`, `::`) are only checked: lama does not call local `systemctl`, does not start `ollama serve`, does not send signals, and does not present a local log as the log for that address. An arbitrary hostname is not treated as local and is not resolved.
- The local backend is chosen from facts. Both the system and user services are queried, along with lama's own process. Only a service with `ActiveState=active` counts as active. An active user service outranks a system service that is merely installed and inactive. An API response by itself does not prove that this service is the one answering: if the services are quiet and the API answers, it is an external process. Two active services, or an active service together with lama's own process, is ambiguous: start, stop, and restart are blocked. When the server is stopped, the start method is: an installed system service, otherwise a user service, otherwise lama's own `ollama serve`. If `systemctl` or the user bus is missing, those options are simply unavailable, with no `sudo` and no search for a process by name.
- Lama's own process is identified by a state file: PID, start time from `/proc/<pid>/stat` (the `comm` field may contain spaces and parentheses), program path, uid, and the selected address. Matching the PID and the `ollama serve` command is not enough. The record is checked again before a signal and once more before SIGKILL. An old file that contains only a PID, and a damaged record, do not allow the process to be stopped. If the process is already gone, or it is already a different process, the foreign PID does not receive a signal.
- Resources: CPU load from `/proc/stat` and the load average from `/proc/loadavg`, RAM from `/proc/meminfo`, loaded models, and `nvidia-smi` (GPU memory and utilization) when the utility is present and responds. Without an NVIDIA GPU that block is simply skipped. `lama monitor` refreshes the screen every 2 seconds; quit with `q` or Esc.

## Server address

The order is: the `--host` flag, the `OLLAMA_HOST` variable, the file `~/.config/lama/config.toml`, otherwise `http://127.0.0.1:11434`.

```bash
lama --host 127.0.0.1:11434 list
OLLAMA_HOST=10.0.0.2:11434 lama ps
```

The address from the menu "Server → Connection address" is written to the config. If `OLLAMA_HOST` is set in the environment, it takes priority over the file again on the next launch. For plain HTTP, the child `ollama` process receives the address without a scheme.

This priority applies only when `config.toml` could be read and the known fields in it are valid. A syntax error, a read failure, a non-string `host`, a `keep_alive` value that is not a table, or an invalid duration stops the launch with a non-zero code even with `--host` and `OLLAMA_HOST`: a damaged file is not replaced with the default address and is not overwritten. If the file is missing, the ordinary defaults are used.

Locally, lama controls the server only for loopback. `http://10.0.0.2:11434`, a name such as `remote.example` or `my-pc`, and the address `0.0.0.0` / `::` do not grant the right to stop the local service. The name `localhost` is local, and `localhost.example` is not.

The chosen keep-alive values are stored there as well:

```toml
host = "http://127.0.0.1:11434"

[keep_alive]
"my-qwen" = "10m"
```

The state directory for a process started by lama is `~/.local/state/lama/`. `ollama.log` is the output of that process. `ollama.pid` is JSON with its identity (`version`, `pid`, `start_time`, `executable`, `uid`, `endpoint`), not a single number. `XDG_CONFIG_HOME` and `XDG_STATE_HOME` are honored.

## Errors

Expected situations — an unreachable server, a bad name, a registry refusal, a cancelled wizard, a read or parse error in `config.toml` — are printed as a panel without a Python traceback. For the full traceback: `LAMA_DEBUG=1`.

## Development

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Tests do not download models and do not talk to a live Ollama: the HTTP API and system commands are replaced.
