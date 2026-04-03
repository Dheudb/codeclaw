import asyncio
import os
import sys
import time
from datetime import datetime
from collections import deque
from contextlib import contextmanager

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from codeclaw.core.engine import QueryEngine
from codeclaw.core.config import load_config_into_env, save_current_env_to_config

console = Console()


class InteractiveTerminalTUI:
    """Flicker-free streaming TUI.

    Instead of rich.Live (erase-N-lines + rewrite = flicker), we:
    - Print a one-line status bar that updates in-place via carriage return
    - Stream text_delta content directly to stdout (append-only, zero clearing)
    This matches how Claude Code (Ink/React) renders: only the changed part repaints.
    """

    HIDE_CURSOR = "\x1b[?25l"
    SHOW_CURSOR = "\x1b[?25h"
    ERASE_LINE  = "\x1b[2K"

    def __init__(self, console: Console, engine_getter):
        self.console = console
        self.engine_getter = engine_getter
        self._stream = sys.stderr if sys.stderr.isatty() else sys.stdout
        self.reset_turn()

    @property
    def engine(self):
        return self.engine_getter()

    def reset_turn(self, user_input: str = ""):
        self.user_input = user_input.strip()
        self.current_turn_text = ""
        self.running = False
        self.turn_number = 0
        self.tool_count = 0
        self.last_tool = ""
        self.last_transition = ""
        self.last_stop_reason = ""
        self.pending_auto_commit = False
        self.replayed = False
        self.event_log = deque(maxlen=10)
        self._status_visible = False
        self._streaming_text = False

    def _write_status_line(self):
        """Overwrite the current line with a compact status bar."""
        icon = "● RUNNING" if self.running else "○ IDLE"
        mode = self.engine.get_mode()
        parts = [icon, f"Mode: {mode}", f"Turn: {self.turn_number or '-'}",
                 f"Tools: {self.tool_count}"]
        if self.last_tool:
            parts.append(f"Active: {self.last_tool}")
        line = " │ ".join(parts)
        try:
            cols = os.get_terminal_size().columns
        except OSError:
            cols = 100
        line = line[:cols - 1]
        self._stream.write(f"\r{self.ERASE_LINE}{line}")
        self._stream.flush()
        self._status_visible = True

    def _clear_status_line(self):
        if self._status_visible:
            self._stream.write(f"\r{self.ERASE_LINE}\r")
            self._stream.flush()
            self._status_visible = False

    def start_streaming(self):
        self._stream.write(self.HIDE_CURSOR)
        self._stream.flush()
        self._write_status_line()

    def stop_streaming(self):
        self._clear_status_line()
        self._stream.write(self.SHOW_CURSOR)
        self._stream.flush()

    def track_stream_event(self, event: dict):
        event_type = event.get("type")
        if event_type == "message_start":
            self.running = True
            self.turn_number = int(event.get("turn", 0) or 0)
            self.current_turn_text = ""
            self.last_stop_reason = ""
            self._write_status_line()
        elif event_type == "text_delta":
            text = event.get("text", "")
            if text:
                self.current_turn_text += text
                self.replayed = bool(event.get("replayed"))
                if not self._streaming_text:
                    self._clear_status_line()
                    self._streaming_text = True
                self._stream.write(text)
                self._stream.flush()
        elif event_type == "tool_scheduled":
            self.tool_count += 1
            self.last_tool = event.get("tool_name", "") or ""
            if self._streaming_text:
                self._stream.write("\n")
                self._streaming_text = False
            self._clear_status_line()
            self.console.print(f"  [dim cyan]⚡ {self.last_tool}[/dim cyan]")
            self._write_status_line()
        elif event_type == "loop_transition":
            self.last_transition = event.get("reason", "") or ""
            if self._streaming_text:
                self._stream.write("\n")
                self._streaming_text = False
            self._clear_status_line()
            self.console.print(f"  [dim]⚡ State transition: {self.last_transition}[/dim]")
            self._write_status_line()
        elif event_type == "auto_commit_proposal":
            self.pending_auto_commit = True
        elif event_type == "message_stop":
            self.running = False
            self.last_stop_reason = event.get("stop_reason", "") or ""
            if self._streaming_text:
                self._stream.write("\n")
                self._streaming_text = False
            self._clear_status_line()
            if self.last_stop_reason:
                self.console.print(f"  [dim]⚡ Message stopped: {self.last_stop_reason}[/dim]")

    @contextmanager
    def suspend_streaming(self):
        was_visible = self._status_visible
        self._clear_status_line()
        self._stream.write(self.SHOW_CURSOR)
        self._stream.flush()
        try:
            yield
        finally:
            if was_visible:
                self._stream.write(self.HIDE_CURSOR)
                self._stream.flush()
                self._write_status_line()

    def choose_menu(self, *, title: str, body_lines: list[str], options: list[dict], prompt_label: str, default_value=None):
        with self.suspend_streaming():
            self.console.print()
            self.console.print(
                Panel("\n".join(body_lines), title=title, border_style="yellow", expand=False)
            )
            for index, option in enumerate(options, start=1):
                self.console.print(f"[cyan]{index}[/cyan]. {option['label']}")

            while True:
                raw = self.console.input(
                    f"[bold yellow]{prompt_label}[/bold yellow] [1-{len(options)}]: "
                ).strip().lower()
                if raw == "" and default_value is not None:
                    return default_value
                for index, option in enumerate(options, start=1):
                    aliases = set(str(alias).lower() for alias in option.get("aliases", []))
                    aliases.add(str(option.get("value", "")).lower())
                    aliases.add(str(index))
                    if raw in aliases:
                        return option.get("value")
                self.console.print("[red]Invalid selection.[/red] Please enter a valid number.")



def print_welcome_screen():
    console.print()
    
    art_lines = [
        "  \u2584\u2588\u2588\u2584              \u2584\u2588\u2588\u2584  ",
        "  \u2588\u2588\u2588\u2588  \u2584\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2584  \u2588\u2588\u2588\u2588  ",
        "  \u2588\u2588\u2588\u2588 \u2588  [bold cyan]\u2588\u2588[/]  [bold cyan]\u2588\u2588[/]  \u2588 \u2588\u2588\u2588\u2588  ",
        "   \u2580\u2588\u2588 \u2588 [#f08080]\u2592\u2592[/] vv [#f08080]\u2592\u2592[/] \u2588 \u2588\u2588\u2580   ",
        "   \u2580\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2580           ",
    ]
    bot_art = Text()
    for line in art_lines:
        bot_art.append_text(Text.from_markup(line, style="white"))
        bot_art.append("\n")
    bot_art.append("   CodeClaw               ", style="bold deep_sky_blue1")

    display_provider = os.environ.get("CODECLAW_MODEL_PROVIDER", "").strip() or ""
    display_model = ""
    if display_provider == "openai":
        display_model = os.environ.get("OPENAI_MODEL", "")
    elif display_provider == "anthropic":
        display_model = os.environ.get("ANTHROPIC_MODEL", "")
    if not display_provider:
        display_model = os.environ.get("ANTHROPIC_MODEL", "") or os.environ.get("OPENAI_MODEL", "")

    model_str = f"[cyan]{display_model}[/cyan]" if display_model else "[dim](not set — use /model)[/dim]"
    provider_str = f"[green]{display_provider}[/green]" if display_provider else "[dim](not set — use /model)[/dim]"

    info_table = Table.grid(padding=(0, 2))
    info_table.add_column()
    info_table.add_row("[bold white]Welcome to CodeClaw CLI![/bold white]")
    info_table.add_row("[dim]Send /help or /model to configure.[/dim]\n")
    info_table.add_row(f"[dim]Model:[/dim]    {model_str}")
    info_table.add_row(f"[dim]Provider:[/dim] {provider_str}\n")

    main_grid = Table.grid(padding=(1, 4))
    main_grid.add_column(justify="center", vertical="top")
    main_grid.add_column(justify="left", vertical="top")
    main_grid.add_row(bot_art, info_table)

    lower_table = Table.grid(padding=(0, 4, 0, 0))
    lower_table.add_column(justify="left", min_width=15)
    lower_table.add_column(justify="left", min_width=12)
    lower_table.add_column(justify="left", min_width=10)
    lower_table.add_column(justify="left")

    lower_table.add_row(
        "[bold white]Protocols[/bold white]",
        "[bold white]Shortcuts[/bold white]",
        "[bold white]Status[/bold white]",
        "[bold white]Date & Time[/bold white]"
    )
    lower_table.add_row("", "", "", "")

    protocols = ["anthropic", "openai", "local"]
    def _p_indicator(name):
        if display_provider and name.startswith(display_provider):
            return f"[green]● {name}[/green]"
        return f"[dim]○ {name}[/dim]"

    has_key = bool(
        os.environ.get("ANTHROPIC_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    ) if display_provider else False
    status_str = "[green]● Ready[/green]" if (display_provider and display_model and has_key) else "[yellow]● Not configured[/yellow]"

    lower_table.add_row(
        _p_indicator("anthropic"),
        "[cyan]/palette[/cyan]",
        status_str,
        f"[cyan]{datetime.now().strftime('%b %d')}[/cyan]",
    )
    lower_table.add_row(
        _p_indicator("openai"),
        "[cyan]/model[/cyan]",
        "",
        f"[cyan]{datetime.now().strftime('%I:%M %p')}[/cyan]",
    )
    lower_table.add_row(
        _p_indicator("local"),
        "[cyan]/quit[/cyan]",
        "",
        "",
    )

    content = Group(
        main_grid,
        "\n",
        lower_table
    )
    
    panel = Panel(
        content,
        border_style="deep_sky_blue1",
        expand=False,
        padding=(1, 2)
    )
    console.print(panel)
    console.print()


def run_command_palette(tui: InteractiveTerminalTUI) -> str:
    choice = tui.choose_menu(
        title="Command Palette",
        body_lines=[
            "Select a built-in command.",
            "Press Enter to cancel and return to input mode.",
        ],
        options=[
            {"value": "/help", "label": "Show help"},
            {"value": "/status", "label": "View runtime status"},
            {"value": "/tools", "label": "View tool activity"},
            {"value": "/todos", "label": "View structured todos"},
            {"value": "/mode", "label": "View current mode"},
            {"value": "/plan", "label": "Enable plan mode"},
            {"value": "/plan off", "label": "Disable plan mode"},
            {"value": "/coordinator", "label": "Enable coordinator mode"},
            {"value": "/coordinator off", "label": "Disable coordinator mode"},
            {"value": "/team", "label": "View team status"},
            {"value": "/model", "label": "Switch model/provider"},
            {"value": "/model status", "label": "Show current model runtime"},
            {"value": "/sessions", "label": "List session history"},
            {"value": "/resume", "label": "Resume a session"},
            {"value": "/quit", "label": "Quit CLI"},
        ],
        prompt_label="Command",
        default_value="",
    )
    if choice == "/resume":
        with tui.suspend_streaming():
            sid = console.input("[bold yellow]Session ID[/bold yellow]: ").strip()
        return f"/resume {sid}" if sid else ""
    return choice or ""


PROVIDER_PRESETS = {
    "anthropic": {
        "label": "Anthropic protocol",
        "key_env": "ANTHROPIC_API_KEY",
        "base_url_env": "ANTHROPIC_BASE_URL",
        "model_env": "ANTHROPIC_MODEL",
        "effective_provider": "anthropic",
    },
    "openai": {
        "label": "OpenAI protocol",
        "key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "model_env": "OPENAI_MODEL",
        "effective_provider": "openai",
    },
    "local": {
        "label": "Local (OpenAI-compatible)",
        "key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "model_env": "OPENAI_MODEL",
        "effective_provider": "openai",
    },
}


async def _interactive_model_switch(engine, tui):
    """Interactive wizard to switch protocol, model, API base URL, and API key."""

    current_model = engine.primary_model or "(empty)"
    current_provider = engine.model_provider or "(empty)"
    current_base = engine.api_base_url or os.environ.get("ANTHROPIC_BASE_URL", "") or "(empty)"

    provider_choice = tui.choose_menu(
        title="Configure Model",
        body_lines=[
            f"Current protocol: [cyan]{current_provider}[/cyan]",
            f"Current model:    [cyan]{current_model}[/cyan]",
            f"Current base URL: [dim]{current_base}[/dim]",
            "",
            "Select protocol:",
        ],
        options=[
            {"value": "anthropic", "label": "Anthropic protocol"},
            {"value": "openai",    "label": "OpenAI protocol"},
            {"value": "local",     "label": "Local (OpenAI-compatible)"},
            {"value": "", "label": "Cancel"},
        ],
        prompt_label="Protocol",
        default_value="",
    )
    if not provider_choice:
        console.print("[dim]Cancelled.[/dim]")
        return

    preset = PROVIDER_PRESETS.get(provider_choice, {})
    effective_provider = preset.get("effective_provider", provider_choice)
    base_url_env = preset.get("base_url_env", "")
    key_env = preset.get("key_env", "")
    model_env = preset.get("model_env", "")

    with tui.suspend_streaming():
        model_input = console.input(
            f"[bold yellow]Model name[/bold yellow] [dim](current: {current_model}, Enter to keep)[/dim]: "
        ).strip()

    with tui.suspend_streaming():
        url_input = console.input(
            f"[bold yellow]API Base URL[/bold yellow] [dim](current: {current_base}, Enter to keep)[/dim]: "
        ).strip()

    current_key_set = bool(os.environ.get(key_env, ""))
    key_hint = "set" if current_key_set else "not set"
    with tui.suspend_streaming():
        key_input = console.input(
            f"[bold yellow]API Key ({key_env})[/bold yellow] [dim](currently {key_hint}, Enter to keep)[/dim]: "
        ).strip()

    if url_input and base_url_env:
        os.environ[base_url_env] = url_input
    if model_input and model_env:
        os.environ[model_env] = model_input
    os.environ["CODECLAW_MODEL_PROVIDER"] = effective_provider

    result = engine.switch_model_runtime(
        provider=effective_provider,
        model=model_input or None,
        api_base_url=url_input if effective_provider == "openai" else None,
        api_key=key_input or None,
    )

    info = Table(title="Model Runtime Updated", border_style="green", expand=False)
    info.add_column("Setting", style="bold")
    info.add_column("Value", style="cyan")
    info.add_row("Provider", result["provider"])
    info.add_row("Model", result["model"])
    info.add_row("API Base URL", result["api_base_url"])
    info.add_row("API Key", "✓ set" if result["has_api_key"] else "✗ not set")
    console.print(info)

    save_current_env_to_config()
    console.print("[dim]Configuration saved to ~/.codeclaw/config.json[/dim]")


async def interactive_loop(resume_session_id: str = None, initial_prompt: str = None, auto_approve: bool = False):
    load_config_into_env()
    print_welcome_screen()

    engine = None
    tui = InteractiveTerminalTUI(console, lambda: engine)

    def permission_prompt(request):
        if auto_approve:
            console.print(f"[dim]Auto-approved: {request.tool_name}[/dim]")
            return "y"
        decision = tui.choose_menu(
            title="Permission Required",
            body_lines=[
                f"Tool: {request.tool_name}",
                f"Risk: {request.risk_level}",
                "",
                request.summary,
            ],
            options=[
                {"value": "y", "label": "Allow once", "aliases": ["allow", "once", "y"]},
                {"value": "a", "label": "Always allow this tool in current session", "aliases": ["always", "a"]},
                {"value": "n", "label": "Deny", "aliases": ["deny", "n"]},
            ],
            prompt_label="授权选择",
            default_value="n",
        )
        console.print(f"权限决策: {request.tool_name} -> {decision}")
        return decision

    async def prompt_auto_commit(proposal: dict):
        if not proposal:
            return
        files = proposal.get("files", []) or []
        lines = [
            f"Branch: {proposal.get('branch') or '<detached>'}",
            f"Repo: {proposal.get('repo_root') or '<unknown>'}",
            f"Message source: {proposal.get('message_source') or 'unknown'}",
            "",
            "Commit message:",
            proposal.get("message") or "<empty>",
            "",
            "Files:",
        ]
        for item in files[:12]:
            lines.append(f"- {item.get('relative_path')} [{item.get('operation')}]")
        if proposal.get("skipped_dirty_overlap_count"):
            lines.extend([
                "",
                f"Skipped pre-existing dirty files: {proposal.get('skipped_dirty_overlap_count')}",
            ])

        decision = tui.choose_menu(
            title="Auto-Commit Proposal",
            body_lines=lines,
            options=[
                {"value": "commit", "label": "Commit now", "aliases": ["y", "yes", "commit"]},
                {"value": "skip", "label": "Skip this proposal", "aliases": ["n", "no", "skip"]},
            ],
            prompt_label="提交选择",
            default_value="commit",
        )
        result = await engine.resolve_auto_commit_proposal(
            proposal.get("proposal_id", ""),
            decision=decision,
        )
        if result.get("ok"):
            console.print(f"[green]{result.get('content')}[/green]")
            console.print(result.get("content", "Auto-commit handled."))
        else:
            console.print(f"[red]{result.get('content')}[/red]")
            console.print(result.get("content", "Auto-commit failed."))
        tui.pending_auto_commit = False

    engine = QueryEngine(permission_handler=permission_prompt)

    if not engine.is_configured:
        console.print(
            "[bold yellow]Warning:[/bold yellow] Model not configured. "
            "Use [cyan]/model[/cyan] to set protocol, model name, base URL, and API key.\n"
        )
    else:
        provider_key_name = "OPENAI_API_KEY" if engine.model_provider == "openai" else "ANTHROPIC_API_KEY"
        api_key = os.environ.get(provider_key_name)
        if not api_key:
            console.print(f"[bold yellow]Warning:[/bold yellow] {provider_key_name} is not set in environment.")
            console.print("Use [cyan]/model[/cyan] to configure, or set the environment variable.\n")

    def agent_events_callback(text: str):
        console.print(text)

    with console.status("[bold cyan]Bootstrapping Advanced Protocols...", spinner="dots"):
        await engine.start_protocols(sys_print=agent_events_callback)

    # Auto-resume if requested via CLI flag
    if resume_session_id:
        if resume_session_id == "__latest__":
            ids = engine.session_manager.list_session_ids()
            if ids:
                resume_session_id = ids[0]
            else:
                console.print("[dim]No previous sessions found. Starting fresh.[/dim]")
                resume_session_id = None
        if resume_session_id and resume_session_id != "__latest__":
            if engine.load_session(resume_session_id):
                console.print(
                    Panel(
                        engine.get_resume_summary(),
                        title=f"Session '{resume_session_id[:8]}...' resumed",
                        border_style="green",
                        expand=False,
                    )
                )
            else:
                console.print(f"[red]Failed to resume session '{resume_session_id}'. Starting fresh.[/red]")

    last_interrupt_time = 0
    _pending_initial_prompt = initial_prompt
    while True:
        try:
            if _pending_initial_prompt:
                user_input = _pending_initial_prompt
                _pending_initial_prompt = None
                console.print(f"\n[bold green]❯[/bold green] [dim]{user_input[:120]}{'...' if len(user_input) > 120 else ''}[/dim]")
            else:
                user_input = console.input("\n[bold green]❯[/bold green] ")
            last_interrupt_time = 0
            if not user_input.strip():
                continue

            if user_input.strip().lower() in {"/palette", "/menu", "/commands"}:
                selected = run_command_palette(tui)
                if not selected:
                    continue
                user_input = selected

            normalized_input = user_input.strip().lower()

            if normalized_input in {"/quit", "/q", "/exit"}:
                console.print("[dim]Session terminated.[/dim]")
                break
            elif normalized_input == "/help":
                help_text = (
                    "[bold white]Session[/bold white]\n"
                    "  /new              Start a fresh session\n"
                    "  /sessions         List saved sessions\n"
                    "  /resume <id>      Resume a saved session\n"
                    "  /delete <id>      Delete a saved session\n"
                    "  /clear sessions   Delete all saved sessions\n\n"
                    "[bold white]Planning[/bold white]\n"
                    "  /plan             Enable plan mode\n"
                    "  /plan off         Disable plan mode\n"
                    "  /plan show        Show current plan\n"
                    "  /plan clear       Clear current plan\n"
                    "  /ultraplan <desc> Deep planning via dedicated sub-agent\n\n"
                    "[bold white]Multi-Agent[/bold white]\n"
                    "  /team             View team status\n"
                    "  /coordinator      Toggle coordinator mode\n"
                    "  /coordinator off  Disable coordinator mode\n\n"
                    "[bold white]Inspect[/bold white]\n"
                    "  /mode             View current mode\n"
                    "  /todos            View structured todos\n"
                    "  /tools            View tool activity\n"
                    "  /agents           View subagent registry\n"
                    "  /sandbox          View sandbox status\n"
                    "  /status           View runtime status\n\n"
                    "[bold white]Model & Provider[/bold white]\n"
                    "  /model            Interactive model/provider switch\n"
                    "  /model status     Show current model runtime\n\n"
                    "[bold white]Other[/bold white]\n"
                    "  /palette          Open command palette\n"
                    "  /quit             Exit CLI\n"
                )
                console.print(Panel(help_text, title="Commands", border_style="cyan", expand=False))
                continue
            elif normalized_input == "/new":
                # Save current session, then reset
                engine.persist_session_state()
                old_sid = engine.session_id
                import uuid as _uuid
                engine.session_id = str(_uuid.uuid4())
                engine.messages = []
                engine.plan_manager.clear_plan()
                engine.todo_manager.load([])
                engine.content_replacement_manager.set_session(engine.session_id)
                console.print(f"[bold green]New session started.[/bold green] Previous session saved as [dim]{old_sid[:8]}...[/dim]")
                continue
            elif normalized_input == "/sessions":
                console.print(engine.session_manager.get_recent_sessions())
                continue
            elif normalized_input.startswith("/delete "):
                parts = user_input.strip().split(maxsplit=1)
                if len(parts) > 1:
                    target_sid = parts[1].strip()
                    if target_sid == engine.session_id:
                        console.print("[red]Cannot delete the active session. Use /new first.[/red]")
                    elif engine.session_manager.delete_session(target_sid):
                        console.print(f"[green]Session [bold]{target_sid[:8]}...[/bold] deleted.[/green]")
                    else:
                        console.print(f"[red]Session '{target_sid}' not found.[/red]")
                continue
            elif normalized_input == "/clear sessions":
                confirm = console.input("[bold yellow]Delete ALL saved sessions? (y/N): [/bold yellow]").strip().lower()
                if confirm == "y":
                    count = engine.session_manager.clear_all_sessions()
                    console.print(f"[green]Cleared {count} saved session(s).[/green]")
                else:
                    console.print("[dim]Cancelled.[/dim]")
                continue
            elif normalized_input.startswith("/resume "):
                items = user_input.split(" ")
                if len(items) > 1:
                    sid = items[1]
                    if engine.load_session(sid):
                        console.print(
                            Panel(
                                engine.get_resume_summary(),
                                title=f"Session '{sid[:8]}...' recovered",
                                border_style="green",
                                expand=False,
                            )
                        )
                        pending_auto_commit = engine.get_pending_auto_commit_proposal()
                        if pending_auto_commit:
                            await prompt_auto_commit(pending_auto_commit)
                    else:
                        console.print(f"[red]Error: Failed to resume session '{sid}'.[/red]")
                continue
            elif normalized_input.startswith("/plan"):
                raw = user_input.strip()
                parts = raw.split(maxsplit=1)
                arg = parts[1].strip().lower() if len(parts) > 1 else ""

                if arg in {"", "on"}:
                    engine.set_mode("plan")
                    engine.persist_session_state()
                    console.print("[bold green]Plan mode enabled.[/bold green] Mutating workspace tools will now be blocked.")
                elif arg in {"off", "exit", "normal"}:
                    engine.set_mode("normal")
                    engine.persist_session_state()
                    console.print("[bold green]Returned to normal mode.[/bold green]")
                elif arg in {"show", "status"}:
                    current_plan = engine.get_plan() or "[dim]Current plan is empty.[/dim]"
                    console.print(Panel(current_plan, title=f"Plan Mode: {engine.get_mode()}", border_style="cyan", expand=False))
                elif arg == "clear":
                    engine.plan_manager.clear_plan()
                    engine.persist_session_state()
                    console.print("[bold green]Current plan cleared.[/bold green]")
                else:
                    console.print("[red]Unknown /plan argument.[/red] Use: /plan, /plan off, /plan show, /plan clear")
                continue
            elif normalized_input == "/mode":
                console.print(Panel(engine.get_mode_summary(), title="Session Mode", border_style="cyan", expand=False))
                continue
            elif normalized_input == "/todos":
                console.print(Panel(engine.get_todo_summary(), title="Structured Todos", border_style="cyan", expand=False))
                continue
            elif normalized_input == "/sandbox":
                console.print(Panel(engine.get_sandbox_summary(), title="Sandbox Status", border_style="cyan", expand=False))
                continue
            elif normalized_input == "/agents":
                console.print(Panel(engine.get_agents_summary(), title="Subagent Registry", border_style="cyan", expand=False))
                continue
            elif normalized_input == "/tools":
                console.print(Panel(engine.get_tools_summary(), title="Tool Activity", border_style="cyan", expand=False))
                continue
            elif normalized_input == "/status":
                console.print(Panel(engine.get_runtime_summary(), title="Runtime Status", border_style="cyan", expand=False))
                continue
            elif normalized_input.startswith("/ultraplan"):
                raw = user_input.strip()
                parts = raw.split(maxsplit=1)
                description = parts[1].strip() if len(parts) > 1 else ""
                if not description:
                    console.print("[yellow]Usage:[/yellow] /ultraplan <task description>")
                    console.print("[dim]Launches a dedicated planning sub-agent that produces a comprehensive plan.[/dim]")
                    continue
                seed_plan = engine.get_plan() or ""
                console.print(f"[bold cyan]Launching Ultraplan...[/bold cyan] [dim]{description[:80]}{'...' if len(description) > 80 else ''}[/dim]")
                with console.status("[bold cyan]Ultraplan running — deep analysis in progress...", spinner="dots"):
                    result = await engine.launch_ultraplan(
                        description=description,
                        seed_plan=seed_plan,
                    )
                if result.get("ok"):
                    console.print(Panel(
                        result.get("plan", "")[:3000] or "[dim]Empty plan returned.[/dim]",
                        title=f"[bold green]Ultraplan Complete[/bold green] ({result.get('elapsed_s', 0):.0f}s)",
                        border_style="green",
                        expand=False,
                    ))
                    console.print("[dim]Plan has been written to session. Use /plan show to view full plan.[/dim]")
                else:
                    console.print(f"[bold red]Ultraplan failed:[/bold red] {result.get('error', 'Unknown error')}")
                continue
            elif normalized_input == "/team":
                console.print(Panel(engine.get_team_summary(), title="Team Status", border_style="cyan", expand=False))
                continue
            elif normalized_input.startswith("/coordinator"):
                raw = user_input.strip()
                parts = raw.split(maxsplit=1)
                arg = parts[1].strip().lower() if len(parts) > 1 else ""
                if arg in {"off", "disable"}:
                    engine.disable_coordinator_mode()
                    engine.persist_session_state()
                    console.print("[bold green]Coordinator mode disabled.[/bold green] Full tool set restored.")
                else:
                    engine.enable_coordinator_mode()
                    engine.persist_session_state()
                    console.print(
                        "[bold green]Coordinator mode enabled.[/bold green]\n"
                        "[dim]Tool set restricted to orchestration tools only.\n"
                        "Use agent_tool to spawn workers, send_message_tool to coordinate.\n"
                        "Type /coordinator off to restore full tool access.[/dim]"
                    )
                continue
            elif normalized_input.startswith("/model"):
                raw = user_input.strip()
                parts = raw.split(maxsplit=1)
                arg = parts[1].strip().lower() if len(parts) > 1 else ""

                if arg == "status":
                    info = Table(title="Model Runtime", border_style="cyan", expand=False)
                    info.add_column("Setting", style="bold")
                    info.add_column("Value", style="cyan")
                    info.add_row("Provider", engine.model_provider)
                    info.add_row("Model", engine.primary_model)
                    info.add_row("Fallback", engine.fallback_model or "(same)")
                    info.add_row("API Base URL", engine.api_base_url or "(default)")
                    key_name = "OPENAI_API_KEY" if engine.model_provider == "openai" else "ANTHROPIC_API_KEY"
                    info.add_row("API Key", "✓ set" if os.environ.get(key_name) else "✗ not set")
                    console.print(info)
                else:
                    await _interactive_model_switch(engine, tui)
                continue

            final_answer = ""
            turn_was_aborted = False
            tui.reset_turn(user_input)
            tui.start_streaming()

            try:
                async for event in engine.submit_message(user_input, sys_print_callback=agent_events_callback):
                    tui.track_stream_event(event)
                    event_type = event.get("type")
                    if event_type == "auto_commit_proposal":
                        with tui.suspend_streaming():
                            await prompt_auto_commit(event.get("proposal") or {})
                    elif event_type == "final":
                        final_answer = event.get("text", "")
            except (KeyboardInterrupt, asyncio.CancelledError):
                last_interrupt_time = 0
                try:
                    await engine.request_abort()
                except Exception:
                    pass
                turn_was_aborted = True
            finally:
                tui.stop_streaming()

            # Show whatever content was received (partial on abort, full on completion)
            display_text = final_answer or tui.current_turn_text.strip()
            if turn_was_aborted:
                if display_text:
                    console.print(Panel(
                        Markdown(display_text),
                        title="[yellow]⚡ Interrupted[/yellow]",
                        border_style="yellow",
                        expand=False,
                    ))
                console.print("[dim]Current turn aborted. You can continue or start a new prompt.[/dim]")
                continue

            if display_text:
                console.print(Panel(Markdown(display_text), border_style="green", expand=False))

        except KeyboardInterrupt:
            current_time = time.time()
            if current_time - last_interrupt_time < 2.0:
                console.print("\n[dim]Received interrupt. Goodbye![/dim]")
                break
            else:
                last_interrupt_time = current_time
                console.print("\n[dim](Press Ctrl+C again within 2 seconds to exit)[/dim]")
                continue
        except asyncio.CancelledError:
            continue
        except Exception as e:
            console.print(f"\n[bold red]Critical Error:[/bold red] {e}")
        except BaseException as e:
            console.print(f"\n[bold red]Fatal System Error:[/bold red] {type(e).__name__}")
            break


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CodeClaw Agentic CLI Engine")
    parser.add_argument("-c", "--continue", dest="continue_session", action="store_true",
                        help="Resume the most recent session")
    parser.add_argument("-r", "--resume", dest="resume_id", type=str, default=None,
                        help="Resume a specific session by ID")
    parser.add_argument("-p", "--prompt", dest="prompt_text", type=str, default=None,
                        help="Send an initial prompt directly (non-interactive start)")
    parser.add_argument("-y", "--dangerously-skip-permissions", dest="auto_approve",
                        action="store_true",
                        help="Auto-approve all tool calls (use for unattended/overnight runs)")
    parser.add_argument("prompt", nargs="*", help="Optional initial prompt (positional)")
    args = parser.parse_args()

    resume_id = None
    if args.continue_session:
        resume_id = "__latest__"
    elif args.resume_id:
        resume_id = args.resume_id

    initial_prompt = args.prompt_text or (" ".join(args.prompt) if args.prompt else None)

    # Suppress Windows ProactorEventLoop cleanup noise on exit.
    # Transport.__del__ may fire after the loop is closed, which is harmless.
    if sys.platform == "win32":
        from asyncio.proactor_events import _ProactorBasePipeTransport  # type: ignore[attr-defined]
        _orig_del = _ProactorBasePipeTransport.__del__

        def _silent_del(self, *args, **kwargs):
            try:
                _orig_del(self, *args, **kwargs)
            except RuntimeError:
                pass
        _ProactorBasePipeTransport.__del__ = _silent_del

    try:
        asyncio.run(interactive_loop(
            resume_session_id=resume_id,
            initial_prompt=initial_prompt,
            auto_approve=args.auto_approve,
        ))
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
