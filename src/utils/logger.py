"""Structured console logger powered by Rich for MLOps observability."""
from rich.console import Console
from rich.theme import Theme

custom_theme = Theme({
    "info": "cyan",
    "warning": "bold yellow",
    "error": "bold red",
    "success": "bold green",
    "step": "bold magenta",
})

console = Console(theme=custom_theme)


def log_step(step_name: str, detail: str = "") -> None:
    console.print(f"[step]▶ [{step_name}][/step] {detail}")


def log_info(msg: str) -> None:
    console.print(f"  [info]ℹ[/info] {msg}")


def log_success(msg: str) -> None:
    console.print(f"  [success]✔[/success] {msg}")


def log_warning(msg: str) -> None:
    console.print(f"  [warning]⚠[/warning] {msg}")


def log_error(msg: str) -> None:
    console.print(f"  [error]✖[/error] {msg}")
