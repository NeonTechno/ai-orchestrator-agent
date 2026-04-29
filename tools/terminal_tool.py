"""Terminal Tool — safe subprocess execution with timeout and output capture."""
import re
import shlex
import logging
import subprocess

logger = logging.getLogger(__name__)

# Patterns checked against the *normalised* (lowercase, collapsed-whitespace) command
BLOCKED_PATTERNS = [
    r"rm\s+-rf\s+/",          # rm -rf / and variants
    r":\(\)\{.*\}",            # fork bomb
    r"\bmkfs\b",               # filesystem format
    r"\bdd\b.*if=",            # disk-destroyer
    r"shutdown\s+-[hrp]",      # shutdown / reboot
    r"format\s+c:",            # Windows format
    r"drop\s+table",           # SQL destructive
    r"delete\s+from",          # SQL destructive
]

OUTPUT_LIMIT = 2000  # chars, applied to both stdout and stderr


def _normalise(command: str) -> str:
    """Lowercase and collapse whitespace for consistent blocklist matching."""
    return re.sub(r"\s+", " ", command.lower().strip())


def is_safe_command(command: str) -> tuple[bool, str]:
    """Return (is_safe, reason). Checks normalised command against BLOCKED_PATTERNS."""
    norm = _normalise(command)
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, norm):
            return False, f"matches blocked pattern: {pattern!r}"
    return True, ""


def run_command(command: str, timeout: int = 30) -> dict:
    """
    Execute a shell command safely.

    - Normalises and checks against the blocklist before execution.
    - Parses the command with shlex to avoid shell injection (shell=False).
    - Caps stdout and stderr at OUTPUT_LIMIT characters each.
    - Enforces a hard timeout.

    Returns a dict with keys: success, stdout, stderr, returncode.
    On failure: success=False, error=<reason>.
    """
    logger.info(f"[Terminal] Requested: {command!r}")

    safe, reason = is_safe_command(command)
    if not safe:
        logger.warning(f"[Terminal] Blocked — {reason}: {command!r}")
        return {
            "success": False,
            "error": f"Command blocked by safety filter ({reason})",
            "stdout": "",
            "stderr": "",
        }

    try:
        args = shlex.split(command)
    except ValueError as exc:
        return {"success": False, "error": f"Command parse error: {exc}", "stdout": "", "stderr": ""}

    try:
        proc = subprocess.run(
            args,
            shell=False,           # no shell injection possible
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        logger.info(f"[Terminal] Exit {proc.returncode}: {command!r}")
        return {
            "success": proc.returncode == 0,
            "stdout": proc.stdout[:OUTPUT_LIMIT],
            "stderr": proc.stderr[:OUTPUT_LIMIT],
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        logger.warning(f"[Terminal] Timed out after {timeout}s: {command!r}")
        return {
            "success": False,
            "error": f"Command timed out after {timeout}s",
            "stdout": "",
            "stderr": "",
        }
    except FileNotFoundError as exc:
        return {"success": False, "error": f"Command not found: {exc}", "stdout": "", "stderr": ""}
    except Exception as exc:
        logger.error(f"[Terminal] Unexpected error: {exc}")
        return {"success": False, "error": str(exc), "stdout": "", "stderr": ""}
