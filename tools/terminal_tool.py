"""Terminal Tool - safe subprocess execution with timeout and output capture"""
import subprocess
import logging
import shlex

logger = logging.getLogger(__name__)

BLOCKED_COMMANDS = ["rm -rf /", ":(){ :|:& };:", "mkfs", "dd if="]

def is_safe_command(command: str) -> bool:
    for blocked in BLOCKED_COMMANDS:
        if blocked in command:
            return False
    return True


def run_command(command: str, timeout: int = 30) -> dict:
    """
    Execute a shell command safely with timeout.
    Returns stdout, stderr, and return code.
    """
    logger.info(f"[Terminal] Running: {command}")
    if not is_safe_command(command):
        logger.warning(f"[Terminal] Blocked unsafe command: {command}")
        return {"success": False, "error": "Command blocked by safety filter", "stdout": "", "stderr": ""}

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        logger.info(f"[Terminal] Return code: {result.returncode}")
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout[:2000],
            "stderr": result.stderr[:500],
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Command timed out after {timeout}s", "stdout": "", "stderr": ""}
    except Exception as e:
        logger.error(f"[Terminal] Exception: {e}")
        return {"success": False, "error": str(e), "stdout": "", "stderr": ""}
