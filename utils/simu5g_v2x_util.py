"""Utilities for launching the Simu5G V2X bridge."""

import os
import shutil
import subprocess
import time
from pathlib import Path


def start_simu5g_bridge_in_terminal(
    repo_root=None,
    terminal_cmd=None,
    wait_seconds=2.0,
):
    """Start the Simu5G bridge in a separate terminal window."""
    repo_root = Path(repo_root or Path(__file__).resolve().parents[1]).resolve()
    bridge_dir = repo_root / "veins_bridge" / "omnetpp"
    run_script = bridge_dir / "run_sim5g_uu.sh"
    if not run_script.exists():
        raise FileNotFoundError(f"Simu5G run script not found: {run_script}")

    shell_command = (
        'source "$OMNETPP_HOME/setenv"; '
        f'cd "{bridge_dir}"; '
        "bash ./run_sim5g_uu.sh"
    )

    terminal_cmd = terminal_cmd or _detect_terminal()
    if terminal_cmd is None:
        raise RuntimeError(
            "No terminal emulator was found. Set terminal_cmd, or run "
            f"`source \"$OMNETPP_HOME/setenv\" && cd {bridge_dir} && bash ./run_sim5g_uu.sh` manually."
        )

    process = subprocess.Popen(_terminal_args(terminal_cmd, shell_command), cwd=str(repo_root))
    if wait_seconds:
        time.sleep(float(wait_seconds))
    return process


def _detect_terminal():
    configured = os.environ.get("METSR_TERMINAL")
    if configured:
        return configured
    for candidate in ("gnome-terminal", "x-terminal-emulator", "konsole", "xfce4-terminal"):
        if shutil.which(candidate):
            return candidate
    return None


def _terminal_args(terminal_cmd, shell_command):
    if isinstance(terminal_cmd, (list, tuple)):
        executable = terminal_cmd[0]
        prefix = list(terminal_cmd)
    else:
        executable = terminal_cmd
        prefix = [terminal_cmd]

    if executable.endswith("gnome-terminal"):
        return prefix + ["--", "bash", "-lc", shell_command + "; exec bash"]
    if executable.endswith("konsole"):
        return prefix + ["-e", "bash", "-lc", shell_command + "; exec bash"]
    if executable.endswith("xfce4-terminal"):
        return prefix + ["--hold", "-e", "bash -lc " + repr(shell_command)]
    return prefix + ["-e", "bash", "-lc", shell_command + "; exec bash"]
