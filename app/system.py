"""The computer the station runs on: what it is, how it is doing, its journal,
and turning it off.

Everything here asks the operating system, and gives nothing rather than
failing where it cannot answer, as on a laptop without vcgencmd or systemd.
"""

import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import time

from app.config import ROOT

UNIT = "kvihtai"                 # the station's systemd service, see deploy/kvihtai.service
HOTSPOT = "kvihtai-hotspot"      # the NetworkManager connection deploy/setup_pi.sh makes

# vcgencmd get_throttled: the low bits say what is happening now, the same
# bits 16 higher whether it has happened since boot.
THROTTLE_BITS = {0: "undervoltage", 1: "freq_capped", 2: "throttled", 3: "temp_limit"}

# What a line of a log looks like when something went wrong. The journal gives
# every line the station writes the same priority, and libcamera writes its own
# WARN and ERROR lines straight to it, so the words are what tells.
PROBLEM = re.compile(r"\b(WARN|WARNING|ERROR|CRITICAL|Traceback|Exception|failed|fails|failure|"
                     r"killed|oom-kill|watchdog|undervoltage|segfault|timed out|cannot|could not)\b",
                     re.IGNORECASE)


def _run(cmd, timeout=3.0):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None


def _read(path):
    try:
        with open(path) as fh:
            return fh.read().strip().strip("\x00")
    except OSError:
        return None


def boot_id():
    """Changes at every boot of the computer, and not otherwise."""
    return _read("/proc/sys/kernel/random/boot_id")


def throttled():
    out = _run(["vcgencmd", "get_throttled"], 2)
    if out is None or out.returncode != 0:
        return None
    try:
        value = int(out.stdout.strip().split("=")[1], 16)
    except (IndexError, ValueError):
        return None
    return {
        "now": {name: bool(value >> bit & 1) for bit, name in THROTTLE_BITS.items()},
        "since_boot": {name: bool(value >> (bit + 16) & 1) for bit, name in THROTTLE_BITS.items()},
        "raw": hex(value),
    }


def memory():
    """Total and available memory in MB, from /proc/meminfo."""
    text = _read("/proc/meminfo")
    if not text:
        return None
    kb = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key in ("MemTotal", "MemAvailable"):
            kb[key] = int(rest.split()[0])
    if len(kb) < 2:
        return None
    return {"total_mb": kb["MemTotal"] // 1024, "available_mb": kb["MemAvailable"] // 1024}


def vitals(path):
    """What the status reports every few seconds: heat, power, space, memory, load."""
    state = {}
    temp = _read("/sys/class/thermal/thermal_zone0/temp")
    if temp:
        try:
            state["temp_c"] = round(int(temp) / 1000, 1)
        except ValueError:
            pass
    t = throttled()
    if t is not None:
        state["throttled_now"] = t["now"]["throttled"]
        state["throttled_since_boot"] = t["since_boot"]["throttled"]
        state["undervoltage"] = t["now"]["undervoltage"]
        state["undervoltage_since_boot"] = t["since_boot"]["undervoltage"]
    try:
        st = os.statvfs(path)
        state["free_bytes"] = st.f_bavail * st.f_frsize
    except OSError:
        pass
    mem = memory()
    if mem is not None:
        state["mem_total_mb"] = mem["total_mb"]
        state["mem_available_mb"] = mem["available_mb"]
    state["load"] = round(os.getloadavg()[0], 2)
    return state


def version():
    """Which code runs: deploy/push.sh writes VERSION, a git checkout knows itself."""
    text = _read(ROOT / "VERSION")
    if text:
        return text
    out = _run(["git", "-C", str(ROOT), "describe", "--always", "--dirty"])
    return out.stdout.strip() if out is not None and out.returncode == 0 else None


def network():
    net = {"addresses": [], "wifi": None, "hotspot": False}
    out = _run(["ip", "-j", "-4", "addr", "show"])
    if out is not None and out.returncode == 0:
        try:
            for iface in json.loads(out.stdout):
                if iface.get("ifname") == "lo":
                    continue
                for a in iface.get("addr_info", []):
                    net["addresses"].append({"interface": iface.get("ifname"), "address": a.get("local")})
        except ValueError:
            pass
    out = _run(["nmcli", "-t", "-f", "DEVICE,TYPE,STATE,CONNECTION", "device"])
    if out is not None and out.returncode == 0:
        for line in out.stdout.splitlines():
            device, kind, state, conn = (line.split(":") + [""] * 4)[:4]
            if kind == "wifi" and state == "connected":
                net["wifi"] = conn
                net["hotspot"] = conn == HOTSPOT
    return net


def directory_bytes(path):
    total = 0
    for base, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(base, name))
            except OSError:
                pass
    return total


def info(data_dir):
    """Everything about the computer worth showing once, on the station page."""
    os_name = None
    for line in (_read("/etc/os-release") or "").splitlines():
        if line.startswith("PRETTY_NAME="):
            os_name = line.split("=", 1)[1].strip('"')
    up = _read("/proc/uptime")
    try:
        disk = shutil.disk_usage(data_dir)
        space = {"total_gb": round(disk.total / 1e9, 1), "free_gb": round(disk.free / 1e9, 1)}
    except OSError:
        space = None
    return {
        "hostname": socket.gethostname(),
        "model": _read("/proc/device-tree/model") or platform.machine(),
        "os": os_name or platform.platform(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "version": version(),
        "booted_s": round(float(up.split()[0])) if up else None,
        "memory": memory(),
        "disk": space,
        "throttled": throttled(),
        "network": network(),
    }


# --- the journal -------------------------------------------------------------------------

JOURNAL_SOURCES = {
    "station": ["-u", UNIT],
    "kernel": ["-k"],
    "system": [],
}


def journal(source="station", boot=0, lines=300, problems=False):
    """The newest lines of the journal; boot 0 is this boot, -1 the one before."""
    if source not in JOURNAL_SOURCES:
        raise ValueError(f"unknown journal {source!r}")
    # look further back when only the problems are wanted, so that a few remain
    wanted = lines * 20 if problems else lines
    cmd = ["journalctl", "--no-pager", "-q", "-o", "short-iso", "-n", str(wanted), "-b", str(int(boot)),
           *JOURNAL_SOURCES[source]]
    out = _run(cmd, timeout=15)
    if out is None:
        return {"available": False, "lines": [], "error": "There is no journal on this computer."}
    if out.returncode != 0:
        return {"available": False, "lines": [], "error": out.stderr.strip()[:300] or "journalctl failed"}
    got = out.stdout.splitlines()
    if problems:
        got = [line for line in got if PROBLEM.search(line)]
    note = None
    if not got and "insufficient permissions" in out.stderr:
        note = "The station's user may not read the system journal; add it to the adm group."
    return {"available": True, "lines": got[-lines:], "error": note}


def boots(limit=10):
    """The boots the journal remembers, newest first. One only, if it is kept in memory."""
    out = _run(["journalctl", "--no-pager", "--list-boots", "-o", "json"], timeout=10)
    if out is None or out.returncode != 0:
        return []
    try:
        listed = json.loads(out.stdout)
    except ValueError:
        return []
    return [{"index": b.get("index"), "id": b.get("boot_id"),
             "first": _iso(b.get("first_entry")), "last": _iso(b.get("last_entry"))}
            for b in reversed(listed[-limit:])]


def _iso(us):
    if not us:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(us / 1e6))


def last_exit_reason():
    """Why the station's previous run in this boot ended, if systemd said: '' if it did not."""
    out = _run(["journalctl", "--no-pager", "-q", "-o", "cat", "-n", "40", "-b", "0", "-u", UNIT], timeout=5)
    text = out.stdout if out is not None and out.returncode == 0 else ""
    if "oom-kill" in text or "OOM killer" in text:
        return "out of memory"
    if "Watchdog timeout" in text:
        return "stopped responding"
    if "core-dump" in text or "SIGSEGV" in text or "signal=SEGV" in text:
        return "crashed"
    return ""


# --- turning it off ------------------------------------------------------------------------

def under_systemd():
    """True when systemd started this process as a service, and starts it again when it ends.

    INVOCATION_ID alone is not enough: everything started from a desktop
    session inherits it. A system service's own process has PID 1 as parent.
    """
    return bool(os.environ.get("INVOCATION_ID")) and os.getppid() == 1


def _systemctl():
    return shutil.which("systemctl") or "/usr/bin/systemctl"


def _sudo_allows(*cmd):
    out = _run(["sudo", "-n", "-l", *cmd], timeout=5)
    return out is not None and out.returncode == 0


def power_abilities(enabled):
    """What the app may do, and in words why not where it may not."""
    why = {}
    if not under_systemd():
        why["restart"] = "The station does not run as a service here, so nothing would start it again."
    if not enabled:
        why["shutdown"] = why["reboot"] = ("Turning the computer off from the app is not switched on here "
                                           "(KVIHTAI_POWER_CONTROL).")
    else:
        for action, verb in (("shutdown", "poweroff"), ("reboot", "reboot")):
            if not _sudo_allows(_systemctl(), verb):
                why[action] = "The station may not turn the Pi off: run deploy/setup_pi.sh again, which allows it."
    return {"restart": "restart" not in why, "shutdown": "shutdown" not in why, "reboot": "reboot" not in why,
            "why": why}


def power(action):
    """Shut down or reboot the computer. None if systemd took it, else what went wrong."""
    verb = {"shutdown": "poweroff", "reboot": "reboot"}[action]
    out = _run(["sudo", "-n", _systemctl(), verb], timeout=15)
    if out is None:
        return "systemctl could not be run"
    return None if out.returncode == 0 else (out.stderr.strip()[:300] or f"exit status {out.returncode}")


def restart_self():
    """End this process the way systemd stops it; systemd then starts it again."""
    os.kill(os.getpid(), signal.SIGTERM)


def notify(message):
    """Tell systemd something (READY=1, WATCHDOG=1), if it is listening."""
    path = os.environ.get("NOTIFY_SOCKET")
    if not path:
        return False
    if path.startswith("@"):
        path = "\0" + path[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.connect(path)
            sock.sendall(message.encode())
        return True
    except OSError:
        return False
