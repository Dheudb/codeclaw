"""Native shell identity, output decoding and cleanup for owned processes."""
import asyncio
import codecs
import locale
import os
import signal
import subprocess


def shell_guidance():
    if os.name == "nt":
        return ("The actual shell is Windows cmd.exe, NOT Bash or PowerShell. Use cmd syntax; "
                "mkdir -p, heredocs and Unix quoting are not supported. "
                "Prefer command_tool with argv and UTF-8 stdin for multiline code. ")
    return "Commands use the native /bin/sh shell. Prefer command_tool argv for exact arguments. "


def process_options():
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def output_encoding(data, *, final=False):
    try:
        codecs.getincrementaldecoder("utf-8")("strict").decode(data, final=final)
        return "utf-8"
    except UnicodeDecodeError:
        if os.name == "nt":
            import ctypes
            for page in (ctypes.windll.kernel32.GetConsoleOutputCP(), ctypes.windll.kernel32.GetOEMCP(),
                         ctypes.windll.kernel32.GetACP()):
                if page and page != 65001:
                    encoding = "cp" + str(page)
                    try:
                        codecs.getincrementaldecoder(encoding)("strict").decode(data, final=final)
                        return encoding
                    except (UnicodeError, LookupError):
                        pass
        return locale.getpreferredencoding(False) or "utf-8"


def decode_output(data):
    return data.decode(output_encoding(data, final=True), errors="replace")


async def terminate_process_tree(process):
    if process is None or process.returncode is not None:
        return
    if os.name == "nt":
        killer = await asyncio.create_subprocess_exec("taskkill", "/PID", str(process.pid), "/T", "/F",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW)
        await killer.wait()
        if process.returncode is None:
            try: process.kill()
            except ProcessLookupError: pass
    else:
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: pass
    await process.wait()
