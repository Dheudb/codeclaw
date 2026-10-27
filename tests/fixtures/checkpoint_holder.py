"""Hold this disposable Windows test checkpoint readable but not replaceable."""
import ctypes
from ctypes import wintypes
from pathlib import Path
import sys

kernel = ctypes.WinDLL("kernel32", use_last_error=True)
kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
kernel.CreateFileW.restype = wintypes.HANDLE
kernel.CloseHandle.argtypes = [wintypes.HANDLE]
handle = kernel.CreateFileW(str(Path(sys.argv[1]).resolve()), 0x80000000, 1, None, 3, 0x80, None)
if handle == wintypes.HANDLE(-1).value:
    raise ctypes.WinError(ctypes.get_last_error())
print("OWNED", flush=True)
try:
    sys.stdin.readline()
finally:
    kernel.CloseHandle(handle)
