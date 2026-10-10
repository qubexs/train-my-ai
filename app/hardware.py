"""Hardware-aware auto budget: max_tokens/ctx/threads from RAM + CPU (stdlib only)."""
import os


def total_ram_gb():
    try:
        if os.name == "nt":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MS()
            st.dwLength = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return st.ullTotalPhys / 1e9
        else:
            return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / 1e9
    except Exception:
        pass
    return 8.0


def cpu_count():
    return os.cpu_count() or 4


def auto_budget():
    """Returns (ram_gb, max_tokens, ctx, threads) sensible for 0.5B CPU box."""
    ram = total_ram_gb()
    cpus = cpu_count()
    if ram >= 16:
        max_tokens, ctx = 1000, 4096
    elif ram >= 8:
        max_tokens, ctx = 700, 4096
    elif ram >= 4:
        max_tokens, ctx = 400, 2048
    else:
        max_tokens, ctx = 250, 1024
    threads = max(2, cpus - 1)
    return ram, max_tokens, ctx, threads
