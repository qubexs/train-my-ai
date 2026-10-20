"""Hardware-aware auto budget: max_tokens/ctx/threads from RAM + CPU (stdlib only)."""
import os
import sys


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


def physical_cores():
    """Physical cores (hyperthreads hurt llama.cpp). Stdlib-only per OS."""
    try:
        if os.name == "nt":
            import ctypes
            import ctypes.wintypes as _w
            # Byte-walk GetLogicalProcessorInformation (no struct defs):
            # entry = [mask:ULONG_PTR][rel:DWORD][pad+union 20B] = 32B x64, 24B x86.
            # RelationProcessorCore = 0. Verified: size % stride == 0.
            k32 = ctypes.windll.kernel32
            size = _w.DWORD(0)
            k32.GetLogicalProcessorInformation(None, ctypes.byref(size))
            if size.value <= 0:
                raise ValueError("no size")
            ptr = ctypes.sizeof(ctypes.c_void_p)
            stride = 32 if ptr == 8 else 24
            if size.value % stride:
                raise ValueError("stride mismatch")
            buf = ctypes.create_string_buffer(size.value)
            if not k32.GetLogicalProcessorInformation(buf, ctypes.byref(size)):
                raise ValueError("query failed")
            cores = 0
            for off in range(0, size.value, stride):
                rel = int.from_bytes(bytes(buf[off + ptr:off + ptr + 4]), "little")
                if rel == 0:  # RelationProcessorCore
                    cores += 1
            if cores > 0:
                return cores
        elif sys.platform == "darwin":
            import subprocess
            out = subprocess.run(["sysctl", "-n", "hw.physicalcpu"],
                                 capture_output=True, text=True, timeout=5)
            n = int((out.stdout or "").strip())
            if n > 0:
                return n
        else:
            pairs = set()
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
                phys, core = None, None
                for line in f:
                    if line.startswith("physical id"):
                        phys = line.split(":", 1)[1].strip()
                    elif line.startswith("core id"):
                        core = line.split(":", 1)[1].strip()
                    elif not line.strip() and phys is not None:
                        pairs.add((phys, core))
                        phys, core = None, None
            if pairs:
                return len(pairs)
    except Exception:
        pass
    return max(1, cpu_count() // 2)


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
    threads = max(2, physical_cores())
    return ram, max_tokens, ctx, threads


def _gpu_vendor(name):
    n = (name or "").lower()
    if any(k in n for k in ("nvidia", "geforce", "rtx", "quadro", "tesla")):
        return "nvidia"
    if any(k in n for k in ("intel", "uhd", "iris", "arc")):
        return "intel"
    if any(k in n for k in ("amd", "radeon", "ati ")):
        return "amd"
    return "other"


def gpus():
    """Senarai adapter paparan [{name, vendor, vram_gb}] — stdlib-only."""
    out = []
    try:
        if os.name == "nt":
            import winreg
            base = (r"SYSTEM\CurrentControlSet\Control\Class"
                    r"\{4d36e968-e325-11ce-bfc1-08002be10318}")
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as k:
                i = 0
                while True:
                    try:
                        sub = winreg.EnumKey(k, i)
                    except OSError:
                        break
                    i += 1
                    if not sub.isdigit():
                        continue
                    try:
                        with winreg.OpenKey(k, sub) as sk:
                            name = winreg.QueryValueEx(sk, "DriverDesc")[0]
                            vram = 0
                            try:
                                raw = winreg.QueryValueEx(
                                    sk, "HardwareInformation.MemorySize")[0]
                                if isinstance(raw, (bytes, bytearray)) and len(raw) >= 4:
                                    vram = int.from_bytes(raw[:8 if len(raw) >= 8 else 4],
                                                          "little")
                            except OSError:
                                pass
                            out.append({"name": name,
                                        "vendor": _gpu_vendor(name),
                                        "vram_gb": round(vram / 1e9, 2) if vram else None})
                    except OSError:
                        continue
    except Exception:
        pass
    return out


def recommended_gpu():
    """Backend GPU paling serasi: 'cuda' | 'vulkan' | 'off' (stdlib-only)."""
    gs = gpus()
    vendors = {g["vendor"] for g in gs}
    if "nvidia" in vendors:
        return "cuda"
    if "intel" in vendors or "amd" in vendors:
        return "vulkan"
    return "off"


def report():
    """Ringkasan perkakasan + backend disyorkan (untuk /hardware atau startup)."""
    gs = gpus()
    return {"cpu": physical_cores(), "threads": cpu_count(),
            "ram_gb": round(total_ram_gb(), 1),
            "gpus": gs, "recommended_gpu": recommended_gpu()}

