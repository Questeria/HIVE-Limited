"""Minimal ctypes binding to libcuda — the NVIDIA *driver*, not the CUDA toolkit.

Purpose-written for HIVE Limited rather than copied from the private engine, for the same
reason the decode driver is: a fresh binding covers only what this demo calls, so it
carries no surface from anything it does not do.

This is the whole of HIVE Limited's dependency on NVIDIA software. There is no cuBLAS,
no cuDNN, no CUDA runtime, no TensorRT, no cuda-python package. Kernels are prebuilt PTX
loaded through cuModuleLoadData and launched through cuLaunchKernel. You can check that
claim on your own machine while the engine runs: the /receipt panel in the demo lists
the shared libraries actually mapped into the process, and doctor.sh checks the same.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import sys

_CUDA_SUCCESS = 0


def _find_libcuda() -> ctypes.CDLL:
    candidates = []
    if sys.platform == "win32":
        candidates = ["nvcuda.dll"]
    else:
        found = ctypes.util.find_library("cuda")
        if found:
            candidates.append(found)
        candidates += [
            "libcuda.so.1",
            "libcuda.so",
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
            "/usr/lib/wsl/lib/libcuda.so.1",
        ]
    errors = []
    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError as exc:                       # pragma: no cover - platform dependent
            errors.append(f"{name}: {exc}")
    raise RuntimeError(
        "could not load libcuda (the NVIDIA driver library).\n"
        "HIVE Limited needs the driver only — NOT the CUDA toolkit.\n"
        "Tried:\n  " + "\n  ".join(errors)
    )


_L = _find_libcuda()


class CudaError(RuntimeError):
    """A libcuda call returned non-zero. Carries the driver's own name for the code."""

    def __init__(self, fn: str, code: int):
        name = ctypes.c_char_p()
        try:
            _L.cuGetErrorName(ctypes.c_int(code), ctypes.byref(name))
            detail = name.value.decode() if name.value else f"code {code}"
        except Exception:                            # pragma: no cover - defensive
            detail = f"code {code}"
        super().__init__(f"{fn} failed: {detail} ({code})")
        self.fn, self.code = fn, code


def _check(fn: str, code: int) -> None:
    if code != _CUDA_SUCCESS:
        raise CudaError(fn, code)


# ---- context ---------------------------------------------------------------

def init() -> None:
    _check("cuInit", _L.cuInit(0))


def device(index: int = 0) -> int:
    dev = ctypes.c_int()
    _check("cuDeviceGet", _L.cuDeviceGet(ctypes.byref(dev), ctypes.c_int(index)))
    return dev.value


def device_name(dev: int) -> str:
    buf = ctypes.create_string_buffer(128)
    _check("cuDeviceGetName", _L.cuDeviceGetName(buf, ctypes.c_int(128), ctypes.c_int(dev)))
    return buf.value.decode(errors="replace")


def driver_version() -> str:
    """The installed NVIDIA driver's CUDA version, e.g. '12.8'.

    Read from the driver itself so the receipt panel reports what is actually loaded rather
    than whatever a toolkit elsewhere on the machine claims.
    """
    v = ctypes.c_int()
    _check("cuDriverGetVersion", _L.cuDriverGetVersion(ctypes.byref(v)))
    return f"{v.value // 1000}.{(v.value % 1000) // 10}"


def device_attribute(dev: int, attr: int) -> int:
    val = ctypes.c_int()
    _check("cuDeviceGetAttribute",
           _L.cuDeviceGetAttribute(ctypes.byref(val), ctypes.c_int(attr), ctypes.c_int(dev)))
    return val.value


def compute_capability(dev: int) -> tuple[int, int]:
    major, minor = ctypes.c_int(), ctypes.c_int()
    # 75 = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 76 = ..._MINOR
    _check("cuDeviceGetAttribute",
           _L.cuDeviceGetAttribute(ctypes.byref(major), ctypes.c_int(75), ctypes.c_int(dev)))
    _check("cuDeviceGetAttribute",
           _L.cuDeviceGetAttribute(ctypes.byref(minor), ctypes.c_int(76), ctypes.c_int(dev)))
    return major.value, minor.value


def retain_primary_context(dev: int) -> int:
    ctx = ctypes.c_void_p()
    _check("cuDevicePrimaryCtxRetain",
           _L.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), ctypes.c_int(dev)))
    _check("cuCtxSetCurrent", _L.cuCtxSetCurrent(ctx))
    return ctx.value or 0


def set_current(ctx: int) -> None:
    """Make `ctx` current on THIS thread.

    CUDA contexts are per-thread: a thread that never called this has no current
    context and every driver call fails with CUDA_ERROR_INVALID_CONTEXT. Any thread
    other than the one that retained the context must bind it first.
    """
    _check("cuCtxSetCurrent", _L.cuCtxSetCurrent(ctypes.c_void_p(ctx)))


def synchronize() -> None:
    _check("cuCtxSynchronize", _L.cuCtxSynchronize())


def mem_info() -> tuple[int, int]:
    """(free, total) device memory in bytes."""
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    _check("cuMemGetInfo_v2", _L.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)))
    return free.value, total.value


# ---- memory ----------------------------------------------------------------

def alloc(nbytes: int) -> int:
    ptr = ctypes.c_void_p()
    _check("cuMemAlloc_v2", _L.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(nbytes)))
    return ptr.value or 0


def free(ptr: int) -> None:
    _check("cuMemFree_v2", _L.cuMemFree_v2(ctypes.c_void_p(ptr)))


def upload(dst: int, host_ptr: int, nbytes: int) -> None:
    _check("cuMemcpyHtoD_v2",
           _L.cuMemcpyHtoD_v2(ctypes.c_void_p(dst), ctypes.c_void_p(host_ptr),
                              ctypes.c_size_t(nbytes)))


def download(host_ptr: int, src: int, nbytes: int) -> None:
    _check("cuMemcpyDtoH_v2",
           _L.cuMemcpyDtoH_v2(ctypes.c_void_p(host_ptr), ctypes.c_void_p(src),
                              ctypes.c_size_t(nbytes)))


def memset8(ptr: int, value: int, nbytes: int) -> None:
    _check("cuMemsetD8_v2",
           _L.cuMemsetD8_v2(ctypes.c_void_p(ptr), ctypes.c_ubyte(value),
                            ctypes.c_size_t(nbytes)))


# ---- modules and launches --------------------------------------------------

def load_module(ptx: bytes) -> int:
    mod = ctypes.c_void_p()
    _check("cuModuleLoadData", _L.cuModuleLoadData(ctypes.byref(mod), ptx))
    return mod.value or 0


def get_function(module: int, name: str) -> int:
    fn = ctypes.c_void_p()
    _check("cuModuleGetFunction",
           _L.cuModuleGetFunction(ctypes.byref(fn), ctypes.c_void_p(module), name.encode()))
    return fn.value or 0


def launch(fn: int, grid: int, block: int, args: list, shared: int = 0,
           stream: int = 0) -> None:
    """args: ints are passed as pointers (device addresses); Scalar wraps a 32-bit int.

    `stream` defaults to 0 (the default stream), which is what the unaccelerated path uses.
    Graph capture requires a real stream, so the engine passes one during capture.
    """
    boxed = []
    for a in args:
        boxed.append(ctypes.c_int(a.value) if isinstance(a, Scalar)
                     else ctypes.c_void_p(int(a)))
    arr = (ctypes.c_void_p * len(boxed))()
    for i, b in enumerate(boxed):
        arr[i] = ctypes.cast(ctypes.pointer(b), ctypes.c_void_p)
    _check("cuLaunchKernel",
           _L.cuLaunchKernel(ctypes.c_void_p(fn),
                             ctypes.c_uint(grid), ctypes.c_uint(1), ctypes.c_uint(1),
                             ctypes.c_uint(block), ctypes.c_uint(1), ctypes.c_uint(1),
                             ctypes.c_uint(shared), ctypes.c_void_p(stream),
                             ctypes.byref(arr), ctypes.c_void_p(0)))


# ---- streams and graphs ----------------------------------------------------
#
# Why this exists: a decode step is ~538 kernel launches, and issuing one through ctypes
# costs ~25 us of HOST time. That is ~13.7 ms per token spent in Python while the GPU waits
# — measured, not assumed. A CUDA graph records that sequence once and replays it with a
# single driver call, so the host stops being the bottleneck.
#
# Nothing about our kernels is revealed by this. A graph is a recording of launches the
# driver already saw; the capture API is public NVIDIA driver surface, and replaying is the
# ordinary way anyone runs a fixed sequence. What makes the engine fast is not this file.

def stream_create() -> int:
    """A non-blocking stream. Capture is illegal on the default stream."""
    s = ctypes.c_void_p()
    _check("cuStreamCreate", _L.cuStreamCreate(ctypes.byref(s), ctypes.c_uint(1)))
    return s.value or 0


def stream_synchronize(stream: int) -> None:
    _check("cuStreamSynchronize", _L.cuStreamSynchronize(ctypes.c_void_p(stream)))


def capture_begin(stream: int) -> None:
    # 0 = CU_STREAM_CAPTURE_MODE_GLOBAL
    _check("cuStreamBeginCapture_v2",
           _L.cuStreamBeginCapture_v2(ctypes.c_void_p(stream), ctypes.c_int(0)))


def capture_end(stream: int) -> int:
    graph = ctypes.c_void_p()
    _check("cuStreamEndCapture",
           _L.cuStreamEndCapture(ctypes.c_void_p(stream), ctypes.byref(graph)))
    return graph.value or 0


def graph_instantiate(graph: int) -> int:
    exec_ = ctypes.c_void_p()
    # cuGraphInstantiateWithFlags exists on every driver new enough to matter and avoids
    # the older signature's error-log buffers.
    _check("cuGraphInstantiateWithFlags",
           _L.cuGraphInstantiateWithFlags(ctypes.byref(exec_), ctypes.c_void_p(graph),
                                          ctypes.c_ulonglong(0)))
    return exec_.value or 0


def graph_launch(exec_: int, stream: int) -> None:
    _check("cuGraphLaunch",
           _L.cuGraphLaunch(ctypes.c_void_p(exec_), ctypes.c_void_p(stream)))


def graph_destroy(graph: int, exec_: int) -> None:
    for fn, handle in (("cuGraphExecDestroy", exec_), ("cuGraphDestroy", graph)):
        if handle:
            try:
                getattr(_L, fn)(ctypes.c_void_p(handle))
            except Exception:
                pass


class Scalar:
    """Marks a kernel argument as a by-value 32-bit int rather than a device pointer."""

    __slots__ = ("value",)

    def __init__(self, value: int):
        self.value = int(value)
