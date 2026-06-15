"""
src/classification/tflite_ctypes.py
Wrapper ctypes sobre libtensorflow-lite.so del sistema.
Funciona con cualquier versión de Python sin pip install.
Instalación del .so: sudo apt install libtensorflow-lite2.20.0
"""

import ctypes
import logging
import numpy as np

logger   = logging.getLogger(__name__)
_SO_PATH = "/usr/lib/arm-linux-gnueabihf/libtensorflow-lite.so"

# TfLiteType enum
_FLOAT32, _UINT8, _INT8 = 1, 3, 9
_NP = {_FLOAT32: np.float32, _UINT8: np.uint8, _INT8: np.int8}


class _QuantParams(ctypes.Structure):
    _fields_ = [("scale", ctypes.c_float), ("zero_point", ctypes.c_int32)]


def _setup(lib: ctypes.CDLL) -> None:
    v, vp, i32, sz, ci = ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_size_t, ctypes.c_int

    lib.TfLiteModelCreateFromFile.restype, lib.TfLiteModelCreateFromFile.argtypes = v, [ctypes.c_char_p]
    lib.TfLiteInterpreterOptionsCreate.restype, lib.TfLiteInterpreterOptionsCreate.argtypes = v, []
    lib.TfLiteInterpreterOptionsSetNumThreads.restype  = None
    lib.TfLiteInterpreterOptionsSetNumThreads.argtypes = [v, i32]
    lib.TfLiteInterpreterOptionsDelete.restype  = None
    lib.TfLiteInterpreterOptionsDelete.argtypes = [v]
    lib.TfLiteInterpreterCreate.restype, lib.TfLiteInterpreterCreate.argtypes = v, [v, v]
    lib.TfLiteInterpreterAllocateTensors.restype, lib.TfLiteInterpreterAllocateTensors.argtypes = ci, [v]
    lib.TfLiteInterpreterInvoke.restype, lib.TfLiteInterpreterInvoke.argtypes = ci, [v]
    lib.TfLiteInterpreterGetInputTensor.restype,  lib.TfLiteInterpreterGetInputTensor.argtypes  = v, [v, i32]
    lib.TfLiteInterpreterGetOutputTensor.restype, lib.TfLiteInterpreterGetOutputTensor.argtypes = v, [v, i32]
    lib.TfLiteTensorType.restype, lib.TfLiteTensorType.argtypes           = ci, [v]
    lib.TfLiteTensorByteSize.restype, lib.TfLiteTensorByteSize.argtypes   = sz, [v]
    lib.TfLiteTensorCopyFromBuffer.restype, lib.TfLiteTensorCopyFromBuffer.argtypes = ci, [v, v, sz]
    lib.TfLiteTensorCopyToBuffer.restype,  lib.TfLiteTensorCopyToBuffer.argtypes  = ci, [v, v, sz]
    lib.TfLiteTensorQuantizationParams.restype  = _QuantParams
    lib.TfLiteTensorQuantizationParams.argtypes = [v]
    lib.TfLiteInterpreterDelete.restype  = None
    lib.TfLiteInterpreterDelete.argtypes = [v]
    lib.TfLiteModelDelete.restype  = None
    lib.TfLiteModelDelete.argtypes = [v]


class Interpreter:
    """
    Drop-in mínimo para tflite_runtime.interpreter.Interpreter.
    Soporta set_tensor / invoke / get_tensor / get_input_details / get_output_details.
    """

    def __init__(self, model_path: str, num_threads: int = 2):
        self._lib = ctypes.CDLL(_SO_PATH)
        _setup(self._lib)

        self._model = self._lib.TfLiteModelCreateFromFile(model_path.encode())
        if not self._model:
            raise RuntimeError(f"No se pudo cargar el modelo: {model_path}")

        opts = self._lib.TfLiteInterpreterOptionsCreate()
        self._lib.TfLiteInterpreterOptionsSetNumThreads(opts, num_threads)
        self._interp = self._lib.TfLiteInterpreterCreate(self._model, opts)
        self._lib.TfLiteInterpreterOptionsDelete(opts)
        if not self._interp:
            raise RuntimeError("No se pudo crear el intérprete TFLite")

        logger.info("TFLite ctypes intérprete creado: %s", model_path)

    def allocate_tensors(self) -> None:
        if self._lib.TfLiteInterpreterAllocateTensors(self._interp) != 0:
            raise RuntimeError("TfLiteInterpreterAllocateTensors falló")

    # ── tflite_runtime compat ──────────────────────────────────────────────

    def get_input_details(self) -> list[dict]:
        t     = self._lib.TfLiteInterpreterGetInputTensor(self._interp, 0)
        dtype = self._lib.TfLiteTensorType(t)
        q     = self._lib.TfLiteTensorQuantizationParams(t)
        return [{"index": 0, "dtype": _NP.get(dtype, np.float32),
                 "quantization": (q.scale, q.zero_point)}]

    def get_output_details(self) -> list[dict]:
        t     = self._lib.TfLiteInterpreterGetOutputTensor(self._interp, 0)
        dtype = self._lib.TfLiteTensorType(t)
        q     = self._lib.TfLiteTensorQuantizationParams(t)
        return [{"index": 0, "dtype": _NP.get(dtype, np.float32),
                 "quantization": (q.scale, q.zero_point)}]

    def set_tensor(self, tensor_index: int, data: np.ndarray) -> None:
        t       = self._lib.TfLiteInterpreterGetInputTensor(self._interp, tensor_index)
        ttype   = self._lib.TfLiteTensorType(t)
        np_type = _NP.get(ttype, np.float32)

        if np_type == np.float32:
            arr = data.astype(np.float32)
        else:
            q   = self._lib.TfLiteTensorQuantizationParams(t)
            lo, hi = (-128, 127) if np_type == np.int8 else (0, 255)
            arr = np.clip(
                np.round(data.astype(np.float32) / (q.scale or 1.0) + q.zero_point),
                lo, hi
            ).astype(np_type)

        raw = arr.flatten().tobytes()
        self._lib.TfLiteTensorCopyFromBuffer(t, raw, len(raw))

    def invoke(self) -> None:
        if self._lib.TfLiteInterpreterInvoke(self._interp) != 0:
            raise RuntimeError("TFLite invoke falló")

    def get_tensor(self, tensor_index: int) -> np.ndarray:
        t       = self._lib.TfLiteInterpreterGetOutputTensor(self._interp, tensor_index)
        ttype   = self._lib.TfLiteTensorType(t)
        size    = self._lib.TfLiteTensorByteSize(t)
        np_type = _NP.get(ttype, np.float32)

        buf = (ctypes.c_uint8 * size)()
        self._lib.TfLiteTensorCopyToBuffer(t, buf, size)
        raw = np.frombuffer(bytes(buf), dtype=np_type)

        if np_type != np.float32:
            q   = self._lib.TfLiteTensorQuantizationParams(t)
            raw = (raw.astype(np.float32) - q.zero_point) * (q.scale or 1.0)

        return np.expand_dims(raw, axis=0)   # shape (1, num_classes) como tflite_runtime

    def __del__(self):
        lib = getattr(self, "_lib", None)
        if lib:
            if getattr(self, "_interp", None):
                lib.TfLiteInterpreterDelete(self._interp)
            if getattr(self, "_model", None):
                lib.TfLiteModelDelete(self._model)
