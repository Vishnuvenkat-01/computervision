"""
OpenCV compatibility wrapper for GateID Vision.

Imports native OpenCV (cv2) if installed. If cv2 is unavailable in the environment
(e.g., during headless CI or lightweight test environments), provides a mock shim
so that pipeline logic, unit tests, and mocking frameworks operate cleanly.
"""
from typing import Any
import sys

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

    class _OpenCVMock:
        """Mock OpenCV module when opencv-python is not yet installed."""
        IMWRITE_JPEG_QUALITY = 1
        COLOR_BGR2GRAY = 6
        CV_64F = 6
        FONT_HERSHEY_SIMPLEX = 0
        LINE_AA = 16
        LINE_4 = 4
        INTER_AREA = 3
        CAP_PROP_FRAME_COUNT = 7
        CAP_PROP_FPS = 5
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4

        @staticmethod
        def VideoWriter_fourcc(*args: Any) -> int:
            return 0x34504D6D  # Mock fourcc

        @staticmethod
        def cvtColor(src: Any, code: int) -> Any:
            return src

        class _Laplacian:
            @staticmethod
            def var() -> float:
                return 100.0

        @staticmethod
        def Laplacian(src: Any, ddepth: int) -> Any:
            import numpy as np
            # Return a simple array that has .var()
            return np.array([100.0])

        @staticmethod
        def rectangle(img: Any, pt1: Any, pt2: Any, color: Any, thickness: int = 1, lineType: int = 8) -> Any:
            return img

        @staticmethod
        def putText(img: Any, text: str, org: Any, fontFace: int, fontScale: float, color: Any, thickness: int = 1, lineType: int = 8) -> Any:
            return img

        @staticmethod
        def addWeighted(src1: Any, alpha: float, src2: Any, beta: float, gamma: float, dst: Any = None) -> Any:
            return src1

        @staticmethod
        def resize(src: Any, dsize: Any, interpolation: int = 0) -> Any:
            import numpy as np
            w, h = dsize
            if hasattr(src, "shape"):
                c = src.shape[2] if len(src.shape) > 2 else 1
                return np.zeros((h, w, c), dtype=getattr(src, "dtype", np.uint8))
            return src

        @staticmethod
        def imwrite(filename: str, img: Any, params: Any = None) -> bool:
            import os
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "wb") as f:
                f.write(b"MOCK_IMAGE_DATA")
            return True

        class VideoCapture:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self._opened = True
                self._frame_count = 10
                self._current = 0

            def isOpened(self) -> bool:
                return self._opened

            def get(self, propId: int) -> float:
                if propId == 7:  # FRAME_COUNT
                    return float(self._frame_count)
                if propId == 5:  # FPS
                    return 25.0
                if propId == 3:  # WIDTH
                    return 640.0
                if propId == 4:  # HEIGHT
                    return 480.0
                return 0.0

            def read(self) -> Any:
                import numpy as np
                if self._current < self._frame_count:
                    self._current += 1
                    return True, np.zeros((480, 640, 3), dtype=np.uint8)
                return False, None

            def release(self) -> None:
                self._opened = False

        class VideoWriter:
            def __init__(self, filename: str, fourcc: int, fps: float, frameSize: Any, *args: Any) -> None:
                self.filename = filename
                self._opened = True
                import os
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, "wb") as f:
                    f.write(b"MOCK_VIDEO_DATA")

            def isOpened(self) -> bool:
                return self._opened

            def write(self, image: Any) -> None:
                pass

            def release(self) -> None:
                self._opened = False

    cv2 = _OpenCVMock()  # type: ignore

__all__ = ["cv2", "HAS_OPENCV"]
