"""
Asynchronous Camera Stream Module
Cross-platform support: macOS (AVFoundation), Raspberry Pi (Picamera2), Linux (V4L2),
and graceful synthetic test fallback.
"""

import platform
import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np

# Detect Raspberry Pi Picamera2 availability
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False


class AsyncCameraStream:
    """
    Decoupled daemon camera reader for high-throughput frame acquisition.
    Eliminates camera I/O wait from the perception thread.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        device_index: int = 0,
        use_picam: bool = False,
        synthetic_fallback: bool = True,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.device_index = device_index
        self.use_picam = use_picam and PICAMERA2_AVAILABLE
        self.synthetic_fallback = synthetic_fallback

        self.cap: Optional[cv2.VideoCapture] = None
        self.picam2: Optional[Any] = None
        self.frame: Optional[np.ndarray] = None
        self.running = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None
        self.is_synthetic = False

        self._init_camera()

    def _init_camera(self) -> None:
        """Initialize the appropriate camera backend based on platform."""
        current_os = platform.system()

        if self.use_picam:
            try:
                self.picam2 = Picamera2()
                config = self.picam2.create_video_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"}
                )
                self.picam2.configure(config)
                self.picam2.start()
                return
            except Exception as e:
                print(f"[CAMERA] Picamera2 init failed ({e}), falling back to OpenCV...")

        # macOS: Try AVFoundation backend first
        if current_os == "Darwin":
            self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_AVFOUNDATION)
            if not self.cap.isOpened():
                # Fallback to default backend
                self.cap = cv2.VideoCapture(self.device_index)
        else:
            self.cap = cv2.VideoCapture(self.device_index)

        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            # Test frame read
            ret, test_frame = self.cap.read()
            if ret and test_frame is not None:
                print(f"[CAMERA] Successfully opened physical camera (device {self.device_index}, {self.width}x{self.height}).")
                return
            else:
                print(f"[CAMERA] Camera opened but frame read failed. (Check macOS camera permissions)")
                self.cap.release()
                self.cap = None

        if self.synthetic_fallback:
            print("[CAMERA] Physical camera unavailable. Activating synthetic test frame generator.")
            self.is_synthetic = True
        else:
            raise RuntimeError(
                f"Failed to open video capture device {self.device_index}. "
                "On macOS, verify camera access in System Settings > Privacy & Security > Camera."
            )

    def start(self) -> "AsyncCameraStream":
        """Start the background frame capture thread."""
        self.running = True
        self.thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.thread.start()
        return self

    def _capture_worker(self) -> None:
        """Background thread worker pulling latest frames."""
        t_start = time.time()
        while self.running:
            if self.is_synthetic:
                # Generate a clean dark synthetic mirror frame with a moving test guide
                t = time.time() - t_start
                syn = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                # Subtle dark gradient background
                cv2.rectangle(syn, (10, 10), (self.width - 10, self.height - 10), (35, 30, 30), 1)
                cv2.putText(
                    syn,
                    "SYNTHETIC MIRROR MODE (Press 's' for demo gestures)",
                    (20, self.height - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (150, 150, 150),
                    1,
                )
                with self.lock:
                    self.frame = syn
                time.sleep(1.0 / self.fps)
                continue

            if self.use_picam and self.picam2 is not None:
                raw = self.picam2.capture_array()
                bgr = raw
            elif self.cap is not None:
                ret, bgr = self.cap.read()
                if not ret or bgr is None:
                    time.sleep(0.005)
                    continue
            else:
                time.sleep(0.02)
                continue

            # Mirror mode horizontal flip
            flipped = cv2.flip(bgr, 1)

            with self.lock:
                self.frame = flipped

    def read(self) -> Optional[np.ndarray]:
        """Retrieve the most recent frame."""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self) -> None:
        """Stop capture thread and release hardware resources."""
        self.running = False
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)

        if self.use_picam and self.picam2 is not None:
            try:
                self.picam2.stop()
            except Exception:
                pass

        if self.cap is not None:
            self.cap.release()
            self.cap = None
