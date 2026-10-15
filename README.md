# Ultra-Lightweight Zero-Training Gesture Control Architecture

### Production-Grade Implementation for macOS (Apple Silicon / Intel) & Raspberry Pi (4/5)
Based on the architecture detailed in *"Ultra-Lightweight Zero-Training Gesture Control Architecture for Raspberry Pi Virtual Try-On Mirrors"*.

---

## 🎯 Architecture Overview

This project implements a **zero-training KISS hybrid architecture** that completely replaces heavy 3D CNNs (I3D, ResNet) and monolithic classifier bundles with a mathematically deterministic edge pipeline:

```
                  ┌──────────────────────────────────────────────┐
                  │ 1. Asynchronous Dual-Mode Camera Stream      │
                  │    • macOS: AVFoundation (Hardware Accel)    │
                  │    • Raspberry Pi: Picamera2 / V4L2          │
                  │    • Zero CPU-wait daemon frame capture      │
                  └──────────────────────┬───────────────────────┘
                                         │  (RGB 640x480 @ 30+ FPS)
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │ 2. MediaPipe Hands Lite Skeletal Perception  │
                  │    • hand_landmark_lite.tflite (1.98 MB)     │
                  │    • model_complexity=0, max_num_hands=1     │
                  │    • Inter-frame ROI tracking (sub-cm acc)   │
                  └──────────────────────┬───────────────────────┘
                                         │  (21 3D Anatomical Keypoints)
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │ 3. Speed-Adaptive 1€ Filter (One Euro)       │
                  │    • Casiez et al. (CHI 2012) formulation    │
                  │    • Low cutoff (1.2 Hz) when stationary     │
                  │    • High cutoff (adaptive) on fast motion   │
                  │    • Eliminates jitter with zero lag         │
                  └──────────────────────┬───────────────────────┘
                                         │  (Stabilized Coordinates)
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │ 4. Deterministic Kinematic FSM               │
                  │    • Scale-invariant palm normalization      │
                  │    • Schmitt-trigger pinch hysteresis        │
                  │    • 4-gate temporal directional swipes      │
                  │    • Rebound suppression cooldown (0.45s)    │
                  └──────────────────────┬───────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
     [Spatial Interactions]                          [Temporal Swipes]
     • HOVER (Virtual Cursor)                        • SWIPE_LEFT  (Prev Item)
     • CLICK (Toggle / Select)                       • SWIPE_RIGHT (Next Item)
     • HOLD  (Fabric Zoom / Confirm)                 • SWIPE_UP    (Catalog Up)
     • DRAG  (3D Rotation / Sliders)                 • SWIPE_DOWN  (Catalog Down)
```

---

## 🚀 Quickstart on Mac

### 1. Requirements & Dependencies
The environment on this Mac already has:
- Python 3.12+
- `mediapipe` (0.10.14)
- `opencv-python` (5.0.0 / 4.x)
- `numpy`

### 2. Run Live Webcam Gesture Control
```bash
python3 main.py
```
*Note: Make sure Camera permissions are granted to your terminal or IDE in **macOS System Settings > Privacy & Security > Camera**.*

### 3. Run Automated Interactive Demo
To run without requiring a physical camera, or to tour through all 7 gestures automatically:
```bash
python3 main.py --mode demo
```

### 4. Zero-Setup In-Browser Integration (100% Client-Side Wasm / No Python Needed!)
Any user can visit your hosted website directly: the browser downloads the lightweight MediaPipe model (~2MB) into their local cache and runs the perception + kinematic FSM directly in JavaScript/WebAssembly:
1. Open the web interface in any modern browser:
   ```bash
   open web_demo/index.html
   ```
2. Click **"Start Camera Gestures"** and grant camera permission.
3. MediaPipe runs locally in the user's browser at 30-60 FPS with zero backend dependencies, zero installation, and complete video privacy.

### 5. Connect to Edge Hardware / Local Python Bridge (Optional)
If running on a dedicated kiosk or Raspberry Pi smart mirror with a local Python daemon:
1. Start the Python gesture engine:
   ```bash
   python3 main.py
   ```
2. In `web_demo/index.html`, toggle the top switch to **"🔌 Python Bridge (WS)"** to listen on `ws://127.0.0.1:8765`.

### 6. React / Next.js Client-Side Hook (e.g., IRIS Frontend)
Import the client-side React hook into any frontend component:
```tsx
import { useClientGesture } from "./useClientGesture";

export function RealtimeTryOn() {
  const { isActive, cursor, state, start, stop } = useClientGesture({
    onSwipeLeft: () => nextGarment(),
    onSwipeRight: () => prevGarment(),
    onSwipeDown: () => nextCategory(),
    onClick: (cursor) => triggerTryOn(cursor),
  });

  return (
    <div>
      <button onClick={start}>Enable Touchless Gestures</button>
    </div>
  );
}
```

### 6. Run the Full Test Suite
Executes unit tests verifying the 1€ filter, anatomical scale invariance, hysteresis, and all 7 state transitions:
```bash
python3 test_suite.py
```

---

## 🎮 Supported Gestures & Mappings

| Gesture | Tracked Landmarks | Kinematic Condition | Thresholds & Timing | Try-On Mirror UI Action |
| :--- | :--- | :--- | :--- | :--- |
| **HOVER** | Thumb Tip (4) + Index Tip (8) | Open hand, stabilized midpoint | $r_{pinch} \ge 0.38$, palm visible & co-planar | Position virtual cursor on screen |
| **CLICK** | Thumb Tip (4), Index Tip (8), Palm (0, 9) | Crisp pinch engage and release | Engage $< 0.28$, Release $> 0.38$, $0.04s \le \Delta t < 0.40s$, disp $< 0.25$ | Select garment, toggle size/color |
| **HOLD** | Thumb Tip (4), Index Tip (8), Palm (0, 9) | Stationary pinch in deadzone | Pinch $< 0.28$, disp $< 0.25$, $\Delta t \ge 0.45s$ | Zoom fabric detail, confirm selection |
| **DRAG** | Thumb Tip (4), Index Tip (8) | Pinch maintained while moving | Pinch $< 0.28$, disp $\ge 0.25 \times L_{palm}$ | Rotate 3D fit avatar, adjust sliders |
| **SWIPE_LEFT** | Middle MCP (9), Wrist (0) | Rapid horizontal stroke left | Window 9 frames, disp $\ge 0.90 \times L_{palm}$, $v \ge 0.38$, lin $\ge 0.68$ | Previous garment in carousel |
| **SWIPE_RIGHT**| Middle MCP (9), Wrist (0) | Rapid horizontal stroke right| Window 9 frames, disp $\ge 0.90 \times L_{palm}$, $v \ge 0.38$, lin $\ge 0.68$ | Next garment in carousel |
| **SWIPE_UP**   | Middle MCP (9), Wrist (0) | Rapid vertical stroke up | Window 9 frames, disp $\ge 0.90 \times L_{palm}$, $v \ge 0.38$, exit guard | Scroll catalog up (Categories) |
| **SWIPE_DOWN** | Middle MCP (9), Wrist (0) | Rapid vertical stroke down | Window 9 frames, disp $\ge 0.90 \times L_{palm}$, $v \ge 0.38$, exit guard | Scroll catalog down (Categories) |

---

## 📁 Repository Structure

```
gesture_control/
├── gesture_engine/
│   ├── __init__.py          # Package exports
│   ├── filter.py            # Speed-adaptive 1€ Filter (OneEuroFilter1D & Point2DEuroFilter)
│   ├── fsm.py               # Scale-invariant Kinematic Finite State Machine
│   ├── camera.py            # Asynchronous threaded camera capture (macOS + Pi + Synthetic)
│   ├── visualizer.py        # Smart mirror UI HUD, cursor, dwell rings, carousel
│   └── synthetic.py         # Deterministic 21-keypoint hand pose simulator
├── main.py                  # Main CLI application runner
├── test_suite.py            # 100% passing unit & kinematic integration test suite
└── README.md                # Technical documentation and guides
```

---

## ⌨️ Interactive Keyboard Controls

- `q` : Exit the application
- `r` : Reset the Finite State Machine and filter state
- `s` : Trigger automated gesture demo tour
- `h` : Help overlay toggle
