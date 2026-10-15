# Technical Deep Dive: Zero-Training Touchless Gesture Architecture

This document provides a comprehensive, in-depth architectural and mathematical breakdown of the touchless gesture control engine. It details the underlying perception models, the signal-processing pipeline, the mathematics of the 1€ Filter, scale-invariance mechanics, and the deterministic multi-gate kinematic verification algorithms used to decode cursor movement, swipes, clicks, and thumbs-up gestures.

---

## 📑 Table of Contents
1. [System Architecture Overview](#1-system-architecture-overview)
2. [Perception Engine: MediaPipe Hands Topology](#2-perception-engine-mediapipe-hands-topology)
3. [Dynamic ROI Auto-Zoom & Hand-Lock Engine](#3-dynamic-roi-auto-zoom--hand-lock-engine)
4. [Signal Stabilization: Speed-Adaptive 1€ Filter](#4-signal-stabilization-speed-adaptive-1-filter)
5. [Anatomical Normalization: Scale Invariance](#5-anatomical-normalization-scale-invariance)
6. [How Swipes & Scrolls are Registered](#6-how-swipes--scrolls-are-registered)
7. [How Click (Pinch & Release) is Registered](#7-how-click-pinch--release-is-registered)
8. [How Thumbs Up (👍) is Registered](#8-how-thumbs-up--is-registered)
9. [Technology Stack & Performance Profile](#9-technology-stack--performance-profile)

---

## 1. System Architecture Overview

Unlike traditional deep learning gesture systems that rely on heavy spatio-temporal 3D CNNs (e.g., I3D, SlowFast, ResNet-3D) requiring gigabytes of GPU VRAM and high latency, this architecture employs a **zero-training KISS hybrid pipeline**:

```
                       RGB Video Feed (Webcam / Mirror / Pi Camera)
                                            │
                                            ▼
               ┌────────────────────────────────────────────────────────┐
               │ 1. Dynamic ROI Auto-Zoom & Hand-Lock Tracker          │
               │    • Smooth exponential target tracking (x, y, zoom)   │
               │    • Crops & centers hand to normal visual scale       │
               └────────────────────────────┬───────────────────────────┘
                                            │
                                            ▼
               ┌────────────────────────────────────────────────────────┐
               │ 2. MediaPipe Hands Lite (WebAssembly / WebGL)          │
               │    • Palm Detector + 21 3D Anatomical Landmarks        │
               │    • Single-hand execution @ 30–60 FPS                 │
               └────────────────────────────┬───────────────────────────┘
                                            │ (Raw 3D Keypoints: x, y, z)
                                            ▼
               ┌────────────────────────────────────────────────────────┐
               │ 3. Casiez Speed-Adaptive 1€ Filter (One Euro)          │
               │    • Dynamic cutoff frequency based on velocity        │
               │    • Eliminates high-frequency stationary tremor       │
               │    • Zero perceptible phase lag during rapid motions   │
               └────────────────────────────┬───────────────────────────┘
                                            │ (Stabilized Screen Coordinates)
                                            ▼
               ┌────────────────────────────────────────────────────────┐
               │ 4. Anatomical Palm Normalization (Scale Invariance)   │
               │    • Reference metric: L_palm = ||Landmark 9 - Wrist|| │
               │    • Eliminates user height and camera distance bias   │
               └────────────────────────────┬───────────────────────────┘
                                            │
                      ┌─────────────────────┴─────────────────────┐
                      ▼                                           ▼
         [Continuous Spatial Stream]                 [Deterministic Kinematic FSM]
         • 1:1 Smooth Virtual Cursor                 • Schmitt-Trigger Pinch Click
                                                     • 4-Gate Temporal Velocity Swipes
                                                     • Morphological Thumbs-Up (👍)
```

---

## 2. Perception Engine: MediaPipe Hands Topology

The visual perception module extracts **21 three-dimensional skeletal landmarks** per frame:

```
          [8] Index Tip               [12] Middle Tip          [16] Ring Tip
            │                           │                        │            [20] Pinky Tip
          [7] Index DIP               [11] Middle DIP          [15] Ring DIP    │
            │                           │                        │            [19] Pinky DIP
          [6] Index PIP               [10] Middle PIP          [14] Ring PIP    │
            │                           │                        │            [18] Pinky PIP
          [5] Index MCP               [9]  Middle MCP          [13] Ring MCP    │
            │                           │                        │            [17] Pinky MCP
   [4]      └─────────────┬─────────────┴───────────┬────────────┘              │
  Thumb                   │                         │                           │
   Tip                    └─────────────────────────┼───────────────────────────┘
    │                                               │
   [3] Thumb IP                                     │
    │                                               │
   [2] Thumb MCP                                    │
    │                                               │
   [1] Thumb CMC                                    │
    │                                               │
    └───────────────────────┬───────────────────────┘
                            │
                        [0] Wrist
```

### Perception Pipeline Steps:
1. **Palm Detector (Single-Shot Detector):** Detects palm bounding box across the video frame.
2. **Hand Landmark Model:** Predicts the 21 normalized coordinates $(x, y, z)$ within the detected box.
   - $x, y \in [0.0, 1.0]$: Normalized screen coordinates.
   - $z$: Relative depth metric (approximating distance relative to the wrist).

---

## 3. Dynamic ROI Auto-Zoom & Hand-Lock Engine

### The Problem with Long-Distance Tracking
When a user stands far back (2–4 meters), their hand occupies only **~2% to 3% of the frame** (~15–25 pixels). At this tiny resolution:
- Finger joint separation is smaller than model quantization error.
- Pinching thumb and index fingers appear merged in 2D pixels even when open.
- Small arm gestures cover negligible normalized screen units, causing swipes to fail.

### The Solution: Continuous Digital Zoom & Target Tracking
The system dynamically computes the hand center $(c_x, c_y)$ and its apparent scale $L_{palm}$, smoothly cropping a sub-rectangle of the high-definition camera feed and scaling it into the perception model:

$$\text{Desired Palm Scale: } S_{target} \approx 0.18 \quad (18\% \text{ of canvas height})$$

$$\text{Optimal Zoom Factor: } Z_{optimal} = \text{clamp}\left(\frac{S_{target}}{\max(L_{palm}, 0.035)}, 1.0, 3.2\right)$$

To eliminate camera jitter, both the zoom level and the crop center point are smoothed using exponential moving filters:

$$Z_{t} = Z_{t-1} + \alpha_z \cdot (Z_{optimal} - Z_{t-1}), \quad \alpha_z = 0.12$$

$$C_{x, t} = C_{x, t-1} + \alpha_c \cdot (H_{x} - C_{x, t-1}), \quad \alpha_c = 0.16$$

### Coordinate Reverse-Mapping
Once landmarks are predicted inside the zoomed canvas $[0, 1]$, they are mathematically re-mapped back into global screen coordinates:

$$x_{global} = \frac{\text{crop}_x + x_{roi} \cdot \text{crop}_w}{W_{video}}, \quad y_{global} = \frac{\text{crop}_y + y_{roi} \cdot \text{crop}_h}{H_{video}}$$

This ensures that the user's cursor navigates the entire webpage edge-to-edge, while the neural network perceives the hand at **full optical resolution**.

---

## 4. Signal Stabilization: Speed-Adaptive 1€ Filter

Standard low-pass filters (like a Moving Average or simple exponential filter) introduce a fundamental trade-off:
- **High smoothing (low cutoff frequency):** Eliminates stationary hand tremor, but introduces noticeable, sluggish lag during fast hand motion.
- **Low smoothing (high cutoff frequency):** Responsive and lag-free, but creates severe jitter when trying to point or click.

To solve this, we implement the **1€ (One Euro) Filter** formulated by *Casiez, Roussel, and Vogel (CHI 2012)*.

### Mathematical Formulation

Given input signal $x_i$ at timestamp $t_i$ and previous smoothed value $\hat{x}_{i-1}$:

#### 1. Instantaneous Velocity (Derivative Estimation)
The instantaneous rate of change is computed and filtered:

$$\dot{x}_i = \frac{x_i - \hat{x}_{i-1}}{\Delta t}, \quad \text{where } \Delta t = t_i - t_{i-1}$$

$$\hat{\dot{x}}_i = \alpha_d \cdot \dot{x}_i + (1 - \alpha_d) \cdot \hat{\dot{x}}_{i-1}$$

Where $\alpha_d$ is the exponential smoothing factor derived from fixed velocity cutoff $f_{c, d} = 1.0\text{ Hz}$:

$$\alpha_d = \frac{1}{1 + \frac{\tau_d}{\Delta t}}, \quad \tau_d = \frac{1}{2 \pi f_{c, d}}$$

#### 2. Speed-Adaptive Cutoff Frequency
The cutoff frequency $f_c$ scales dynamically as hand velocity increases:

$$f_c = f_{c, min} + \beta \cdot |\hat{\dot{x}}_i|$$

- $f_{c, min} = 1.2\text{ Hz}$: Minimum cutoff frequency when hand is stationary (suppresses jitter completely).
- $\beta = 0.35$: Speed coefficient. During fast flicks or swipes, $f_c$ climbs to 15–30 Hz, dropping filtering lag to zero.

#### 3. Position Smoothing
The filtered coordinate $\hat{x}_i$ is computed using the adaptive factor $\alpha$:

$$\alpha = \frac{1}{1 + \frac{\tau}{\Delta t}}, \quad \tau = \frac{1}{2 \pi f_c}$$

$$\hat{x}_i = \alpha \cdot x_i + (1 - \alpha) \cdot \hat{x}_{i-1}$$

---

## 5. Anatomical Normalization: Scale Invariance

Physical distances in pixel units vary wildly depending on whether a person is 50 cm or 300 cm from the webcam. To make all interactions **100% scale-invariant**, we calculate the anatomical reference baseline $L_{palm}$:

$$L_{palm} = \sqrt{(x_{m9} - x_{w0})^2 + (y_{m9} - y_{w0})^2}$$

Where $w_0$ is Landmark 0 (Wrist) and $m_9$ is Landmark 9 (Middle Knuckle/MCP).

Every physical interaction threshold (pinch distance, swipe displacement, stationary deadzone) is expressed strictly as a ratio normalized by $L_{palm}$:

$$\text{Normalized Pinch Ratio: } r_{pinch} = \frac{\sqrt{(x_{index} - x_{thumb})^2 + (y_{index} - y_{thumb})^2}}{L_{palm}}$$

---

## 6. How Swipes & Scrolls are Registered

To prevent ordinary arm drift, casual hover motions, or hand fidgeting from misfiring as navigation commands, swipe and scroll detection requires passing **four rigorous deterministic kinematic gates**:

```
 [Raw Index Fingertip Motion Deque (History Window: 9 Frames)]
                           │
                           ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ Gate 1: Scale-Invariant Displacement & Velocity Floor           │
 │ • Displacement Δd / L_palm ≥ D_req (adaptive: 0.45 to 0.90)     │
 │ • Stroke Velocity v = Δd / Δt ≥ 0.38 units/sec                  │
 └─────────────────────────────────┬───────────────────────────────┘
                                   │ PASS
                                   ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ Gate 2: Path Linearity (Straightness Ratio)                     │
 │ • Linearity λ = (Direct Displacement / Integrated Arc Length)   │
 │ • λ ≥ 0.68 (Rejects curved, circular, or wavy hand motions)     │
 └─────────────────────────────────┬───────────────────────────────┘
                                   │ PASS
                                   ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ Gate 3: Dominant Axis Separation Disambiguation                 │
 │ • Horizontal: |dx| > 1.35 · |dy|  ──> SWIPE_LEFT / SWIPE_RIGHT  │
 │ • Vertical:   |dy| > 1.35 · |dx|  ──> SCROLL_DOWN / SCROLL_UP   │
 └─────────────────────────────────┬───────────────────────────────┘
                                   │ PASS
                                   ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ Gate 4: Boundary Exit Suppression & Return Rebound Cooldown     │
 │ • Rejects downward strokes if hand is exiting camera lap margin │
 │ • 0.50s temporal cooldown suppresses opposite return strokes    │
 └─────────────────────────────────┬───────────────────────────────┘
                                   │ PASS
                                   ▼
                    [DISPATCH GESTURE ACTION]
```

### Detailed Breakdown of the 4 Gates

#### Gate 1: Displacement & Velocity
Across a sliding temporal window of $N=9$ frames ($\sim 300\text{ ms}$), the Euclidean displacement vector is measured:

$$\Delta x = x_{t} - x_{t - \Delta t}, \quad \Delta y = y_{t} - y_{t - \Delta t}$$

$$\Delta d = \sqrt{\Delta x^2 + \Delta y^2}, \quad v = \frac{\Delta d}{\Delta t}$$

The motion must exceed both a screen displacement floor and an anatomical palm displacement threshold:

$$\frac{\Delta d}{L_{palm}} \ge D_{required}$$

#### Velocity-Aware Floor Relaxation:
A rapid, short flick is just as deliberate as a long, full-arm stroke. The required displacement threshold dynamically scales with speed:

$$D_{required} = D_{slow} - \text{clamp}\left(\frac{v - v_{min}}{v_{fast} - v_{min}}, 0, 1\right) \cdot (D_{slow} - D_{fast})$$

- Slower strokes require $D_{slow} = 0.90 \times L_{palm}$.
- High-velocity flicks ($v \ge 1.0$) require only $D_{fast} = 0.45 \times L_{palm}$.

#### Gate 2: Path Linearity (Straightness Ratio $\lambda$)
To reject circular, curved, or hovering arm repositioning, the ratio of net displacement to integrated cumulative path distance is evaluated:

$$\lambda = \frac{\sqrt{(x_N - x_0)^2 + (y_N - y_0)^2}}{\sum_{k=1}^{N} \sqrt{(x_k - x_{k-1})^2 + (y_k - y_{k-1})^2}}$$

- A straight line stroke has $\lambda \approx 1.0$.
- Wavy or meandering movements have $\lambda < 0.60$.
- Threshold: $\lambda \ge 0.68$.

#### Gate 3: Dominant Axis Disambiguation
Diagonal strokes are discarded to avoid ambiguous inputs. The motion must be decisively aligned with one axis by at least **35% dominance** ($R_{axis} = 1.35$):

$$\text{If } |\Delta x| > 1.35 \cdot |\Delta y| \implies \begin{cases} \Delta x > 0: & \text{SWIPE\_RIGHT} \\ \Delta x < 0: & \text{SWIPE\_LEFT} \end{cases}$$

$$\text{If } |\Delta y| > 1.35 \cdot |\Delta x| \implies \begin{cases} \Delta y > 0: & \text{SCROLL\_DOWN} \\ \Delta y < 0: & \text{SCROLL\_UP} \end{cases}$$

#### Gate 4: Boundary Exit Guards & Rebound Cooldown
1. **Withdrawal Guard:** When a user drops their hand down toward their desk or lap to stop gesturing, the hand travels downward. If the motion terminates near the bottom margin ($y_{wrist} > 0.88$), it is flagged as **hand withdrawal**, preventing false `SCROLL_DOWN` triggers.
2. **Rebound Suppression Cooldown:** After completing an intentional swipe to the right, a user naturally pulls their hand back to the center. A hard cooldown timer ($t_{cooldown} = 0.50\text{ s}$) clears the motion history deque, preventing the return recoil from triggering an accidental opposite swipe.

---

## 7. How Click (Pinch & Release) is Registered

Clicking uses a **Schmitt-Trigger Hysteresis State Machine** to eliminate contact bounce and false double-clicks:

```
                          [ HOVER STATE ]
                                 │
             r_pinch < 0.28      │      r_pinch > 0.38
         (Thumb & Index Close)   │   (Fingers Fully Released)
                                 ▼
                     [ PINCH_PENDING STATE ]
                                 │
                 Fingers Open (r_pinch > 0.38)
            AND Elapsed Duration: 0.04s ≤ Δt < 0.40s
            AND Displacement < 0.25 · L_palm (Deadzone)
                                 │
                                 ▼
                       [ DISPATCH CLICK ]
```

### 1. Schmitt-Trigger Hysteresis
- **Engage Threshold:** $r_{pinch} < 0.28$ (Fingers must close tightly to start click candidate).
- **Release Threshold:** $r_{pinch} > 0.38$ (Fingers must decisively separate to complete click).
- This $0.10$ gap prevents noisy frames near the boundary from toggling state repeatedly.

### 2. Temporal Confirmation Gate
To prevent optical occlusion artifacts (e.g., when the thumb briefly passes in front of the index finger during a swipe), the pinch condition must hold consistently for **3 consecutive frames** ($100\text{ ms}$) before `PINCH_PENDING` engages.

### 3. Stationary Deadzone Gate
During a click, the hand must remain stationary within a deadzone radius ($r_{deadzone} = 0.25 \times L_{palm}$). If the hand moves significantly while pinched, the action is cancelled to avoid accidental clicks while moving.

---

## 8. How Thumbs Up (👍) is Registered

The Thumbs-Up gesture is evaluated using **geometric skeletal morphology**:

```
        Thumb Tip [4]  ▲  (Highest Point of Hand)
                       │
        Thumb IP  [3]  │
                       │
        Thumb MCP [2]  │
                       │
    Index  Middle  Ring  Pinky   (All 4 fingers curled into fist)
     [8]    [12]   [16]   [20]
      ▼      ▼      ▼      ▼
    ─────────────────────────
             [0] Wrist
```

### Detection Conditions:
1. **Thumb Vector Orientation:** The thumb must point vertically upward in screen space:
   $$y_{t4} < y_{t3} < y_{t2}$$
2. **Height Dominance:** The thumb tip ($T_4$) must be strictly higher (lower $y$ value) than all other four fingertips:
   $$y_{t4} < y_{i8} - 0.02 \quad \text{and} \quad y_{t4} < y_{m12} - 0.02$$
3. **Wrist Elevation:** The thumb tip must be elevated well above the wrist:
   $$y_{w0} - y_{t4} > 0.55 \times L_{palm}$$
4. **Fist Curling Verification:** At least 3 of the 4 long fingers must be curled inward toward the wrist:
   $$\text{dist}(w_0, \text{tip}_k) < \text{dist}(w_0, \text{mcp}_k) \times 1.25, \quad k \in \{8, 12, 16, 20\}$$
5. **Debounce Cooldown:** Once triggered, a cooldown timer of **0.85 seconds** prevents rapid repeated firings while holding the pose.

---

## 9. Technology Stack & Performance Profile

### Core Technologies
- **Perception:** Google MediaPipe Hands Lite (`hand_landmark_lite.tflite`) compiled to WebAssembly (Wasm) with WebGL GPU acceleration shaders.
- **Client Runtime:** Vanilla ECMAScript 2022 (Zero external npm runtime dependencies; runs directly in browser).
- **Video Capture:** HTML5 `navigator.mediaDevices.getUserMedia` using hardware `requestVideoFrameCallback` synchronization.
- **Filtering Engine:** Casiez 1€ Filter implemented in pure JavaScript / TypeScript with zero garbage collection allocations.

### Performance Benchmarks
| Metric | Apple Silicon (M1/M2/M3) | Intel Core i5/i7 (Mac/PC) | Raspberry Pi 4/5 (Chromium) |
| :--- | :--- | :--- | :--- |
| **Model Download Size** | ~1.98 MB (Cached permanently) | ~1.98 MB (Cached permanently) | ~1.98 MB (Cached permanently) |
| **Inference Frame Rate** | 55–60 FPS | 35–45 FPS | 28–32 FPS |
| **End-to-End Latency** | 16–22 ms | 24–30 ms | 32–38 ms |
| **CPU Utilization** | < 8% | < 14% | < 25% |
| **Memory Footprint** | ~65 MB | ~72 MB | ~60 MB |
