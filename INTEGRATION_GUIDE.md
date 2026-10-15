# Website Integration Guide: Touchless Gesture Control

This guide provides end-to-end instructions for integrating the zero-training touchless gesture engine into any website or web application (Vanilla HTML/JS, React, Next.js, Vue, Angular, or Shopify).

---

## 🌟 How It Works

- **100% Client-Side:** Everything runs locally inside the visitor's browser using MediaPipe WebAssembly (Wasm) and WebGL.
- **Zero Python / Zero Server:** No backend dependencies or installations are needed for end users.
- **Ultra Lightweight:** The hand landmark model is only **~2 MB** and is automatically cached in the browser for instant reloads.
- **100% Private:** Camera frames are processed strictly in GPU memory on the user's device and are never transmitted to any server.

---

## 📋 Table of Contents
1. [Prerequisites & Requirements](#prerequisites--requirements)
2. [Method 1: Vanilla HTML & JavaScript (Any Website)](#method-1-vanilla-html--javascript-any-website)
3. [Method 2: React & Next.js Integration](#method-2-react--nextjs-integration)
4. [Supported Gestures & Event Payloads](#supported-gestures--event-payloads)
5. [Tuning Sensitivity & Thresholds](#tuning-sensitivity--thresholds)
6. [Production Deployment Best Practices](#production-deployment-best-practices)

---

## Prerequisites & Requirements

- **HTTPS Protocol:** Browsers require a secure context (`https://` or `http://localhost`) to allow camera access via `navigator.mediaDevices.getUserMedia`.
- **Supported Browsers:** Chrome, Safari, Edge, Firefox, Brave, and Chromium-based mobile browsers.

---

## Method 1: Vanilla HTML & JavaScript (Any Website)

### Step 1: Copy `gesture-engine.js`
Copy `web_demo/gesture-engine.js` into your website's static public assets folder (e.g., `/js/gesture-engine.js` or `/public/gesture-engine.js`).

### Step 2: Add Scripts and Cursor Element
Add the following scripts to the `<head>` or before `</body>` of your HTML page:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>My Gesture-Controlled Website</title>

  <!-- 1. Load MediaPipe dependencies from CDN -->
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js" crossorigin="anonymous"></script>

  <!-- 2. Load the Gesture Engine -->
  <script src="./js/gesture-engine.js"></script>

  <style>
    /* Styling for the virtual floating hand cursor */
    #gesture-cursor {
      position: fixed;
      width: 26px;
      height: 26px;
      border-radius: 50%;
      border: 2px solid #00e5ff;
      background: radial-gradient(circle, rgba(0, 229, 255, 0.6) 0%, transparent 70%);
      pointer-events: none; /* Allows clicks to pass through to underlying elements */
      z-index: 999999;
      transform: translate(-50%, -50%);
      display: none;
      box-shadow: 0 0 16px rgba(0, 229, 255, 0.4);
      transition: width 0.12s, height 0.12s, background-color 0.12s;
    }

    #gesture-cursor.pinching {
      width: 18px;
      height: 18px;
      border-color: #ff9100;
      background: #ff9100;
      box-shadow: 0 0 20px rgba(255, 145, 0, 0.8);
    }
  </style>
</head>
<body>

  <!-- Virtual Floating Cursor -->
  <div id="gesture-cursor"></div>

  <!-- Button to prompt the user for camera access -->
  <button id="enable-gestures-btn" onclick="initGestures()">Enable Touchless Gestures</button>

  <!-- Optional preview canvas to show hand tracking / skeleton -->
  <canvas id="gesture-preview" width="240" height="180" style="position:fixed; bottom:16px; right:16px; border-radius:12px; z-index:9999; display:none;"></canvas>

  <script>
    let controller = null;
    const cursor = document.getElementById("gesture-cursor");
    const previewCanvas = document.getElementById("gesture-preview");

    async function initGestures() {
      const btn = document.getElementById("enable-gestures-btn");
      btn.textContent = "Loading Model...";
      btn.disabled = true;

      controller = new TouchlessWebController({
        previewCanvas: previewCanvas,
        drawSkeleton: true, // Set to false if you don't want the visual skeleton
        onStatus: ({ status, message }) => {
          console.log(`[Gesture Engine] ${status}: ${message}`);
          if (status === "active") {
            btn.style.display = "none";
            previewCanvas.style.display = "block";
          }
        },
        onUpdate: (event) => {
          // Continuous cursor tracking (smoothed with 1€ Filter)
          const screenX = event.cursor.x * window.innerWidth;
          const screenY = event.cursor.y * window.innerHeight;
          
          cursor.style.left = `${screenX}px`;
          cursor.style.top = `${screenY}px`;
          cursor.style.display = "block";
          cursor.classList.toggle("pinching", event.pinchRatio < 0.28);
        },
        onGesture: (event) => {
          console.log("Triggered Gesture:", event.action);
          
          if (event.action === "CLICK") {
            // Find whatever element is under the cursor and click it
            const screenX = event.cursor.x * window.innerWidth;
            const screenY = event.cursor.y * window.innerHeight;
            const target = document.elementFromPoint(screenX, screenY);
            if (target && typeof target.click === "function") {
              target.click();
            }
          } else if (event.action === "SWIPE_LEFT") {
            // Next item / Next slide / Scroll carousel right
            if (window.nextSlide) window.nextSlide();
          } else if (event.action === "SWIPE_RIGHT") {
            // Previous item / Prev slide / Scroll carousel left
            if (window.prevSlide) window.prevSlide();
          } else if (event.action === "SWIPE_DOWN") {
            window.scrollBy({ top: 300, behavior: "smooth" });
          } else if (event.action === "SWIPE_UP") {
            window.scrollBy({ top: -300, behavior: "smooth" });
          }
        }
      });

      await controller.start();
    }
  </script>
</body>
</html>
```

---

## Method 2: React & Next.js Integration

### Step 1: Copy Files
1. Copy `web_demo/gesture-engine.js` into your `public/` directory (e.g. `/public/gesture-engine.js`).
2. Copy `web_demo/useClientGesture.ts` into your hooks or components folder (e.g. `src/hooks/useClientGesture.ts`).

### Step 2: Use the Hook in Your Component

```tsx
"use client"; // For Next.js App Router

import React, { useRef } from "react";
import { useClientGesture } from "@/hooks/useClientGesture";

export function TouchlessProductViewer() {
  const previewCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const { isActive, statusMessage, cursor, state, isPinching, start, stop } = useClientGesture({
    previewCanvas: previewCanvasRef.current,
    onSwipeLeft: () => {
      console.log("Swiped Left: Show Next Image");
    },
    onSwipeRight: () => {
      console.log("Swiped Right: Show Previous Image");
    },
    onSwipeDown: () => {
      console.log("Swiped Down: Next Category");
    },
    onSwipeUp: () => {
      console.log("Swiped Up: Previous Category");
    },
    onClick: (coords) => {
      const screenX = coords.x * window.innerWidth;
      const screenY = coords.y * window.innerHeight;
      const target = document.elementFromPoint(screenX, screenY) as HTMLElement;
      if (target && typeof target.click === "function") {
        target.click();
      }
    },
    onDragMove: (coords) => {
      console.log("Pinch & Drag active at:", coords.x, coords.y);
    }
  });

  return (
    <div className="relative min-h-screen">
      {/* Virtual Hand Cursor */}
      {isActive && (
        <div
          style={{
            position: "fixed",
            left: `${cursor.x * 100}vw`,
            top: `${cursor.y * 100}vh`,
            width: isPinching ? 18 : 28,
            height: isPinching ? 18 : 28,
            borderRadius: "50%",
            border: `2px solid ${isPinching ? "#ff9100" : "#00e5ff"}`,
            backgroundColor: isPinching ? "#ff9100" : "rgba(0, 229, 255, 0.4)",
            transform: "translate(-50%, -50%)",
            pointerEvents: "none",
            zIndex: 9999,
            transition: "width 0.1s, height 0.1s",
            boxShadow: "0 0 16px rgba(0, 229, 255, 0.5)",
          }}
        />
      )}

      {/* Control Bar */}
      <div className="p-4 flex items-center justify-between bg-gray-900 text-white">
        <div>
          <span className="font-bold">Status:</span> {statusMessage} ({state})
        </div>
        {!isActive ? (
          <button
            onClick={start}
            className="px-4 py-2 bg-cyan-400 text-black font-semibold rounded-lg hover:bg-cyan-300 transition"
          >
            Start Gestures
          </button>
        ) : (
          <button
            onClick={stop}
            className="px-4 py-2 bg-red-500 text-white font-semibold rounded-lg hover:bg-red-600 transition"
          >
            Stop
          </button>
        )}
      </div>

      {/* Optional PiP Skeleton Preview */}
      <canvas
        ref={previewCanvasRef}
        width={320}
        height={240}
        className={`fixed bottom-4 right-4 rounded-xl border border-white/20 shadow-xl ${
          isActive ? "block" : "hidden"
        }`}
      />
    </div>
  );
}
```

---

## Supported Gestures & Event Payloads

| Gesture Action | Trigger Mechanism | Typical Website Mapping |
| :--- | :--- | :--- |
| **`CURSOR`** | Moving index fingertip across field of view | Smooth, stabilized virtual cursor (1€ Filter) |
| **`CLICK`** | Crisp index-thumb pinch & release | Click active button, garment, or link |
| **`SWIPE_LEFT`** | Rapid horizontal stroke to user's left | Next carousel slide / next product |
| **`SWIPE_RIGHT`** | Rapid horizontal stroke to user's right | Previous carousel slide / prev product |
| **`SWIPE_DOWN`** | Deliberate downward stroke | Scroll page down / next category |
| **`SWIPE_UP`** | Deliberate upward stroke | Scroll page up / previous category |
| **`THUMBS_UP`** | Thumb upright with 4 fingers curled into fist (👍) | Like garment, quick confirm, add to bag |

---

## Tuning Sensitivity & Thresholds

When initializing `TouchlessWebController`, you can pass custom options to tune sensitivity for your specific user interface:

```javascript
const controller = new TouchlessWebController({
  // Pinch Sensitivity
  pinchEngageThresh: 0.28,   // Lower = requires tighter pinch; Higher = easier to pinch
  pinchReleaseThresh: 0.38,  // Hysteresis release ceiling (prevents click chatter)
  clickMinSec: 0.04,         // Minimum pinch duration to register as click (seconds)
  clickMaxSec: 0.40,         // Maximum duration for click before transitioning to Hold/Drag

  // Swipe Sensitivity
  swipeMinVelocity: 0.38,    // Minimum hand stroke velocity
  swipeCooldownSec: 0.50,    // Cooldown interval between consecutive swipes
  swipeMinScreenDisp: 0.09,  // Minimum stroke distance in screen units
  swipeAxisRatio: 1.35,      // Dominant axis separation ratio (horizontal vs vertical)
});
```

---

## Production Deployment Best Practices

1. **Host on Any Static Hosting Platform:**
   - The entire frontend can be hosted on **Vercel, Netlify, Cloudflare Pages, AWS S3/CloudFront, or GitHub Pages**.
   - No Docker containers, WebSockets, or GPU instances required on your server.

2. **Camera Permission UX:**
   - Always prompt the user with a clean, branded modal or button before calling `navigator.mediaDevices.getUserMedia`.
   - Explain that camera processing is **100% on-device and private**.

3. **Performance & Battery Efficiency:**
   - MediaPipe Hands Lite (`modelComplexity: 0`) consumes minimal CPU/GPU overhead.
   - Automatically call `controller.stop()` when the user switches tabs (`document.hidden` listener) or closes the try-on modal.

4. **Kiosk / Smart Mirror Mode (Dual Mode):**
   - For standalone web visitors, use **Browser Direct Mode** (default).
   - If deploying a physical Raspberry Pi / Apple Silicon smart mirror, you can seamlessly connect to the local Python WebSocket daemon (`ws://127.0.0.1:8765`) using `web_demo/useGestureBridge.ts`.
