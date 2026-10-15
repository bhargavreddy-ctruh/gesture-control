/**
 * In-Browser Zero-Training Gesture Perception & Kinematic Engine
 * 
 * Runs 100% client-side in the browser using MediaPipe Hands WebAssembly/WebGL
 * and a pure JavaScript port of the Speed-Adaptive 1€ Filter + Deterministic Kinematic FSM.
 * 
 * Works on any static website or web app with zero backend/Python dependencies.
 */

class OneEuroFilter1D {
  constructor(minCutoff = 1.2, beta = 0.35, dCutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.xPrev = null;
    this.dxPrev = 0.0;
    this.tPrev = null;
  }

  static alpha(cutoff, dt) {
    if (cutoff <= 0.0 || dt <= 0.0) return 1.0;
    const tau = 1.0 / (2.0 * Math.PI * cutoff);
    return 1.0 / (1.0 + tau / dt);
  }

  reset() {
    this.xPrev = null;
    this.dxPrev = 0.0;
    this.tPrev = null;
  }

  filter(x, t) {
    if (this.tPrev === null || this.xPrev === null) {
      this.xPrev = x;
      this.tPrev = t;
      this.dxPrev = 0.0;
      return x;
    }

    const dt = t - this.tPrev;
    if (dt <= 1e-6) return this.xPrev;

    const dx = (x - this.xPrev) / dt;
    const aD = OneEuroFilter1D.alpha(this.dCutoff, dt);
    const dxHat = aD * dx + (1.0 - aD) * this.dxPrev;

    const cutoff = this.minCutoff + this.beta * Math.abs(dxHat);
    const a = OneEuroFilter1D.alpha(cutoff, dt);
    const xHat = a * x + (1.0 - a) * this.xPrev;

    this.xPrev = xHat;
    this.dxPrev = dxHat;
    this.tPrev = t;
    return xHat;
  }
}

class Point2DEuroFilter {
  constructor(minCutoff = 1.2, beta = 0.35, dCutoff = 1.0) {
    this.fx = new OneEuroFilter1D(minCutoff, beta, dCutoff);
    this.fy = new OneEuroFilter1D(minCutoff, beta, dCutoff);
  }

  reset() {
    this.fx.reset();
    this.fy.reset();
  }

  filter(x, y, t) {
    return [this.fx.filter(x, t), this.fy.filter(y, t)];
  }
}

class BrowserGestureFSM {
  constructor(options = {}) {
    this.pinchEngageThresh = options.pinchEngageThresh ?? 0.28;
    this.pinchReleaseThresh = options.pinchReleaseThresh ?? 0.38;
    this.deadzonePalmRatio = options.deadzonePalmRatio ?? 0.25;
    this.clickMinSec = options.clickMinSec ?? 0.04;
    this.clickMaxSec = options.clickMaxSec ?? 0.40;
    this.holdTriggerSec = options.holdTriggerSec ?? 0.45;

    this.swipeWindow = options.swipeWindow ?? 9;
    this.swipeMinPalmDisp = options.swipeMinPalmDisp ?? 0.90;
    this.swipeMinScreenDisp = options.swipeMinScreenDisp ?? 0.09;
    this.swipeMinVelocity = options.swipeMinVelocity ?? 0.38;
    this.swipeLinearityMin = options.swipeLinearityMin ?? 0.68;
    this.swipeAxisRatio = options.swipeAxisRatio ?? 1.35;
    this.swipeCooldownSec = options.swipeCooldownSec ?? 0.50;
    this.swipeMinPalmDispFast = options.swipeMinPalmDispFast ?? 0.45;
    this.swipeLinearityMinFast = options.swipeLinearityMinFast ?? 0.55;
    this.swipeFastVelocity = options.swipeFastVelocity ?? 1.0;
    this.swipeSpeedDecayTau = options.swipeSpeedDecayTau ?? 0.15;
    this.pinchEntryBlockSec = options.pinchEntryBlockSec ?? 0.12;
    this.pinchConfirmFrames = options.pinchConfirmFrames ?? 3;

    // Thumbs Up gesture parameters
    this.thumbsUpCooldownSec = options.thumbsUpCooldownSec ?? 0.85;
    this.lastThumbsUpTime = -100.0;
    this.thumbsUpCandidateFrames = 0;

    this.state = "IDLE";
    this.pinchStartTime = 0.0;
    this.anchorXY = [0.5, 0.5];
    this.lastSwipeTime = -100.0;
    this.swipeHistory = [];
    this.smoothedSpeed = 0.0;
    this.lastUpdateTime = 0.0;
    this.lastHistoryClearTime = -100.0;
    this.pinchCandidateFrames = 0;

    this.thumbFilter = new Point2DEuroFilter(1.2, 0.35);
    this.indexFilter = new Point2DEuroFilter(1.2, 0.35);
  }

  reset() {
    this.state = "IDLE";
    this.pinchStartTime = 0.0;
    this.swipeHistory = [];
    this.lastSwipeTime = -100.0;
    this.smoothedSpeed = 0.0;
    this.lastUpdateTime = 0.0;
    this.lastHistoryClearTime = -100.0;
    this.pinchCandidateFrames = 0;
    this.thumbsUpCandidateFrames = 0;
    this.thumbFilter.reset();
    this.indexFilter.reset();
  }

  clearSwipeHistory(now) {
    this.swipeHistory = [];
    this.lastHistoryClearTime = now;
  }

  isThumbsUpPose(lm, palmScale) {
    const w0 = lm[0];
    const t4 = lm[4];
    const t3 = lm[3];
    const t2 = lm[2];
    const i8 = lm[8];
    const m12 = lm[12];
    const r16 = lm[16];
    const p20 = lm[20];

    // Thumb pointing upward: tip is higher than IP joint and MCP
    const thumbUpright = (t4.y < t3.y - 0.015) && (t3.y < t2.y + 0.02);
    if (!thumbUpright) return false;

    // Thumb tip clearly highest point of hand (above index, middle, ring, pinky)
    const thumbAboveFingers = (t4.y < i8.y - 0.02) && (t4.y < m12.y - 0.02) && (t4.y < r16.y - 0.02);
    if (!thumbAboveFingers) return false;

    // Thumb well-extended above wrist
    const thumbExtended = (w0.y - t4.y) > palmScale * 0.55;
    if (!thumbExtended) return false;

    // The other 4 fingers curled into the palm
    const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
    const curledIndex = dist(w0, i8) < dist(w0, lm[5]) * 1.25;
    const curledMiddle = dist(w0, m12) < dist(w0, lm[9]) * 1.25;
    const curledRing = dist(w0, r16) < dist(w0, lm[13]) * 1.25;
    const curledPinky = dist(w0, p20) < dist(w0, lm[17]) * 1.25;

    const curledCount = Number(curledIndex) + Number(curledMiddle) + Number(curledRing) + Number(curledPinky);
    return curledCount >= 3;
  }

  isPointingPose(lm) {
    const w0 = lm[0];
    const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

    const indexExtended = dist(w0, lm[8]) > dist(w0, lm[5]) * 1.15;
    if (!indexExtended) return false;

    const curledMiddle = dist(w0, lm[12]) < dist(w0, lm[9]) * 1.25;
    const curledRing = dist(w0, lm[16]) < dist(w0, lm[13]) * 1.25;
    const curledPinky = dist(w0, lm[20]) < dist(w0, lm[17]) * 1.25;

    return (Number(curledMiddle) + Number(curledRing) + Number(curledPinky)) >= 2;
  }

  update(landmarks, now) {
    const w0 = landmarks[0];
    const t4 = landmarks[4];
    const i8 = landmarks[8];
    const m9 = landmarks[9];

    const wz = w0.z || 0.0;
    const tz = t4.z || 0.0;
    const iz = i8.z || 0.0;
    const mz = m9.z || 0.0;

    // Filter raw fingertips
    const [tx, ty] = this.thumbFilter.filter(t4.x, t4.y, now);
    const [ix, iy] = this.indexFilter.filter(i8.x, i8.y, now);

    // Mirrored screen coordinates for intuitive interaction
    const screenIx = 1.0 - ix;
    const screenIy = iy;
    const cursorXY = [screenIx, screenIy];

    // Palm scale normalization
    const palmScale2D = Math.hypot(m9.x - w0.x, m9.y - w0.y);
    const palmScale3D = Math.sqrt((m9.x - w0.x) ** 2 + (m9.y - w0.y) ** 2 + (mz - wz) ** 2);
    const palmScale = Math.max(palmScale2D, 1e-4);

    // Euclidean pinch distance
    const pinchDist2D = Math.hypot(ix - tx, iy - ty);
    const pinchRatio = pinchDist2D / palmScale;

    // Depth coplanar test with distance adaptation (distant hands have higher depth noise)
    const isDistant = palmScale < 0.07;
    const depthDisparity = Math.abs(iz - tz) / palmScale;
    const maxDepthDisparity = isDistant ? 0.65 : 0.38;
    const depthCoplanar = depthDisparity <= maxDepthDisparity;

    // Edge guards (in original camera frame)
    const borderX = 0.05;
    const borderYTop = 0.05;
    const borderYBottom = 0.10;
    const isNearEdge = (
      w0.x < borderX || w0.x > (1.0 - borderX) ||
      m9.x < borderX || m9.x > (1.0 - borderX) ||
      w0.y < borderYTop || w0.y > (1.0 - borderYBottom) ||
      m9.y < borderYTop || m9.y > (1.0 - borderYBottom)
    );

    // Minimum physical palm size: lowered from 0.045 to 0.015 so users can stand 2-4 meters away
    const palmSizeOk = palmScale >= 0.015;
    const tiltRatio = palmScale2D / Math.max(1e-4, palmScale3D);
    const minTilt = isDistant ? 0.20 : 0.32;
    const tiltOk = tiltRatio >= minTilt;
    const palmValid = (!isNearEdge) && palmSizeOk && tiltOk;

    // Distance-adaptive pinch engage & release thresholds
    const activeEngageThresh = isDistant
      ? Math.min(0.36, this.pinchEngageThresh + (0.07 - palmScale) * 1.5)
      : this.pinchEngageThresh;
    const activeReleaseThresh = isDistant
      ? Math.min(0.48, this.pinchReleaseThresh + (0.07 - palmScale) * 1.5)
      : this.pinchReleaseThresh;

    // Pinch geometric gate
    const distT4M9 = Math.hypot(t4.x - m9.x, t4.y - m9.y);
    const distT4I8 = Math.hypot(t4.x - i8.x, t4.y - i8.y);
    const thumbNotTucked = (distT4I8 < distT4M9 * 1.35);
    const pinchValidToEngage = (
      palmValid &&
      (pinchRatio < activeEngageThresh) &&
      depthCoplanar &&
      thumbNotTucked
    );

    let action = null;
    let dwellProgress = 0.0;
    const pointing = this.isPointingPose(landmarks);

    // Mirrored index tip for swipe direction tracking
    const rawScreenIx = 1.0 - i8.x;
    const rawScreenIy = i8.y;

    if (isNearEdge) {
      this.clearSwipeHistory(now);
    } else if (!pointing) {
      this.swipeHistory = [];
    } else {
      this.swipeHistory.push([rawScreenIx, rawScreenIy, now]);
      if (this.swipeHistory.length > this.swipeWindow) {
        this.swipeHistory.shift();
      }
    }

    // Diagnostics
    let currentSpeed = 0.0;
    if (this.swipeHistory.length >= 2) {
      const [x0, y0, t0] = this.swipeHistory[0];
      const [x1, y1, t1] = this.swipeHistory[this.swipeHistory.length - 1];
      const dt = Math.max(1e-4, t1 - t0);
      const netD = Math.hypot(x1 - x0, y1 - y0);
      currentSpeed = netD / dt;
    }

    const dtFrame = this.lastUpdateTime > 0.0 ? Math.max(1e-4, now - this.lastUpdateTime) : (1.0 / 30.0);
    this.lastUpdateTime = now;
    const decay = Math.exp(-dtFrame / this.swipeSpeedDecayTau);
    if (this.swipeHistory.length >= 2) {
      this.smoothedSpeed = Math.max(currentSpeed, this.smoothedSpeed * decay);
    } else {
      this.smoothedSpeed *= decay;
    }

    // --- GESTURE EVALUATION & STATE MACHINE ---
    // 1. Check for Thumbs Up gesture
    const isThumbsUp = this.isThumbsUpPose(landmarks, palmScale);
    if (isThumbsUp) {
      this.thumbsUpCandidateFrames++;
      if (this.thumbsUpCandidateFrames >= 3 && (now - this.lastThumbsUpTime) > this.thumbsUpCooldownSec) {
        this.lastThumbsUpTime = now;
        action = "THUMBS_UP";
        this.state = "THUMBS_UP";
        this.clearSwipeHistory(now);
        this.pinchCandidateFrames = 0;
        return {
          state: this.state,
          action,
          cursor: { x: cursorXY[0], y: cursorXY[1] },
          pinchRatio,
          palmScale,
          dwellProgress: 0.0,
          currentSpeed,
          palmValid
        };
      }
    } else {
      this.thumbsUpCandidateFrames = 0;
    }

    // 2. Active states: HOVER (cursor tracking + swipes), PINCH_PENDING (clicking)
    if (this.state === "IDLE" || this.state === "HOVER" || this.state === "THUMBS_UP") {
      this.state = "HOVER";

      // Directional swipes (Left / Right / Up / Down)
      if (pointing) {
        action = this.evaluateSwipe(palmScale, now, isNearEdge, 1.0 - m9.x, m9.y, 1.0 - w0.x, w0.y);
      }

      // Pinch engage
      if (action === null && pinchValidToEngage) {
        const recentlyCleared = (now - this.lastHistoryClearTime) < this.pinchEntryBlockSec;
        const maxSpeed = Math.min(this.swipeMinVelocity * 0.75, palmScale * 3.8);
        if (this.smoothedSpeed < maxSpeed && !recentlyCleared) {
          this.pinchCandidateFrames += 1;
          if (this.pinchCandidateFrames >= this.pinchConfirmFrames) {
            this.state = "PINCH_PENDING";
            this.pinchStartTime = now;
            this.anchorXY = [cursorXY[0], cursorXY[1]];
          }
        } else {
          this.pinchCandidateFrames = 0;
        }
      } else {
        this.pinchCandidateFrames = 0;
      }
    } else if (this.state === "PINCH_PENDING") {
      const elapsed = now - this.pinchStartTime;
      const dispRatio = Math.hypot(
        cursorXY[0] - this.anchorXY[0],
        cursorXY[1] - this.anchorXY[1]
      ) / palmScale;

      // Crisp click upon opening fingers (Hold and Drag completely removed)
      if (pinchRatio > activeReleaseThresh) {
        if (elapsed >= this.clickMinSec && dispRatio < this.deadzonePalmRatio * 1.5) {
          action = "CLICK";
        }
        this.state = "HOVER";
      } else if (isNearEdge || !palmSizeOk) {
        this.state = "HOVER";
      }
    }

    return {
      state: this.state,
      action,
      cursor: { x: cursorXY[0], y: cursorXY[1] },
      pinchRatio,
      palmScale,
      dwellProgress,
      currentSpeed,
      palmValid
    };
  }

  evaluateSwipe(palmScale, now, isNearEdge, screenM9x, screenM9y, screenW0x, screenW0y) {
    if (isNearEdge) {
      this.clearSwipeHistory(now);
      return null;
    }
    const history = this.swipeHistory;
    if (history.length < 4) return null;
    if ((now - this.lastSwipeTime) < this.swipeCooldownSec) return null;

    const [x1, y1, t1] = history[history.length - 1];
    let bestCandidate = null;
    let maxStrength = 0.0;

    const minStart = Math.max(0, history.length - this.swipeWindow);
    const maxStart = history.length - 3;

    for (let i = minStart; i < maxStart; i++) {
      const [x0, y0, t0] = history[i];
      const dt = t1 - t0;
      if (dt <= 1e-4) continue;

      const dx = x1 - x0;
      const dy = y1 - y0;
      const netDisp = Math.hypot(dx, dy);

      // Distance-adaptive screen displacement & velocity floors
      // When far away, a physical hand stroke covers less normalized screen distance
      const minScreenDisp = Math.min(this.swipeMinScreenDisp, palmScale * 0.70);
      if (netDisp < minScreenDisp) continue;

      const velocity = netDisp / dt;
      const minVelocity = Math.min(this.swipeMinVelocity, palmScale * 3.8);
      if (velocity < minVelocity) continue;

      const fastVelocity = Math.min(this.swipeFastVelocity, minVelocity * 2.5);
      const frac = Math.min(1.0, Math.max(0.0, (velocity - minVelocity) / Math.max(1e-4, fastVelocity - minVelocity)));
      const requiredDisp = this.swipeMinPalmDisp - frac * (this.swipeMinPalmDisp - this.swipeMinPalmDispFast);
      const requiredLinearity = this.swipeLinearityMin - frac * (this.swipeLinearityMin - this.swipeLinearityMinFast);

      if ((netDisp / palmScale) < requiredDisp) continue;

      // Linearity
      let pathLen = 0.0;
      for (let k = i + 1; k < history.length; k++) {
        pathLen += Math.hypot(history[k][0] - history[k - 1][0], history[k][1] - history[k - 1][1]);
      }
      if (pathLen <= 1e-5 || (netDisp / pathLen) < requiredLinearity) continue;

      // Axis disambiguation
      let swipeCmd = null;
      if (Math.abs(dx) > this.swipeAxisRatio * Math.abs(dy)) {
        swipeCmd = dx > 0 ? "SWIPE_RIGHT" : "SWIPE_LEFT";
      } else if (Math.abs(dy) > this.swipeAxisRatio * Math.abs(dx)) {
        swipeCmd = dy > 0 ? "SWIPE_DOWN" : "SWIPE_UP";
      }
      if (!swipeCmd) continue;

      // Boundary exit guards
      if (swipeCmd === "SWIPE_DOWN" && (screenM9y > 0.88 || screenW0y > 0.90)) {
        this.clearSwipeHistory(now);
        return null;
      }
      if (swipeCmd === "SWIPE_UP" && (screenM9y < 0.08 || screenW0y < 0.06)) {
        this.clearSwipeHistory(now);
        return null;
      }
      if (swipeCmd === "SWIPE_LEFT" && (screenM9x < 0.08 || screenW0x < 0.06)) {
        this.clearSwipeHistory(now);
        return null;
      }
      if (swipeCmd === "SWIPE_RIGHT" && (screenM9x > 0.92 || screenW0x > 0.94)) {
        this.clearSwipeHistory(now);
        return null;
      }

      const strength = (netDisp / palmScale) * velocity;
      if (strength > maxStrength) {
        maxStrength = strength;
        bestCandidate = swipeCmd;
      }
    }

    if (bestCandidate !== null) {
      this.lastSwipeTime = now;
      this.clearSwipeHistory(now);
      return bestCandidate;
    }
    return null;
  }
}

/**
 * High-level Touchless Gesture Controller for Web
 * Features Dynamic Region-of-Interest (ROI) Auto-Zoom & Hand-Lock:
 * When the user is far away, the camera automatically zooms in and locks onto
 * the hand so the hand appears at normal size to MediaPipe without losing tracking.
 */
class TouchlessWebController {
  constructor(options = {}) {
    this.fsm = new BrowserGestureFSM(options);
    this.videoElement = null;
    this.canvasElement = options.previewCanvas || null;
    this.canvasCtx = this.canvasElement ? this.canvasElement.getContext("2d") : null;
    this.hands = null;
    this.isRunning = false;
    this.onUpdate = options.onUpdate || null;
    this.onGesture = options.onGesture || null;
    this.onStatus = options.onStatus || null;
    this.drawSkeleton = options.drawSkeleton ?? true;
    this.lastFrameTime = performance.now();
    this.fps = 0;

    // --- Dynamic ROI Auto-Zoom & Hand-Lock System ---
    this.autoZoom = options.autoZoom ?? true;
    this.zoomLevel = 1.0;
    this.targetZoom = 1.0;
    this.centerX = 0.5;
    this.centerY = 0.5;
    this.targetCenterX = 0.5;
    this.targetCenterY = 0.5;
    this.isLocked = false;
    this.lostFrames = 0;
    this.currentCrop = { x: 0, y: 0, w: 640, h: 480, vw: 640, vh: 480 };

    // Offscreen canvas for zoomed ROI inference
    this.procCanvas = document.createElement("canvas");
    this.procCanvas.width = 640;
    this.procCanvas.height = 480;
    this.procCtx = this.procCanvas.getContext("2d", { willReadFrequently: true });
  }

  setAutoZoom(enabled) {
    this.autoZoom = enabled;
    if (!enabled) {
      this.targetZoom = 1.0;
      this.targetCenterX = 0.5;
      this.targetCenterY = 0.5;
    }
  }

  async loadMediaPipe() {
    this.notifyStatus("loading", "Loading MediaPipe Hands WebAssembly...");

    if (!window.Hands) {
      await this.loadScript("https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js");
      await this.loadScript("https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js");
    }

    this.notifyStatus("loading", "Initializing Hands model...");
    this.hands = new window.Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });

    this.hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 0, // Lite model for maximum frame rates
      minDetectionConfidence: 0.45,
      minTrackingConfidence: 0.45
    });

    this.hands.onResults((results) => this.handleResults(results));
    this.notifyStatus("ready", "Model ready");
  }

  loadScript(src) {
    return new Promise((resolve, reject) => {
      const existing = document.querySelector(`script[src="${src}"]`);
      if (existing) return resolve();
      const s = document.createElement("script");
      s.src = src;
      s.crossOrigin = "anonymous";
      s.onload = () => resolve();
      s.onerror = (e) => reject(new Error(`Failed to load ${src}`));
      document.head.appendChild(s);
    });
  }

  notifyStatus(status, message) {
    if (this.onStatus) this.onStatus({ status, message });
  }

  async start(videoEl) {
    if (this.isRunning) return;
    this.videoElement = videoEl || document.createElement("video");
    this.videoElement.playsInline = true;
    this.videoElement.muted = true;
    this.videoElement.autoplay = true;

    try {
      if (!this.hands) {
        await this.loadMediaPipe();
      }

      this.notifyStatus("camera", "Requesting camera access...");
      // High-res stream: ideal 720p without restrictive min constraints
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user"
        },
        audio: false
      });

      this.videoElement.srcObject = stream;
      await this.videoElement.play();

      this.isRunning = true;
      this.notifyStatus("active", "Gesture control active");

      // Robust frame loop without double-calling getUserMedia
      let isBusy = false;
      const processLoop = async () => {
        if (!this.isRunning) return;

        if (!isBusy && this.hands && this.videoElement.readyState >= 2) {
          isBusy = true;
          try {
            await this.processNextFrame();
          } catch (e) {
            console.error("Frame processing error:", e);
          } finally {
            isBusy = false;
          }
        }

        if (this.videoElement && "requestVideoFrameCallback" in this.videoElement) {
          this.videoElement.requestVideoFrameCallback(processLoop);
        } else {
          requestAnimationFrame(processLoop);
        }
      };

      if (this.videoElement && "requestVideoFrameCallback" in this.videoElement) {
        this.videoElement.requestVideoFrameCallback(processLoop);
      } else {
        requestAnimationFrame(processLoop);
      }
    } catch (err) {
      this.notifyStatus("error", err.message || "Camera permission denied");
      throw err;
    }
  }

  async processNextFrame() {
    const vw = this.videoElement.videoWidth || 640;
    const vh = this.videoElement.videoHeight || 480;

    // Smooth exponential tracker for Auto-Zoom & Hand-Lock
    const zoomSmoothing = 0.12;
    const centerSmoothing = 0.16;
    this.zoomLevel += (this.targetZoom - this.zoomLevel) * zoomSmoothing;
    this.centerX += (this.targetCenterX - this.centerX) * centerSmoothing;
    this.centerY += (this.targetCenterY - this.centerY) * centerSmoothing;

    // Calculate crop window in video coordinates
    const cropW = vw / this.zoomLevel;
    const cropH = vh / this.zoomLevel;
    let cropX = (this.centerX * vw) - (cropW / 2);
    let cropY = (this.centerY * vh) - (cropH / 2);

    cropX = Math.max(0, Math.min(vw - cropW, cropX));
    cropY = Math.max(0, Math.min(vh - cropH, cropY));

    this.currentCrop = { x: cropX, y: cropY, w: cropW, h: cropH, vw, vh };

    // Render cropped ROI directly into 640x480 processing canvas
    this.procCtx.drawImage(
      this.videoElement,
      cropX, cropY, cropW, cropH,
      0, 0, 640, 480
    );

    // Send high-resolution zoomed hand into MediaPipe
    await this.hands.send({ image: this.procCanvas });
  }

  stop() {
    this.isRunning = false;
    if (this.videoElement && this.videoElement.srcObject) {
      const tracks = this.videoElement.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      this.videoElement.srcObject = null;
    }
    this.fsm.reset();
    this.targetZoom = 1.0;
    this.zoomLevel = 1.0;
    this.isLocked = false;
    this.notifyStatus("stopped", "Engine stopped");
  }

  handleResults(results) {
    const nowSec = performance.now() / 1000.0;

    // Calculate FPS
    const nowMs = performance.now();
    const frameDelta = nowMs - this.lastFrameTime;
    this.lastFrameTime = nowMs;
    if (frameDelta > 0) {
      this.fps = Math.round(1000 / frameDelta);
    }

    let event = null;
    const crop = this.currentCrop;

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      this.lostFrames = 0;
      const roiLandmarks = results.multiHandLandmarks[0];

      // Map landmarks from zoomed ROI canvas [0, 1] back into full camera space [0, 1]
      const fullLandmarks = roiLandmarks.map(p => ({
        x: (crop.x + p.x * crop.w) / crop.vw,
        y: (crop.y + p.y * crop.h) / crop.vh,
        z: p.z || 0.0
      }));

      // Update Auto-Zoom & Hand-Lock tracking target
      const handCenterX = (fullLandmarks[0].x + fullLandmarks[9].x) / 2;
      const handCenterY = (fullLandmarks[0].y + fullLandmarks[9].y) / 2;
      this.targetCenterX = handCenterX;
      this.targetCenterY = handCenterY;

      // Calculate palm scale in full frame coordinates
      const palmScaleFull = Math.hypot(
        fullLandmarks[9].x - fullLandmarks[0].x,
        fullLandmarks[9].y - fullLandmarks[0].y
      );

      if (this.autoZoom) {
        // Target palm scale ~ 0.18 of frame for optimal perception
        const desiredPalm = 0.18;
        const optimalZoom = desiredPalm / Math.max(palmScaleFull, 0.035);
        this.targetZoom = Math.min(3.2, Math.max(1.0, optimalZoom));
      } else {
        this.targetZoom = 1.0;
      }

      this.isLocked = this.zoomLevel > 1.25;

      // Update Kinematic FSM with scale-stabilized coordinates
      event = this.fsm.update(fullLandmarks, nowSec);

      if (this.canvasElement && this.canvasCtx && this.drawSkeleton) {
        this.renderPreview(this.procCanvas, roiLandmarks, event);
      }
    } else {
      this.lostFrames++;
      // If hand lost for > 15 frames (~0.5s), zoom back out to wide-angle to search
      if (this.lostFrames > 15) {
        this.targetZoom = 1.0;
        this.targetCenterX = 0.5;
        this.targetCenterY = 0.5;
        this.isLocked = false;
      }

      this.fsm.reset();
      event = {
        state: "IDLE",
        action: null,
        cursor: { x: 0.5, y: 0.5 },
        pinchRatio: 1.0,
        palmScale: 0.1,
        dwellProgress: 0.0,
        currentSpeed: 0.0,
        palmValid: false
      };
      if (this.canvasElement && this.canvasCtx) {
        this.renderPreview(this.procCanvas, null, event);
      }
    }

    event.fps = this.fps;
    event.zoom = this.zoomLevel;
    event.isLocked = this.isLocked;

    if (this.onUpdate) {
      this.onUpdate(event);
    }
    if (event.action && this.onGesture) {
      this.onGesture(event);
    }
  }

  renderPreview(image, landmarks, event) {
    const ctx = this.canvasCtx;
    const canvas = this.canvasElement;
    if (!ctx || !canvas) return;

    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw mirrored zoomed video frame
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    if (image) {
      ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    }
    ctx.restore();

    // Subtle dark vignette
    ctx.fillStyle = "rgba(10, 12, 18, 0.25)";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (landmarks) {
      // Connect skeletal bones
      const connections = [
        [0, 1], [1, 2], [2, 3], [3, 4],       // thumb
        [0, 5], [5, 6], [6, 7], [7, 8],       // index
        [0, 9], [9, 10], [10, 11], [11, 12],  // middle
        [0, 13], [13, 14], [14, 15], [15, 16],// ring
        [0, 17], [17, 18], [18, 19], [19, 20],// pinky
        [5, 9], [9, 13], [13, 17]              // palm
      ];

      ctx.lineWidth = 2.5;
      ctx.strokeStyle = "rgba(0, 229, 255, 0.75)";
      for (const [a, b] of connections) {
        const ptA = landmarks[a];
        const ptB = landmarks[b];
        ctx.beginPath();
        ctx.moveTo((1.0 - ptA.x) * canvas.width, ptA.y * canvas.height);
        ctx.lineTo((1.0 - ptB.x) * canvas.width, ptB.y * canvas.height);
        ctx.stroke();
      }

      // Draw keypoints
      for (let i = 0; i < landmarks.length; i++) {
        const pt = landmarks[i];
        const px = (1.0 - pt.x) * canvas.width;
        const py = pt.y * canvas.height;
        ctx.beginPath();
        if (i === 8) {
          ctx.arc(px, py, 6, 0, 2 * Math.PI);
          ctx.fillStyle = "#00e5ff";
        } else if (i === 4) {
          ctx.arc(px, py, 5, 0, 2 * Math.PI);
          ctx.fillStyle = event.pinchRatio < 0.28 ? "#ff9100" : "#00ff88";
        } else {
          ctx.arc(px, py, 3, 0, 2 * Math.PI);
          ctx.fillStyle = "rgba(255, 255, 255, 0.85)";
        }
        ctx.fill();
      }
    }

    // Top status HUD badge
    ctx.fillStyle = "rgba(0, 0, 0, 0.75)";
    ctx.fillRect(8, 8, 150, 24);
    
    if (this.isLocked) {
      ctx.fillStyle = "#00ff88";
      ctx.font = "bold 11px sans-serif";
      ctx.fillText(`🔒 LOCKED (${this.zoomLevel.toFixed(1)}x) | ${this.fps} FPS`, 14, 24);
    } else {
      ctx.fillStyle = event.palmValid ? "#00e5ff" : "#ff9100";
      ctx.font = "bold 11px sans-serif";
      ctx.fillText(`🔍 WIDE (1.0x) | ${this.fps} FPS`, 14, 24);
    }
  }
}

// Global window attachment for easy CDN / Vanilla usage
if (typeof window !== "undefined") {
  window.OneEuroFilter1D = OneEuroFilter1D;
  window.Point2DEuroFilter = Point2DEuroFilter;
  window.BrowserGestureFSM = BrowserGestureFSM;
  window.TouchlessWebController = TouchlessWebController;
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    OneEuroFilter1D,
    Point2DEuroFilter,
    BrowserGestureFSM,
    TouchlessWebController
  };
}
