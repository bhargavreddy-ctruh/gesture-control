/**
 * React / Next.js Client-Side In-Browser Gesture Hook
 * 
 * Runs 100% in the user's browser using MediaPipe Hands WebAssembly/WebGL.
 * No Python backend or local WebSocket required!
 * 
 * Example:
 * ```tsx
 * import { useClientGesture } from "./useClientGesture";
 * 
 * export function VirtualTryOn() {
 *   const { isReady, isActive, cursor, state, start, stop } = useClientGesture({
 *     onSwipeLeft: () => console.log("Swipe Left"),
 *     onSwipeRight: () => console.log("Swipe Right"),
 *     onClick: (c) => console.log("Click at", c),
 *   });
 * 
 *   return (
 *     <div>
 *       <button onClick={start}>Enable Camera Gestures</button>
 *     </div>
 *   );
 * }
 * ```
 */

import { useState, useEffect, useRef, useCallback } from "react";

export interface GesturePoint {
  x: number;
  y: number;
}

export interface ClientGestureOptions {
  autoStart?: boolean;
  previewCanvas?: HTMLCanvasElement | null;
  autoZoom?: boolean;
  onSwipeLeft?: () => void;
  onSwipeRight?: () => void;
  onSwipeUp?: () => void;
  onSwipeDown?: () => void;
  onClick?: (cursor: GesturePoint) => void;
  onThumbsUp?: () => void;
  onCursorMove?: (cursor: GesturePoint, isPinching: boolean, state: string) => void;
}

export function useClientGesture(options: ClientGestureOptions = {}) {
  const [isActive, setIsActive] = useState(false);
  const [statusMessage, setStatusMessage] = useState("Idle");
  const [fps, setFps] = useState(0);
  const [cursor, setCursor] = useState<GesturePoint>({ x: 0.5, y: 0.5 });
  const [state, setState] = useState<string>("IDLE");
  const [isPinching, setIsPinching] = useState(false);

  const controllerRef = useRef<any>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const optionsRef = useRef(options);
  optionsRef.current = options;

  const start = useCallback(async () => {
    if (typeof window === "undefined") return;

    try {
      // Lazy load controller script if not present
      if (!(window as any).TouchlessWebController) {
        setStatusMessage("Loading gesture engine script...");
        await new Promise<void>((resolve, reject) => {
          const s = document.createElement("script");
          s.src = "./gesture-engine.js";
          s.onload = () => resolve();
          s.onerror = () => reject(new Error("Failed to load gesture-engine.js"));
          document.head.appendChild(s);
        });
      }

      if (!videoRef.current) {
        videoRef.current = document.createElement("video");
      }

      const Controller = (window as any).TouchlessWebController;
      controllerRef.current = new Controller({
        previewCanvas: optionsRef.current.previewCanvas,
        autoZoom: optionsRef.current.autoZoom ?? true,
        onStatus: ({ status, message }: { status: string; message: string }) => {
          setStatusMessage(message);
        },
        onUpdate: (event: any) => {
          setCursor(event.cursor);
          setState(event.state);
          const pinching = event.pinchRatio < 0.28;
          setIsPinching(pinching);
          setFps(event.fps || 0);

          optionsRef.current.onCursorMove?.(event.cursor, pinching, event.state);
        },
        onGesture: (event: any) => {
          const act = event.action;
          switch (act) {
            case "SWIPE_LEFT":
              optionsRef.current.onSwipeLeft?.();
              break;
            case "SWIPE_RIGHT":
              optionsRef.current.onSwipeRight?.();
              break;
            case "SWIPE_UP":
              optionsRef.current.onSwipeUp?.();
              break;
            case "SWIPE_DOWN":
              optionsRef.current.onSwipeDown?.();
              break;
            case "CLICK":
              optionsRef.current.onClick?.(event.cursor);
              break;
            case "THUMBS_UP":
              optionsRef.current.onThumbsUp?.();
              break;
          }
        }
      });

      await controllerRef.current.start(videoRef.current);
      setIsActive(true);
    } catch (err: any) {
      setStatusMessage(`Error: ${err.message}`);
      setIsActive(false);
    }
  }, []);

  const stop = useCallback(() => {
    if (controllerRef.current) {
      controllerRef.current.stop();
      setIsActive(false);
      setStatusMessage("Stopped");
    }
  }, []);

  useEffect(() => {
    if (options.autoStart) {
      start();
    }
    return () => {
      stop();
    };
  }, [options.autoStart, start, stop]);

  return {
    isActive,
    statusMessage,
    fps,
    cursor,
    state,
    isPinching,
    start,
    stop
  };
}
