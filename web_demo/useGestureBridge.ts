/**
 * React / Next.js Hook for Gesture Control Integration
 * Connects to the local Python Gesture Engine WebSocket bridge (ws://127.0.0.1:8765)
 * and provides real-time cursor coordinates and gesture action callbacks.
 *
 * Example Usage in RealtimeTryOn.tsx:
 * ```tsx
 * import { useGestureBridge } from "./useGestureBridge";
 *
 * export function RealtimeTryOn() {
 *   const { isConnected, cursor, lastAction } = useGestureBridge({
 *     onSwipeLeft: () => nextGarment(),
 *     onSwipeRight: () => prevGarment(),
 *     onSwipeUp: () => prevCategory(),
 *     onSwipeDown: () => nextCategory(),
 *     onClick: (x, y) => console.log("Pinched at", x, y),
 *     onDrag: (x, y) => rotateModel(x),
 *   });
 *   ...
 * }
 * ```
 */

import { useState, useEffect, useRef, useCallback } from "react";

export interface GestureCursor {
  x: number; // Normalized [0.0, 1.0] across screen width
  y: number; // Normalized [0.0, 1.0] across screen height
}

export interface GestureBridgeOptions {
  wsUrl?: string; // default: "ws://127.0.0.1:8765"
  onSwipeLeft?: () => void;
  onSwipeRight?: () => void;
  onSwipeUp?: () => void;
  onSwipeDown?: () => void;
  onClick?: (cursor: GestureCursor) => void;
  onHoldStart?: (cursor: GestureCursor) => void;
  onHoldEnd?: () => void;
  onDragStart?: (cursor: GestureCursor) => void;
  onDragMove?: (cursor: GestureCursor) => void;
  onDragEnd?: () => void;
  onCursorMove?: (cursor: GestureCursor, isPinching: boolean) => void;
}

export function useGestureBridge(options: GestureBridgeOptions = {}) {
  const {
    wsUrl = "ws://127.0.0.1:8765",
    onSwipeLeft,
    onSwipeRight,
    onSwipeUp,
    onSwipeDown,
    onClick,
    onHoldStart,
    onHoldEnd,
    onDragStart,
    onDragMove,
    onDragEnd,
    onCursorMove,
  } = options;

  const [isConnected, setIsConnected] = useState(false);
  const [cursor, setCursor] = useState<GestureCursor>({ x: 0.5, y: 0.5 });
  const [state, setState] = useState<string>("IDLE");
  const [lastAction, setLastAction] = useState<string | null>(null);
  const [isPinching, setIsPinching] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const optionsRef = useRef(options);
  optionsRef.current = options;

  useEffect(() => {
    let reconnectTimeout: any;

    function connect() {
      try {
        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        ws.onopen = () => {
          setIsConnected(true);
        };

        ws.onclose = () => {
          setIsConnected(false);
          reconnectTimeout = setTimeout(connect, 2000);
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);

            if (data.type === "cursor") {
              const c = data.cursor;
              setCursor(c);
              setState(data.state);
              const pinching = data.pinch_ratio < 0.28;
              setIsPinching(pinching);
              optionsRef.current.onCursorMove?.(c, pinching);
            }

            if (data.type === "gesture") {
              const act = data.action;
              setLastAction(act);

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
                  optionsRef.current.onClick?.(data.cursor);
                  break;
                case "HOLD_START":
                  optionsRef.current.onHoldStart?.(data.cursor);
                  break;
                case "HOLD_END":
                  optionsRef.current.onHoldEnd?.();
                  break;
                case "DRAG_START":
                  optionsRef.current.onDragStart?.(data.cursor);
                  break;
                case "DRAG_MOVE":
                  optionsRef.current.onDragMove?.(data.cursor);
                  break;
                case "DRAG_END":
                  optionsRef.current.onDragEnd?.();
                  break;
              }
            }
          } catch (e) {
            // Ignore parse errors
          }
        };
      } catch (err) {
        reconnectTimeout = setTimeout(connect, 2000);
      }
    }

    connect();

    return () => {
      clearTimeout(reconnectTimeout);
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [wsUrl]);

  return {
    isConnected,
    cursor,
    state,
    isPinching,
    lastAction,
  };
}
