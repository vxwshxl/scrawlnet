"use client";

import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef } from "react";

export type DrawPadHandle = { clear: () => void; canvas: () => HTMLCanvasElement | null };

const SIZE = 560; // internal pixels; displayed at up to 320 CSS px
const BRUSH = 38;

/** Ink is drawn as alpha only — preprocessing reads the alpha channel, so it works in any theme. */
export const DrawPad = forwardRef<DrawPadHandle, { onChange: () => void; disabled?: boolean }>(
  function DrawPad({ onChange, disabled }, ref) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const last = useRef<{ x: number; y: number } | null>(null);
    const frame = useRef(0);

    const schedule = useCallback(() => {
      cancelAnimationFrame(frame.current);
      frame.current = requestAnimationFrame(onChange);
    }, [onChange]);

    useImperativeHandle(ref, () => ({
      clear: () => {
        const c = canvasRef.current;
        c?.getContext("2d")?.clearRect(0, 0, SIZE, SIZE);
        onChange();
      },
      canvas: () => canvasRef.current,
    }));

    useEffect(() => () => cancelAnimationFrame(frame.current), []);

    const point = (e: React.PointerEvent<HTMLCanvasElement>) => {
      const r = e.currentTarget.getBoundingClientRect();
      return { x: ((e.clientX - r.left) / r.width) * SIZE, y: ((e.clientY - r.top) / r.height) * SIZE };
    };

    const stroke = (to: { x: number; y: number }) => {
      const ctx = canvasRef.current?.getContext("2d");
      if (!ctx) return;
      const from = last.current ?? to;
      ctx.strokeStyle = getComputedStyle(canvasRef.current!).color;
      ctx.lineWidth = BRUSH;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x + 0.01, to.y);
      ctx.stroke();
      last.current = to;
      schedule();
    };

    return (
      <canvas
        ref={canvasRef}
        width={SIZE}
        height={SIZE}
        aria-label="Drawing pad. Draw one digit from 0 to 9 with a mouse, finger or stylus."
        className="aspect-square w-full touch-none rounded-xl border border-border bg-subtle text-foreground data-[disabled=true]:opacity-50 cursor-crosshair"
        data-disabled={disabled}
        onPointerDown={(e) => {
          if (disabled) return;
          e.currentTarget.setPointerCapture(e.pointerId);
          last.current = null;
          stroke(point(e));
        }}
        onPointerMove={(e) => {
          if (last.current) stroke(point(e));
        }}
        onPointerUp={() => (last.current = null)}
        onPointerCancel={() => (last.current = null)}
      />
    );
  },
);
