"use client";

import { useTheme } from "next-themes";
import { useEffect, useRef } from "react";

type Props = {
  data: Float32Array;
  width: number;
  height: number;
  /** Values at or above this are full ink. */
  max: number;
  /** Signed data: negatives draw in the warm `--kernel-neg` colour. */
  signed?: boolean;
  className?: string;
  label: string;
};

function rgb(el: Element, prop: string): [number, number, number] {
  const probe = document.createElement("span");
  probe.style.color = `var(${prop})`;
  el.appendChild(probe);
  const m = getComputedStyle(probe).color.match(/[\d.]+/g) ?? ["0", "0", "0"];
  probe.remove();
  return [Number(m[0]), Number(m[1]), Number(m[2])];
}

/** One small activation map / kernel, drawn pixel-for-pixel and scaled up crisply by CSS. */
export function HeatCanvas({ data, width, height, max, signed, className, label }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const { resolvedTheme } = useTheme();

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    // Colours come from theme tokens, re-read on every draw so a theme switch repaints.
    const pos = rgb(canvas.parentElement ?? canvas, signed ? "--ds-accent" : "--ds-fg");
    const neg = rgb(canvas.parentElement ?? canvas, "--kernel-neg");
    const img = ctx.createImageData(width, height);
    const scale = max > 0 ? max : 1;
    for (let i = 0; i < width * height; i++) {
      const v = data[i] ?? 0;
      const c = v >= 0 ? pos : neg;
      img.data[i * 4] = c[0];
      img.data[i * 4 + 1] = c[1];
      img.data[i * 4 + 2] = c[2];
      img.data[i * 4 + 3] = Math.min(255, (Math.abs(v) / scale) * 255);
    }
    ctx.putImageData(img, 0, 0);
  }, [data, width, height, max, signed, resolvedTheme]);

  return (
    <canvas
      ref={ref}
      width={width}
      height={height}
      role="img"
      aria-label={label}
      className={className}
      style={{ imageRendering: "pixelated" }}
    />
  );
}
