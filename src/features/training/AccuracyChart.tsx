"use client";

import { useState } from "react";

import type { ModelMeta } from "../recognizer/cnn";

const W = 560;
const H = 240;
const PAD = { l: 44, r: 64, t: 12, b: 32 };

/** Train vs test accuracy per epoch. Two series: solid teal (test) and dashed gray (train), both direct-labelled. */
export function AccuracyChart({ history }: { history: ModelMeta["history"] }) {
  const [hover, setHover] = useState<number | null>(null);
  const lo = Math.floor(Math.min(...history.map((h) => Math.min(h.trainAcc, h.testAcc))) * 10) / 10;
  const n = history.length;
  const x = (i: number) => PAD.l + (i / (n - 1)) * (W - PAD.l - PAD.r);
  const y = (v: number) => PAD.t + ((1 - v) / (1 - lo)) * (H - PAD.t - PAD.b);
  const path = (k: "trainAcc" | "testAcc") => history.map((h, i) => `${i ? "L" : "M"}${x(i)},${y(h[k])}`).join("");
  const ticks = Array.from({ length: Math.round((1 - lo) / 0.1) + 1 }, (_, i) => lo + i * 0.1);
  const last = history[n - 1]!;
  const h = hover !== null ? history[hover]! : null;
  // Keep the two end labels at least 14 px apart.
  const testY = Math.min(y(last.testAcc), y(last.trainAcc) - 14);
  const trainY = Math.max(y(last.trainAcc), testY + 14);

  return (
    <figure className="flex flex-col gap-2">
      <figcaption className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
        <span className="font-medium">Accuracy per epoch</span>
        <span className="flex items-center gap-1.5 text-muted-foreground">
          <svg width="16" height="4" aria-hidden><line x1="0" y1="2" x2="16" y2="2" className="stroke-primary" strokeWidth="2" /></svg>
          Test (unseen digits)
        </span>
        <span className="flex items-center gap-1.5 text-muted-foreground">
          <svg width="16" height="4" aria-hidden><line x1="0" y1="2" x2="16" y2="2" className="stroke-muted-foreground" strokeWidth="2" strokeDasharray="4 3" /></svg>
          Train (augmented)
        </span>
      </figcaption>
      <div className="relative">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="w-full overflow-visible"
          role="img"
          aria-label={`Test accuracy rose from ${(history[0]!.testAcc * 100).toFixed(1)}% to ${(last.testAcc * 100).toFixed(2)}% over ${n} epochs`}
          onPointerLeave={() => setHover(null)}
          onPointerMove={(e) => {
            const r = e.currentTarget.getBoundingClientRect();
            const px = ((e.clientX - r.left) / r.width) * W;
            setHover(Math.max(0, Math.min(n - 1, Math.round(((px - PAD.l) / (W - PAD.l - PAD.r)) * (n - 1)))));
          }}
        >
          {ticks.map((t) => (
            <g key={t}>
              <line x1={PAD.l} x2={W - PAD.r} y1={y(t)} y2={y(t)} className="stroke-border" strokeWidth="1" />
              <text x={PAD.l - 8} y={y(t)} dy="0.32em" textAnchor="end" className="fill-muted-foreground font-mono text-[11px]">
                {Math.round(t * 100)}%
              </text>
            </g>
          ))}
          {history.map((hh, i) => (
            <text key={hh.epoch} x={x(i)} y={H - 10} textAnchor="middle" className="fill-muted-foreground font-mono text-[11px]">
              {hh.epoch}
            </text>
          ))}
          <text x={W - PAD.r} y={H - 10} dx="8" className="fill-muted-foreground text-[11px]">epoch</text>
          {hover !== null && (
            <line x1={x(hover)} x2={x(hover)} y1={PAD.t} y2={H - PAD.b} className="stroke-border-strong" strokeWidth="1" />
          )}
          <path d={path("trainAcc")} fill="none" className="stroke-muted-foreground" strokeWidth="2" strokeDasharray="5 4" />
          <path d={path("testAcc")} fill="none" className="stroke-primary" strokeWidth="2" />
          {hover !== null && h && (
            <>
              <circle cx={x(hover)} cy={y(h.trainAcc)} r="4" className="fill-muted-foreground stroke-card" strokeWidth="2" />
              <circle cx={x(hover)} cy={y(h.testAcc)} r="4" className="fill-primary stroke-card" strokeWidth="2" />
            </>
          )}
          <text x={x(n - 1) + 8} y={testY} dy="0.32em" className="fill-foreground font-mono text-[11px]">
            test {(last.testAcc * 100).toFixed(1)}%
          </text>
          <text x={x(n - 1) + 8} y={trainY} dy="0.32em" className="fill-muted-foreground font-mono text-[11px]">
            train {(last.trainAcc * 100).toFixed(1)}%
          </text>
        </svg>
        {h && hover !== null && (
          <div
            className="pointer-events-none absolute top-0 rounded-md border border-border bg-popover px-2.5 py-1.5 text-xs shadow-sm"
            style={{ left: `${(x(hover) / W) * 100}%`, transform: `translateX(${hover > n / 2 ? "-110%" : "10%"})` }}
          >
            <p className="font-medium">Epoch {h.epoch}</p>
            <p className="font-mono tabular-nums">test {(h.testAcc * 100).toFixed(2)}% · loss {h.testLoss}</p>
            <p className="font-mono tabular-nums text-muted-foreground">train {(h.trainAcc * 100).toFixed(2)}% · loss {h.trainLoss}</p>
          </div>
        )}
      </div>
    </figure>
  );
}
