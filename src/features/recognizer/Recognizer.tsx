"use client";

import { Eraser, Shuffle } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { forward, parseWeights, type Activations, type ModelMeta, type Weights } from "./cnn";
import { DrawPad, type DrawPadHandle } from "./DrawPad";
import { HeatCanvas } from "./HeatCanvas";
import { canvasToMnist, normalize } from "./preprocess";

type Sample = { label: number; input: number[] };
type Model = { meta: ModelMeta; weights: Weights; samples: Sample[] };

const maxOf = (a: Float32Array) => a.reduce((m, v) => (v > m ? v : m), 0);
const absMaxOf = (a: Float32Array) => a.reduce((m, v) => Math.max(m, Math.abs(v)), 0);

export function Recognizer() {
  const pad = useRef<DrawPadHandle>(null);
  const [model, setModel] = useState<Model | null>(null);
  const [error, setError] = useState(false);
  const [raw, setRaw] = useState<Float32Array | null>(null);
  const [acts, setActs] = useState<Activations | null>(null);
  const [sampleLabel, setSampleLabel] = useState<number | null>(null);
  const sampleIdx = useRef(0);

  useEffect(() => {
    let live = true;
    Promise.all([
      fetch("/model/model.json").then((r) => r.json() as Promise<ModelMeta>),
      fetch("/model/weights.bin").then((r) => r.arrayBuffer()),
      fetch("/model/samples.json").then((r) => r.json() as Promise<Sample[]>),
    ])
      .then(([meta, buf, samples]) => live && setModel({ meta, weights: parseWeights(meta, buf), samples }))
      .catch(() => live && setError(true));
    return () => {
      live = false;
    };
  }, []);

  const run = useCallback(() => {
    const canvas = pad.current?.canvas();
    if (!canvas || !model) return;
    const digit = canvasToMnist(canvas);
    setRaw(digit);
    if (!digit) {
      setActs(null);
      return;
    }
    const { mean, std } = model.meta.normalize;
    setActs(forward(model.weights, normalize(digit, mean, std)));
  }, [model]);

  const onDraw = useCallback(() => {
    setSampleLabel(null);
    run();
  }, [run]);

  /** Paint a real MNIST test digit onto the pad, so the pipeline runs on it end to end. */
  const loadSample = () => {
    const canvas = pad.current?.canvas();
    const ctx = canvas?.getContext("2d");
    if (!model || !canvas || !ctx) return;
    const s = model.samples[sampleIdx.current++ % model.samples.length]!;
    const { mean, std } = model.meta.normalize;
    const tiny = document.createElement("canvas");
    tiny.width = tiny.height = 28;
    const tctx = tiny.getContext("2d")!;
    const img = tctx.createImageData(28, 28);
    const [r, g, b] = (getComputedStyle(canvas).color.match(/[\d.]+/g) ?? ["0", "0", "0"]).map(Number);
    s.input.forEach((v, i) => {
      img.data[i * 4] = r!;
      img.data[i * 4 + 1] = g!;
      img.data[i * 4 + 2] = b!;
      img.data[i * 4 + 3] = Math.max(0, Math.min(1, v * std + mean)) * 255;
    });
    tctx.putImageData(img, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(tiny, 0, 0, canvas.width, canvas.height);
    setSampleLabel(s.label);
    run();
  };

  const best = acts ? acts.probs.indexOf(Math.max(...acts.probs)) : null;
  const conf = acts && best !== null ? acts.probs[best]! : 0;

  return (
    <div className="flex flex-col gap-12">
      <section aria-labelledby="try" className="grid gap-8 md:grid-cols-[minmax(0,22rem)_1fr] md:gap-12">
        <div className="flex flex-col gap-3">
          <h2 id="try" className="sr-only">
            Draw a digit
          </h2>
          <div className="relative">
            <DrawPad ref={pad} onChange={onDraw} disabled={!model} />
            {!raw && (
              <p className="pointer-events-none absolute inset-0 grid place-items-center text-sm text-muted-foreground">
                {error ? "Couldn't load the model — refresh to retry" : model ? "Draw a digit here" : "Loading model…"}
              </p>
            )}
          </div>
          <div className="flex gap-2">
            <button type="button" onClick={() => pad.current?.clear()} disabled={!raw} className="btn-secondary flex-1">
              <Eraser aria-hidden className="size-4" /> Clear
            </button>
            <button type="button" onClick={loadSample} disabled={!model} className="btn-secondary flex-1">
              <Shuffle aria-hidden className="size-4" /> Real MNIST digit
            </button>
          </div>
          {sampleLabel !== null && (
            <p className="text-sm text-muted-foreground">
              From the test set — never seen in training. True label: <span className="font-mono text-foreground">{sampleLabel}</span>
            </p>
          )}
        </div>

        <div className="flex flex-col gap-6" aria-live="polite">
          <div className="flex items-end gap-5">
            <div className="grid size-28 place-items-center rounded-xl bg-accent-subtle font-mono text-7xl font-semibold text-primary tabular-nums">
              {best ?? "?"}
            </div>
            <div className="pb-1">
              <p className="text-sm text-muted-foreground">Prediction</p>
              <p className="text-2xl font-semibold tabular-nums">
                {acts ? `${(conf * 100).toFixed(1)}% sure` : "Waiting for ink"}
              </p>
            </div>
          </div>
          <ul className="flex flex-col gap-1.5" aria-label="Probability for each digit">
            {Array.from({ length: 10 }, (_, d) => {
              const p = acts?.probs[d] ?? 0;
              return (
                <li key={d} className="grid grid-cols-[1.25rem_1fr_3.5rem] items-center gap-3 text-sm">
                  <span className="font-mono tabular-nums">{d}</span>
                  <span className="h-3 rounded-full bg-muted">
                    <span
                      className={`block h-full rounded-full transition-transform duration-150 origin-left ${d === best ? "bg-primary" : "bg-muted-foreground/50"}`}
                      style={{ transform: `scaleX(${p})` }}
                    />
                  </span>
                  <span className="text-right font-mono tabular-nums text-muted-foreground">{(p * 100).toFixed(1)}%</span>
                </li>
              );
            })}
          </ul>
        </div>
      </section>

      <section aria-labelledby="inside" className="flex flex-col gap-6">
        <div>
          <h2 id="inside" className="text-xl font-semibold tracking-tight">
            Inside the network, live
          </h2>
          <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
            Every layer below is computed in your browser by about 100 lines of TypeScript as you draw — no ML
            library. Brighter means a stronger activation.
          </p>
        </div>

        <ol className="grid gap-4 lg:grid-cols-[auto_1fr_1fr]">
          <Stage n={1} title="Input" note="28 × 28, centred like MNIST">
            <HeatCanvas
              data={raw ?? new Float32Array(784)}
              width={28}
              height={28}
              max={1}
              label="The 28 by 28 pixel image the network receives"
              className="size-28 rounded-md bg-muted"
            />
          </Stage>
          <Stage n={2} title="Conv 1 → 16 feature maps" note="3 × 3 filters spot edges and strokes · pooled to 14 × 14">
            <MapGrid data={acts?.conv1} count={16} side={14} cols="grid-cols-8" label="conv 1" />
          </Stage>
          <Stage n={3} title="Conv 2 → 32 feature maps" note="combines strokes into curves and corners · 7 × 7">
            <MapGrid data={acts?.conv2} count={32} side={7} cols="grid-cols-8" label="conv 2" />
          </Stage>
        </ol>

        <ol className="grid gap-4 lg:grid-cols-[1fr_1fr]" start={4}>
          <Stage n={4} title="Dense → 64 neurons" note="1,568 numbers squeezed into 64 learned features">
            <HeatCanvas
              data={acts?.fc1 ?? new Float32Array(64)}
              width={16}
              height={4}
              max={acts ? maxOf(acts.fc1) : 1}
              label="64 dense-layer activations"
              className="h-12 w-full rounded-md bg-muted"
            />
          </Stage>
          <Stage n={5} title="What layer 1 learned" note="the 16 conv-1 filters · teal adds, orange subtracts">
            {model && (
              <div className="grid grid-cols-8 gap-1.5">
                {Array.from({ length: 16 }, (_, i) => {
                  const k = model.weights.conv1.w.subarray(i * 9, i * 9 + 9);
                  return (
                    <HeatCanvas
                      key={i}
                      data={k}
                      width={3}
                      height={3}
                      max={absMaxOf(k)}
                      signed
                      label={`Filter ${i + 1}`}
                      className="aspect-square w-full rounded-sm bg-muted"
                    />
                  );
                })}
              </div>
            )}
          </Stage>
        </ol>
      </section>
    </div>
  );
}

function Stage({ n, title, note, children }: { n: number; title: string; note: string; children: React.ReactNode }) {
  return (
    <li className="flex flex-col gap-3 rounded-xl border border-border bg-card p-4">
      <div>
        <p className="text-sm font-medium">
          <span className="mr-2 font-mono text-muted-foreground">{n}</span>
          {title}
        </p>
        <p className="text-xs text-muted-foreground">{note}</p>
      </div>
      {children}
    </li>
  );
}

function MapGrid({ data, count, side, cols, label }: { data?: Float32Array; count: number; side: number; cols: string; label: string }) {
  const max = data ? maxOf(data) : 1;
  const area = side * side;
  return (
    <div className={`grid ${cols} gap-1`}>
      {Array.from({ length: count }, (_, i) => (
        <HeatCanvas
          key={i}
          data={data ? data.subarray(i * area, (i + 1) * area) : new Float32Array(area)}
          width={side}
          height={side}
          max={max}
          label={`${label} map ${i + 1}`}
          className="aspect-square w-full rounded-sm bg-muted"
        />
      ))}
    </div>
  );
}
