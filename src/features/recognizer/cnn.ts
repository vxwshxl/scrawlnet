/**
 * scrawlnet's CNN, run by hand in plain TypeScript — no ML runtime.
 * Mirrors trainer/train.py exactly:
 *   28×28 → conv3×3(16)+ReLU → maxpool2 → conv3×3(32)+ReLU → maxpool2 → dense(64)+ReLU → dense(10)
 * Kept dependency-free (no `@/` imports) so `node scripts/check-parity.ts` can run it.
 */

export type Shape = number[];
type TensorRef = { shape: Shape; offset: number; length: number };
type LayerRef = { name: string; weight: TensorRef; bias: TensorRef };

export type ModelMeta = {
  name: string;
  input: Shape;
  normalize: { mean: number; std: number };
  params: number;
  bytes: number;
  layers: LayerRef[];
  history: { epoch: number; trainLoss: number; trainAcc: number; testLoss: number; testAcc: number }[];
  testAcc: number;
  confusion: number[][];
};

type Layer = { w: Float32Array; b: Float32Array; shape: Shape };
export type Weights = { conv1: Layer; conv2: Layer; fc1: Layer; fc2: Layer };

/** Everything the network computed for one image, for visualising. */
export type Activations = {
  input: Float32Array; // 1 × 28 × 28 (normalised)
  conv1: Float32Array; // 16 × 14 × 14 (after ReLU + pool)
  conv2: Float32Array; // 32 × 7 × 7
  fc1: Float32Array; // 64
  logits: Float32Array; // 10
  probs: Float32Array; // 10, softmax
};

export function parseWeights(meta: ModelMeta, buf: ArrayBuffer): Weights {
  const get = (name: string): Layer => {
    const l = meta.layers.find((x) => x.name === name);
    if (!l) throw new Error(`model.json is missing layer ${name}`);
    return {
      w: new Float32Array(buf, l.weight.offset, l.weight.length),
      b: new Float32Array(buf, l.bias.offset, l.bias.length),
      shape: l.weight.shape,
    };
  };
  return { conv1: get("conv1"), conv2: get("conv2"), fc1: get("fc1"), fc2: get("fc2") };
}

/** 3×3 convolution, stride 1, zero padding 1, then ReLU. in: C×H×W → out: O×H×W */
function conv3x3Relu(input: Float32Array, c: number, h: number, w: number, layer: Layer): Float32Array {
  const o = layer.shape[0]!;
  const out = new Float32Array(o * h * w);
  for (let oc = 0; oc < o; oc++) {
    const bias = layer.b[oc]!;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let sum = bias;
        for (let ic = 0; ic < c; ic++) {
          const kBase = (oc * c + ic) * 9;
          const iBase = ic * h * w;
          for (let ky = -1; ky <= 1; ky++) {
            const yy = y + ky;
            if (yy < 0 || yy >= h) continue;
            for (let kx = -1; kx <= 1; kx++) {
              const xx = x + kx;
              if (xx < 0 || xx >= w) continue;
              sum += input[iBase + yy * w + xx]! * layer.w[kBase + (ky + 1) * 3 + (kx + 1)]!;
            }
          }
        }
        out[oc * h * w + y * w + x] = sum > 0 ? sum : 0;
      }
    }
  }
  return out;
}

/** 2×2 max pool, stride 2. C×H×W → C×(H/2)×(W/2) */
function maxPool2(input: Float32Array, c: number, h: number, w: number): Float32Array {
  const oh = h >> 1;
  const ow = w >> 1;
  const out = new Float32Array(c * oh * ow);
  for (let ch = 0; ch < c; ch++) {
    for (let y = 0; y < oh; y++) {
      for (let x = 0; x < ow; x++) {
        const i = ch * h * w + 2 * y * w + 2 * x;
        out[ch * oh * ow + y * ow + x] = Math.max(input[i]!, input[i + 1]!, input[i + w]!, input[i + w + 1]!);
      }
    }
  }
  return out;
}

/** Fully connected: out = W·x + b (PyTorch stores W as out × in). */
function dense(input: Float32Array, layer: Layer, relu: boolean): Float32Array {
  const [o, n] = layer.shape as [number, number];
  const out = new Float32Array(o);
  for (let i = 0; i < o; i++) {
    let sum = layer.b[i]!;
    const base = i * n;
    for (let j = 0; j < n; j++) sum += layer.w[base + j]! * input[j]!;
    out[i] = relu && sum < 0 ? 0 : sum;
  }
  return out;
}

function softmax(logits: Float32Array): Float32Array {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / total);
}

/** Forward pass. `input` is 784 normalised pixels, row-major. */
export function forward(weights: Weights, input: Float32Array): Activations {
  const c1 = maxPool2(conv3x3Relu(input, 1, 28, 28, weights.conv1), 16, 28, 28);
  const c2 = maxPool2(conv3x3Relu(c1, 16, 14, 14, weights.conv2), 32, 14, 14);
  const f1 = dense(c2, weights.fc1, true); // flatten is free: C×H×W row-major matches torch.flatten
  const logits = dense(f1, weights.fc2, false);
  return { input, conv1: c1, conv2: c2, fc1: f1, logits, probs: softmax(logits) };
}
