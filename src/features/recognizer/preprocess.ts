/**
 * Turn a free-hand drawing into an MNIST-style input, the way the original dataset was made:
 * crop to the ink, scale the longer side to 20 px (keeping aspect ratio), drop it on a
 * 28×28 field, then shift so the ink's centre of mass sits at the centre.
 * The model only ever saw digits prepared like this, so skipping a step costs accuracy.
 */

const INK_THRESHOLD = 20; // alpha 0–255

/** Returns 784 raw intensities in [0, 1] (white ink on black, like MNIST), or null if blank. */
export function canvasToMnist(source: HTMLCanvasElement): Float32Array | null {
  const ctx = source.getContext("2d", { willReadFrequently: true });
  if (!ctx) return null;
  const { width, height } = source;
  const alpha = ctx.getImageData(0, 0, width, height).data;

  let minX = width, minY = height, maxX = -1, maxY = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (alpha[(y * width + x) * 4 + 3]! > INK_THRESHOLD) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < 0) return null;

  const bw = maxX - minX + 1;
  const bh = maxY - minY + 1;
  const scale = 20 / Math.max(bw, bh);
  const dw = Math.max(1, bw * scale);
  const dh = Math.max(1, bh * scale);

  // Downscale in two steps; one big jump makes thin strokes vanish.
  const mid = document.createElement("canvas");
  mid.width = mid.height = 112;
  const mctx = mid.getContext("2d")!;
  mctx.imageSmoothingQuality = "high";
  mctx.drawImage(source, minX, minY, bw, bh, (112 - dw * 4) / 2, (112 - dh * 4) / 2, dw * 4, dh * 4);

  const small = document.createElement("canvas");
  small.width = small.height = 28;
  const sctx = small.getContext("2d", { willReadFrequently: true })!;
  sctx.imageSmoothingQuality = "high";
  sctx.drawImage(mid, 0, 0, 28, 28);
  const px = sctx.getImageData(0, 0, 28, 28).data;

  const raw = new Float32Array(784);
  let mass = 0, cx = 0, cy = 0;
  for (let i = 0; i < 784; i++) {
    const v = px[i * 4 + 3]! / 255;
    raw[i] = v;
    mass += v;
    cx += (i % 28) * v;
    cy += Math.floor(i / 28) * v;
  }
  const sx = Math.round(13.5 - cx / mass);
  const sy = Math.round(13.5 - cy / mass);

  const out = new Float32Array(784);
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const nx = x + sx;
      const ny = y + sy;
      if (nx >= 0 && nx < 28 && ny >= 0 && ny < 28) out[ny * 28 + nx] = raw[y * 28 + x]!;
    }
  }
  return out;
}

export function normalize(raw: Float32Array, mean: number, std: number): Float32Array {
  return raw.map((v) => (v - mean) / std);
}
