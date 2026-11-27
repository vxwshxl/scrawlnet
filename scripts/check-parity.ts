// Checks the hand-written TypeScript forward pass against PyTorch's logits.
// Run after training: pnpm test:parity
import { readFileSync } from "node:fs";
import { forward, parseWeights, type ModelMeta } from "../src/features/recognizer/cnn.ts";

const root = new URL("..", import.meta.url);
const meta = JSON.parse(readFileSync(new URL("public/model/model.json", root), "utf8")) as ModelMeta;
const bin = readFileSync(new URL("public/model/weights.bin", root));
const buf = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
const weights = parseWeights(meta, buf);
const samples = JSON.parse(readFileSync(new URL("public/model/samples.json", root), "utf8")) as {
  label: number;
  input: number[];
  logits: number[];
}[];

let worst = 0;
let correct = 0;
for (const s of samples) {
  const { logits } = forward(weights, Float32Array.from(s.input));
  logits.forEach((v, i) => (worst = Math.max(worst, Math.abs(v - s.logits[i]!))));
  if (logits.indexOf(Math.max(...logits)) === s.label) correct++;
}
console.log(`samples ${samples.length} · correct ${correct} · max |Δlogit| ${worst.toExponential(2)}`);
if (worst > 1e-3) {
  console.error("FAIL: TypeScript forward pass disagrees with PyTorch");
  process.exit(1);
}
console.log("PASS");
