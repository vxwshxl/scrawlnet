# scrawlnet

Draw a digit, and a convolutional neural network (CNN) trained on MNIST reads it, live in your browser. Every layer's activations are shown as you draw.

**Live:** https://scrawlnet.vercel.app

## How it works

| Part | What it does |
|------|--------------|
| `trainer/train.py` | PyTorch. Trains the CNN on the 60,000 MNIST training digits (with random rotate/shift/scale/shear augmentation), evaluates on the 10,000 test digits, and exports the weights as raw float32 to `public/model/`. |
| `src/features/recognizer/cnn.ts` | The same network's forward pass, about 100 lines of plain TypeScript (conv → ReLU → max-pool ×2, then two dense layers). There's no ML runtime, so every intermediate layer can be shown. |
| `src/features/recognizer/preprocess.ts` | Turns your drawing into an MNIST-style input: crop to the ink, fit it into 20×20, place it on a 28×28 field, then centre it by centre of mass. |
| `scripts/check-parity.ts` | Checks the TypeScript forward pass against PyTorch's logits on real test digits. |

### Architecture

```
28×28 input
 → conv 3×3, 16 filters + ReLU → max-pool 2×2   (16×14×14)
 → conv 3×3, 32 filters + ReLU → max-pool 2×2   (32×7×7)
 → dense 1568→64 + ReLU (+ dropout in training)
 → dense 64→10 → softmax
```

105,866 parameters (414 KiB). **99.21 %** test accuracy after 8 epochs (~1 min on Apple Silicon).

## Run it

```bash
pnpm install
pnpm dev                 # http://localhost:3000

# retrain (needs uv: https://docs.astral.sh/uv/)
pnpm train               # writes public/model/{weights.bin,model.json,samples.json}
pnpm test:parity         # TS forward pass must match PyTorch
```
