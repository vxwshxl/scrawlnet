# scrawlnet — project notes

Handwritten digit recognizer: PyTorch CNN trained on MNIST (`trainer/`, uv, Python 3.13), inference by hand in TypeScript (`src/features/recognizer/cnn.ts`). Single Next.js app, static, no backend. Live at scrawlnet.vercel.app; repo vxwshxl/scrawlnet.

## Invariants
- `cnn.ts` must mirror `ScrawlNet` in `trainer/train.py` exactly (layer order, padding 1, pool 2, PyTorch's out×in dense layout). After changing either, retrain and run `pnpm test:parity`.
- `cnn.ts` stays free of `@/` imports so Node can run it directly for the parity script.
- Preprocessing (`preprocess.ts`) copies how MNIST was made (20×20 box, centre of mass at the centre). Skipping a step costs real-world accuracy even though test accuracy doesn't change.
- The drawing pad stores ink as alpha only, and preprocessing reads alpha, so drawing works in either theme.
