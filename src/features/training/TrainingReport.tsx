import meta from "../../../public/model/model.json";
import type { ModelMeta } from "../recognizer/cnn";
import { AccuracyChart } from "./AccuracyChart";

const model = meta as ModelMeta;

export function TrainingReport() {
  const confusion = model.confusion;
  const totals = confusion.map((row) => row.reduce((a, b) => a + b, 0));
  const offMax = Math.max(...confusion.flatMap((row, i) => row.filter((_, j) => j !== i)));
  const errors = totals.reduce((a, b) => a + b, 0) - confusion.reduce((a, row, i) => a + row[i]!, 0);

  return (
    <section aria-labelledby="training" className="flex flex-col gap-6">
      <div>
        <h2 id="training" className="text-xl font-semibold tracking-tight">
          How it was trained
        </h2>
        <p className="mt-1 max-w-2xl text-sm text-muted-foreground">
          PyTorch, {model.history.length} epochs over the 60,000 MNIST training images, in under a minute on a laptop.
          Each training image was randomly rotated, shifted, scaled and sheared so the network copes with how
          people actually draw — that&apos;s why the train line sits <em>below</em> the test line.
        </p>
      </div>

      <dl className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        {[
          ["Test accuracy", `${(model.testAcc * 100).toFixed(2)}%`],
          ["Mistakes of 10,000", `${errors}`],
          ["Parameters", model.params.toLocaleString("en")],
          ["Model size", `${Math.round(model.bytes / 1024)} KiB`],
        ].map(([k, v]) => (
          <div key={k} className="rounded-xl border border-border bg-card p-4">
            <dt className="text-sm text-muted-foreground">{k}</dt>
            <dd className="mt-1 font-mono text-2xl font-semibold tabular-nums">{v}</dd>
          </div>
        ))}
      </dl>

      <div className="grid gap-6 lg:grid-cols-[3fr_2fr]">
        <div className="rounded-xl border border-border bg-card p-4">
          <AccuracyChart history={model.history} />
        </div>
        <div className="rounded-xl border border-border bg-card p-4">
          <p className="text-sm font-medium">Where it goes wrong</p>
          <p className="mb-3 text-xs text-muted-foreground">
            10,000 test digits. Rows are the true digit, columns the guess; the diagonal is correct, darker
            off-diagonal cells are common mix-ups.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full border-separate border-spacing-0.5 font-mono text-[11px] tabular-nums">
              <caption className="sr-only">Confusion matrix of true digit against predicted digit</caption>
              <thead>
                <tr>
                  <th scope="col" className="text-left font-normal text-muted-foreground">
                    <span className="sr-only">True digit</span>
                  </th>
                  {confusion.map((_, j) => (
                    <th key={j} scope="col" className="font-normal text-muted-foreground">
                      {j}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {confusion.map((row, i) => (
                  <tr key={i}>
                    <th scope="row" className="pr-1 text-right font-normal text-muted-foreground">
                      {i}
                    </th>
                    {row.map((v, j) => {
                      const diag = i === j;
                      const strength = diag ? 0 : v / offMax;
                      return (
                        <td
                          key={j}
                          title={diag ? `${v} ${i}s read correctly` : `${v} × true ${i} read as ${j}`}
                          className={`h-6 min-w-6 rounded-sm text-center ${diag ? "bg-muted text-foreground" : strength > 0.5 ? "text-primary-foreground" : "text-muted-foreground"}`}
                          style={
                            diag || v === 0
                              ? undefined
                              : { background: `color-mix(in oklab, var(--ds-accent) ${Math.round(15 + strength * 85)}%, var(--ds-bg-elevated))` }
                          }
                        >
                          {v === 0 ? "·" : v}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>
  );
}
