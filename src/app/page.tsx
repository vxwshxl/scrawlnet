import Link from "next/link";

import { ThemeToggle } from "@/components/ThemeToggle";
import { Recognizer } from "@/features/recognizer/Recognizer";
import { TrainingReport } from "@/features/training/TrainingReport";

const REPO = "https://github.com/vxwshxl/scrawlnet";

export default function Home() {
  return (
    <>
      <header className="mx-auto flex max-w-6xl items-center justify-between px-5 py-4">
        <Link href="/" className="flex items-center gap-2 font-semibold tracking-tight">
          <svg viewBox="0 0 32 32" className="size-7" aria-hidden>
            <rect width="32" height="32" rx="8" className="fill-primary" />
            <path d="M10 11c2-3 9-3 9 1s-9 7-9 10h12" fill="none" className="stroke-primary-foreground" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          scrawlnet
        </Link>
        <div className="flex items-center gap-3">
          <a href={REPO} className="text-sm text-muted-foreground underline-offset-4 hover:text-foreground hover:underline">
            Source
          </a>
          <ThemeToggle />
        </div>
      </header>

      <main className="mx-auto flex max-w-6xl flex-col gap-16 px-5 pb-20 pt-6">
        <div className="max-w-2xl">
          <h1 className="text-balance text-4xl font-semibold tracking-tight sm:text-5xl">Draw a digit. Watch a CNN read it.</h1>
          <p className="mt-3 text-pretty text-muted-foreground">
            A small convolutional neural network, trained on 60,000 handwritten digits from MNIST, running right here
            in your browser — with every layer on show.
          </p>
        </div>
        <Recognizer />
        <TrainingReport />
      </main>

      <footer className="border-t border-border">
        <p className="mx-auto max-w-6xl px-5 py-6 text-sm text-muted-foreground">
          Trained with PyTorch, inference in plain TypeScript.{" "}
          <a href={REPO} className="underline underline-offset-4 hover:text-foreground">
            Read the code on GitHub
          </a>
          .
        </p>
      </footer>
    </>
  );
}
