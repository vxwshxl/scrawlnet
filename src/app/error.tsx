"use client";

export default function Error({ reset }: { error: Error; reset: () => void }) {
  return (
    <main className="grid min-h-dvh place-items-center px-5 text-center">
      <div>
        <p className="text-lg font-semibold">Something broke while drawing the page.</p>
        <button type="button" onClick={reset} className="btn-secondary mt-6">Try again</button>
      </div>
    </main>
  );
}
