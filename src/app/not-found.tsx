import Link from "next/link";

export default function NotFound() {
  return (
    <main className="grid min-h-dvh place-items-center px-5 text-center">
      <div>
        <p className="font-mono text-6xl font-semibold text-primary">404</p>
        <p className="mt-2 text-muted-foreground">Even a CNN can&apos;t read this page — it doesn&apos;t exist.</p>
        <Link href="/" className="btn-secondary mt-6">Back to the drawing pad</Link>
      </div>
    </main>
  );
}
