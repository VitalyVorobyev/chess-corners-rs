// chess-corners demo: top-level shell.
//
// Fully static: no backend, no network calls beyond loading bundled sample
// images and the WASM module itself. Detection runs entirely in the browser.

import { useEffect, useState } from "react";
import { Badge, Callout, ThemeToggle } from "@vitavision/ui";
import { Workspace } from "./views/Workspace";
import { initialize, isReady } from "./lib/detector";

function Header() {
  return (
    <div className="flex h-11 items-center gap-3 px-4">
      <span className="flex items-center gap-2 text-sm font-semibold tracking-tight">
        <span className="text-signal" aria-hidden>◇</span>
        chess-corners
      </span>
      <span className="text-xs text-fg-subtle">WebAssembly playground</span>
      <div className="ml-auto flex items-center gap-2">
        <Badge tone="normal">WASM ready</Badge>
        <ThemeToggle />
      </div>
    </div>
  );
}

export default function App() {
  const [ready, setReady] = useState(isReady());
  const [initError, setInitError] = useState<string | null>(null);

  useEffect(() => {
    if (isReady()) return;
    initialize()
      .then(() => setReady(true))
      .catch((e: unknown) => setInitError(e instanceof Error ? e.message : String(e)));
  }, []);

  if (initError) {
    return (
      <div className="grid h-dvh place-items-center bg-ground p-6">
        <Callout tone="error" title="Failed to load WASM module">
          {initError}
        </Callout>
      </div>
    );
  }
  if (!ready) return <div className="grid h-dvh place-items-center bg-ground text-sm text-fg-muted">Loading WASM module…</div>;
  return <Workspace header={<Header />} />;
}
