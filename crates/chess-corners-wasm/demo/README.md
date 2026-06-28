# chess-corners-rs — Interactive WASM Demo

A single-page browser app that runs the `chess-corners-rs` detector in
WebAssembly. Supports image upload / drag-and-drop, bundled sample
images, live parameter tuning, and webcam-streaming detection.

## What it shows

The demo drives the detector through the typed `DetectorConfig` tree
exposed by `chess-corners-wasm`. The on-screen sliders edit a Rust-
backed config and call `detector.applyConfig(cfg)` to commit the
edits.

The user-visible controls correspond to fields on the typed config:

- ChESS vs Radon — `cfg.strategy = DetectionStrategy.fromChess(...)` or
  `.fromRadon(...)`.
- ChESS ring (Canonical / Broad) — `cfg.strategy.chess.ring`.
- Threshold — `cfg.threshold = v`.
- NMS radius / min cluster size — on the active strategy variant.
- Pyramid levels and min size — `cfg.multiscale =
  MultiscaleConfig.pyramid(levels, minSize, 3)` (or `.singleScale()`).
- Upscale factor — `cfg.upscale = UpscaleConfig.disabled()` or
  `UpscaleConfig.fixed(k)` with `k ∈ {2, 3, 4}`.
- Refiner (ChESS) — `ChessRefiner.withCenterOfMass(...)`,
  `.withForstner(...)`, or `.withSaddlePoint(...)`. The Radon
  detector's subpixel step is its built-in peak fit (`PeakFitMode`),
  not a pluggable refiner.

For each detected corner it overlays:

- a white dot at the subpixel location,
- a teal arrow along `axes[0].angle` (line direction ∈ [0, π)),
- an orange arrow along `axes[1].angle` (polar, dark sector runs CCW
  from axis 0 to axis 1).

Hovering a corner shows its response and both axis angles ± 1σ.

## Prerequisites

- `wasm-pack` (`cargo install wasm-pack`)
- Any local static HTTP server (Python, Node, `basic-http-server`, …)
- A modern browser with WebAssembly + ES module support

## Build and run

From the repository root:

```bash
# 1. Build the WASM package (outputs crates/chess-corners-wasm/pkg/)
wasm-pack build crates/chess-corners-wasm --target web --release

# 2. Serve the repository root so the demo can load /pkg/ and /testimages/
python3 -m http.server 8080

# 3. Open the demo
open http://localhost:8080/crates/chess-corners-wasm/demo/
```

The demo expects the compiled `pkg/` to sit at
`crates/chess-corners-wasm/pkg/` (its default wasm-pack output path).
No extra copy step is needed.

## Layout

```
crates/chess-corners-wasm/demo/
├── index.html       Two-column UI (sidebar controls + canvas)
├── app.js           ES module that wires the UI to ChessDetector
├── testimages/      Bundled samples (copies of repo-root testimages/)
└── README.md        This file
```

## Controls

### Image source

- **From file** — file picker (PNG/JPEG/WebP)
- **Sample** — bundled small/mid/large test images
- **Drop zone** — drag-and-drop any image file from the OS

### Detector

| Control     | Config path                                  | Notes                                                |
|-------------|----------------------------------------------|------------------------------------------------------|
| Threshold   | `cfg.threshold`                              | Relative threshold (fraction of max response)        |
| NMS radius  | `cfg.strategy.chess.nmsRadius` (or radon)    | Non-max suppression window radius, in pixels         |
| Min cluster | `cfg.strategy.chess.minClusterSize`          | Reject isolated responses with fewer neighbours      |
| Broad ring  | `cfg.strategy.chess.ring = ChessRing.Broad`  | Use the r=10 ChESS ring instead of the canonical r=5 |

### Pyramid / upscale

| Control        | Config path                                                 | Notes                                                        |
|----------------|-------------------------------------------------------------|--------------------------------------------------------------|
| Levels         | `cfg.multiscale = MultiscaleConfig.pyramid(n, ...)`         | `1` selects `singleScale()`; `≥ 2` selects `pyramid(...)`    |
| Min size       | `cfg.multiscale = MultiscaleConfig.pyramid(_, minSize, _)`  | Stop the pyramid when either dim drops below this            |
| Upscale factor | `cfg.upscale = UpscaleConfig.fixed(k)`                       | `0`/`1` selects `disabled()`; `2`/`3`/`4` enables upscale    |

Upscaling runs **ahead of the pyramid**. Output coordinates are always
reported in the original input pixel frame — the facade divides `x, y`
by the factor before returning.

### Advanced

- **Refiner** — pick the subpixel refiner.
  `ChessRefiner.withCenterOfMass(...)` is the default and cheapest
  benchmarked option. `ChessRefiner.withForstner(...)` uses a local
  structure-tensor solve. `ChessRefiner.withSaddlePoint(...)` fits a
  quadratic surface.
- **Auto rerun** — re-runs detection whenever a slider changes.

### Overlay

- **Show axes** — toggle the two-arrow overlay.
- **Arrow length** — arrow length in image pixels.
- **Corner radius** — corner dot radius in pixels.

## Webcam mode

The **Webcam** button toggles live detection on the default camera.
The demo uses a single `ChessDetector` instance across frames, so
pyramid and upscale buffers are reused — detection runs every
`requestAnimationFrame` tick.

Stop the webcam with the same button. Configuration sliders stay live
during streaming.

## Embedding in documentation

To embed this demo in an mdBook build, copy the three files
(`index.html`, `app.js`, plus the `testimages/` directory) next to the
book's `pkg/` output and wrap them in an `<iframe>` from a book page.
See `book/build.sh` (if present) or mirror this layout under
`book/src/demo/` with paths rewritten from `../pkg/` to `./pkg/` and
the testimages placed alongside.

## Troubleshooting

- **Blank page / Run button stays disabled**: open the browser
  console. The most common cause is a missing `pkg/` build — rerun
  `wasm-pack build` or verify the path.
- **CORS errors on sample images**: the demo must be served over HTTP,
  not opened via `file://`. Use the Python one-liner above.
- **Webcam fails with `NotAllowedError`**: the browser must be on a
  secure origin (`https://` or `localhost`) and the user must grant
  camera permission.
- **Stale WASM after rebuild**: hard-refresh (Cmd+Shift+R or
  Ctrl+Shift+R) to bust the browser cache.
