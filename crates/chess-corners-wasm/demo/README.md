# chess-corners-rs — Interactive WASM Demo

A single-page browser app that runs the `chess-corners-rs` detector in
WebAssembly. Supports image upload / drag-and-drop, bundled sample
images, live parameter tuning, and webcam-streaming detection.

## What it shows

The demo exposes every `ChessDetector` setter from
`chess-corners-wasm`, including the v0.6.0 additions (two-axis
descriptor, optional integer upscaling):

- `set_threshold`, `set_nms_radius`, `set_min_cluster_size`,
  `set_broad_mode`
- `set_pyramid_levels`, `set_pyramid_min_size`
- **`set_upscale_factor`** (new in 0.6.0 — 0/1 disables, 2/3/4 upscale
  bilinearly before detection)
- `set_refiner` — `center_of_mass` / `forstner` / `saddle_point`

For each detected corner it overlays:

- a white dot at the subpixel location,
- a teal arrow along `axes[0].angle` (line direction ∈ [0, π)),
- an orange arrow along `axes[1].angle` (polar, dark sector runs CCW
  from axis 0 to axis 1).

Hovering a corner shows its contrast, fit RMS, and both angles ± 1σ.

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

| Control        | WASM setter             | Notes                                                |
|----------------|-------------------------|------------------------------------------------------|
| Threshold      | `set_threshold`         | Relative threshold (fraction of max response)        |
| NMS radius     | `set_nms_radius`        | Non-max suppression window radius, in pixels         |
| Min cluster    | `set_min_cluster_size`  | Reject isolated responses with fewer neighbours      |
| Broad ring     | `set_broad_mode`        | Use the r=10 ChESS ring instead of the canonical r=5 |

### Pyramid / upscale

| Control        | WASM setter                 | Notes                                                        |
|----------------|-----------------------------|--------------------------------------------------------------|
| Levels         | `set_pyramid_levels(n: u8)` | `1` = single-scale; `≥2` = coarse-to-fine                    |
| Min size       | `set_pyramid_min_size`      | Stop the pyramid when either dim drops below this            |
| Upscale factor | `set_upscale_factor`        | `0`/`1` = off; `2`/`3`/`4` = bilinear upscale before detect  |

Upscaling runs **ahead of the pyramid**. Output coordinates are always
reported in the original input pixel frame — the facade divides `x, y`
by the factor before returning.

### Advanced

- **Refiner** — pick the subpixel refiner. `center_of_mass` is the
  default, fastest, and least precise. `forstner` uses the structure
  tensor of the ring-sampled image and is typically the most accurate
  for distorted boards. `saddle_point` fits a quadratic surface.
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
