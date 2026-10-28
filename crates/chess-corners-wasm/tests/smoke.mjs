// Node smoke test for the JSON config path of the built wasm package.
//
// Usage (after `wasm-pack build crates/chess-corners-wasm --target web`):
//   node crates/chess-corners-wasm/tests/smoke.mjs [pkg-dir]
//
// Exercises the real wasm module (the Rust unit tests cannot call
// `JSON.parse` / `JSON.stringify`, which the JSON glue delegates to).
import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const pkg = path.resolve(process.argv[2] ?? path.join(here, "..", "pkg"));
const mod = await import(pathToFileURL(path.join(pkg, "chess_corners_wasm.js")).href);
const {
  default: init,
  DetectorConfig,
  ChessDetector,
  RadonRefiner,
  SaddlePointConfig,
  Uncertainty,
  default_detector_config_json,
  defaultDetectorConfigJson,
  estimateNoiseSigma,
  estimateNoiseSigmaRgba,
} = mod;
await init({ module_or_path: readFileSync(path.join(pkg, "chess_corners_wasm_bg.wasm")) });

const throwsWith = (fn, needle) => {
  assert.throws(fn, (e) => {
    assert.ok(e instanceof Error, `expected an Error, got ${typeof e}`);
    assert.ok(e.message.includes(needle), `message ${JSON.stringify(e.message)} lacks ${JSON.stringify(needle)}`);
    return true;
  });
};

// Default JSON: parseable, snake_case, f32 values print shortest.
const defaults = JSON.parse(default_detector_config_json());
assert.equal(default_detector_config_json(), defaultDetectorConfigJson());
assert.equal(defaults.threshold, 30);
assert.equal(defaults.merge_radius, 3);
assert.deepEqual(defaults.strategy, { chess: { ring: "canonical", refiner: { center_of_mass: { radius: 2 } } } });
assert.equal(defaults.multiscale, "single_scale");
assert.equal(defaults.upscale, "disabled");
assert.equal(defaults.orientation_method, "ring_fit");

// Every preset round-trips and f32 numbers keep their short form.
for (const name of ["chess", "chessMultiscale", "radon", "radonMultiscale"]) {
  const cfg = DetectorConfig[name]();
  const json = cfg.toJson();
  assert.equal(DetectorConfig.fromJson(json).toJson(), json, name);
}
assert.equal(JSON.parse(DetectorConfig.radon().toJson()).threshold, 0.28);
assert.equal(
  JSON.parse(DetectorConfig.fromJson('{"strategy":{"chess":{"refiner":{"forstner":{}}}}}').toJson()).strategy.chess.refiner.forstner.min_det,
  0.001,
);

// Empty object = defaults; partial objects only override what they name.
assert.equal(DetectorConfig.fromJson("{}").toJson(), default_detector_config_json());
const partial = JSON.parse(
  DetectorConfig.fromJson('{"threshold": 55, "strategy": {"radon": {"ray_radius": 6}}}').toJson(),
);
assert.equal(partial.threshold, 55);
assert.equal(partial.strategy.radon.ray_radius, 6);
assert.equal(partial.strategy.radon.image_upsample, 2);

// The typed class API and the JSON path agree.
assert.equal(
  DetectorConfig.chessMultiscale().toJson(),
  JSON.stringify({ ...defaults, multiscale: { pyramid: { levels: 3, min_size: 128, refinement_radius: 3 } } }),
);

// A JSON-built config drives a detector.
const detector = ChessDetector.withConfig(DetectorConfig.fromJson('{"upscale": {"fixed": 2}}'));
assert.equal(JSON.parse(detector.getConfig().toJson()).upscale.fixed, 2);

// Errors are real JS errors with a useful message.
throwsWith(() => DetectorConfig.fromJson("{not json"), "JSON");
throwsWith(() => DetectorConfig.fromJson('{"threshold": "high"}'), "expected f32");
throwsWith(() => DetectorConfig.fromJson('{"strategy": {"sobel": {}}}'), "sobel");
throwsWith(() => DetectorConfig.fromJson('{"upscale": {"fixed": 2.5}}'), "invalid type");
// Semantic validation (allowed upscale factors) happens at detector construction,
// where the existing API throws the message as a plain string.
assert.throws(
  () => ChessDetector.withConfig(DetectorConfig.fromJson('{"upscale": {"fixed": 7}}')),
  (e) => String(e).includes("7"),
);

// Noise level: JSON and typed accessors agree; `undefined` estimates per image.
assert.equal(DetectorConfig.chess().noiseSigma, undefined);
assert.equal(DetectorConfig.fromJson('{"noise_sigma": 2.5}').noiseSigma, 2.5);
assert.equal(JSON.parse(DetectorConfig.chess().withNoiseSigma(2.5).toJson()).noise_sigma, 2.5);

// Position covariance: stride 3 per corner, aligned with `detect`; finite with
// every localizer under `uncertainty: "propagated"`, NaN under the default
// `"off"`.
const size = 128;
const board = new Uint8Array(size * size);
let seed = 1;
for (let y = 0; y < size; y++) {
  for (let x = 0; x < size; x++) {
    seed = (seed * 1103515245 + 12345) % 2147483648;
    const noise = (seed / 2147483648 - 0.5) * 8;
    const bright = (Math.floor(x / 16) + Math.floor(y / 16)) % 2 === 0;
    board[y * size + x] = Math.round((bright ? 200 : 40) + noise);
  }
}
const saddle = ChessDetector.withConfig(
  DetectorConfig.fromJson(
    '{"strategy": {"chess": {"refiner": {"saddle_point": {}}}}, "uncertainty": "propagated"}',
  ),
);
const found = saddle.detect(board, size, size);
const cov = saddle.positionCovariance();
assert.ok(found.length > 0);
assert.equal(cov.length, (found.length / 7) * 3);
assert.ok(cov.some((v) => Number.isFinite(v) && v > 0), "saddle point reports covariances");
const plain = ChessDetector.withConfig(DetectorConfig.chess().withUncertainty(Uncertainty.Propagated));
const plainFound = plain.detect(board, size, size);
assert.equal(plain.positionCovariance().length, (plainFound.length / 7) * 3);
assert.ok(plain.positionCovariance().every((v) => Number.isFinite(v)), "default refiner reports covariances");
for (const cfg of [
  DetectorConfig.chess(),
  DetectorConfig.chessMultiscale(),
  DetectorConfig.radon(),
  DetectorConfig.radonMultiscale(),
]) {
  assert.equal(cfg.uncertainty, Uncertainty.Off, "Off is the default tier");
}
assert.equal(JSON.parse(DetectorConfig.chess().toJson()).uncertainty, "off");
assert.equal(DetectorConfig.fromJson('{"uncertainty": "propagated"}').uncertainty, Uncertainty.Propagated);
assert.equal(DetectorConfig.fromJson("{}").uncertainty, Uncertainty.Off);
assert.equal(
  JSON.parse(DetectorConfig.chess().withUncertainty(Uncertainty.Propagated).toJson()).uncertainty,
  "propagated",
);
const quiet = new ChessDetector();
const quietFound = quiet.detect(board, size, size);
assert.equal(quiet.positionCovariance().length, (quietFound.length / 7) * 3);
assert.ok(quiet.positionCovariance().every((v) => Number.isNaN(v)), "the default reports none");
assert.deepEqual(Array.from(quietFound), Array.from(plainFound), "positions identical across tiers");
// The covariance buffer never outlives the detection it belongs to.
assert.throws(() => saddle.detect(board, size + 1, size));
assert.equal(saddle.positionCovariance().length, 0, "cleared by a failed detection");
saddle.detect(board, size, size);
assert.ok(saddle.positionCovariance().length > 0);
saddle.applyConfig(DetectorConfig.chess());
assert.equal(saddle.positionCovariance().length, 0, "cleared by applyConfig");

// Radon keeps its peak fit by default, which reports a covariance as well as
// the saddle-point refinement.
const radonCfg = DetectorConfig.radon().withUncertainty(Uncertainty.Propagated);
assert.equal(JSON.parse(radonCfg.toJson()).strategy.radon.refiner, "peak_fit");
const peakFit = ChessDetector.withConfig(radonCfg);
peakFit.detect(board, size, size);
assert.ok(peakFit.positionCovariance().every((v) => Number.isFinite(v)));
const saddleCfg = radonCfg.withRadonRefiner(RadonRefiner.withSaddlePoint(new SaddlePointConfig()));
assert.equal(JSON.parse(saddleCfg.toJson()).strategy.radon.refiner.saddle_point.radius, 2);
const radon = ChessDetector.withConfig(saddleCfg);
radon.detect(board, size, size);
assert.ok(radon.positionCovariance().some((v) => Number.isFinite(v) && v > 0));

// Noise estimate: the board carries ±4 gray levels of uniform noise (σ ≈ 2.3).
const sigma = estimateNoiseSigma(board, size, size);
assert.ok(sigma > 1.5 && sigma < 3.5, `estimated σ ${sigma}`);
const rgba = new Uint8Array(size * size * 4);
for (let i = 0; i < size * size; i++) {
  rgba.set([board[i], board[i], board[i], 255], 4 * i);
}
// Same grayscale conversion as `detectRgba`, which truncates to integers.
const sigmaRgba = estimateNoiseSigmaRgba(rgba, size, size);
assert.ok(Math.abs(sigmaRgba / sigma - 1) < 0.1, `RGBA σ ${sigmaRgba} vs gray σ ${sigma}`);
assert.throws(() => estimateNoiseSigmaRgba(board, size, size));
assert.equal(estimateNoiseSigma(new Uint8Array(4 * 4).fill(128), 4, 4), undefined);
assert.throws(() => estimateNoiseSigma(board, size + 1, size));

// The noise σ behind a detection's covariances: estimated, supplied, or none.
saddle.applyConfig(DetectorConfig.fromJson('{"uncertainty": "propagated"}'));
assert.equal(saddle.noiseSigma(), undefined, "cleared by applyConfig");
saddle.detect(board, size, size);
assert.equal(saddle.noiseSigma(), sigma, "the per-image estimate");
assert.throws(() => saddle.detect(board, size + 1, size));
assert.equal(saddle.noiseSigma(), undefined, "cleared by a failed detection");
const supplied = ChessDetector.withConfig(DetectorConfig.fromJson('{"uncertainty": "propagated", "noise_sigma": 1.25}'));
supplied.detect(board, size, size);
assert.equal(supplied.noiseSigma(), 1.25);
quiet.detect(board, size, size);
assert.equal(quiet.noiseSigma(), undefined, "none under Uncertainty.Off");

console.log("chess-corners-wasm JSON smoke test: ok");
