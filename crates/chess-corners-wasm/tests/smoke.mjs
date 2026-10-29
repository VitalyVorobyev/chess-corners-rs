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
  ImageRefiner,
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
assert.deepEqual(defaults.strategy, { chess: { ring: "canonical", center_of_mass: { radius: 2 } } });
assert.equal("refiner" in defaults, false, "no refiner key when none is set");
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
  JSON.parse(DetectorConfig.fromJson('{"refiner":{"saddle_point":{}}}').toJson()).refiner.saddle_point.min_abs_det,
  0.0001,
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
assert.throws(() => DetectorConfig.fromJson("{not json"), SyntaxError);
// Stale 1.x keys are rejected at every level, and the message names the key.
throwsWith(() => DetectorConfig.fromJson('{"noise_sigma": 1.5}'), "unknown field `noise_sigma`");
throwsWith(
  () => DetectorConfig.fromJson('{"strategy": {"chess": {"refiner": "center_of_mass"}}}'),
  "unknown field `refiner`",
);
throwsWith(
  () => DetectorConfig.fromJson('{"multiscale": {"pyramid": {"levels": 3, "min_size": 128, "refinement_radius": 3, "x": 1}}}'),
  "unknown field `x`",
);
// Semantic validation (allowed upscale factors) happens at detector construction,
// where the existing API throws the message as a plain string.
assert.throws(
  () => ChessDetector.withConfig(DetectorConfig.fromJson('{"upscale": {"fixed": 7}}')),
  (e) => String(e).includes("7"),
);

// Uncertainty: an immutable value carrying its noise level; JSON and typed
// accessors agree.
const off = Uncertainty.off();
const est = Uncertainty.propagated();
const known = Uncertainty.propagated(2.5);
assert.deepEqual([off.kind, off.noiseSigma], ["off", undefined]);
assert.deepEqual([est.kind, est.noiseSigma], ["propagated", undefined]);
assert.deepEqual([known.kind, known.noiseSigma], ["propagated", 2.5]);
assert.equal(DetectorConfig.chess().noiseSigma, undefined, "no noise-sigma field");
assert.equal(DetectorConfig.chess().withNoiseSigma, undefined, "no noise-sigma builder");
const jsonOf = (u) => JSON.parse(DetectorConfig.chess().withUncertainty(u).toJson()).uncertainty;
assert.equal(jsonOf(off), "off");
assert.deepEqual(jsonOf(est), { propagated: { noise: "estimate" } });
assert.deepEqual(jsonOf(known), { propagated: { noise: { known: 2.5 } } });
const knownBack = DetectorConfig.fromJson('{"uncertainty": {"propagated": {"noise": {"known": 2.5}}}}');
assert.deepEqual([knownBack.uncertainty.kind, knownBack.uncertainty.noiseSigma], ["propagated", 2.5]);
for (const bad of ['"propagated"', '{"propagated": {}}', '{"propagated": {"noise": "known"}}', '"bogus"']) {
  assert.throws(() => DetectorConfig.fromJson(`{"uncertainty": ${bad}}`), undefined, bad);
}

// Position covariance: stride 3 per corner, aligned with `detect`; finite with
// every localizer under `Uncertainty.propagated()`, NaN under the default
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
    '{"refiner": {"saddle_point": {}}, "uncertainty": {"propagated": {"noise": "estimate"}}}',
  ),
);
const found = saddle.detect(board, size, size);
const cov = saddle.positionCovariance();
assert.ok(found.length > 0);
assert.equal(cov.length, (found.length / 7) * 3);
assert.ok(cov.some((v) => Number.isFinite(v) && v > 0), "saddle point reports covariances");
const plain = ChessDetector.withConfig(DetectorConfig.chess().withUncertainty(Uncertainty.propagated()));
const plainFound = plain.detect(board, size, size);
assert.equal(plain.positionCovariance().length, (plainFound.length / 7) * 3);
assert.ok(plain.positionCovariance().every((v) => Number.isFinite(v)), "default refiner reports covariances");
for (const cfg of [
  DetectorConfig.chess(),
  DetectorConfig.chessMultiscale(),
  DetectorConfig.radon(),
  DetectorConfig.radonMultiscale(),
]) {
  assert.equal(cfg.uncertainty.kind, "off", "off is the default");
}
assert.equal(JSON.parse(DetectorConfig.chess().toJson()).uncertainty, "off");
assert.equal(DetectorConfig.fromJson('{"uncertainty": {"propagated": {"noise": "estimate"}}}').uncertainty.kind, "propagated");
assert.equal(DetectorConfig.fromJson("{}").uncertainty.kind, "off");
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

// The image refiner is a detector-independent value; Radon keeps its peak fit
// without one, and both localizers report a covariance.
const radonCfg = DetectorConfig.radon().withUncertainty(Uncertainty.propagated());
assert.equal(radonCfg.refiner, undefined);
assert.equal(JSON.parse(radonCfg.toJson()).refiner, undefined);
const peakFit = ChessDetector.withConfig(radonCfg);
peakFit.detect(board, size, size);
assert.ok(peakFit.positionCovariance().every((v) => Number.isFinite(v)));
const sp = new SaddlePointConfig();
sp.maxOffset = 1.5;
const refiner = ImageRefiner.saddlePointWith(sp);
sp.maxOffset = 0.5; // the refiner holds a copy
assert.equal(refiner.kind, "saddle_point");
assert.equal(refiner.saddlePointConfig.maxOffset, 1.5);
assert.equal(ImageRefiner.saddlePoint().saddlePointConfig.maxOffset, new SaddlePointConfig().maxOffset);
const saddleCfg = radonCfg.withRefiner(refiner);
assert.equal(radonCfg.refiner, undefined, "withRefiner returns a copy");
assert.equal(saddleCfg.refiner.kind, "saddle_point");
assert.equal(JSON.parse(saddleCfg.toJson()).refiner.saddle_point.max_offset, 1.5);
assert.equal(JSON.parse(saddleCfg.toJson()).refiner.saddle_point.radius, 2);
const radon = ChessDetector.withConfig(saddleCfg);
radon.detect(board, size, size);
assert.ok(radon.positionCovariance().some((v) => Number.isFinite(v) && v > 0));
// The refiner survives strategy switches and clears with withoutRefiner.
assert.equal(saddleCfg.withChess({}).refiner.kind, "saddle_point");
assert.equal(DetectorConfig.chess().withRefiner(refiner).withRadon({}).refiner.kind, "saddle_point");
assert.equal(saddleCfg.withoutRefiner().refiner, undefined);
assert.equal(JSON.parse(saddleCfg.withoutRefiner().toJson()).refiner, undefined);
// The center-of-mass radius is edited through the shared chess payload.
const comCfg = DetectorConfig.chess();
comCfg.strategy.chess.centerOfMass.radius = 3;
assert.equal(JSON.parse(comCfg.toJson()).strategy.chess.center_of_mass.radius, 3);
// A refiner on a ChESS detector reports covariances too.
const chessRefined = ChessDetector.withConfig(
  DetectorConfig.chess().withUncertainty(Uncertainty.propagated()).withRefiner(ImageRefiner.saddlePoint()),
);
chessRefined.detect(board, size, size);
assert.ok(chessRefined.positionCovariance().some((v) => Number.isFinite(v) && v > 0));
// A refiner key in the strategy option objects is rejected.
assert.throws(() => DetectorConfig.chess().withChess({ refiner: 1 }), (e) => String(e).includes("unexpected option"));

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
saddle.applyConfig(DetectorConfig.fromJson('{"uncertainty": {"propagated": {"noise": "estimate"}}}'));
assert.equal(saddle.noiseSigma(), undefined, "cleared by applyConfig");
saddle.detect(board, size, size);
assert.equal(saddle.noiseSigma(), sigma, "the per-image estimate");
assert.throws(() => saddle.detect(board, size + 1, size));
assert.equal(saddle.noiseSigma(), undefined, "cleared by a failed detection");
const supplied = ChessDetector.withConfig(DetectorConfig.chess().withUncertainty(Uncertainty.propagated(1.25)));
supplied.detect(board, size, size);
assert.equal(supplied.noiseSigma(), 1.25);
quiet.detect(board, size, size);
assert.equal(quiet.noiseSigma(), undefined, "none under Uncertainty.off()");
assert.throws(
  () => ChessDetector.withConfig(DetectorConfig.chess().withUncertainty(Uncertainty.propagated(-1))),
  (e) => String(e).includes("noise sigma"),
);

console.log("chess-corners-wasm JSON smoke test: ok");
