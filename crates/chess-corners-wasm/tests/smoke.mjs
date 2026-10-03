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
const { default: init, DetectorConfig, ChessDetector, default_detector_config_json, defaultDetectorConfigJson } = mod;
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

console.log("chess-corners-wasm JSON smoke test: ok");
