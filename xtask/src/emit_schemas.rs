//! Emit the JSON Schema of the detector config.
//!
//! Output goes to `schemas/detector_config.json` at the repo root. With
//! `--check`, the command instead verifies that the committed file matches
//! what would be generated from the current source; CI runs this to catch
//! drift between the config types and the shipped schema.

use anyhow::{bail, Context, Result};
use chess_corners::DetectorConfig;
use schemars::schema_for;
use serde_json::Value;
use std::path::Path;

/// Path of the schema relative to the workspace root.
pub const SCHEMA_PATH: &str = "schemas/detector_config.json";

/// Schema of [`DetectorConfig`] as a JSON value, after the cosmetic
/// clean-ups in [`tidy`].
pub fn detector_config_schema() -> Value {
    let mut schema = serde_json::to_value(schema_for!(DetectorConfig))
        .expect("JsonSchema serialization is infallible");
    tidy(&mut schema);
    schema
}

/// Pretty-printed schema text with a trailing newline.
pub fn render() -> Result<String> {
    let mut text = serde_json::to_string_pretty(&detector_config_schema())
        .context("rendering schema as pretty JSON")?;
    text.push('\n');
    Ok(text)
}

/// Cosmetic, validation-neutral clean-ups so the schema reads well in a
/// form UI:
///
/// - `f32` defaults/bounds widen to `f64` when derived (`0.001_f32` becomes
///   `0.0010000000474974513`); numbers that are exactly an `f32` are
///   rewritten to that `f32`'s shortest decimal form.
/// - rustdoc intra-doc links (``[`Foo`]`` and ``[`Foo`](path)``) in
///   `description` strings are reduced to inline code (`` `Foo` ``).
/// - a `oneOf` made only of string `const` branches (a unit-variant enum)
///   also gets `type: "string"` and an `enum` list, which simple form
///   generators recognise as a closed set. The `oneOf` stays so the
///   per-variant descriptions are not lost; the two are equivalent.
fn tidy(value: &mut Value) {
    match value {
        Value::Number(n) => {
            if let Some(x) = n.as_f64() {
                if n.is_f64() && x.is_finite() && (x as f32) as f64 == x {
                    if let Ok(short) = format!("{}", x as f32).parse::<f64>() {
                        if let Some(num) = serde_json::Number::from_f64(short) {
                            *n = num;
                        }
                    }
                }
            }
        }
        Value::Array(items) => items.iter_mut().for_each(tidy),
        Value::Object(map) => {
            for (key, child) in map.iter_mut() {
                if key == "description" {
                    if let Value::String(text) = child {
                        *text = strip_intra_doc_links(text);
                    }
                } else {
                    tidy(child);
                }
            }
            add_enum_for_const_one_of(map);
        }
        _ => {}
    }
}

fn add_enum_for_const_one_of(map: &mut serde_json::Map<String, Value>) {
    let Some(Value::Array(branches)) = map.get("oneOf") else {
        return;
    };
    let consts: Option<Vec<Value>> = branches
        .iter()
        .map(|b| match (b.get("const"), b.get("type")) {
            (Some(c @ Value::String(_)), Some(Value::String(t))) if t == "string" => {
                Some(c.clone())
            }
            _ => None,
        })
        .collect();
    if let Some(consts) = consts.filter(|c| !c.is_empty()) {
        map.insert("type".into(), Value::String("string".into()));
        map.insert("enum".into(), Value::Array(consts));
    }
}

/// Turn ``[`Foo`]`` / ``[`Foo`](target)`` into `` `Foo` ``.
fn strip_intra_doc_links(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("[`") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 1..];
        // `after` begins with the opening backtick of the code span.
        let Some(end) = after[1..].find('`').map(|i| i + 1) else {
            out.push_str(&rest[start..]);
            return out;
        };
        let code = &after[..=end];
        let tail = &after[end + 1..];
        if let Some(tail) = tail.strip_prefix(']') {
            out.push_str(code);
            rest = match tail.strip_prefix('(') {
                Some(link) => link.split_once(')').map_or(tail, |(_, r)| r),
                None => tail,
            };
        } else {
            out.push('[');
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

pub fn run(workspace_root: &Path, check: bool) -> Result<()> {
    let path = workspace_root.join(SCHEMA_PATH);
    let text = render()?;

    if check {
        let on_disk = std::fs::read_to_string(&path).unwrap_or_default();
        if on_disk == text {
            println!("schema up to date ({SCHEMA_PATH})");
            return Ok(());
        }
        if on_disk.replace("\r\n", "\n") == text {
            bail!(
                "{SCHEMA_PATH} has CRLF line endings; re-checkout with LF \
                 (`git add --renormalize .`), do not re-emit"
            );
        }
        bail!("{SCHEMA_PATH} is out of date; run `cargo xtask emit-schemas` and commit the result");
    }

    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    std::fs::write(&path, &text).with_context(|| format!("writing {}", path.display()))?;
    println!("emitted {}", path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess_corners::UpscaleConfig;
    use serde_json::json;

    #[test]
    fn strips_intra_doc_links() {
        assert_eq!(
            strip_intra_doc_links("see [`Foo`](Self::foo) and [`Bar`] or [x] [`unclosed"),
            "see `Foo` and `Bar` or [x] [`unclosed"
        );
        assert_eq!(strip_intra_doc_links("plain"), "plain");
    }

    #[test]
    fn f32_numbers_are_shortened() {
        let mut v = json!({"a": 0.001_f32 as f64, "b": 0.5, "c": 3});
        tidy(&mut v);
        assert_eq!(v, json!({"a": 0.001, "b": 0.5, "c": 3}));
    }

    fn validator() -> jsonschema::Validator {
        jsonschema::validator_for(&detector_config_schema()).expect("schema compiles")
    }

    fn presets() -> Vec<(&'static str, DetectorConfig)> {
        vec![
            ("default", DetectorConfig::default()),
            ("chess", DetectorConfig::chess()),
            ("chess_multiscale", DetectorConfig::chess_multiscale()),
            ("radon", DetectorConfig::radon()),
            ("radon_multiscale", DetectorConfig::radon_multiscale()),
            (
                "forstner_upscale",
                DetectorConfig::chess()
                    .with_chess(|c| c.refiner = chess_corners::ChessRefiner::forstner())
                    .with_upscale(UpscaleConfig::fixed(3))
                    .without_orientation(),
            ),
            (
                "saddle_disk",
                DetectorConfig::chess_multiscale()
                    .with_chess(|c| c.refiner = chess_corners::ChessRefiner::saddle_point())
                    .with_orientation_method(chess_corners::OrientationMethod::DiskFit),
            ),
        ]
    }

    #[test]
    fn serialized_presets_validate() {
        let validator = validator();
        for (name, cfg) in presets() {
            let instance = serde_json::to_value(cfg).unwrap();
            let errors: Vec<String> = validator
                .iter_errors(&instance)
                .map(|e| format!("{} at {}", e, e.instance_path))
                .collect();
            assert!(errors.is_empty(), "{name}: {errors:?}\n{instance}");
            // The schema must describe what serde accepts, too.
            let back: DetectorConfig = serde_json::from_value(instance).unwrap();
            assert_eq!(back, cfg, "{name}: serde round-trip");
        }
    }

    #[test]
    fn empty_object_is_valid_like_serde() {
        // `#[serde(default)]` on the container: `{}` deserializes to the default.
        assert!(validator().is_valid(&json!({})));
        let cfg: DetectorConfig = serde_json::from_value(json!({})).unwrap();
        assert_eq!(cfg, DetectorConfig::default());
    }

    #[test]
    fn unknown_keys_follow_serde() {
        // No `deny_unknown_fields` anywhere: serde ignores unknown keys, so
        // the schema must not reject them either.
        let instance = json!({"threshold": 30.0, "not_a_field": 1});
        assert!(serde_json::from_value::<DetectorConfig>(instance.clone()).is_ok());
        assert!(validator().is_valid(&instance));
    }

    #[test]
    fn out_of_range_values_are_rejected() {
        let validator = validator();
        for bad in [
            json!({"threshold": -1.0}),
            json!({"upscale": {"fixed": 7}}),
            json!({"multiscale": {"pyramid": {"levels": 0, "min_size": 128, "refinement_radius": 3}}}),
            json!({"strategy": {"radon": {"image_upsample": 3}}}),
            json!({"strategy": {"radon": {"ray_radius": 0}}}),
            json!({"strategy": {"sobel": {}}}),
            json!({"orientation_method": "nope"}),
        ] {
            assert!(!validator.is_valid(&bad), "should reject {bad}");
        }
    }

    /// `true` when cargo's feature unification turned on `chess-corners/ml-refiner`
    /// (e.g. `cargo test --workspace --features ml-refiner`), which adds the
    /// `Ml` refiner variant to the derived schema. The shipped schema is
    /// always generated by `cargo run -p xtask`, where that feature is off.
    fn ml_refiner_compiled_in() -> bool {
        serde_json::from_str::<chess_corners::ChessRefiner>("\"ml\"").is_ok()
    }

    #[test]
    fn ml_refiner_variant_is_absent_from_default_build() {
        let text = render().unwrap();
        assert_eq!(
            text.contains("\"ml\""),
            ml_refiner_compiled_in(),
            "schema must list the Ml refiner exactly when ml-refiner is compiled in"
        );
    }

    #[test]
    fn committed_schema_is_current() {
        if ml_refiner_compiled_in() {
            eprintln!("skipped: ml-refiner is unified into this build; CI checks via `cargo run -p xtask`");
            return;
        }
        let on_disk = std::fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join(SCHEMA_PATH),
        )
        .expect("schemas/detector_config.json exists; run `cargo xtask emit-schemas`");
        assert_eq!(
            on_disk.replace("\r\n", "\n"),
            render().unwrap(),
            "schema drift; run `cargo xtask emit-schemas`"
        );
    }
}
