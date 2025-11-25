use anyhow::Context;
use chess::{find_corners_image, ChessParams};
use image::{
    imageops::{resize, FilterType},
    ImageBuffer, ImageReader, Luma,
};
use serde::Serialize;
use std::{fs::File, io::Write, path::PathBuf};

#[derive(Serialize)]
struct CornerOut {
    x: f32,
    y: f32,
    strength: f32,
    scale: u8,
}

#[derive(Serialize)]
struct CornerDump {
    image: String,
    width: u32,
    height: u32,
    downsample: u32,
    corners: Vec<CornerOut>,
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let input: PathBuf = args
        .next()
        .expect("usage: dump_corners <image> [--downsample N]")
        .into();

    let mut downsample: u32 = 1;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--downsample" => {
                let v = args
                    .next()
                    .context("expected an integer after --downsample")?;
                downsample = v
                    .parse()
                    .context("could not parse downsample factor (use integer >= 1)")?;
                if downsample == 0 {
                    anyhow::bail!("downsample factor must be >= 1");
                }
            }
            other => anyhow::bail!("unknown argument: {other}"),
        }
    }

    let img = ImageReader::open(&input)?.decode()?.to_luma8();
    let work_img = if downsample > 1 {
        let w = (img.width() + downsample - 1) / downsample;
        let h = (img.height() + downsample - 1) / downsample;
        resize(&img, w, h, FilterType::Triangle)
    } else {
        img.clone()
    };

    let params = ChessParams::default();
    let mut corners = find_corners_image(&work_img, &params);
    if downsample > 1 {
        let s = downsample as f32;
        for c in &mut corners {
            c.xy[0] *= s;
            c.xy[1] *= s;
        }
    }

    println!("Detected {} corners (downsample={})", corners.len(), downsample);

    let json_out = input.with_extension("corners.json");
    let dump = CornerDump {
        image: input.to_string_lossy().into_owned(),
        width: img.width(),
        height: img.height(),
        downsample,
        corners: corners
            .iter()
            .map(|c| CornerOut {
                x: c.xy[0],
                y: c.xy[1],
                strength: c.strength,
                scale: c.scale,
            })
            .collect(),
    };
    let mut json_file = File::create(&json_out)?;
    serde_json::to_writer_pretty(&mut json_file, &dump)?;
    json_file.write_all(b"\n")?;
    println!("Saved JSON dump to {}", json_out.display());

    // simple visualization: draw small 3x3 white squares around corners
    let mut vis: ImageBuffer<Luma<u8>, _> = img.clone();
    for c in &corners {
        let x = c.xy[0].round() as i32;
        let y = c.xy[1].round() as i32;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let xx = x + dx;
                let yy = y + dy;
                if xx >= 0 && yy >= 0 && xx < vis.width() as i32 && yy < vis.height() as i32 {
                    vis.put_pixel(xx as u32, yy as u32, Luma([255u8]));
                }
            }
        }
    }

    let out = input.with_extension("corners.png");
    vis.save(&out)?;
    println!("Saved visualization to {}", out.display());

    Ok(())
}
