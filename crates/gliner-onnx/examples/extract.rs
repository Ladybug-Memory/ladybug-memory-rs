//! Sample graph extraction over the ladybug test texts.
//!
//! ```sh
//! cargo run -p gliner-onnx --example extract -- <snapshot-dir> [threshold]
//! ```
//!
//! (needs `lbm fetch-model` once for the model snapshot + ORT lib)
//!
//! Prints entity mentions (graph nodes) that `LadybugMemory::store_with_entities`
//! will link via `MentionedIn` edges.

use gliner_onnx::GlinerOnnx;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let snap = std::env::args()
        .nth(1)
        .unwrap_or_else(|| {
            format!(
                "{}/.cache/huggingface/hub/models--DanKau--gliner2.5-base-v1-onnx/snapshots/b4920ab2b4f39b2da60c9761802e3d8cc6ec9993",
                std::env::var("HOME").unwrap()
            )
        });
    let threshold: f32 = std::env::args()
        .nth(2)
        .map(|s| s.parse().expect("threshold must be a float"))
        .unwrap_or(0.5);

    let t0 = Instant::now();
    let mut model = GlinerOnnx::load(&snap)?;
    println!("loaded in {:.2}s", t0.elapsed().as_secs_f32());

    let labels = vec![
        "person".to_string(),
        "organization".to_string(),
        "location".to_string(),
        "product".to_string(),
        "event".to_string(),
        "date".to_string(),
        "technology".to_string(),
        "concept".to_string(),
    ];
    let texts = [
        "Apple Inc. announced today that Tim Cook will be presenting at their headquarters in Cupertino, California. The event is scheduled for March 15, 2024. Microsoft and Google are also expected to make announcements during the same week.",
        "Elon Musk founded SpaceX in 2002 and Tesla's headquarters are in Austin, Texas.",
    ];

    for text in texts {
        let t1 = Instant::now();
        let out = model.extract(text, &labels, threshold)?;
        println!("\n{text}\n{} mentions in {:.1} ms", out.mentions.len(), t1.elapsed().as_secs_f32() * 1000.0);
        for m in &out.mentions {
            println!(
                "  {:<12} {:<22?} {:>6.2}%  chars [{},{}) words [{},{})",
                m.label, m.text, m.confidence * 100.0, m.start, m.end, m.word_start, m.word_end,
            );
        }
    }
    Ok(())
}
