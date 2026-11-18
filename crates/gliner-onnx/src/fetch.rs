//! First-run model fetch for the GLiNER2.5 ONNX snapshot.
//!
//! The CLI depends on the `nicolasembleton/gliner2.5-base-v1-onnx`
//! single-graph export (~780 MB), which is too large to vendor. On a fresh
//! machine the snapshot directory does not exist, so [`ensure_model`]
//! downloads the missing files before the first inference runs.
//!
//! Strategy (std-only, no new HTTP dependencies):
//!
//! 1. If every expected file is already present, do nothing.
//! 2. If `HF_HUB_OFFLINE=1` is set, fail with a clear message instead of
//!    touching the network.
//! 3. Prefer the `hf` CLI (`hf download … --local-dir <dir>`) when it is on
//!    `PATH` — it handles auth, resume, and the CDN for free.
//! 4. Otherwise fall back to `curl -fSL` against `{HF_ENDPOINT}/<repo>/`
//!    `resolve/<revision>/<file>`, one file at a time, writing atomically
//!    via `.part` + rename so interrupted downloads never poison the cache.
//!
//! Knobs (all optional):
//!
//! | env | default |
//! |---|---|
//! | `LB_MEMORY_SNAPSHOT` (target dir) | `~/.cache/ladybug-memory/models/gliner2.5-base-v1-nico-onnx` |
//! | `LB_MEMORY_MODEL_REPO` | `nicolasembleton/gliner2.5-base-v1-onnx` |
//! | `LB_MEMORY_MODEL_REVISION` | `main` |
//! | `HF_ENDPOINT` | `https://huggingface.co` |
//! | `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | (none — repo is public) |
//! | `HF_HUB_OFFLINE=1` | fail fast instead of downloading |

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};

/// Hugging Face repo holding the single-graph ONNX export
/// (`onnx/model.onnx` + `onnx/heads.onnx`, the `SparseRelationScorer`).
pub const MODEL_REPO: &str = "nicolasembleton/gliner2.5-base-v1-onnx";

/// Files every snapshot directory must hold for the Nico layout.
/// (Deliberately not the whole repo: `attrs.onnx` / `records.onnx` would
/// double the download for graphs this crate never loads.)
pub const MODEL_FILES: &[&str] = &[
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "export_config.json",
    "onnx/model.onnx",
    "onnx/heads.onnx",
];

/// Minimum accepted size for the two ONNX graphs (bytes). Catches truncated
/// downloads that still leave a file behind: `model.onnx` is ~746 MB and
/// `heads.onnx` ~23 MB, so 10 MB / 1 MB floors only reject garbage.
fn min_size(rel: &str) -> u64 {
    match rel {
        "onnx/model.onnx" => 10 * 1024 * 1024,
        "onnx/heads.onnx" => 1024 * 1024,
        _ => 1,
    }
}

/// Default snapshot directory: `~/.cache/ladybug-memory/models/…`.
/// Honors `LB_MEMORY_SNAPSHOT` like the rest of the CLI.
pub fn default_model_dir() -> PathBuf {
    std::env::var("LB_MEMORY_SNAPSHOT").map(PathBuf::from).unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        PathBuf::from(format!(
            "{home}/.cache/ladybug-memory/models/gliner2.5-base-v1-nico-onnx"
        ))
    })
}

fn repo() -> String {
    std::env::var("LB_MEMORY_MODEL_REPO").unwrap_or_else(|_| MODEL_REPO.into())
}

fn revision() -> String {
    std::env::var("LB_MEMORY_MODEL_REVISION").unwrap_or_else(|_| "main".into())
}

fn endpoint() -> String {
    std::env::var("HF_ENDPOINT")
        .map(|e| e.trim_end_matches('/').to_string())
        .unwrap_or_else(|_| "https://huggingface.co".into())
}

/// Direct download URL for one repo-relative file.
pub fn file_url(rel: &str) -> String {
    format!("{}/{}/resolve/{}/{}", endpoint(), repo(), revision(), rel)
}

/// Repo-relative paths of the files missing (or suspiciously small) in `dir`.
pub fn missing_files(dir: &Path) -> Vec<String> {
    MODEL_FILES
        .iter()
        .filter(|rel| {
            let p = dir.join(rel);
            match std::fs::metadata(&p) {
                Ok(m) => m.len() < min_size(rel),
                Err(_) => true,
            }
        })
        .map(|s| s.to_string())
        .collect()
}

/// True when `dir` holds the DanKau 3-graph split (no relation head).
/// Loadable as-is; nothing to fetch.
pub fn has_dankau_layout(dir: &Path) -> bool {
    ["tokenizer.json", "onnx/encoder.onnx", "onnx/boundary.onnx", "onnx/classifier.onnx"]
        .iter()
        .all(|rel| dir.join(rel).is_file())
}

/// True when `dir` holds every expected file at a plausible size.
pub fn is_complete(dir: &Path) -> bool {
    missing_files(dir).is_empty()
}

/// True when the directory is loadable: the Nico export or the DanKau
/// split. (Keep in sync with `GlinerOnnx::load`'s layout detection.)
pub fn is_loadable(dir: &Path) -> bool {
    missing_files(dir).is_empty() || has_dankau_layout(dir)
}

/// Bearer token for gated repos, when set.
fn auth_token() -> Option<String> {
    std::env::var("HF_TOKEN")
        .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
        .ok()
        .filter(|t| !t.trim().is_empty())
}

/// Download the missing snapshot files into `dir` (creating it), then verify
/// completeness. A no-op when the directory is already loadable (Nico or
/// DanKau layout).
pub fn ensure_model(dir: &Path) -> Result<()> {
    if is_loadable(dir) {
        return Ok(());
    }
    let missing = missing_files(dir);
    if std::env::var("HF_HUB_OFFLINE").as_deref() == Ok("1") {
        return Err(anyhow!(
            "model snapshot incomplete at {} (missing: {}) and HF_HUB_OFFLINE=1; \
             download {} or unset HF_HUB_OFFLINE",
            dir.display(),
            missing.join(", "),
            file_url("onnx/model.onnx"),
        ));
    }
    std::fs::create_dir_all(dir)
        .with_context(|| format!("creating {}", dir.display()))?;
    // Rough total for the status line (model ~746 + heads ~23 + tok ~8 MB).
    eprintln!(
        "lbm: fetching GLiNER model into {} ({} file(s) missing, ~780 MB on first run)…",
        dir.display(),
        missing.len(),
    );
    let via_hf = ensure_via_hf_cli(dir, &missing)
        .with_context(|| format!("fetching {MODEL_REPO} via `hf download`"));
    let fetch_err = match via_hf {
        Ok(()) => None,
        Err(e) => {
            eprintln!("lbm: `hf` download failed ({e:#}); falling back to curl…");
            None::<anyhow::Error>.filter(|_| false).or_else(|| {
                match ensure_via_curl(dir, &missing) {
                    Ok(()) => None,
                    Err(e2) => Some(anyhow!(
                        "model fetch failed.\n- hf: {e:#}\n- curl: {e2:#}\n\
                         Install the `hf` CLI (brew install hf), or download \
                         {} into {} manually.",
                        file_url("onnx/model.onnx"),
                        dir.display(),
                    )),
                }
            })
        }
    };
    if let Some(e) = fetch_err {
        return Err(e);
    }
    let still_missing = missing_files(dir);
    if !still_missing.is_empty() {
        return Err(anyhow!(
            "model fetch incomplete at {} (still missing: {})",
            dir.display(),
            still_missing.join(", ")
        ));
    }
    eprintln!("lbm: model ready at {}", dir.display());
    Ok(())
}

fn hf_cli() -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths).find_map(|dir| {
            let full = dir.join("hf");
            if full.is_file() {
                Some(full)
            } else {
                None
            }
        })
    })
}

/// `hf download <repo> <files…> --local-dir <dir>`: the CLI lays files out
/// preserving repo-relative paths, which matches our snapshot layout exactly.
fn ensure_via_hf_cli(dir: &Path, missing: &[String]) -> Result<()> {
    let hf = hf_cli().ok_or_else(|| anyhow!("`hf` CLI not found on PATH"))?;
    let mut cmd = std::process::Command::new(&hf);
    cmd.arg("download").arg(repo());
    for f in missing {
        cmd.arg(f);
    }
    cmd.arg("--local-dir").arg(dir);
    if revision() != "main" {
        cmd.arg("--revision").arg(revision());
    }
    if let Some(tok) = auth_token() {
        cmd.arg("--token").arg(tok);
    }
    let status = cmd.status().context("spawning `hf download`")?;
    if !status.success() {
        return Err(anyhow!("`hf download` exited with {status}"));
    }
    Ok(())
}

pub(crate) fn curl_bin() -> Option<&'static str> {
    for cand in ["curl"] {
        // `command -v`-style probe without a shell.
        if std::process::Command::new(cand)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
        {
            return Some(cand);
        }
    }
    None
}

/// One file at a time via `curl -fSL`, resumed into `<file>.part` and
/// renamed on success so a killed download never leaves a plausible file.
fn ensure_via_curl(dir: &Path, missing: &[String]) -> Result<()> {
    let curl = curl_bin().ok_or_else(|| anyhow!("`curl` not found on PATH"))?;
    for rel in missing {
        let dest = dir.join(rel);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating {}", parent.display()))?;
        }
        let part = dest.with_extension(
            dest.extension()
                .map(|e| format!("{}.part", e.to_string_lossy()))
                .unwrap_or_else(|| "part".into()),
        );
        // Human size hint for the progress line.
        let hint = match rel.as_str() {
            "onnx/model.onnx" => " (~746 MB)",
            "onnx/heads.onnx" => " (~24 MB)",
            "tokenizer.json" => " (~8 MB)",
            _ => "",
        };
        eprintln!("lbm: downloading {rel}{hint}…");
        let mut cmd = std::process::Command::new(curl);
        cmd.arg("-fSL")
            .arg("--retry")
            .arg("3")
            .arg("--retry-all-errors")
            .arg("-C")
            .arg("-") // resume the .part file when present
            .arg("-o")
            .arg(&part)
            .arg(file_url(rel));
        if let Some(tok) = auth_token() {
            cmd.arg("-H").arg(format!("Authorization: Bearer {tok}"));
        }
        // Inherit stderr so curl's progress bar reaches the terminal.
        let status = cmd.status().with_context(|| format!("spawning curl for {rel}"))?;
        if !status.success() {
            return Err(anyhow!("curl exited with {status} downloading {rel}"));
        }
        std::fs::rename(&part, &dest)
            .with_context(|| format!("promoting {} to {}", part.display(), dest.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> PathBuf {
        let mut d = std::env::temp_dir();
        d.push(format!("gliner-fetch-test-{}-{}", std::process::id(), tag));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn touch(dir: &Path, rel: &str, size: u64) {
        let p = dir.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(&p, vec![7u8; size as usize]).unwrap();
    }

    #[test]
    fn empty_dir_reports_all_missing() {
        let d = tmpdir("empty");
        let missing = missing_files(&d);
        assert_eq!(missing.len(), MODEL_FILES.len());
        assert!(!is_complete(&d));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn complete_dir_passes_and_truncated_onnx_fails() {
        let d = tmpdir("full");
        for f in MODEL_FILES {
            // Satisfy the size floors (model/heads need MB-scale dummies).
            touch(&d, f, min_size(f).max(8));
        }
        assert!(is_complete(&d));
        // Truncate the big graph: flagged as missing again.
        touch(&d, "onnx/model.onnx", 12);
        assert!(!is_complete(&d));
        assert!(missing_files(&d).contains(&"onnx/model.onnx".to_string()));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn file_url_respects_endpoint_and_revision_overrides() {
        std::env::set_var("HF_ENDPOINT", "https://mirror.example/");
        std::env::set_var("LB_MEMORY_MODEL_REVISION", "abc123");
        let url = file_url("onnx/model.onnx");
        assert_eq!(
            url,
            "https://mirror.example/nicolasembleton/gliner2.5-base-v1-onnx/resolve/abc123/onnx/model.onnx"
        );
        std::env::remove_var("HF_ENDPOINT");
        std::env::remove_var("LB_MEMORY_MODEL_REVISION");
    }

    #[test]
    fn offline_mode_errors_without_network() {
        let d = tmpdir("offline");
        std::env::set_var("HF_HUB_OFFLINE", "1");
        let err = ensure_model(&d).unwrap_err();
        assert!(err.to_string().contains("HF_HUB_OFFLINE"), "{err:#}");
        std::env::remove_var("HF_HUB_OFFLINE");
        let _ = std::fs::remove_dir_all(&d);
    }
}
