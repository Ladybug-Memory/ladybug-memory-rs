//! ONNX Runtime shared-library provisioning.
//!
//! `ort` is built with `load-dynamic`, so the dylib is opened from an
//! explicit absolute path — there is deliberately **no** `PATH` /
//! `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH` / shell-profile / registry setup
//! on any OS. `lbm fetch-model` (or the first inference, via
//! [`GlinerOnnx::load_or_fetch`](super::GlinerOnnx::load_or_fetch)) drops the
//! file into a cache dir once, and every later run opens it by path:
//!
//! | OS | file | opened via |
//! |---|---|---|
//! | macOS | `libonnxruntime.dylib` | `dlopen` (absolute path; quarantine xattr cleared best-effort) |
//! | Linux | `libonnxruntime.so` | `dlopen` (absolute path) |
//! | Windows | `onnxruntime.dll` | `LoadLibrary` (absolute path, no side-by-side manifest) |
//!
//! Resolution order ([`resolve_dylib`]):
//!
//! 1. `$GLINER_ORT_DYLIB` — explicit file, wins over everything.
//! 2. `$ORT_DYLIB_PATH` — shared convention, kept for back-compat
//!    (venv copies, system installs).
//! 3. `<lib-dir>/<platform file>` provisioned by [`ensure_ort`]
//!    (`$LB_MEMORY_ORT_DIR`, else `~/.cache/ladybug-memory/lib`).
//!
//! [`ensure_ort`] downloads the lib from the PyPI `onnxruntime` wheel
//! matching [`ORT_VERSION`] — the exact build the compiled `ort-sys`
//! links against, so any "latest" is never picked blindly. The wheel is a
//! plain zip; only the `capi/` shared library is extracted (curl +
//! `unzip`/`python3`/`tar`, all best-effort probed).
//!
//! Knobs (all optional):
//!
//! | env | default |
//! |---|---|
//! | `LB_MEMORY_ORT_DIR` | `~/.cache/ladybug-memory/lib` |
//! | `LB_MEMORY_ORT_VERSION` | [`ORT_VERSION`] (must stay ABI-compatible with the `ort` crate) |
//! | `LB_MEMORY_ORT_INDEX` | `https://pypi.org/simple/onnxruntime/` |
//! | `LB_MEMORY_ORT_URL` | (none — force one wheel/archive URL, for mirrors) |
//! | `HF_HUB_OFFLINE=1` / `LB_MEMORY_OFFLINE=1` | fail fast instead of downloading |

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};

/// ORT release the compiled `ort-sys` expects (its `ms@1.31.0` prebuilt
/// ABI). Override with `LB_MEMORY_ORT_VERSION` only for an ABI-compatible
/// build — a newer upstream "latest" may fail `ort::init_from`.
pub const ORT_VERSION: &str = "1.31.0";

/// Simple index listing every `onnxruntime` wheel (what `pip` reads).
const DEFAULT_INDEX: &str = "https://pypi.org/simple/onnxruntime/";

/// Platform file opened by `ort::init_from`.
pub fn lib_filename() -> &'static str {
    if cfg!(target_os = "windows") {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    }
}

/// Minimum plausible size for the ORT shared library (the real one is
/// 30–90 MB depending on platform). Catches truncated downloads that still
/// leave a file behind.
fn min_lib_size() -> u64 {
    5 * 1024 * 1024
}

fn version() -> String {
    std::env::var("LB_MEMORY_ORT_VERSION")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| ORT_VERSION.into())
}

fn offline() -> bool {
    std::env::var("HF_HUB_OFFLINE").as_deref() == Ok("1")
        || std::env::var("LB_MEMORY_OFFLINE").as_deref() == Ok("1")
}

/// Directory holding the provisioned ORT lib: `$LB_MEMORY_ORT_DIR`, else
/// `~/.cache/ladybug-memory/lib`.
pub fn default_lib_dir() -> PathBuf {
    if let Ok(d) = std::env::var("LB_MEMORY_ORT_DIR") {
        if !d.trim().is_empty() {
            return PathBuf::from(d);
        }
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(format!("{home}/.cache/ladybug-memory/lib"))
}

/// Path [`ensure_ort`] provisions (whether or not it exists yet).
pub fn cached_dylib() -> PathBuf {
    default_lib_dir().join(lib_filename())
}

fn plausible(p: &Path) -> bool {
    std::fs::metadata(p).map(|m| m.is_file() && m.len() >= min_lib_size()).unwrap_or(false)
}

fn env_file(var: &str) -> Option<PathBuf> {
    std::env::var(var).ok().map(PathBuf::from).filter(|p| p.is_file())
}

/// Resolve the ORT dylib without touching the network:
/// `$GLINER_ORT_DYLIB` > `$ORT_DYLIB_PATH` > provisioned cache file.
/// Returns `None` when nothing usable exists (see [`ensure_ort`]).
pub fn resolve_dylib() -> Option<PathBuf> {
    if let Some(p) = env_file("GLINER_ORT_DYLIB") {
        return Some(p);
    }
    if let Some(p) = env_file("ORT_DYLIB_PATH") {
        return Some(p);
    }
    let p = cached_dylib();
    plausible(&p).then_some(p)
}

/// `(path, ready)` for `lbm status`: the resolved lib, or where
/// `fetch-model` would put it when nothing resolves.
pub fn status() -> (String, bool) {
    match resolve_dylib() {
        Some(p) => (p.display().to_string(), true),
        None => (cached_dylib().display().to_string(), false),
    }
}

/// Ensure a usable ORT dylib exists, downloading it on first run.
/// Returns its path. A no-op when [`resolve_dylib`] already hits.
pub fn ensure_ort() -> Result<PathBuf> {
    if let Some(p) = resolve_dylib() {
        return Ok(p);
    }
    let ver = version();
    let dest = cached_dylib();
    if offline() {
        return Err(anyhow!(
            "ONNX Runtime {ver} lib not found at {} and downloads are disabled \
             (HF_HUB_OFFLINE=1 or LB_MEMORY_OFFLINE=1); run `lbm fetch-model` online once, \
             or set ORT_DYLIB_PATH to an existing libonnxruntime",
            dest.display()
        ));
    }
    std::fs::create_dir_all(dest.parent().expect("lib file has a parent"))
        .with_context(|| format!("creating {}", dest.display()))?;
    eprintln!("lbm: fetching ONNX Runtime {ver} lib into {}…", dest.display());
    if let Some(url) = std::env::var("LB_MEMORY_ORT_URL").ok().filter(|u| !u.trim().is_empty()) {
        install_from_url(&url, &ver, &dest)?;
    } else if let Some(plat) = platform_for(std::env::consts::OS, std::env::consts::ARCH, &ver) {
        match install_from_pypi(&plat, &ver, &dest) {
            Ok(()) => {}
            Err(e) => {
                eprintln!("lbm: PyPI install failed ({e:#}); trying microsoft release archive…");
                install_from_microsoft(&plat, &ver, &dest)?;
            }
        }
    } else {
        return Err(anyhow!(
            "no ONNX Runtime {ver} build known for {}-{}; set ORT_DYLIB_PATH to an \
             existing libonnxruntime (1.23+), or LB_MEMORY_ORT_URL to a wheel/archive URL",
            std::env::consts::OS,
            std::env::consts::ARCH
        ));
    }
    if !plausible(&dest) {
        return Err(anyhow!(
            "ORT install produced no usable lib at {} (missing or < {} MB)",
            dest.display(),
            min_lib_size() / 1024 / 1024
        ));
    }
    finish_install(&dest);
    eprintln!("lbm: ORT lib ready at {}", dest.display());
    Ok(dest)
}

/// Per-platform install facts for one ORT version.
struct Platform {
    /// Substrings a PyPI wheel filename must contain.
    wheel_needles: &'static [&'static str],
    /// The shared library inside the wheel (`onnxruntime/capi/…`).
    wheel_member: String,
    /// Microsoft release asset infix (`onnxruntime-{asset}-{ver}.tgz/zip`).
    ms_asset: &'static str,
}

fn platform_for(os: &str, arch: &str, ver: &str) -> Option<Platform> {
    let (needles, member, asset): (&[_], String, _) = match (os, arch) {
        ("macos", "aarch64") => (
            &["macosx", "arm64"],
            format!("onnxruntime/capi/libonnxruntime.{ver}.dylib"),
            "osx-arm64",
        ),
        // No PyPI intel-mac wheels for 1.31: filename match fails, the
        // microsoft archive fallback below still gets a try.
        ("macos", "x86_64") => (
            &["macosx", "x86_64"],
            format!("onnxruntime/capi/libonnxruntime.{ver}.dylib"),
            "osx-x86_64",
        ),
        ("linux", "x86_64") => (
            &["manylinux", "x86_64"],
            format!("onnxruntime/capi/libonnxruntime.so.{ver}"),
            "linux-x64",
        ),
        ("linux", "aarch64") => (
            &["manylinux", "aarch64"],
            format!("onnxruntime/capi/libonnxruntime.so.{ver}"),
            "linux-aarch64",
        ),
        ("windows", "x86_64") => (&["win_amd64"], "onnxruntime/capi/onnxruntime.dll".into(), "win-x64"),
        ("windows", "aarch64") => (&["win_arm64"], "onnxruntime/capi/onnxruntime.dll".into(), "win-arm64"),
        _ => return None,
    };
    Some(Platform { wheel_needles: needles, wheel_member: member, ms_asset: asset })
}

fn index_url() -> String {
    std::env::var("LB_MEMORY_ORT_INDEX")
        .ok()
        .filter(|u| !u.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_INDEX.into())
}

/// Pick a wheel URL from a PyPI simple-index page: filenames matching
/// `onnxruntime-{ver}-…{needles}….whl`, musl excluded, free-threaded (`t-`)
/// deprioritized, highest CP tag wins.
fn pick_wheel<'a>(html: &'a str, ver: &str, needles: &[&str]) -> Option<String> {
    let prefix = format!("onnxruntime-{ver}-");
    let mut cands: Vec<&str> = Vec::new();
    let mut rest = html;
    while let Some(i) = rest.find("href=\"") {
        rest = &rest[i + 6..];
        let end = rest.find('"')?;
        let url = &rest[..end];
        rest = &rest[end + 1..];
        let name = url.rsplit('/').next().unwrap_or(url);
        let name = name.split('#').next().unwrap_or(name);
        if !name.starts_with(&prefix) || !name.ends_with(".whl") {
            continue;
        }
        if needles.iter().all(|n| name.contains(n)) && !name.contains("musl") {
            cands.push(url.into());
        }
    }
    // Python tag is the second `-` component (`cp313`, `cp313t`, `pp310`…);
    // prefer non-free-threaded, then the highest tag (string order is fine
    // for same-width `cp3xx` tags).
    cands.sort_by_key(|u| {
        let name = u.rsplit('/').next().unwrap_or(u);
        let tag = name.split('-').nth(1).unwrap_or("");
        (tag.ends_with('t'), std::cmp::Reverse(name.to_string()))
    });
    cands.into_iter().next().map(str::to_string)
}

fn curl_download(url: &str, dest: &Path) -> Result<()> {
    let curl = super::fetch::curl_bin().ok_or_else(|| anyhow!("`curl` not found on PATH"))?;
    let part = dest.with_extension("part");
    let status = std::process::Command::new(curl)
        .arg("-fSL")
        .arg("--retry")
        .arg("3")
        .arg("--retry-all-errors")
        .arg("-o")
        .arg(&part)
        .arg(url)
        .status()
        .with_context(|| format!("spawning curl for {url}"))?;
    if !status.success() {
        return Err(anyhow!("curl exited with {status} downloading {url}"));
    }
    std::fs::rename(&part, dest)
        .with_context(|| format!("promoting {} to {}", part.display(), dest.display()))?;
    Ok(())
}

fn tmpdir() -> Result<PathBuf> {
    let d = std::env::temp_dir().join(format!("lb-ort-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).with_context(|| format!("creating {}", d.display()))?;
    Ok(d)
}

/// Extract one `member` from a zip archive into `dest`, trying
/// `unzip -p`, then `python3 -m zipfile`, then `tar -xOf` (bsdtar reads
/// zips), then PowerShell `Expand-Archive` on Windows.
fn extract_zip_member(archive: &Path, member: &str, dest: &Path) -> Result<()> {
    let mut attempts: Vec<String> = Vec::new();
    // unzip -p: member bytes to stdout.
    match std::process::Command::new("unzip").arg("-p").arg(archive).arg(member).output() {
        Ok(out) if out.status.success() && !out.stdout.is_empty() => {
            std::fs::write(dest, &out.stdout)
                .with_context(|| format!("writing {}", dest.display()))?;
            return Ok(());
        }
        Ok(out) => attempts.push(format!(
            "unzip: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        )),
        Err(e) => attempts.push(format!("unzip: {e}")),
    }
    for py in ["python3", "python"] {
        let script = "import sys,zipfile; open(sys.argv[3],'wb').write(zipfile.ZipFile(sys.argv[1]).read(sys.argv[2]))";
        match std::process::Command::new(py).arg("-c").arg(script).arg(archive).arg(member).arg(dest).output() {
            Ok(out) if out.status.success() && plausible(dest) => return Ok(()),
            Ok(out) => attempts.push(format!(
                "{py}: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            )),
            Err(e) => attempts.push(format!("{py}: {e}")),
        }
    }
    match std::process::Command::new("tar").arg("-xOf").arg(archive).arg(member).output() {
        Ok(out) if out.status.success() && !out.stdout.is_empty() => {
            std::fs::write(dest, &out.stdout)
                .with_context(|| format!("writing {}", dest.display()))?;
            return Ok(());
        }
        Ok(out) => attempts.push(format!("tar: {}", String::from_utf8_lossy(&out.stderr).trim())),
        Err(e) => attempts.push(format!("tar: {e}")),
    }
    #[cfg(target_os = "windows")]
    {
        let tmp = tmpdir()?;
        let ps = std::process::Command::new("powershell")
            .arg("-NoProfile")
            .arg("-Command")
            .arg(format!(
                "Expand-Archive -Path '{}' -DestinationPath '{}' -Force",
                archive.display(),
                tmp.display()
            ))
            .output();
        match ps {
            Ok(out) if out.status.success() => {
                let got = tmp.join(member);
                if got.is_file() {
                    std::fs::copy(&got, dest)
                        .with_context(|| format!("copying {} to {}", got.display(), dest.display()))?;
                    let _ = std::fs::remove_dir_all(&tmp);
                    return Ok(());
                }
                attempts.push("powershell Expand-Archive: member not found after extract".into());
            }
            Ok(out) => attempts.push(format!(
                "powershell: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            )),
            Err(e) => attempts.push(format!("powershell: {e}")),
        }
        let _ = std::fs::remove_dir_all(&tmp);
    }
    Err(anyhow!(
        "could not extract {member} from {} (tried unzip/python/tar{}): {}",
        archive.display(),
        if cfg!(target_os = "windows") { "/powershell" } else { "" },
        attempts.join("; ")
    ))
}

fn install_from_pypi(plat: &Platform, ver: &str, dest: &Path) -> Result<()> {
    let index = index_url();
    eprintln!("lbm: resolving onnxruntime=={ver} wheel from {index}…");
    let curl = super::fetch::curl_bin().ok_or_else(|| anyhow!("`curl` not found on PATH"))?;
    let out = std::process::Command::new(curl).arg("-fSL").arg(&index).output().context("fetching PyPI index")?;
    if !out.status.success() {
        return Err(anyhow!("curl exited with {} fetching {index}", out.status));
    }
    let html = String::from_utf8_lossy(&out.stdout);
    let url = pick_wheel(&html, ver, plat.wheel_needles).ok_or_else(|| {
        anyhow!("no onnxruntime=={ver} wheel for this platform in {index} (needles: {})", plat.wheel_needles.join("+"))
    })?;
    let url = url.split('#').next().unwrap_or(&url).to_string();
    eprintln!("lbm: downloading {}…", url.rsplit('/').next().unwrap_or(&url));
    let tmp = tmpdir()?;
    let wheel = tmp.join("ort.whl");
    let r = curl_download(&url, &wheel).and_then(|_| extract_zip_member(&wheel, &plat.wheel_member, dest));
    let _ = std::fs::remove_dir_all(&tmp);
    r.with_context(|| format!("installing ORT lib from {url}"))
}

/// Explicit URL override (`LB_MEMORY_ORT_URL`): a wheel/zip when it ends in
/// `.whl`/`.zip`, else a `.tar.gz`/`.tgz` release archive.
fn install_from_url(url: &str, ver: &str, dest: &Path) -> Result<()> {
    let tmp = tmpdir()?;
    let file = tmp.join("ort-dl");
    curl_download(url, &file)?;
    let r = if url.ends_with(".whl") || url.ends_with(".zip") {
        let plat = platform_for(std::env::consts::OS, std::env::consts::ARCH, ver).ok_or_else(|| {
            anyhow!("LB_MEMORY_ORT_URL is a wheel but this platform is unknown; name the member via a direct download + ORT_DYLIB_PATH")
        })?;
        extract_zip_member(&file, &plat.wheel_member, dest)
    } else {
        extract_lib_from_tarball(&file, dest)
    };
    let _ = std::fs::remove_dir_all(&tmp);
    r
}

/// Microsoft release archive fallback
/// (`onnxruntime-{asset}-{ver}.tgz`, `.zip` on Windows): extract the whole
/// archive to a temp dir and promote the largest `*onnxruntime*` shared
/// library. Used for platforms PyPI doesn't serve (e.g. Intel macs).
fn install_from_microsoft(plat: &Platform, ver: &str, dest: &Path) -> Result<()> {
    let ext = if cfg!(target_os = "windows") { "zip" } else { "tgz" };
    let url = format!(
        "https://github.com/microsoft/onnxruntime/releases/download/v{ver}/onnxruntime-{}-{ver}.{ext}",
        plat.ms_asset
    );
    eprintln!("lbm: downloading {url}…");
    let tmp = tmpdir()?;
    let archive = tmp.join(format!("ort.{ext}"));
    let r = curl_download(&url, &archive).and_then(|_| extract_lib_from_tarball(&archive, dest));
    let _ = std::fs::remove_dir_all(&tmp);
    r.with_context(|| format!("installing ORT lib from {url}"))
}

/// Extract `archive` (tar.gz/tgz/zip) fully to a temp dir and promote the
/// largest shared-library-shaped regular file to `dest`.
fn extract_lib_from_tarball(archive: &Path, dest: &Path) -> Result<()> {
    let tmp = tmpdir()?;
    let dir = tmp.join("x");
    std::fs::create_dir_all(&dir)?;
    let name = archive.to_string_lossy();
    let mut attempts: Vec<String> = Vec::new();
    let mut ok = false;
    if name.ends_with(".zip") || name.ends_with(".whl") {
        for tool in ["unzip", "tar"] {
            let mut cmd = std::process::Command::new(tool);
            if tool == "unzip" {
                cmd.arg("-q").arg(archive).arg("-d").arg(&dir);
            } else {
                cmd.arg("-xf").arg(archive).arg("-C").arg(&dir);
            }
            match cmd.output() {
                Ok(out) if out.status.success() => { ok = true; break; }
                Ok(out) => attempts.push(format!("{tool}: {}", String::from_utf8_lossy(&out.stderr).trim())),
                Err(e) => attempts.push(format!("{tool}: {e}")),
            }
        }
    } else {
        match std::process::Command::new("tar").arg("-xzf").arg(archive).arg("-C").arg(&dir).output() {
            Ok(out) if out.status.success() => ok = true,
            Ok(out) => attempts.push(format!("tar: {}", String::from_utf8_lossy(&out.stderr).trim())),
            Err(e) => attempts.push(format!("tar: {e}")),
        }
    }
    if !ok {
        let _ = std::fs::remove_dir_all(&tmp);
        return Err(anyhow!("could not unpack {}: {}", archive.display(), attempts.join("; ")));
    }
    let want_dll = cfg!(target_os = "windows");
    let mut best: Option<(u64, PathBuf)> = None;
    let mut stack = vec![dir.clone()];
    while let Some(d) = stack.pop() {
        let rd = match std::fs::read_dir(&d) {
            Ok(rd) => rd,
            Err(_) => continue,
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            let n = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            let hit = if want_dll {
                n.eq_ignore_ascii_case("onnxruntime.dll")
            } else {
                (n.starts_with("libonnxruntime.") || n == "libonnxruntime.dylib" || n == "libonnxruntime.so")
                    && !p.is_symlink()
            };
            if hit {
                if let Ok(m) = std::fs::metadata(&p) {
                    if best.as_ref().map(|(s, _)| m.len() > *s).unwrap_or(true) {
                        best = Some((m.len(), p));
                    }
                }
            }
        }
    }
    let (_, src) = best.ok_or_else(|| {
        anyhow!("no onnxruntime shared library found in {}", archive.display())
    })?;
    std::fs::copy(&src, dest)
        .with_context(|| format!("copying {} to {}", src.display(), dest.display()))?;
    let _ = std::fs::remove_dir_all(&tmp);
    Ok(())
}

/// Post-install niceties: unix exec bit; clear macOS quarantine (downloaded
/// files carry it, and it can break `dlopen` under ad-hoc signatures).
/// Best-effort — failures are ignored.
fn finish_install(dest: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(m) = std::fs::metadata(dest) {
            let mut p = m.permissions();
            p.set_mode(0o755);
            let _ = std::fs::set_permissions(dest, p);
        }
    }
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("xattr").arg("-c").arg(dest).output();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    struct EnvGuard {
        saved: Vec<(String, Option<String>)>,
    }

    impl EnvGuard {
        fn hold(vars: &[&str]) -> (std::sync::MutexGuard<'static, ()>, Self) {
            let guard = env_lock().lock().unwrap();
            let saved = vars.iter().map(|v| (v.to_string(), std::env::var(v).ok())).collect();
            (guard, Self { saved })
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (k, v) in self.saved.drain(..) {
                match v {
                    Some(val) => std::env::set_var(&k, val),
                    None => std::env::remove_var(&k),
                }
            }
        }
    }

    #[test]
    fn platform_matrix() {
        let v = "1.31.0";
        let m = platform_for("macos", "aarch64", v).unwrap();
        assert_eq!(m.wheel_member, "onnxruntime/capi/libonnxruntime.1.31.0.dylib");
        assert_eq!(m.ms_asset, "osx-arm64");
        let m = platform_for("linux", "x86_64", v).unwrap();
        assert_eq!(m.wheel_member, "onnxruntime/capi/libonnxruntime.so.1.31.0");
        let m = platform_for("windows", "x86_64", v).unwrap();
        assert_eq!(m.wheel_member, "onnxruntime/capi/onnxruntime.dll");
        assert!(platform_for("freebsd", "x86_64", v).is_none());
    }

    #[test]
    fn wheel_picker_prefers_highest_non_freethreaded_cp() {
        let html = r#"
<a href="https://f/p/onnxruntime-1.31.0-cp311-cp311-manylinux_2_28_x86_64.whl#sha256=a">x</a>
<a href="https://f/p/onnxruntime-1.31.0-cp313-cp313t-manylinux_2_28_x86_64.whl#sha256=b">x</a>
<a href="https://f/p/onnxruntime-1.31.0-cp314-cp314-manylinux_2_28_x86_64.whl#sha256=c">x</a>
<a href="https://f/p/onnxruntime-1.31.0-cp314-cp314-musllinux_1_2_x86_64.whl#sha256=d">x</a>
<a href="https://f/p/onnxruntime-1.31.0-cp314-cp314-macosx_14_0_arm64.whl#sha256=e">x</a>
"#;
        let got = pick_wheel(html, "1.31.0", &["manylinux", "x86_64"]).unwrap();
        assert!(got.contains("cp314-cp314-manylinux"), "{got}");
        // No intel-mac wheel at this version: other-platform files never match.
        assert!(pick_wheel(html, "1.31.0", &["macosx", "x86_64"]).is_none());
        assert!(pick_wheel(html, "2.0.0", &["manylinux", "x86_64"]).is_none());
    }

    #[test]
    fn resolve_order_env_then_cache() {
        let (_lock, _env) = EnvGuard::hold(&["GLINER_ORT_DYLIB", "ORT_DYLIB_PATH", "LB_MEMORY_ORT_DIR", "HOME"]);
        let base = std::env::temp_dir().join(format!("lb-ort-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let cached = base.join("cache");
        std::fs::create_dir_all(&cached).unwrap();
        std::env::set_var("LB_MEMORY_ORT_DIR", &cached);
        std::env::remove_var("GLINER_ORT_DYLIB");
        std::env::remove_var("ORT_DYLIB_PATH");

        // Nothing provisioned: no resolution.
        assert!(resolve_dylib().is_none());

        // Cache file wins once plausible-sized.
        let lib = cached.join(lib_filename());
        std::fs::write(&lib, vec![7u8; (min_lib_size() + 8) as usize]).unwrap();
        assert_eq!(resolve_dylib().unwrap(), lib);

        // ORT_DYLIB_PATH beats the cache; GLINER_ORT_DYLIB beats both.
        let other = base.join("other-lib");
        let top = base.join("top-lib");
        std::fs::write(&other, b"x").unwrap();
        std::fs::write(&top, b"x").unwrap();
        std::env::set_var("ORT_DYLIB_PATH", &other);
        assert_eq!(resolve_dylib().unwrap(), other);
        std::env::set_var("GLINER_ORT_DYLIB", &top);
        assert_eq!(resolve_dylib().unwrap(), top);

        // Tiny cache files don't count.
        std::env::remove_var("GLINER_ORT_DYLIB");
        std::env::remove_var("ORT_DYLIB_PATH");
        std::fs::write(&lib, b"tiny").unwrap();
        assert!(resolve_dylib().is_none());

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn offline_errors_without_network() {
        let (_lock, _env) = EnvGuard::hold(&["GLINER_ORT_DYLIB", "ORT_DYLIB_PATH", "LB_MEMORY_ORT_DIR", "HF_HUB_OFFLINE", "LB_MEMORY_OFFLINE"]);
        let base = std::env::temp_dir().join(format!("lb-ort-offline-{}", std::process::id()));
        std::env::set_var("LB_MEMORY_ORT_DIR", &base);
        std::env::remove_var("GLINER_ORT_DYLIB");
        std::env::remove_var("ORT_DYLIB_PATH");
        std::env::set_var("HF_HUB_OFFLINE", "1");
        let err = ensure_ort().unwrap_err();
        assert!(err.to_string().contains("ORT_DYLIB_PATH"), "{err:#}");
        std::env::remove_var("HF_HUB_OFFLINE");
        let _ = std::fs::remove_dir_all(&base);
    }
}
