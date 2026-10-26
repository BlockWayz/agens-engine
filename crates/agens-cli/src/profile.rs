//! `agens serve --profile 12gb|16gb|24gb|48gb` (1.3.6): the consumer profiles of scripts/agens-entry.sh in the binary,
//! so `agens serve` run directly (outside the container, e.g. a Windows build) gets the same flags and AGENS_* defaults.
//! The container entry point passes its --profile through to here; the profile's flags go first, so a flag given on the
//! command line wins (the parsers keep the last value), and an AGENS_* variable already set wins over the profile's.

use anyhow::{bail, Result};

/// The profile's expansion of `args` (`args[0]`: the checkpoint): [ckpt, profile flags..., the other flags...], with
/// the profile's environment defaults as (name, value) pairs. `args` without --profile come back unchanged.
pub fn expand(args: &[String], env: &dyn Fn(&str) -> Option<String>) -> Result<(Vec<String>, Vec<(String, String)>)> {
    let Some(at) = args.iter().skip(1).position(|a| a == "--profile").map(|p| p + 1) else {
        return Ok((args.to_vec(), vec![]));
    };
    let Some(profile) = args.get(at + 1).cloned() else { bail!("--profile 12gb|16gb|24gb|48gb") };
    let mut user: Vec<String> = args[1..at].iter().chain(&args[at + 2..]).cloned().collect();
    // the last value of a flag among the user's flags
    let flag = |u: &[String], f: &str| -> Option<String> {
        u.windows(2).rev().find(|w| w[0] == f).map(|w| w[1].clone())
    };
    let drop_flag = |u: &mut Vec<String>, f: &str| {
        let mut i = 0;
        while i + 1 < u.len() {
            if u[i] == f { u.drain(i..i + 2); } else { i += 1; }
        }
    };
    let tp = flag(&user, "--tp").unwrap_or_else(|| "1".into());
    if !matches!(tp.as_str(), "1" | "2" | "4") {
        bail!("--tp 1, 2 or 4 (got {tp})");
    }
    let images = flag(&user, "--vision").map(|v| v != "0");
    // the --mtp context sizes below are tp1 plans; under --tp the profile keeps its full --wctx and the engine clamps it
    let mtp1 = tp == "1" && flag(&user, "--mtp").is_some();
    let wctx = |dflt: usize| env("AGENS_WCTX").unwrap_or_else(|| dflt.to_string());
    let mut set = vec![];
    let mut setdef = |k: &str, v: &str| {
        if env(k).map_or(true, |x| x.is_empty()) {
            set.push((k.to_string(), v.to_string()));
        }
    };
    let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    let mut p = match profile.as_str() {
        // scripts/agens-entry.sh carries the measurements behind each number
        "16gb" => {
            let w = match (images == Some(true), mtp1) {
                (true, false) => wctx(229380),
                (true, true) => wctx(221188),
                (false, false) => wctx(262148),
                (false, true) => wctx(241668),
            };
            setdef("AGENS_KV_INT8_DENSE", "1");
            s(&["--budget-gib", "16", "--wctx", &w, "--mode", "personal", "--no-fork", "1"])
        }
        "12gb" => {
            setdef("AGENS_KV_INT8_DENSE", "1");
            setdef("AGENS_SPEC_FAR_CACHE", "0");
            if env("AGENS_12GB_DRAFTER").as_deref() != Some("1") {
                drop_flag(&mut user, "--spec-drafter");  // copy-from-context drafting only: no drafter weights
            }
            if flag(&user, "--mtp").is_some() {
                eprintln!("12gb profile: --mtp ignored (the head does not fit the 12 GiB plan)");
                drop_flag(&mut user, "--mtp");
            }
            s(&["--budget-gib", "12", "--wctx", &wctx(65540), "--mode", "personal", "--no-fork", "1", "--spec-copy", "1"])
        }
        "24gb" => s(&["--budget-gib", "23", "--wctx", &wctx(262148), "--mode", "server", "--max-batch", "4",
                      "--max-convs", "4", "--sessions", "4"]),
        "48gb" => s(&["--budget-gib", "46", "--wctx", "262148", "--mode", "server", "--max-batch", "16",
                      "--max-convs", "16"]),
        p => bail!("unknown profile {p} (12gb, 16gb, 24gb, 48gb)"),
    };
    // image input: off by default for 12gb / 16gb; 24gb / 48gb: on when the checkpoint has the tower
    if images.is_none() && matches!(profile.as_str(), "12gb" | "16gb") {
        p.extend(s(&["--vision", "0"]));
    }
    if tp != "1" {
        setdef("AGENS_TP_LOG_DIR", "/tmp");  // follower ranks' logs: /tmp/tp_rank<r>.log
    }
    let mut out = vec![args[0].clone()];
    out.extend(p);
    out.extend(user);
    Ok((out, set))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn v(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }
    fn none(_: &str) -> Option<String> {
        None
    }

    #[test]
    fn no_profile_unchanged() {
        let a = v("/m --port 1 --mode server");
        assert_eq!(expand(&a, &none).unwrap(), (a, vec![]));
    }

    #[test]
    fn p16_defaults_and_overrides() {
        let (a, e) = expand(&v("/m --profile 16gb --port 31000"), &none).unwrap();
        assert_eq!(a, v("/m --budget-gib 16 --wctx 262148 --mode personal --no-fork 1 --vision 0 --port 31000"));
        assert_eq!(e, vec![("AGENS_KV_INT8_DENSE".into(), "1".into())]);
        let (a, _) = expand(&v("/m --vision 1 --mtp /h --profile 16gb"), &none).unwrap();
        assert_eq!(a, v("/m --budget-gib 16 --wctx 221188 --mode personal --no-fork 1 --vision 1 --mtp /h"));
        // --mtp under --tp: the full context (the engine clamps it)
        let (a, e) = expand(&v("/m --profile 16gb --mtp /h --tp 2"), &none).unwrap();
        assert_eq!(a[4], "262148");
        assert!(e.iter().any(|(k, _)| k == "AGENS_TP_LOG_DIR"));
        // a variable already set wins
        let set = |k: &str| (k == "AGENS_KV_INT8_DENSE" || k == "AGENS_WCTX").then(|| "0".to_string());
        let (a, e) = expand(&v("/m --profile 16gb"), &set).unwrap();
        assert_eq!(a[4], "0");
        assert!(e.is_empty());
    }

    #[test]
    fn p12_drops_drafter_and_mtp() {
        let (a, _) = expand(&v("/m --profile 12gb --spec-drafter /d --mtp /h"), &none).unwrap();
        assert_eq!(a, v("/m --budget-gib 12 --wctx 65540 --mode personal --no-fork 1 --spec-copy 1 --vision 0"));
        let keep = |k: &str| (k == "AGENS_12GB_DRAFTER").then(|| "1".to_string());
        let (a, _) = expand(&v("/m --profile 12gb --spec-drafter /d"), &keep).unwrap();
        assert!(a.ends_with(&v("--spec-drafter /d")));
    }

    #[test]
    fn p24_p48_and_errors() {
        let (a, e) = expand(&v("/m --profile 24gb"), &none).unwrap();
        assert_eq!(a, v("/m --budget-gib 23 --wctx 262148 --mode server --max-batch 4 --max-convs 4 --sessions 4"));
        assert!(e.is_empty());
        let (a, _) = expand(&v("/m --profile 48gb --max-batch 8"), &none).unwrap();
        assert_eq!(a.last().unwrap(), "8");
        assert!(expand(&v("/m --profile 8gb"), &none).is_err());
        assert!(expand(&v("/m --profile"), &none).is_err());
        assert!(expand(&v("/m --profile 16gb --tp 3"), &none).is_err());
    }
}
