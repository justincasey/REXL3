//! What this binary is: version, engine commit, compiler, kernels and the CUDA libraries it runs against. Shown by
//! `--version`, logged at start, exported as `rexl3_build_info` and returned to clients as `system_fingerprint`.

use std::sync::OnceLock;

#[derive(Clone, Debug)]
pub struct BuildInfo {
    pub version: &'static str,
    pub commit: &'static str,
    pub cuda: gb_cuda::rt::Versions,
}

impl BuildInfo {
    pub fn get() -> &'static BuildInfo {
        static B: OnceLock<BuildInfo> = OnceLock::new();
        B.get_or_init(|| BuildInfo {
            version: env!("CARGO_PKG_VERSION"),
            commit: env!("REXL3_COMMIT"),
            cuda: gb_cuda::rt::versions(),
        })
    }

    /// `fp_<commit>_<kernel hash>_cb<cuBLAS>`: changes whenever the code, the kernels or the cuBLAS that computes prefill
    /// does, which are the things that can change output bits. A process whose Nemotron prefill settings are not the defaults
    /// (`REXL3_NEMO_PREFILL_CONTRACT`, the boot's effective prefill rows (`--prefill-chunk` at most the contract's cap; under P1
    /// `REXL3_NEMO_PREFILL_ROWS_MAX`), `REXL3_NEMO_PREFILL_ABSENT_FAST`, `REXL3_NEMO_PREFILL_ROUTER_MMA`) carries them as a last `_pf...`
    /// field (`contract::fingerprint_suffix`: since rc2d the default contract is P2 at 4,096 rows, which adds nothing,
    /// `REXL3_NEMO_PREFILL_CONTRACT=p1` adds `_pfp1r512a1` and `--prefill-chunk 1024` adds `_pfp2r1024a1`); a process whose KV cache is
    /// not fp16 (`--kv-dtype int8`, which changes output bits) carries `_kvi8` after it ([`kv_suffix`]); a default process has neither
    /// field.
    ///
    /// A process whose GLM prefill contract is not the default (`REXL3_GLM_PREFILL_CONTRACT=p1`,
    /// the opt-out; P2 has been the default since the owner's w2 decision on 2026-10-19) carries
    /// `_glmpfp1` last (`families::glm::contract::fingerprint_suffix`); a default P2 process has
    /// no such field.
    ///
    /// A process whose GLM latent cache is fp8 (`--kv-dtype fp8`, `REXL3_GLM_KV`, which changes output bits) carries `_glmkv8` after
    /// that; an fp16 GLM process has no such field.
    ///
    /// A process serving DeepSeek-V4.1 whose cache is the q4 plane (`--kv-dtype q4`, `REXL3_DSV41_KV`)
    /// carries `kvq4` (fused with the Nemotron KV suffix when both apply); an f16 DSV4.1 process has
    /// no such field.
    pub fn fingerprint(&self) -> String {
        let kv = gb_cuda::nemotron_attn::configured_kv_dtype().unwrap_or_default();
        // DSV4.1's cache is selected separately (`--kv-dtype q4`, `REXL3_DSV41_KV`): a q4 process says so
        let q4 = gb_cuda::dsv4_kvq4::configured_fmt(gb_cuda::dsv4_kvq4::KvFmt::F16) == Ok(gb_cuda::dsv4_kvq4::KvFmt::Q4);
        let suffix = match (kv_suffix(kv), q4) {
            (None, false) => None,
            (Some(s), false) => Some(s.to_string()),
            (None, true) => Some("kvq4".to_string()),
            (Some(s), true) => Some(format!("{s}q4")),
        };
        let mut s = Self::compose(
            self.commit,
            &self.cuda,
            gb_model::families::nemotron_h_omni::contract::fingerprint_suffix(),
            suffix.as_deref(),
        );
        if let Some(g) = gb_model::families::glm::contract::fingerprint_suffix() {
            s.push('_');
            s.push_str(&g);
        }
        if gb_cuda::mla::configured_kv() == Ok(gb_cuda::mla::LatentKv::Fp8) {
            s.push_str("_glmkv8");
        }
        s
    }

    fn compose(commit: &str, cuda: &gb_cuda::rt::Versions, prefill: Option<String>, kv: Option<&str>) -> String {
        let commit: String = commit.chars().take(12).collect();
        let kernels: String = cuda.kernel_hash.chars().take(8).collect();
        let cublas = cuda.cublas.as_deref().unwrap_or("none").replace('.', "");
        let mut s = format!("fp_{commit}_{kernels}_cb{cublas}");
        for x in [prefill.as_deref(), kv].into_iter().flatten() {
            s.push('_');
            s.push_str(x);
        }
        s
    }

    /// One `name value` pair per line, for `--version`.
    pub fn lines(&self) -> Vec<(&'static str, String)> {
        let c = &self.cuda;
        let or_none = |v: &Option<String>| v.clone().unwrap_or_else(|| "unavailable".into());
        vec![
            ("version", self.version.to_string()),
            ("commit", self.commit.to_string()),
            ("system_fingerprint", self.fingerprint()),
            ("nvcc", c.nvcc.to_string()),
            ("cuda_arch", c.arch.to_string()),
            ("kernel_hash", c.kernel_hash.to_string()),
            ("cudart", or_none(&c.cudart)),
            ("cuda_driver", or_none(&c.driver)),
            ("cublas", or_none(&c.cublas)),
        ]
    }
}

/// The fingerprint field of a KV element type: none for fp16 (the default), `kvi8` for int8.
pub fn kv_suffix(kv: gb_cuda::nemotron_attn::KvDtype) -> Option<&'static str> {
    match kv {
        gb_cuda::nemotron_attn::KvDtype::F16 => None,
        gb_cuda::nemotron_attn::KvDtype::Int8 => Some("kvi8"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fingerprint_names_commit_kernels_and_cublas() {
        let mut b = BuildInfo::get().clone();
        b.commit = "4180196abcdef0123";
        b.cuda.kernel_hash = "0123456789abcdef";
        b.cuda.cublas = Some("13.1.0".into());
        assert_eq!(BuildInfo::compose(b.commit, &b.cuda, None, None), "fp_4180196abcde_01234567_cb1310");
        b.cuda.cublas = None;
        assert!(BuildInfo::compose(b.commit, &b.cuda, None, None).ends_with("_cbnone"));
        // a non-default prefill setting is the last field; since rc2d the default contract is P2 at the shipped 4,096 prefill rows (no
        // field) and P1 is the opt-in; since v0.1.1 the rows are the boot's effective ones
        use gb_model::families::nemotron_h_omni::contract::{suffix_for, PrefillContract};
        assert_eq!(
            BuildInfo::compose(b.commit, &b.cuda, suffix_for(PrefillContract::P2, 4096, true, false), None),
            "fp_4180196abcde_01234567_cbnone"
        );
        assert_eq!(
            BuildInfo::compose(b.commit, &b.cuda, suffix_for(PrefillContract::P2, 1024, true, false), None),
            "fp_4180196abcde_01234567_cbnone_pfp2r1024a1"
        );
        assert_eq!(
            BuildInfo::compose(b.commit, &b.cuda, suffix_for(PrefillContract::P1, 512, true, false), None),
            "fp_4180196abcde_01234567_cbnone_pfp1r512a1"
        );
        assert_eq!(
            BuildInfo::compose(b.commit, &b.cuda, Some("pfp1r256a1".into()), None),
            "fp_4180196abcde_01234567_cbnone_pfp1r256a1"
        );
    }

    #[test]
    fn an_int8_kv_cache_is_a_last_field_and_fp16_adds_nothing() {
        use gb_cuda::nemotron_attn::KvDtype;
        let mut b = BuildInfo::get().clone();
        b.commit = "4180196abcdef0123";
        b.cuda.kernel_hash = "0123456789abcdef";
        b.cuda.cublas = Some("13.1.0".into());
        // the default (fp16) fingerprint is byte-identical to the format before the field existed
        assert_eq!(kv_suffix(KvDtype::F16), None);
        assert_eq!(BuildInfo::compose(b.commit, &b.cuda, None, kv_suffix(KvDtype::F16)), "fp_4180196abcde_01234567_cb1310");
        assert_eq!(BuildInfo::compose(b.commit, &b.cuda, None, kv_suffix(KvDtype::Int8)), "fp_4180196abcde_01234567_cb1310_kvi8");
        assert_eq!(
            BuildInfo::compose(b.commit, &b.cuda, Some("pfp1r512a1".into()), kv_suffix(KvDtype::Int8)),
            "fp_4180196abcde_01234567_cb1310_pfp1r512a1_kvi8"
        );
    }
}
