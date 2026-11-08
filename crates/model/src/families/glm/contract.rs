//! GLM's prefill arithmetic contract (Lane P of the GLM-5.3-Flash performance plan; owner decision 2026-10-18: "P2 for GLM prefill:
//! start now, in parallel with the exact work; P1 stays the default until P2 passes its gates and its quality check"; owner decision
//! 2026-10-19 (w2, option B), after step (g) passed its pre-registered rule: **P2 is the default**, `REXL3_GLM_PREFILL_CONTRACT=p1` opts
//! out, `docs/GLM_PREFILL_P2.md` sections 13 and 14).
//!
//! * [`PrefillContract::P1`] (`REXL3_GLM_PREFILL_CONTRACT=p1`, the opt-out): every prompt row is computed operation for operation as the decode path computes it (the
//!   large-row kernels `exl3_big`, `moe_prefill`, `mla_prefill` reproduce the decode kernels' bits), so a prefill row, a verify-window row
//!   and a decoded row have the same bits, the state after a prompt does not depend on the chunk grid or on `--prefill-chunk`, and a
//!   prefix snapshot is the state the same tokens decoded one at a time would leave (`docs/GLM53.md`, "Prefill"). Nothing about a P1
//!   process changes with this module: its prefix namespace, its digests, its kernels are the ones before it (only its `system_fingerprint`
//!   now carries `_glmpfp1`, being the non-default contract, as Nemotron's `=p1` carries `_pfp1...`).
//! * [`PrefillContract::P2`] (THE DEFAULT; `REXL3_GLM_PREFILL_CONTRACT=p2` selects it explicitly): Nemotron's P2 (`nemotron_h_omni/contract.rs`) applied to GLM's
//!   prompt rows only; decode and verify windows (and the NextN drafter's draft steps) keep P1. (a) a row's output does not depend on the
//!   other rows, the chunk width, a chunk boundary, padding or row order; (b) every K reduction is in an order that is a function of K
//!   alone; (c) activations enter `mma.sync` as fp16, the EXL3 weights decoded exactly as the decode path decodes them, fp32 accumulation;
//!   (d) deterministic run to run. A prompt row has other bits under P2 than under P1 (prefill bits differ from decode bits), so P2 is
//!   another namespace of the prefix cache ([`PrefillContract::arith_tag`]) and has its own goldens. What holds under P2: row and chunk
//!   independence, chunked == whole, determinism, drafted == serial; NOT `windows == prefill` (a row prefilled and the same row decoded
//!   differ).
//!
//! What runs P2 today (steps 2 and 3 of Lane P): the dense EXL3 linears of the prompt rows (KDA qkv / o, MLA q_a / q_b / kv_a / o, the
//! shared expert and the dense MLP of layers 0-2, the NextN layer's eh_proj, MLA and shared expert in its prefill) on the tensor-core GEMM
//! (`gb_cuda::glm_p2`), and the routed experts of every MoE layer and of the NextN layer's prefill on the grouped tensor-core GEMMs
//! (`gb_cuda::glm_p2_moe`: gate/up with the clamped SwiGLU in one block, down, and the two-group combine whose order is the TP=2
//! ownership's, so P2 at TP=2 can equal P2 on one node), and the KDA recurrence of the prompt rows in chunks of 16 rows
//! (`gb_cuda::kda::prefill_chunked`, step (e); the chunks are anchored per prefill SEGMENT, `docs/GLM_PREFILL_P2.md` 8.1: every call of a
//! segment but its last starts a whole number of chunks after the segment's first row, which the model enforces and the backend
//! arranges, so under P2 "chunk-grid independent" means independent of every grid of 16-row multiples from the anchor). What stays P1
//! inside a P2 prefill, by design: the router (logits and selection: a near-tie flip moves an expert), the DSA indexer (wq_b, wk, the
//! gate, weights_proj: a sum over heads that picks the attended pools), the MLA attention (step 4), the KDA projections / conv / norms,
//! mHC and the norms, the lm_head (the last row only). The tag names each.
//!
//! The contract is one process-wide setting read once from [`CONTRACT_VAR`]; a model copies it at construction (`GlmModel::new_chunked`),
//! and under TP it is a fact of the Hello (`GlmFacts::prefill`), so two ranks on different contracts are refused by name. A value that is
//! not `p1` or `p2` is refused at load by name ([`check_selected`]), never silently run as another contract.

use std::sync::OnceLock;

/// The environment variable that selects GLM's prefill contract: unset or empty for the default ([`DEFAULT_CONTRACT`], P2), `p1`, or `p2`;
/// anything else is refused at load by name.
pub const CONTRACT_VAR: &str = "REXL3_GLM_PREFILL_CONTRACT";

/// The contract of a process that does not set [`CONTRACT_VAR`]. **P2** since the owner's decision of 2026-10-19 (w2, option B; Lane P step
/// (g) passed its pre-registered rule, `docs/GLM_PREFILL_P2.md` section 13). It was P1 before; flipping it is this one constant, and the
/// fingerprint suffix and the tests follow it. `REXL3_GLM_PREFILL_CONTRACT=p1` selects the former behaviour exactly.
pub const DEFAULT_CONTRACT: PrefillContract = PrefillContract::P2;

/// Whether this build links GLM's P2 kernels (`gb_cuda::glm_p2`).
pub const P2_BUILT: bool = true;

/// The version of P2's dense arithmetic (the rounding points, the K order and the tile rules of `gb_cuda::glm_p2`): part of the prefix
/// namespace and of the Hello fact. Bump it with any change that moves a P2 bit.
pub const P2_DENSE: &str = gb_cuda::glm_p2::ARITH_ID;

/// The version of P2's routed-expert arithmetic (the K order, the rounding points, the two-group combine of `gb_cuda::glm_p2_moe`): part of
/// the prefix namespace and of the Hello fact. Bump it with any change that moves a P2 expert bit.
pub const P2_EXPERTS: &str = gb_cuda::glm_p2_moe::ARITH_ID;

/// What a split gate/up subbank's pairs run inside a P2 prefill (`gb_cuda::glm_p2_moe::SPLIT_GU_ID`: the decode pairs stage, P1's bits;
/// no P2 GEMM takes gate != up widths): part of the prefix namespace and of the Hello fact.
pub const P2_SPLIT_GU: &str = gb_cuda::glm_p2_moe::SPLIT_GU_ID;

/// The version of P2's KDA arithmetic for prompt rows (Lane P step (e), `gb_cuda::kda::prefill_chunked`: chunks of 16 rows anchored per
/// prefill segment): part of the prefix namespace and of the Hello fact.
pub const P2_KDA: &str = gb_cuda::kda::P2_ARITH_ID;

/// The version of P2's MLA absorb / expand arithmetic for prompt rows (Lane P step (d), `gb_cuda::glm_p2_mla`: tensor cores, fold128):
/// part of the prefix namespace and of the Hello fact. The latent attention between them stays P1's kernel (`mla-attn=p1`).
pub const P2_MLA: &str = gb_cuda::glm_p2_mla::ARITH_ID;

/// The form of P2's tensor-parallel exchange (Lane P step (c), `model/tp/p2.rs`): each rank's part of the routed sum swapped as one fp32
/// row a prompt row, the parts added in rank order; v2: a chunk of two slabs or more is row-split (the peer's rows of the part go, the
/// shared expert, the parts' sum, the dense MLP and the MLA layers' per-row tail run on the rank's rows, their output rows swapped). A fact of
/// the TP Hello under P2 (in [`prefill_fingerprint`], through [`p2_exchange`]), so ranks on another form are refused by name. Bump it with
/// any change to what crosses or how it is summed.
pub const P2_EXCHANGE: &str = "part-swap-v2";

/// [`P2_EXCHANGE`] under the sequence-parallel residual ([`p2_sp_on`], `GlmModel::run`'s `sp`): a row-split chunk keeps only the rank's
/// rows of the mHC streams; the fp16 hidden rows cross before a mixer, the routed experts and the head, the router's picks and weights
/// after it, the KDA core output's peer columns of the rank's rows before its o_proj, and (a drafting group) the drafter's tap rows; no
/// output rows cross.
pub const P2_EXCHANGE_SP: &str = "part-swap-v3-sp";

/// `REXL3_GLM_P2_SP`: the sequence-parallel residual of a row-split P2 prompt chunk at TP=2 (unset, `1`/`on`: on; `0`/`off`: the v2
/// exchange). Never a bit (every op it moves is row-independent): what crosses the link and which rank computes which rows. A fact of
/// the Hello through [`p2_exchange`]. Read once a process.
pub const SP_VAR: &str = "REXL3_GLM_P2_SP";

/// Whether this process runs the sequence-parallel residual ([`SP_VAR`]; any value but `0`, `off`, `false` or `no` is on).
pub fn p2_sp_on() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        !matches!(std::env::var(SP_VAR).as_deref().map(str::trim), Ok("0" | "off" | "false" | "no"))
    });
    *ON
}

/// The P2 exchange id of this process: [`P2_EXCHANGE_SP`] with the sequence-parallel residual on, [`P2_EXCHANGE`] with it off.
pub fn p2_exchange() -> &'static str {
    if p2_sp_on() {
        P2_EXCHANGE_SP
    } else {
        P2_EXCHANGE
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PrefillContract {
    P1,
    P2,
}

impl PrefillContract {
    /// The short id that goes into logs, the TP Hello and the fingerprint.
    pub const fn id(self) -> &'static str {
        match self {
            PrefillContract::P1 => "p1",
            PrefillContract::P2 => "p2",
        }
    }

    /// The tag appended to the prefix cache's arithmetic field (`prefix::namespace_of`): empty for P1, so P1's namespace is the text it
    /// always was; P2's names what runs P2 and what stays P1 inside a P2 prefill, and the dense kernel's version.
    pub fn arith_tag(self) -> String {
        match self {
            PrefillContract::P1 => String::new(),
            PrefillContract::P2 => {
                format!(
                    "+prefill=p2(dense={P2_DENSE},experts={P2_EXPERTS},split-gu={P2_SPLIT_GU},kda={P2_KDA},mla-ae={P2_MLA},mla-attn=p1,router=p1,indexer=p1,lm_head=p1)"
                )
            }
        }
    }

    /// Whether a forward of this kind runs the contract's prefill arithmetic: prefill chunks only, at every width. A verify or decode
    /// window is P1 under every contract.
    pub const fn p2_prefill(self, is_prefill: bool) -> bool {
        is_prefill && matches!(self, PrefillContract::P2)
    }
}

/// The contract a value of [`CONTRACT_VAR`] asks for: unset or empty is [`DEFAULT_CONTRACT`].
pub fn parse(v: Option<&str>) -> Result<PrefillContract, String> {
    match v.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        None | Some("") => Ok(DEFAULT_CONTRACT),
        Some("p1") => Ok(PrefillContract::P1),
        Some("p2") => Ok(PrefillContract::P2),
        Some(other) => Err(format!("{CONTRACT_VAR}={other:?} is not p1 or p2 (unset is the default, {})", DEFAULT_CONTRACT.id())),
    }
}

/// What [`CONTRACT_VAR`] selects in this process (read once, logged once): the contract, or why the value is refused.
fn selection() -> &'static Result<PrefillContract, String> {
    static C: OnceLock<Result<PrefillContract, String>> = OnceLock::new();
    C.get_or_init(|| {
        let r = parse(std::env::var(CONTRACT_VAR).ok().as_deref());
        match &r {
            Ok(c) => eprintln!(
                "rexl3 level=info event=glm_prefill_contract contract={} default={} var={CONTRACT_VAR}{}",
                c.id(),
                (*c == DEFAULT_CONTRACT) as u8,
                if *c == PrefillContract::P2 {
                    format!(" dense={P2_DENSE} experts={P2_EXPERTS} split-gu={P2_SPLIT_GU} kda={P2_KDA} mla-ae={P2_MLA}")
                } else {
                    String::new()
                }
            ),
            Err(e) => eprintln!("rexl3 level=error event=glm_prefill_contract_refused reason=\"{e}\""),
        }
        r
    })
}

/// The contract of this process. A refused value ([`check_selected`] stops the load first) reads as the default here, so nothing that
/// asks before the load (a log line, the fingerprint) panics.
pub fn current() -> PrefillContract {
    selection().clone().unwrap_or(DEFAULT_CONTRACT)
}

/// `Ok(contract)` when [`CONTRACT_VAR`] names one (or is unset), else the refusal by name; the engine calls it before any weight is read.
pub fn check_selected() -> Result<PrefillContract, String> {
    selection().clone()
}

/// `Ok` when this build can run `c`.
pub fn check_built(c: PrefillContract) -> Result<(), String> {
    match c {
        PrefillContract::P1 => Ok(()),
        PrefillContract::P2 if P2_BUILT => Ok(()),
        PrefillContract::P2 => {
            Err(format!("{CONTRACT_VAR}=p2 asks for GLM's tensor-core prefill, which this build does not link"))
        }
    }
}

/// The prefill fact of a rank for the TP Hello (`GlmFacts::prefill`): `contract=p1`, or under P2 also the dense and expert kernels' versions. Two ranks
/// that differ in it are refused by name (they would leave different bits in the replicated layers and the exchange would carry them).
pub fn prefill_fingerprint() -> String {
    prefill_fingerprint_for(current())
}

/// [`prefill_fingerprint`] for a given contract (pure, for the tests).
pub fn prefill_fingerprint_for(c: PrefillContract) -> String {
    match c {
        PrefillContract::P1 => "contract=p1".into(),
        PrefillContract::P2 => {
            format!(
                "contract=p2,dense={P2_DENSE},experts={P2_EXPERTS},split-gu={P2_SPLIT_GU},kda={P2_KDA},mla-ae={P2_MLA},p2x={}",
                p2_exchange()
            )
        }
    }
}

/// The field the server's `system_fingerprint` carries when this process's GLM prefill contract is not the default: `glmpfp1` under
/// `REXL3_GLM_PREFILL_CONTRACT=p1` (the bits the server produced before the default moved to P2), `None` for the default, P2 (as Nemotron's
/// default P2 adds nothing and its `=p1` adds `_pfp1...`).
pub fn fingerprint_suffix() -> Option<String> {
    suffix_for(current())
}

/// [`fingerprint_suffix`] for a given contract (pure, for the tests).
pub fn suffix_for(c: PrefillContract) -> Option<String> {
    (c != DEFAULT_CONTRACT).then(|| format!("glmpf{}", c.id()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_contract_switch_takes_p1_or_p2_and_p2_is_the_default() {
        assert_eq!(DEFAULT_CONTRACT, PrefillContract::P2);
        assert_eq!(parse(None), Ok(PrefillContract::P2));
        assert_eq!(parse(Some("")), Ok(PrefillContract::P2));
        assert_eq!(parse(Some("  ")), Ok(PrefillContract::P2));
        assert_eq!(parse(Some("p1")), Ok(PrefillContract::P1), "the opt-out");
        assert_eq!(parse(Some(" P1 ")), Ok(PrefillContract::P1));
        assert_eq!(parse(Some("p2")), Ok(PrefillContract::P2));
        assert_eq!(parse(Some(" P2")), Ok(PrefillContract::P2));
        for bad in ["p3", "mma", "1", "fp16", "p1,p2"] {
            let e = parse(Some(bad)).unwrap_err();
            // refused by name: the variable and the value are in the message
            assert!(e.contains(CONTRACT_VAR) && e.contains(bad), "{bad}: {e}");
        }
        assert!(check_built(PrefillContract::P1).is_ok());
        assert_eq!(check_built(PrefillContract::P2).is_ok(), P2_BUILT);
        if std::env::var(CONTRACT_VAR).is_err() {
            assert_eq!(current(), PrefillContract::P2);
            assert_eq!(check_selected(), Ok(PrefillContract::P2));
            assert_eq!(fingerprint_suffix(), None);
        }
    }

    #[test]
    fn p1_keeps_its_namespace_and_p2_names_what_it_changes() {
        // P1 adds nothing to the prefix namespace or the Hello beyond its id (its namespace is the text it always was: the prefix cache of
        // a `=p1` process hits what a P1 process wrote before the default moved); being the non-default contract it now marks its
        // `system_fingerprint` (`glmpfp1`), where the default P2 adds nothing
        assert_eq!(PrefillContract::P1.arith_tag(), "");
        assert_eq!(suffix_for(PrefillContract::P1).as_deref(), Some("glmpfp1"));
        assert_eq!(prefill_fingerprint_for(PrefillContract::P1), "contract=p1");
        // P2 is another namespace, names its dense kernel's version and every block that stays P1 inside it
        let t = PrefillContract::P2.arith_tag();
        for want in [P2_DENSE, P2_EXPERTS, "split-gu=p1", P2_KDA, P2_MLA, "router=p1", "indexer=p1", "lm_head=p1", "mla-attn=p1"] {
            assert!(t.contains(want), "{want} not in {t}");
        }
        assert_eq!(suffix_for(PrefillContract::P2), None, "the default adds nothing to the fingerprint");
        assert!(!t.contains("experts=p1"), "the routed experts run P2 (step 3): {t}");
        assert_eq!(
            prefill_fingerprint_for(PrefillContract::P2),
            format!(
                "contract=p2,dense={P2_DENSE},experts={P2_EXPERTS},split-gu={P2_SPLIT_GU},kda={P2_KDA},mla-ae={P2_MLA},p2x={}",
                p2_exchange()
            )
        );
        assert_ne!(prefill_fingerprint_for(PrefillContract::P1), prefill_fingerprint_for(PrefillContract::P2));
    }

    #[test]
    fn the_exl3_workspace_holds_a_p2_chunk_of_every_dense_shape() {
        // P2 needs no scratch of its own: the stream's EXL3 workspace (`alloc::exl3_workspace_bytes`, priced for the decode kernels and the
        // large-row kernels) holds a chunk's rotated fp16 input at every K the P2 GEMM runs (MLA o_proj's 16,384 the widest), at every chunk
        // size the server takes. (`GlmModel::check_p2` checks the same against the real buffers at boot.) A stub build prices nothing.
        if gb_cuda::exl3_big::big_workspace_bytes(128, 4096, 4096, 2) == 0 {
            return;
        }
        let text = include_str!("../../../../../testdata/glm53/config.json");
        let cfg = crate::families::glm::config::GlmConfig::from_json(&serde_json::from_str(text).unwrap()).unwrap();
        let shapes = crate::families::glm::alloc::exl3_shapes(&cfg);
        assert!(shapes.iter().any(|&(k, _)| k == 16384), "MLA o_proj's K is the widest: {shapes:?}");
        for chunk in [128usize, 256, 512, 1024, 2048, 4096, 6144, 8192] {
            let ws = crate::families::glm::alloc::exl3_workspace_bytes(&cfg, chunk);
            for &(k, n) in &shapes {
                assert!(
                    ws >= gb_cuda::glm_p2::workspace_bytes(chunk, k),
                    "chunk {chunk}: {k} x {n} needs more than the {ws} bytes"
                );
            }
        }
    }

    #[test]
    fn under_p2_a_prefill_chunk_of_128_rows_or_fewer_is_refused_by_name_before_the_load() {
        use crate::families::glm::engine::check_prefill_chunk;
        for f in [1usize, 64, 128] {
            let e = check_prefill_chunk(f, PrefillContract::P2).unwrap_err();
            assert!(e.contains("REXL3_GLM_PREFILL_CONTRACT=p2 needs --prefill-chunk above 128 rows"), "{e}");
        }
        // 4096 is the server's default; P2 takes up to 8,192 rows, and above 8,192 is clamped to it
        for f in [144usize, 512, 2048, 4096, 6144, 8192, 9000] {
            assert!(check_prefill_chunk(f, PrefillContract::P2).is_ok(), "{f}");
        }
        // a chunk that is not a whole number of P2's 16-row KDA chunks is refused by name (never rounded silently)
        for f in [129usize, 1000, 4095, 6150] {
            let e = check_prefill_chunk(f, PrefillContract::P2).unwrap_err();
            assert!(e.contains("multiple of 16 rows"), "{e}");
            assert!(check_prefill_chunk(f, PrefillContract::P1).is_ok(), "P1 takes {f}");
        }
        for f in [1usize, 128, 4096] {
            assert!(check_prefill_chunk(f, PrefillContract::P1).is_ok(), "P1 takes any chunk: {f}");
        }
    }

    #[test]
    fn only_prefill_chunks_run_p2() {
        assert!(PrefillContract::P2.p2_prefill(true));
        assert!(!PrefillContract::P2.p2_prefill(false));
        assert!(!PrefillContract::P1.p2_prefill(true));
        assert!(!PrefillContract::P1.p2_prefill(false));
    }
}
