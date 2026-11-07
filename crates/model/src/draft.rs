//! The DFlash2 draft model (spec section 8): weights, the per-slot context cache, and one draft round.
//!
//! A round, for one sequence at position `P` (the pending anchor token is at `P`; the draft context covers `0..P`):
//!
//! 1. Absorb the target's tap rows for the tokens committed since the last round: `ctx = hidden_norm(fc(taps))`, then
//!    per draft layer the context K/V (`k = rope(norm(k_proj(ctx)))`, `v = v_proj(ctx)`) go into the slot's ring.
//! 2. Run the 5-layer block on `[anchor, MASK x 7]` at positions `P..P+7` (dynamic convs, causal in-block attention
//!    over the ring plus the block's own K/V), then the target's own head on rows 1..7, top-16 per row, and the
//!    selector's chain walk, which picks the 7 draft tokens.

use crate::error::ModelError;
use crate::loadctx::{push_linear, LoadCtx};
use crate::pack::Pack;
use core::ffi::c_void;
use gb_cuda::dflash2::{self, DraftCache, DraftCtxLayout, GateDtype, WalkSampling};
use gb_cuda::exl3_linear::{linear_workspace_bytes, prefill_scratch_bytes, Dtype, Exl3Weights};
use gb_cuda::q4_linear::{self, Q4Weights};
use gb_cuda::graph::Graph;
use gb_cuda::{glue, DevBuf, Stream};
use std::collections::HashMap;

/// Round graph shapes kept at once (each a few hundred launches; a new shape past the limit runs eagerly).
const ROUND_GRAPHS: usize = 64;

/// What identifies a captured draft round: every host value its launches take (`Draft::round_launches`): the sequences and the picks
/// each one asks for, the head (restricted or full), and the target's embedding and head it reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct RoundKey {
    n: usize,
    wants: [u8; MAX_DRAFT_SEQS],
    use_sub: bool,
    embed: usize,
    head: usize,
}

enum RoundSlot {
    /// Seen once: the next round of the shape is captured.
    Seen,
    Ready(Graph),
    /// The capture failed: eager rounds.
    Eager,
}

/// The draft's round graphs and the stream they are captured on.
#[derive(Default)]
struct RoundGraphs {
    slots: HashMap<RoundKey, RoundSlot>,
    cap: Option<Stream>,
}

pub const BLOCK: usize = 8;
/// Most sequences one batched draft round carries (its rows go through the decode kernels, which take at most 128).
pub const MAX_DRAFT_SEQS: usize = 16;
pub const DRAFTS: usize = BLOCK - 1;
/// The least and most rows a draft block may run (`REXL3_DRAFT_BLOCK`, for the depth study; the pack was trained with 8,
/// which stays the default and what `BLOCK`/`DRAFTS` say). A 16-row block halves the batched round to 8 sequences
/// (`MAX_DRAFT_SEQS` blocks of rows still fit the decode kernels' 128), so it is opt-in.
pub const MIN_DRAFT_BLOCK: usize = 8;
pub const MAX_DRAFT_BLOCK: usize = 16;

/// `REXL3_DRAFT_BLOCK`: the draft block's rows, 8..=16 (the trained value when unset). Only the depth study sets it;
/// the engine's chains stay at `DRAFTS` deep whatever it says (`arbiter::MAX_DEPTH` does not move).
pub fn draft_block_from_env() -> Result<usize, ModelError> {
    match std::env::var("REXL3_DRAFT_BLOCK").ok().as_deref() {
        None | Some("") => Ok(BLOCK),
        Some(v) => {
            let b = v.trim().parse::<usize>().map_err(|_| {
                ModelError::Config(format!(
                    "REXL3_DRAFT_BLOCK={v:?}: expected a number from {MIN_DRAFT_BLOCK} to {MAX_DRAFT_BLOCK}"
                ))
            })?;
            if (MIN_DRAFT_BLOCK..=MAX_DRAFT_BLOCK).contains(&b) {
                Ok(b)
            } else {
                Err(ModelError::Config(format!(
                    "REXL3_DRAFT_BLOCK={v:?}: expected a number from {MIN_DRAFT_BLOCK} to {MAX_DRAFT_BLOCK}"
                )))
            }
        }
    }
}
/// One sequence's drafts with the draft's own confidence in each: `score` is the winning selector score (on the target head's
/// logit scale), `margin` the winner's lead over the runner-up in what the pick maximised (the noised score in temperature units,
/// or the raw gap when greedy; at most 30). Each is as long as the number of drafts the sequence asked for.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Picked {
    pub tokens: Vec<u32>,
    pub score: Vec<f32>,
    pub margin: Vec<f32>,
    /// The selector's candidate lattice behind these picks, when the draft was asked to capture it (`Draft::capture_lattice`).
    pub lattice: Option<Lattice>,
}

/// One sequence's selector lattice for a draft round (see `gb_spec::tree::Lattice`): per draft row the head's top-16 candidates with their
/// unary scores and the dot terms between a node's token and each candidate of the next row. The walk's score of candidate `c` of row `i`
/// after pick `parent` is `unary[i][c] + edge[i][parent][c]`, bit for bit (`gb_dflash_selector_edges`). The tree policy and the S0 study
/// read it.
pub use gb_spec::tree::Lattice;
/// The shapes the draft kernels are built for: every DFlash2 pack the engine serves has them (its config is checked against them,
/// `shapes::DraftConfig::check_kernels`). The rest of a pack's shapes and constants are its own ([`DraftDims`]).
pub const LAYERS: usize = 5;
pub const Q_HEADS: usize = 32;
pub const KV_HEADS: usize = 8;
pub const HEAD_DIM: usize = 128;
pub const CONV_GROUP: usize = 16;
pub const SELECTOR_RANK: usize = 256;
pub const SELECTOR_TOP_K: usize = 16;
/// Context positions the draft can ever look back over (its window plus a block of slack).
pub const CTX_KEEP: usize = DraftCtxLayout::WINDOW + 16;

/// The shapes and constants a draft pack's `config.json` gives (`shapes::DraftConfig::dims`): its width, its MLP, how many target
/// layers it taps, its mask token, its norms' epsilon and rope's theta, and the token ids the target's head may propose.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DraftDims {
    pub hidden: usize,
    pub intermediate: usize,
    /// Target layers tapped (the fc's input is this many hidden rows side by side).
    pub taps: usize,
    pub mask_token: u32,
    pub eps: f32,
    pub rope_theta: f64,
    /// Head columns that are tokens: a draft never proposes a column at or past it (the padding columns of a head whose
    /// vocabulary is not a multiple of 128). Equal to the head's width when every column is a token.
    pub valid_vocab: usize,
}

impl DraftDims {
    /// The Qwen3.8-27B target's DFlash2 pack.
    pub const QWEN38: DraftDims = DraftDims {
        hidden: 5120,
        intermediate: 17408,
        taps: 5,
        mask_token: 248070,
        eps: 1e-6,
        rope_theta: 1e7,
        valid_vocab: 248320,
    };

    /// Inputs of the fc: the tapped rows side by side.
    pub const fn fc_in(&self) -> usize {
        self.taps * self.hidden
    }

    /// The dynamic convs' width: four taps a group of `CONV_GROUP` channels.
    pub const fn dyn_width(&self) -> usize {
        4 * (self.hidden / CONV_GROUP)
    }

    /// The attention prep of the draft's block (per-head norms and full rope over 128 dims) at this pack's epsilon and theta.
    pub fn attn_prep(&self) -> glue::AttnPrepCfg {
        glue::AttnPrepCfg { eps: self.eps, theta: self.rope_theta as f32, ..glue::AttnPrepCfg::DFLASH2_DRAFT }
    }
}

/// A draft linear: an EXL3 group of a quantized pack (Qwen's), or the affine 4-bit matrix a bf16 pack's linear is quantized to at
/// load (GLM's, [`Draft::declare_bf16`]).
enum Lin {
    Exl3(Exl3Weights),
    Q4(Q4Weights),
}

/// The scratch a draft linear runs in: the EXL3 decode workspace (`<= 128` rows) and prefill scratch, and the 4-bit linear's input
/// group sums.
struct LinScratch {
    ws: DevBuf,
    pws: DevBuf,
    gs: DevBuf,
}

impl Lin {
    /// `y [m, n] = x [m, k] . W^T`, bf16 in and out.
    fn forward(&self, x: &DevBuf, y: &mut DevBuf, m: usize, s: &mut LinScratch, st: &Stream) -> Result<(), ModelError> {
        match self {
            Lin::Exl3(w) if m > 128 => w.forward_prefill(x, y, m, Dtype::BF16, &mut s.pws, st)?,
            Lin::Exl3(w) => w.forward(x, y, m, Dtype::BF16, Dtype::BF16, &mut s.ws, st)?,
            Lin::Q4(w) => w.forward(x, &mut s.gs, y, m, st)?,
        }
        Ok(())
    }

    fn push<'a>(&'a self, v: &mut Vec<(String, &'a DevBuf)>, name: String) {
        match self {
            Lin::Exl3(w) => push_linear(v, name, w),
            Lin::Q4(w) => v.push((format!("{name}.q4"), &w.buf)),
        }
    }
}

/// Linears that read one input: separate ones (an EXL3 pack's), or, for a bf16 pack, ONE 4-bit matrix of their rows stacked
/// ([`load_group`]), which one launch runs, writing each part's output to its own buffer. Each part's rows are quantized exactly
/// as they would be alone; only the K split (a function of the stacked shape) can differ, and a draft only proposes tokens.
enum LinGroup {
    Each(Vec<Lin>),
    Stacked { w: Q4Weights, parts: Vec<usize> },
}

impl LinGroup {
    /// Parts `from..` of the group over `m` rows of `x`: part `from + i` into `outs[i]`, bf16 `[m, part width]`.
    fn forward(
        &self,
        x: &DevBuf,
        from: usize,
        outs: &mut [&mut DevBuf],
        m: usize,
        s: &mut LinScratch,
        st: &Stream,
    ) -> Result<(), ModelError> {
        match self {
            LinGroup::Each(v) => {
                for (l, y) in v[from..].iter().zip(outs.iter_mut()) {
                    l.forward(x, y, m, s, st)?;
                }
            }
            LinGroup::Stacked { w, parts } => {
                let col0 = parts[..from].iter().sum();
                let mut segs: Vec<(&mut DevBuf, usize)> = outs.iter_mut().map(|y| &mut **y).zip(parts[from..].iter().copied()).collect();
                w.forward_cols(x, &mut s.gs, col0, &mut segs, m, st)?;
            }
        }
        Ok(())
    }

    fn push<'a>(&'a self, v: &mut Vec<(String, &'a DevBuf)>, names: &[String]) {
        match self {
            LinGroup::Each(ls) => {
                for (l, n) in ls.iter().zip(names) {
                    l.push(v, n.clone());
                }
            }
            LinGroup::Stacked { w, .. } => v.push((format!("{}.q4", names.join("+")), &w.buf)),
        }
    }
}

struct ConvPair {
    /// f16 `[2, 2, hidden]`
    base: DevBuf,
    /// f16 `[4 hidden / 16, hidden]`
    proj: DevBuf,
}

struct DraftLayer {
    input_norm: DevBuf,
    post_norm: DevBuf,
    /// q, k and v
    qkv: LinGroup,
    o: Lin,
    q_norm: DevBuf,
    k_norm: DevBuf,
    /// gate and up
    gu: LinGroup,
    down: Lin,
    attn_conv: ConvPair,
    mlp_conv: ConvPair,
}

/// One sequence's context rows to absorb: `n` tap rows from `row0` of the target's tap buffer, at positions `pos0..` of
/// `slot`'s ring.
#[derive(Clone, Copy, Debug)]
pub struct Absorb {
    pub row0: usize,
    pub n: usize,
    pub slot: usize,
    pub pos0: usize,
}

/// One sequence in a draft round: the anchor token (the last one, not yet fed) at position `p` of `slot`, and the
/// request's sampling (the walk draws with the target's keyed sampler on sampled requests).
#[derive(Clone, Copy, Debug)]
pub struct DraftReq<'a> {
    pub anchor: u32,
    pub slot: usize,
    pub p: usize,
    pub sampling: &'a gb_sampling::Sampling,
    /// How many drafts to return (1..=7). The block always runs its 8 rows through the layers; only the first `want` picks are
    /// computed (head rows, top-k and walk steps), and they are the same picks a full round would make first.
    pub want: usize,
    /// Share of the sequence's recent tokens that lie inside the draft's restricted vocabulary (see [`Draft::coverage`]); only an
    /// adaptive draft head reads it. 1.0 when nothing is restricted.
    pub coverage: f32,
}

/// Recent tokens an adaptive draft head looks at to tell whether a sequence is writing text its restricted vocabulary can reach.
pub const COVERAGE_WINDOW: usize = 64;

/// Whether a token id lies in `ranges`, `(first id, length)` pairs.
fn in_ranges(ranges: &[(usize, usize)], id: u32) -> bool {
    ranges.iter().any(|&(first, len)| (first..first + len).contains(&(id as usize)))
}

/// Share of `ids` inside `ranges` (1.0 for an empty list: nothing the set cannot reach).
pub fn coverage_of(ranges: &[(usize, usize)], ids: &[u32]) -> f32 {
    if ids.is_empty() {
        return 1.0;
    }
    ids.iter().filter(|&&t| in_ranges(ranges, t)).count() as f32 / ids.len() as f32
}

/// A restricted candidate set for the draft's vocabulary head: the target's `lm_head` columns of a few ranges of token ids, so the
/// head reads a fraction of its 0.95 GB each round. The verify path still uses the full head; this only decides which tokens the
/// draft can propose, so a token outside the set is a draft that cannot match, never a different output.
struct SubHead {
    /// The head over the kept columns (a copy of those strips of the target's `lm_head`).
    w: Exl3Weights,
    /// Token ids covered, as ascending `(first id, length)` ranges laid end to end in the compact numbering.
    ranges: Vec<(usize, usize)>,
    /// bf16 `[max rows, columns]`
    logits: DevBuf,
    ws: DevBuf,
}

/// Wall time of the steps of [`Draft::restrict_vocab`], in seconds: the strip copy ([`gb_cuda::exl3_linear::StripTimes`]), the logits and
/// workspace buffers, and the whole call.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RestrictTimes {
    pub strips: gb_cuda::exl3_linear::StripTimes,
    pub buffers: f64,
    pub total: f64,
}

fn cp(b: &DevBuf) -> *const c_void {
    b.ptr() as *const c_void
}
fn mp(b: &DevBuf) -> *mut c_void {
    b.ptr()
}

pub struct Draft {
    /// The pack's shapes and constants.
    pub dims: DraftDims,
    /// Columns of the target head the draft runs (and rows of its selector codebooks).
    vocab: usize,
    fc: Lin,
    hidden_norm: DevBuf,
    norm: DevBuf,
    layers: Vec<DraftLayer>,
    sel_proj: DevBuf,
    pred: DevBuf,
    succ: DevBuf,
    cache: DraftCache,
    pub slots: usize,
    /// Sequences per batched round: `slots` up to `MAX_DRAFT_SEQS`, further when a 16-row block halves it (8 sequences
    /// keep the round's rows within the decode kernels' 128).
    pub max_seqs: usize,
    /// Rows of the draft's block (`REXL3_DRAFT_BLOCK`, the trained 8 when unset). The layers run this many rows per
    /// sequence; `DRAFTS`-deep chains and the engine's `MAX_DEPTH` stay at 7/8 whatever it says.
    pub block: usize,
    max_ctx_rows: usize,
    // scratch (block rows = 8; context rows up to max_ctx_rows)
    fc_in: DevBuf,
    ctx: DevBuf,
    kproj: DevBuf,
    vproj: DevBuf,
    ctx_pos: DevBuf,
    ctx_slot: DevBuf,
    /// The linears' scratch; its EXL3 decode workspace is the target head's too.
    lin: LinScratch,
    attn_ws: DevBuf,
    x: DevBuf,
    x2: DevBuf,
    h: DevBuf,
    y: DevBuf,
    o: DevBuf,
    dynv: DevBuf,
    qb: DevBuf,
    kb: DevBuf,
    vb: DevBuf,
    q2: DevBuf,
    k2: DevBuf,
    att: DevBuf,
    g: DevBuf,
    u: DevBuf,
    sw: DevBuf,
    sel: DevBuf,
    logits: DevBuf,
    top_ids: DevBuf,
    top_vals: DevBuf,
    top_ws: DevBuf,
    gate: DevBuf,
    ids: DevBuf,
    pos: DevBuf,
    anchor: DevBuf,
    slot_a: DevBuf,
    p_a: DevBuf,
    out_tokens: DevBuf,
    /// f32 `[nd]`: the selector score of each pick (the draft's own confidence in it).
    conf: DevBuf,
    /// f32 `[nd]`: the winner's lead over the runner-up of each pick.
    margin: DevBuf,
    walk: DevBuf,
    /// i32 `[max_seqs + 1]`: where each sequence's picks start in the packed head rows
    row_off: DevBuf,
    sub: Option<SubHead>,
    /// With a restricted vocabulary: `Some(min)` runs the restricted head only in rounds where every sequence has at least `min` of
    /// its recent tokens inside the set (text the set cannot reach gets the full head), `None` runs it every round.
    adaptive_min: Option<f32>,
    /// Where the last [`Draft::restrict_vocab`] spent its time (boot-time diagnostics).
    restrict_times: Option<RestrictTimes>,
    /// Rounds the draft head ran over every column (`[0]`) and over the restricted set (`[1]`).
    pub head_rounds: [u64; 2],
    /// Download every round's selector lattice into [`Picked::lattice`] (tree drafting measurements; one extra kernel and three small
    /// downloads a round). Off for serving.
    pub capture_lattice: bool,
    /// f32 `[max_seqs * DRAFTS, k, k]`: the lattice's edge terms, allocated on first capture.
    edge: Option<DevBuf>,
    /// The host bytes of the round's and the absorbs' index uploads, queued on the stream in its order (the host does not wait for the
    /// work queued before them) and kept until the stream has run them.
    stage: gb_cuda::rt::HostStage,
    /// The draft rounds' CUDA graphs (`Draft::round_graphed`).
    rounds: RoundGraphs,
}

fn load_exl3<'p>(
    ctx: &mut LoadCtx<'p>,
    pack: &'p Pack,
    name: &str,
    k: usize,
    n: usize,
    bytes: &mut usize,
) -> Result<Lin, ModelError> {
    let g = pack.group_meta(name)?;
    if (g.k, g.n) != (k, n) {
        return Err(ModelError::Config(format!("draft {name}: expected K={k} N={n}, found K={} N={}", g.k, g.n)));
    }
    let w = ctx.group(pack, name)?;
    *bytes += w.weight_bytes();
    Ok(Lin::Exl3(w))
}

/// A bf16 `[n, k]` linear (`{name}.weight`) quantized to the affine 4-bit format on a loader thread.
fn load_q4<'p>(
    ctx: &mut LoadCtx<'p>,
    pack: &'p Pack,
    name: &str,
    k: usize,
    n: usize,
    bytes: &mut usize,
) -> Result<Lin, ModelError> {
    let name = format!("{name}.weight");
    pack.entry(&name, "BF16", &[n, k])?;
    if k % q4_linear::GROUP != 0 {
        return Err(ModelError::Config(format!("draft {name}: {k} inputs are not whole groups of {}", q4_linear::GROUP)));
    }
    let len = q4_linear::bytes(n, k);
    let buf = ctx.computed(len, move || Ok(q4_linear::quantize_bf16(&pack.bf16(&name, &[n, k])?, n, k)))?;
    *bytes += len;
    Ok(Lin::Q4(Q4Weights::from_buf(buf, n, k)?))
}

/// The linears `names` (each `[n, k]`, `parts` the n's) that read one input: an EXL3 pack's one by one, a bf16 pack's as one 4-bit
/// matrix of their rows stacked in order ([`LinGroup::Stacked`]; every part but the last whole 64-row tiles).
fn load_group<'p>(
    ctx: &mut LoadCtx<'p>,
    pack: &'p Pack,
    format: Format,
    names: &[String],
    k: usize,
    parts: &[usize],
    bytes: &mut usize,
) -> Result<LinGroup, ModelError> {
    if format == Format::Exl3 {
        let mut v = Vec::with_capacity(names.len());
        for (name, &n) in names.iter().zip(parts) {
            v.push(load_exl3(ctx, pack, name, k, n, bytes)?);
        }
        return Ok(LinGroup::Each(v));
    }
    let tensors: Vec<String> = names.iter().map(|n| format!("{n}.weight")).collect();
    for (t, &n) in tensors.iter().zip(parts) {
        pack.entry(t, "BF16", &[n, k])?;
    }
    if k % q4_linear::GROUP != 0 || parts[..parts.len() - 1].iter().any(|n| n % 64 != 0) {
        return Err(ModelError::Config(format!("draft {names:?}: parts {parts:?} of {k} inputs cannot be stacked")));
    }
    let n: usize = parts.iter().sum();
    let len = q4_linear::bytes(n, k);
    let shapes: Vec<(String, usize)> = tensors.into_iter().zip(parts.iter().copied()).collect();
    let buf = ctx.computed(len, move || {
        let mut w = Vec::with_capacity(n * k);
        for (t, rows) in &shapes {
            w.extend_from_slice(&pack.bf16(t, &[*rows, k])?);
        }
        Ok(q4_linear::quantize_bf16(&w, n, k))
    })?;
    *bytes += len;
    Ok(LinGroup::Stacked { w: Q4Weights::from_buf(buf, n, k)?, parts: parts.to_vec() })
}

fn f32_dev<'p>(ctx: &mut LoadCtx<'p>, pack: &'p Pack, name: &str, shape: &[usize]) -> Result<DevBuf, ModelError> {
    ctx.f32_from_bf16(pack, name, shape, false)
}

/// An fp16 tensor of the draft: stored as fp16 (a quantized pack), or bf16 narrowed on a loader thread (a bf16 pack).
fn f16_dev<'p>(
    ctx: &mut LoadCtx<'p>,
    pack: &'p Pack,
    name: &str,
    shape: &[usize],
    bytes: &mut usize,
) -> Result<DevBuf, ModelError> {
    let b = if pack.entry(name, "BF16", shape).is_ok() {
        ctx.f16_from_bf16(pack, name, shape)?
    } else {
        ctx.raw16(pack, name, "F16", shape)?
    };
    *bytes += b.bytes();
    Ok(b)
}

fn conv_pair<'p>(
    ctx: &mut LoadCtx<'p>,
    pack: &'p Pack,
    dims: &DraftDims,
    p: &str,
    which: &str,
    bytes: &mut usize,
) -> Result<ConvPair, ModelError> {
    let h = dims.hidden;
    Ok(ConvPair {
        base: f16_dev(ctx, pack, &format!("{p}{which}.base_kernel"), &[2, 2, h], bytes)?,
        proj: f16_dev(ctx, pack, &format!("{p}{which}.kernel_projection.weight"), &[dims.dyn_width(), h], bytes)?,
    })
}

/// How a draft's linears are stored in its pack.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    /// EXL3 groups (Qwen's quantized pack).
    Exl3,
    /// bf16 tensors, quantized to the affine 4-bit format at load.
    Bf16,
}

impl Draft {
    /// Load the draft pack in a pipeline of its own. `slots` sequence slots get a context ring each; `max_ctx_rows` is the
    /// most context rows absorbed in one call (a prefill chunk). The block is the trained 8; `Draft::load_block` for the
    /// depth study's.
    pub fn load(pack: &Pack, vocab: usize, slots: usize, max_ctx_rows: usize) -> Result<(Self, usize), ModelError> {
        Self::load_block(pack, vocab, slots, max_ctx_rows, BLOCK)
    }

    /// [`Draft::load`] with an explicit draft block (8..=16, `draft_block_from_env`): a 16-row block halves the batched
    /// round to 8 sequences, so it is opt-in.
    pub fn load_block(
        pack: &Pack,
        vocab: usize,
        slots: usize,
        max_ctx_rows: usize,
        block: usize,
    ) -> Result<(Self, usize), ModelError> {
        let mut ctx = LoadCtx::from_env();
        let r = Self::declare(&mut ctx, pack, vocab, slots, max_ctx_rows, block)?;
        ctx.finish()?;
        Ok(r)
    }

    /// Every weight buffer of the draft (not its scratch), named and in a fixed order, for comparing two loads of one pack.
    pub fn weight_buffers(&self) -> Vec<(String, &DevBuf)> {
        let mut v: Vec<(String, &DevBuf)> = Vec::new();
        self.fc.push(&mut v, "fc".into());
        v.push(("hidden_norm".into(), &self.hidden_norm));
        v.push(("norm".into(), &self.norm));
        for (i, l) in self.layers.iter().enumerate() {
            let p = format!("layers.{i}");
            v.push((format!("{p}.input_norm"), &l.input_norm));
            v.push((format!("{p}.post_norm"), &l.post_norm));
            let names = |ns: &[&str]| ns.iter().map(|n| format!("{p}.{n}")).collect::<Vec<_>>();
            l.qkv.push(&mut v, &names(&["q", "k", "v"]));
            l.o.push(&mut v, format!("{p}.o"));
            l.gu.push(&mut v, &names(&["gate", "up"]));
            l.down.push(&mut v, format!("{p}.down"));
            v.push((format!("{p}.q_norm"), &l.q_norm));
            v.push((format!("{p}.k_norm"), &l.k_norm));
            for (n, c) in [("attn_conv", &l.attn_conv), ("mlp_conv", &l.mlp_conv)] {
                v.push((format!("{p}.{n}.base"), &c.base));
                v.push((format!("{p}.{n}.proj"), &c.proj));
            }
        }
        v.push(("sel_proj".into(), &self.sel_proj));
        v.push(("pred".into(), &self.pred));
        v.push(("succ".into(), &self.succ));
        v
    }

    /// Declare the Qwen3.8 pack's weights (EXL3 linears, fp16 convs and selector) against `ctx` and allocate the draft's scratch.
    /// The weight buffers hold nothing until `ctx.finish()` has run.
    pub fn declare<'p>(
        ctx: &mut LoadCtx<'p>,
        pack: &'p Pack,
        vocab: usize,
        slots: usize,
        max_ctx_rows: usize,
        block: usize,
    ) -> Result<(Self, usize), ModelError> {
        Self::declare_as(ctx, pack, Format::Exl3, DraftDims::QWEN38, vocab, slots, max_ctx_rows, block)
    }

    /// Declare a bf16 pack's weights (the GLM-5.3-Flash DFlash2 checkpoint as published) against `ctx`: every linear quantized to
    /// the affine 4-bit format on a loader thread, the convs, the selector and its codebooks narrowed to fp16, the norms widened
    /// to fp32. `dims` is the pack's (`shapes::DraftConfig::dims`), `vocab` the target head's width. The weight buffers hold
    /// nothing until `ctx.finish()` has run.
    #[allow(clippy::too_many_arguments)]
    pub fn declare_bf16<'p>(
        ctx: &mut LoadCtx<'p>,
        pack: &'p Pack,
        dims: DraftDims,
        vocab: usize,
        slots: usize,
        max_ctx_rows: usize,
        block: usize,
    ) -> Result<(Self, usize), ModelError> {
        Self::declare_as(ctx, pack, Format::Bf16, dims, vocab, slots, max_ctx_rows, block)
    }

    /// Device bytes [`Draft::declare_bf16`] allocates for these settings (weights, context rings and scratch; the restricted
    /// head and the lattice buffer are extra): what a memory plan holds back for the draft before it is loaded.
    pub fn planned_bf16_bytes(dims: &DraftDims, vocab: usize, slots: usize, max_ctx_rows: usize, block: usize) -> u64 {
        let (h, f, kv, q) = (dims.hidden, dims.intermediate, KV_HEADS * HEAD_DIM, Q_HEADS * HEAD_DIM);
        let lin = |n: usize, k: usize| q4_linear::bytes(n, k);
        let per_layer = lin(q, h) + 2 * lin(kv, h) + lin(h, q) + 2 * lin(f, h) + lin(h, f)
            + 2 * (2 * 2 * h + dims.dyn_width() * h) * 2
            + (2 * h + 2 * HEAD_DIM) * 4;
        let weights = lin(h, dims.fc_in()) + 2 * h * 4 + LAYERS * per_layer + SELECTOR_RANK * h * 2 + 2 * vocab * SELECTOR_RANK * 2;
        let ring = DraftCtxLayout::qwen38(slots).total_bytes();
        (weights + ring + Self::scratch_bytes(dims, vocab, slots, max_ctx_rows, block, Format::Bf16)) as u64
    }

    /// Sequences a batched round carries at this many slots and this block.
    fn max_seqs_of(slots: usize, block: usize) -> usize {
        slots.clamp(1, MAX_DRAFT_SEQS.min(128 / block))
    }

    /// The linears' workspaces `(EXL3 decode, EXL3 prefill, 4-bit group sums)` in bytes.
    fn lin_scratch_bytes(dims: &DraftDims, vocab: usize, max_ctx_rows: usize, nb: usize, nd: usize, format: Format) -> [usize; 3] {
        let (h, f) = (dims.hidden, dims.intermediate);
        let shapes =
            [(h, Q_HEADS * HEAD_DIM), (h, KV_HEADS * HEAD_DIM), (Q_HEADS * HEAD_DIM, h), (h, f), (f, h), (dims.fc_in(), h)];
        // the target head's decode workspace always; the EXL3 linears' decode workspace at up to 128 rows and prefill scratch for
        // the fc over a whole chunk, or the 4-bit linears' group sums over the widest input at the most rows a call takes
        let head = linear_workspace_bytes(nd, h, vocab, 6);
        match format {
            Format::Exl3 => [
                shapes.iter().map(|&(k, n)| linear_workspace_bytes(max_ctx_rows.min(128).max(nb), k, n, 4)).max().unwrap_or(0).max(head),
                shapes.iter().map(|&(k, n)| prefill_scratch_bytes(max_ctx_rows, k, n)).max().unwrap_or(0),
                0,
            ],
            Format::Bf16 => {
                let widest = shapes.iter().map(|&(k, _)| k).max().unwrap_or(0);
                [head, 0, q4_linear::gsum_bytes(max_ctx_rows.max(nb), widest)]
            }
        }
    }

    /// Bytes of the scratch [`Draft::declare_as`] allocates.
    fn scratch_bytes(dims: &DraftDims, vocab: usize, slots: usize, max_ctx_rows: usize, block: usize, format: Format) -> usize {
        let max_seqs = Self::max_seqs_of(slots, block);
        let (nb, nd) = (max_seqs * block, max_seqs * (block - 1));
        let (h, f, kv, q) = (dims.hidden, dims.intermediate, KV_HEADS * HEAD_DIM, Q_HEADS * HEAD_DIM);
        let topk = if dims.valid_vocab < vocab {
            glue::topk_valid_workspace_bytes(nd, dims.valid_vocab, SELECTOR_TOP_K)
        } else {
            glue::topk_workspace_bytes(nd, vocab, SELECTOR_TOP_K)
        };
        let lin: usize = Self::lin_scratch_bytes(dims, vocab, max_ctx_rows, nb, nd, format).iter().sum();
        max_ctx_rows * (dims.fc_in() + h + 2 * kv) * 2
            + max_ctx_rows * 8
            + lin
            + dflash2::attn_workspace_bytes(max_seqs, DraftCtxLayout::WINDOW, Q_HEADS, block)
            + nb * (5 * h + dims.dyn_width() + 3 * q + 2 * kv + 3 * f) * 2
            + nd * (h + vocab) * 2
            + nd * SELECTOR_TOP_K * 8
            + topk
            + nd * SELECTOR_RANK * 2
            + nb * 8
            + max_seqs * 12
            + nd * 12
            + max_seqs * std::mem::size_of::<WalkSampling>()
            + (max_seqs + 1) * 4
    }

    #[allow(clippy::too_many_arguments)]
    fn declare_as<'p>(
        ctx: &mut LoadCtx<'p>,
        pack: &'p Pack,
        format: Format,
        dims: DraftDims,
        vocab: usize,
        slots: usize,
        max_ctx_rows: usize,
        block: usize,
    ) -> Result<(Self, usize), ModelError> {
        if !(MIN_DRAFT_BLOCK..=MAX_DRAFT_BLOCK).contains(&block) {
            return Err(ModelError::Config(format!(
                "the draft block is {block} rows, the kernels are built for {MIN_DRAFT_BLOCK} to {MAX_DRAFT_BLOCK}"
            )));
        }
        if dims.valid_vocab > vocab || dims.mask_token as usize >= vocab {
            return Err(ModelError::Config(format!(
                "the draft's vocabulary ({} tokens, mask {}) does not fit the target head's {vocab} columns",
                dims.valid_vocab, dims.mask_token
            )));
        }
        let (h, f) = (dims.hidden, dims.intermediate);
        let mut bytes = 0usize;
        let lin = |ctx: &mut LoadCtx<'p>, name: &str, k: usize, n: usize, bytes: &mut usize| match format {
            Format::Exl3 => load_exl3(ctx, pack, name, k, n, bytes),
            Format::Bf16 => load_q4(ctx, pack, name, k, n, bytes),
        };
        let fc = lin(ctx, "fc", dims.fc_in(), h, &mut bytes)?;
        let hidden_norm = f32_dev(ctx, pack, "hidden_norm.weight", &[h])?;
        let norm = f32_dev(ctx, pack, "norm.weight", &[h])?;
        let mut layers = Vec::new();
        for l in 0..LAYERS {
            let p = format!("layers.{l}.");
            layers.push(DraftLayer {
                input_norm: f32_dev(ctx, pack, &format!("{p}input_layernorm.weight"), &[h])?,
                post_norm: f32_dev(ctx, pack, &format!("{p}post_attention_layernorm.weight"), &[h])?,
                qkv: load_group(
                    ctx,
                    pack,
                    format,
                    &[format!("{p}self_attn.q_proj"), format!("{p}self_attn.k_proj"), format!("{p}self_attn.v_proj")],
                    h,
                    &[Q_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM],
                    &mut bytes,
                )?,
                o: lin(ctx, &format!("{p}self_attn.o_proj"), Q_HEADS * HEAD_DIM, h, &mut bytes)?,
                q_norm: f32_dev(ctx, pack, &format!("{p}self_attn.q_norm.weight"), &[HEAD_DIM])?,
                k_norm: f32_dev(ctx, pack, &format!("{p}self_attn.k_norm.weight"), &[HEAD_DIM])?,
                gu: load_group(ctx, pack, format, &[format!("{p}mlp.gate_proj"), format!("{p}mlp.up_proj")], h, &[f, f], &mut bytes)?,
                down: lin(ctx, &format!("{p}mlp.down_proj"), f, h, &mut bytes)?,
                attn_conv: conv_pair(ctx, pack, &dims, &p, "attention_conv", &mut bytes)?,
                mlp_conv: conv_pair(ctx, pack, &dims, &p, "mlp_conv", &mut bytes)?,
            });
        }
        // the codebooks are `.weight` tensors in the Qwen pack and bare in the GLM checkpoint
        let codebook = |name: &str| {
            let w = format!("candidate_selector.{name}.weight");
            if pack.ck.plain.contains_key(&w) {
                w
            } else {
                format!("candidate_selector.{name}")
            }
        };
        let sel_proj = f16_dev(ctx, pack, "candidate_selector.hidden_projection.weight", &[SELECTOR_RANK, h], &mut bytes)?;
        let pred = f16_dev(ctx, pack, &codebook("predecessor_codebook"), &[vocab, SELECTOR_RANK], &mut bytes)?;
        let succ = f16_dev(ctx, pack, &codebook("successor_codebook"), &[vocab, SELECTOR_RANK], &mut bytes)?;
        let layout = DraftCtxLayout::qwen38(slots);
        let cache = DraftCache::new(layout)?;
        bytes += layout.total_bytes();
        let max_seqs = Self::max_seqs_of(slots, block);
        let (nb, nd) = (max_seqs * block, max_seqs * (block - 1));
        let [ws_bytes, pws_bytes, gs_bytes] = Self::lin_scratch_bytes(&dims, vocab, max_ctx_rows, nb, nd, format);
        let topk_ws = if dims.valid_vocab < vocab {
            glue::topk_valid_workspace_bytes(nd, dims.valid_vocab, SELECTOR_TOP_K)
        } else {
            glue::topk_workspace_bytes(nd, vocab, SELECTOR_TOP_K)
        };
        // a bf16 pack's figure is everything the draft allocates (`planned_bf16_bytes`, which a memory plan holds back); the Qwen
        // pack's keeps what it always reported (its linears, fp16 tensors and rings)
        if format == Format::Bf16 {
            bytes += (2 * h + LAYERS * (2 * h + 2 * HEAD_DIM)) * 4 + Self::scratch_bytes(&dims, vocab, slots, max_ctx_rows, block, format);
        }
        let (kv, q) = (KV_HEADS * HEAD_DIM, Q_HEADS * HEAD_DIM);
        let z = |n: usize| DevBuf::zeroed(n);
        let d = Self {
            dims,
            vocab,
            fc,
            hidden_norm,
            norm,
            layers,
            sel_proj,
            pred,
            succ,
            cache,
            slots,
            max_seqs,
            block,
            max_ctx_rows,
            fc_in: z(max_ctx_rows * dims.fc_in() * 2)?,
            ctx: z(max_ctx_rows * h * 2)?,
            kproj: z(max_ctx_rows * kv * 2)?,
            vproj: z(max_ctx_rows * kv * 2)?,
            ctx_pos: z(max_ctx_rows * 4)?,
            ctx_slot: z(max_ctx_rows * 4)?,
            lin: LinScratch { ws: z(ws_bytes)?, pws: z(pws_bytes)?, gs: z(gs_bytes)? },
            attn_ws: z(dflash2::attn_workspace_bytes(max_seqs, DraftCtxLayout::WINDOW, Q_HEADS, block))?,
            x: z(nb * h * 2)?,
            x2: z(nb * h * 2)?,
            h: z(nb * h * 2)?,
            y: z(nb * h * 2)?,
            o: z(nb * h * 2)?,
            dynv: z(nb * dims.dyn_width() * 2)?,
            qb: z(nb * q * 2)?,
            kb: z(nb * kv * 2)?,
            vb: z(nb * kv * 2)?,
            q2: z(nb * q * 2)?,
            k2: z(nb * kv * 2)?,
            att: z(nb * q * 2)?,
            g: z(nb * f * 2)?,
            u: z(nb * f * 2)?,
            sw: z(nb * f * 2)?,
            sel: z(nd * h * 2)?,
            logits: z(nd * vocab * 2)?,
            top_ids: z(nd * SELECTOR_TOP_K * 4)?,
            top_vals: z(nd * SELECTOR_TOP_K * 4)?,
            top_ws: z(topk_ws)?,
            gate: z(nd * SELECTOR_RANK * 2)?,
            ids: z(nb * 4)?,
            pos: z(nb * 4)?,
            anchor: z(max_seqs * 4)?,
            slot_a: z(max_seqs * 4)?,
            p_a: z(max_seqs * 4)?,
            out_tokens: z(nd * 4)?,
            conf: z(nd * 4)?,
            margin: z(nd * 4)?,
            walk: z(max_seqs * std::mem::size_of::<WalkSampling>())?,
            row_off: z((max_seqs + 1) * 4)?,
            sub: None,
            adaptive_min: None,
            restrict_times: None,
            head_rounds: [0; 2],
            capture_lattice: false,
            edge: None,
            stage: gb_cuda::rt::HostStage::new(),
            rounds: RoundGraphs::default(),
        };
        Ok((d, bytes))
    }

    /// The most context rows one absorb takes.
    pub fn max_ctx_rows(&self) -> usize {
        self.max_ctx_rows
    }

    /// Bytes of one ring row (one position's K or V, all kv heads) of one draft layer.
    pub const RING_ROW_BYTES: usize = KV_HEADS * HEAD_DIM * 2;

    /// Bytes of the context window a prefix-cache snapshot keeps: K and V of every draft layer for the last [`DraftCtxLayout::WINDOW`] positions.
    pub const fn window_snapshot_bytes() -> u64 {
        (LAYERS * 2 * DraftCtxLayout::WINDOW * Self::RING_ROW_BYTES) as u64
    }

    /// The ring segments (`(ring byte offset, bytes, offset in the window copy)`) of positions `lo..lo + n` of `slot`: one, or two when the
    /// positions wrap the ring.
    fn ring_segments(&self, slot: usize, lo: usize, n: usize) -> Vec<(usize, usize, usize)> {
        let l = &self.cache.layout;
        let first = n.min(l.cap - lo % l.cap);
        let mut v = vec![((slot * l.cap + lo % l.cap) * Self::RING_ROW_BYTES, first * Self::RING_ROW_BYTES, 0)];
        if first < n {
            v.push((slot * l.cap * Self::RING_ROW_BYTES, (n - first) * Self::RING_ROW_BYTES, first * Self::RING_ROW_BYTES));
        }
        v
    }

    /// FNV-1a over the ring rows of positions `lo..lo + n` of `slot` (every layer, K then V, in position order), for tests that compare a
    /// restored window with the saved one ([`crate::snapshot::SnapPool::window_hash`] hashes a snapshot's the same way).
    pub fn ring_window_hash(&self, slot: usize, lo: usize, n: usize) -> Result<u64, ModelError> {
        self.cache_sync_hash(slot, lo, n)
    }

    fn cache_sync_hash(&self, slot: usize, lo: usize, n: usize) -> Result<u64, ModelError> {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for layer in 0..LAYERS {
            for buf in [self.cache.k(layer), self.cache.v(layer)] {
                for (off, bytes, _) in self.ring_segments(slot, lo, n) {
                    let mut v = vec![0u8; bytes];
                    buf.download_range(off, &mut v)?;
                    h = v.iter().fold(h, |h, &b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3));
                }
            }
        }
        Ok(h)
    }

    /// Copy the context ring's rows of positions `lo..lo + n` of `slot` into snapshot `snap` of `dk` / `dv` (one buffer per draft layer, each
    /// `[snapshots][WINDOW][kv_heads x head_dim]` bf16): row `i` of the snapshot is position `lo + i`. `n` is at most the window.
    #[allow(clippy::too_many_arguments)] // slot, the positions, both snapshot buffers and which snapshot, the stream
    pub fn save_window(
        &self,
        slot: usize,
        lo: usize,
        n: usize,
        dk: &mut [DevBuf],
        dv: &mut [DevBuf],
        snap: usize,
        stream: &Stream,
    ) -> Result<(), ModelError> {
        assert!(n <= DraftCtxLayout::WINDOW && dk.len() == LAYERS && dv.len() == LAYERS);
        for (layer, (k, v)) in dk.iter_mut().zip(dv.iter_mut()).enumerate() {
            for (off, bytes, at) in self.ring_segments(slot, lo, n) {
                let dst = snap * DraftCtxLayout::WINDOW * Self::RING_ROW_BYTES + at;
                k.copy_range_from(dst, self.cache.k(layer), off, bytes, stream)?;
                v.copy_range_from(dst, self.cache.v(layer), off, bytes, stream)?;
            }
        }
        Ok(())
    }

    /// The reverse of [`Draft::save_window`]: put snapshot `snap`'s rows back at positions `lo..lo + n` of `slot`'s ring.
    #[allow(clippy::too_many_arguments)]
    pub fn load_window(
        &mut self,
        slot: usize,
        lo: usize,
        n: usize,
        dk: &[DevBuf],
        dv: &[DevBuf],
        snap: usize,
        stream: &Stream,
    ) -> Result<(), ModelError> {
        assert!(n <= DraftCtxLayout::WINDOW && dk.len() == LAYERS && dv.len() == LAYERS);
        let segs = self.ring_segments(slot, lo, n);
        for (layer, (k, v)) in dk.iter().zip(dv.iter()).enumerate() {
            let (rk, rv) = self.cache.kv_mut(layer);
            for &(off, bytes, at) in &segs {
                let src = snap * DraftCtxLayout::WINDOW * Self::RING_ROW_BYTES + at;
                rk.copy_range_from(off, k, src, bytes, stream)?;
                rv.copy_range_from(off, v, src, bytes, stream)?;
            }
        }
        Ok(())
    }

    /// Absorb `n` contiguous tap rows (`taps` is the target's `[rows, taps, hidden]` tap buffer; rows `row0..row0+n`) of
    /// `slot` at absolute positions `pos0..pos0+n` into the slot's context ring.
    #[allow(clippy::too_many_arguments)]
    pub fn absorb(
        &mut self,
        taps: &DevBuf,
        row0: usize,
        n: usize,
        slot: usize,
        pos0: usize,
        stream: &Stream,
    ) -> Result<(), ModelError> {
        self.absorb_batch(taps, &[Absorb { row0, n, slot, pos0 }], stream)
    }

    /// Absorb several sequences' committed tap rows in one pass: each item's rows `row0..row0+n` of `taps` land at
    /// positions `pos0..pos0+n` of its slot's ring. The rows share one fc, one K and one V projection per layer and one
    /// ring write per layer, instead of one set per sequence. Every row goes through row-invariant kernels, so the ring
    /// holds the same bits as absorbing the items one at a time.
    pub fn absorb_batch(&mut self, taps: &DevBuf, items: &[Absorb], stream: &Stream) -> Result<(), ModelError> {
        let total: usize = items.iter().map(|a| a.n).sum();
        if total == 0 {
            return Ok(());
        }
        if total > self.max_ctx_rows {
            return Err(ModelError::Request(format!(
                "absorbing {total} context rows at once; the draft takes {}",
                self.max_ctx_rows
            )));
        }
        if items.iter().any(|a| a.n >= DraftCtxLayout::DEFAULT_CAP) {
            return Err(ModelError::Request("one context write must span fewer positions than the ring holds".into()));
        }
        let mut pos: Vec<i32> = Vec::with_capacity(total);
        let mut slots: Vec<i32> = Vec::with_capacity(total);
        for a in items {
            pos.extend((0..a.n).map(|i| (a.pos0 + i) as i32));
            slots.extend(std::iter::repeat_n(a.slot as i32, a.n));
        }
        let n = total;
        // an absorb-only stretch (a serial request) never reaches a round's sync: bound what the stage keeps
        if self.stage.queued() >= 256 {
            self.stage.sync(stream)?;
        }
        self.stage.upload(&mut self.ctx_pos, &pos, stream)?;
        self.stage.upload(&mut self.ctx_slot, &slots, stream)?;
        let Draft { dims, fc, hidden_norm, layers, cache, fc_in, ctx, kproj, vproj, ctx_pos, ctx_slot, lin, .. } = self;
        let (fc_w, h) = (dims.fc_in(), dims.hidden);
        let mut at = 0usize;
        for a in items.iter().filter(|a| a.n > 0) {
            unsafe {
                // fc input rows are the tap rows themselves: the tapped layers' rows side by side
                glue::copy_rows(
                    (taps.ptr() as *const u16).add(a.row0 * fc_w) as *const c_void,
                    fc_w,
                    (fc_in.ptr() as *mut u16).add(at * fc_w) as *mut c_void,
                    fc_w,
                    a.n,
                    fc_w,
                    stream,
                )?;
            }
            at += a.n;
        }
        fc.forward(fc_in, ctx, n, lin, stream)?;
        unsafe {
            glue::rmsnorm(cp(ctx), cp(hidden_norm), mp(ctx), n, h, dims.eps, stream)?;
        }
        for (l, layer) in layers.iter().enumerate() {
            layer.qkv.forward(ctx, 1, &mut [&mut *kproj, &mut *vproj], n, lin, stream)?;
            cache.write_ctx(l, kproj, vproj, &layer.k_norm, ctx_pos, ctx_slot, n, dims.eps, dims.rope_theta, stream)?;
        }
        Ok(())
    }

    /// One draft round for `slot` at position `p` (the anchor token's position; the ring must hold the context
    /// through `p - 1`). Returns the 7 proposed tokens. `embed` and `head` are the TARGET's embedding table and
    /// lm_head (the draft has none of its own).
    #[allow(clippy::too_many_arguments)] // mirrors the kernel inputs one for one
    pub fn propose(
        &mut self,
        embed: &DevBuf,
        head: &Exl3Weights,
        vocab: usize,
        anchor: u32,
        slot: usize,
        p: usize,
        sampling: &gb_sampling::Sampling,
        stream: &Stream,
    ) -> Result<Vec<u32>, ModelError> {
        let want = self.block - 1;
        Ok(self
            .propose_batch(embed, head, vocab, &[DraftReq { anchor, slot, p, sampling, want, coverage: 1.0 }], stream)?
            .swap_remove(0))
    }

    /// One draft round for up to `max_seqs` sequences at once. Their blocks share every weight read (the draft's layers
    /// and the target's lm_head are read once for all of them) and one sync. The kernels are row-invariant and each
    /// block is independent, so a sequence's drafts are the same bits whether it runs alone or in any batch.
    pub fn propose_batch(
        &mut self,
        embed: &DevBuf,
        head: &Exl3Weights,
        vocab: usize,
        reqs: &[DraftReq],
        stream: &Stream,
    ) -> Result<Vec<Vec<u32>>, ModelError> {
        Ok(self.propose_batch_scored(embed, head, vocab, reqs, stream)?.into_iter().map(|p| p.tokens).collect())
    }

    /// Restrict the draft's vocabulary head to the target `head`'s columns of these token-id ranges (each `(first id, end id)`,
    /// widened to whole 128-column strips; overlapping or touching ranges merge; at most 4 remain and at least 64 strips must).
    /// The kept columns are bit-identical to the full head's, so the drafts are the full draft's whenever its choices lie inside
    /// the set. Costs one device-to-device copy of those strips of the head (3.6 ms plus 12 ms of `cudaMalloc` on a GB10; no host
    /// memory is involved) and their memory; returns the bytes added.
    pub fn restrict_vocab(&mut self, head: &Exl3Weights, token_ranges: &[(usize, usize)]) -> Result<usize, ModelError> {
        let t0 = std::time::Instant::now();
        let max_rows = self.max_seqs * (self.block - 1);
        let strips = vocab_strips(token_ranges, head.n)?;
        let (w, strip_times) = head.select_strips_timed(&strips)?;
        let ranges: Vec<(usize, usize)> = strips.iter().map(|&(lo, hi)| (lo * 128, (hi - lo) * 128)).collect();
        let t1 = std::time::Instant::now();
        let logits = DevBuf::zeroed(max_rows * w.n * 2)?;
        let ws = DevBuf::zeroed(linear_workspace_bytes(max_rows, self.dims.hidden, w.n, w.bits))?;
        let mut bytes = w.weight_bytes() + logits.bytes() + ws.bytes();
        // the top-k over the kept columns (its valid columns: `sub_valid`) may need more workspace than the full head's
        let top = glue::topk_valid_workspace_bytes(max_rows, w.n, SELECTOR_TOP_K);
        if top > self.top_ws.bytes() {
            self.top_ws = DevBuf::zeroed(top)?;
            bytes += top;
        }
        self.sub = Some(SubHead { w, ranges, logits, ws });
        self.restrict_times =
            Some(RestrictTimes { strips: strip_times, buffers: t1.elapsed().as_secs_f64(), total: t0.elapsed().as_secs_f64() });
        Ok(bytes)
    }

    /// Capture every round's selector lattice into [`Picked::lattice`] (what draft trees are searched over). Allocates the edge buffer now,
    /// so a server that calls it before its memory plan has the 114 KB counted (the plan measures what is free after it).
    pub fn enable_lattice(&mut self) -> Result<(), ModelError> {
        if self.edge.is_none() {
            self.edge = Some(DevBuf::zeroed(self.max_seqs * (self.block - 1) * SELECTOR_TOP_K * SELECTOR_TOP_K * 4)?);
        }
        self.capture_lattice = true;
        Ok(())
    }

    /// Where the last [`Self::restrict_vocab`] spent its time, or None when the draft has not been restricted.
    pub fn restrict_times(&self) -> Option<&RestrictTimes> {
        self.restrict_times.as_ref()
    }

    /// Back to the full vocabulary.
    pub fn use_full_vocab(&mut self) {
        self.sub = None;
        self.adaptive_min = None;
    }

    /// With a restricted vocabulary, run the restricted head only in rounds where every sequence has at least `min_in_set` of its
    /// last [`COVERAGE_WINDOW`] tokens inside the set, and the full head otherwise: the restricted head saves 2.5-4 ms a round on
    /// English and code (which live inside the set) and loses a quarter of the acceptance on text that does not (Chinese, Russian,
    /// Arabic), which the full head keeps. `None` runs the restricted head every round.
    pub fn set_adaptive(&mut self, min_in_set: Option<f32>) {
        self.adaptive_min = min_in_set;
    }

    /// Share of `ids` (a sequence's recent tokens) inside the restricted vocabulary; 1.0 when nothing is restricted.
    pub fn coverage(&self, ids: &[u32]) -> f32 {
        self.sub.as_ref().map_or(1.0, |s| coverage_of(&s.ranges, ids))
    }

    /// Token-id ranges the draft head covers, `(first id, length)`, or None for the full vocabulary.
    pub fn vocab_ranges(&self) -> Option<&[(usize, usize)]> {
        self.sub.as_ref().map(|s| s.ranges.as_slice())
    }

    /// [`Draft::propose_batch`] with the draft's own confidence in each pick (see [`Picked`]).
    pub fn propose_batch_scored(
        &mut self,
        embed: &DevBuf,
        head: &Exl3Weights,
        vocab: usize,
        reqs: &[DraftReq],
        stream: &Stream,
    ) -> Result<Vec<Picked>, ModelError> {
        let n = reqs.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if n > self.max_seqs {
            return Err(ModelError::Request(format!("a draft round of {n} sequences; the draft takes {}", self.max_seqs)));
        }
        if vocab != self.vocab {
            return Err(ModelError::Config(format!("the draft was built for a head of {} columns, not {vocab}", self.vocab)));
        }
        let block = self.block;
        let mut ids: Vec<i32> = Vec::with_capacity(n * block);
        let mut pos: Vec<i32> = Vec::with_capacity(n * block);
        for r in reqs {
            ids.push(r.anchor as i32);
            ids.extend(std::iter::repeat_n(self.dims.mask_token as i32, block - 1));
            pos.extend((0..block).map(|i| (r.p + i) as i32));
        }
        let anchors: Vec<i32> = reqs.iter().map(|r| r.anchor as i32).collect();
        let slots: Vec<i32> = reqs.iter().map(|r| r.slot as i32).collect();
        let ps: Vec<i32> = reqs.iter().map(|r| r.p as i32).collect();
        // the target samples draft i (absolute position p + 1 + i) with its keyed sampler; the walk uses the same draw
        let walks: Vec<WalkSampling> = reqs
            .iter()
            .map(|r| WalkSampling {
                seed: r.sampling.seed,
                first_position: r.p as u64 + 1,
                temperature: r.sampling.temperature,
                top_p: r.sampling.top_p,
                min_p: r.sampling.min_p,
                keyed: (r.sampling.temperature > 0.0) as u32,
                pad: 0,
            })
            .collect();
        // the first `want` picks of each sequence: its rows 1..=want go through the head and the walk, packed one sequence after
        // another (the block's own 8 rows are all needed by the layers, but row 0 and the rows past `want` are not by the head)
        let wants: Vec<usize> = reqs.iter().map(|r| r.want.clamp(1, block - 1)).collect();
        let mut row_off = Vec::with_capacity(n + 1);
        row_off.push(0i32);
        for &w in &wants {
            row_off.push(row_off.last().unwrap() + w as i32);
        }
        let st = &mut self.stage;
        st.upload(&mut self.ids, &ids, stream)?;
        st.upload(&mut self.pos, &pos, stream)?;
        st.upload(&mut self.anchor, &anchors, stream)?;
        st.upload(&mut self.slot_a, &slots, stream)?;
        st.upload(&mut self.p_a, &ps, stream)?;
        st.upload(&mut self.walk, &walks, stream)?;
        st.upload(&mut self.row_off, &row_off, stream)?;
        // the restricted head runs when there is one and, if it is adaptive, when every sequence is writing text it can reach
        let use_sub = self.sub.is_some() && self.adaptive_min.is_none_or(|min| reqs.iter().all(|r| r.coverage >= min));
        let capture = self.capture_lattice;
        if capture && self.edge.is_none() {
            self.edge = Some(DevBuf::zeroed(self.max_seqs * (self.block - 1) * SELECTOR_TOP_K * SELECTOR_TOP_K * 4)?);
        }
        let nd = row_off[n] as usize;
        self.head_rounds[use_sub as usize] += 1;
        if capture {
            self.round_launches(embed, head, vocab, &wants, use_sub, true, stream)?;
        } else {
            self.round_graphed(embed, head, vocab, &wants, use_sub, stream)?;
        }
        let Draft { stage, out_tokens, conf, margin, top_ids, top_vals, edge, .. } = self;
        stage.sync(stream)?;
        let mut t = vec![0i32; nd];
        out_tokens.download(&mut t)?;
        let mut c = vec![0f32; nd];
        conf.download(&mut c)?;
        let mut m = vec![0f32; nd];
        margin.download(&mut m)?;
        let lat = if capture {
            let mut ids = vec![0i32; nd * SELECTOR_TOP_K];
            top_ids.download(&mut ids)?;
            let mut vals = vec![0f32; nd * SELECTOR_TOP_K];
            top_vals.download(&mut vals)?;
            let mut ed = vec![0f32; nd * SELECTOR_TOP_K * SELECTOR_TOP_K];
            edge.as_ref().expect("allocated above").download(&mut ed)?;
            Some((ids, vals, ed))
        } else {
            None
        };
        Ok((0..n)
            .map(|s| {
                let (a, b) = (row_off[s] as usize, row_off[s + 1] as usize);
                let k = SELECTOR_TOP_K;
                let lattice = lat.as_ref().map(|(ids, vals, ed)| Lattice {
                    rows: b - a,
                    k,
                    ids: ids[a * k..b * k].iter().map(|&x| x as u32).collect(),
                    unary: vals[a * k..b * k].to_vec(),
                    edge: ed[a * k * k..b * k * k].to_vec(),
                    restricted_head: use_sub,
                });
                Picked {
                    tokens: t[a..b].iter().map(|&x| x as u32).collect(),
                    score: c[a..b].to_vec(),
                    margin: m[a..b].to_vec(),
                    lattice,
                }
            })
            .collect())
    }

    /// A round's launches through the round graphs: the first round of a shape runs eagerly, the second is captured (on the draft's
    /// own capture stream) and launched, every later one replays the graph. Only the host values that reach a launcher are in the
    /// key (sequences and each one's picks, the head); every other input of the round is read from the draft's device buffers, which
    /// the round's uploads rewrote in place. A shape whose capture fails runs eagerly from then on.
    fn round_graphed(
        &mut self,
        embed: &DevBuf,
        head: &Exl3Weights,
        vocab: usize,
        wants: &[usize],
        use_sub: bool,
        stream: &Stream,
    ) -> Result<(), ModelError> {
        let mut w = [0u8; MAX_DRAFT_SEQS];
        for (d, &s) in w.iter_mut().zip(wants) {
            *d = s as u8;
        }
        let key = RoundKey { n: wants.len(), wants: w, use_sub, embed: embed.ptr() as usize, head: head as *const _ as usize };
        match self.rounds.slots.get(&key) {
            Some(RoundSlot::Ready(g)) => return Ok(g.launch(stream)?),
            Some(RoundSlot::Eager) => return self.round_launches(embed, head, vocab, wants, use_sub, false, stream),
            Some(RoundSlot::Seen) => {}
            None => {
                if self.rounds.slots.len() < ROUND_GRAPHS {
                    self.rounds.slots.insert(key, RoundSlot::Seen);
                }
                return self.round_launches(embed, head, vocab, wants, use_sub, false, stream);
            }
        }
        let cap = match self.rounds.cap.take() {
            Some(c) => c,
            None => Stream::new()?,
        };
        let captured = Graph::capture(&cap, || self.round_launches(embed, head, vocab, wants, use_sub, false, &cap));
        self.rounds.cap = Some(cap);
        match captured {
            Ok(g) => {
                g.launch(stream)?;
                self.rounds.slots.insert(key, RoundSlot::Ready(g));
                Ok(())
            }
            Err(e) => {
                if let Some(fatal) = e.fatal() {
                    return Err(ModelError::Cuda(fatal.clone()));
                }
                eprintln!("rexl3 level=warn event=draft_round_capture_failed rows={} (this round shape runs eagerly)", wants.len());
                self.rounds.slots.insert(key, RoundSlot::Eager);
                self.round_launches(embed, head, vocab, wants, use_sub, false, stream)
            }
        }
    }

    /// The launches of a draft round on `stream` (the index arrays already uploaded): the block's layers, the head over each sequence's
    /// first `wants[s]` picks, the top-k, the selector walk, and with `edges` the lattice's edge terms.
    #[allow(clippy::too_many_arguments)]
    fn round_launches(
        &mut self,
        embed: &DevBuf,
        head: &Exl3Weights,
        vocab: usize,
        wants: &[usize],
        use_sub: bool,
        edges: bool,
        stream: &Stream,
    ) -> Result<(), ModelError> {
        let n = wants.len();
        let nd: usize = wants.iter().sum();
        let block = self.block;
        let Draft {
            dims,
            layers,
            norm,
            sel_proj,
            pred,
            succ,
            cache,
            lin,
            attn_ws,
            x,
            x2,
            h,
            y,
            o,
            dynv,
            qb,
            kb,
            vb,
            q2,
            k2,
            att,
            g,
            u,
            sw,
            sel,
            logits,
            top_ids,
            top_vals,
            top_ws,
            gate,
            ids: ids_d,
            pos: pos_d,
            anchor: anchor_d,
            slot_a,
            p_a,
            out_tokens,
            conf,
            margin,
            walk,
            row_off: row_off_d,
            sub,
            edge,
            ..
        } = self;
        let capture = edges;
        let rows = n * block;
        let (hd, eps, dw) = (dims.hidden, dims.eps, dims.dyn_width());
        let prep = dims.attn_prep();
        unsafe {
            glue::embed(cp(embed), cp(ids_d), mp(x), rows, hd, vocab, stream)?;
        }
        for (l, layer) in layers.iter().enumerate() {
            unsafe {
                glue::rmsnorm(cp(x), cp(&layer.input_norm), mp(h), rows, hd, eps, stream)?;
                glue::small_linear_f16w(cp(h), cp(&layer.attn_conv.proj), mp(dynv), rows, hd, dw, stream)?;
            }
            dflash2::dynconv(h, dynv, &layer.attn_conv.base, 0, n, block, hd, CONV_GROUP, None, y, stream)?;
            layer.qkv.forward(y, 0, &mut [&mut *qb, &mut *kb, &mut *vb], rows, lin, stream)?;
            unsafe {
                glue::attn_prep(
                    &prep,
                    cp(qb),
                    cp(kb),
                    cp(pos_d),
                    cp(&layer.q_norm),
                    cp(&layer.k_norm),
                    mp(q2),
                    std::ptr::null_mut(),
                    mp(k2),
                    rows,
                    stream,
                )?;
            }
            dflash2::attn(
                q2,
                k2,
                vb,
                cache.k(l),
                cache.v(l),
                slot_a,
                p_a,
                n,
                Q_HEADS,
                KV_HEADS,
                DraftCtxLayout::DEFAULT_CAP,
                DraftCtxLayout::WINDOW,
                block,
                attn_ws,
                att,
                stream,
            )?;
            layer.o.forward(att, o, rows, lin, stream)?;
            // x' = x + conv_finish(o), written to the other buffer, then swapped in
            dflash2::dynconv(o, dynv, &layer.attn_conv.base, 1, n, block, hd, CONV_GROUP, Some(x), x2, stream)?;
            std::mem::swap(x, x2);
            unsafe {
                glue::rmsnorm(cp(x), cp(&layer.post_norm), mp(h), rows, hd, eps, stream)?;
                glue::small_linear_f16w(cp(h), cp(&layer.mlp_conv.proj), mp(dynv), rows, hd, dw, stream)?;
            }
            dflash2::dynconv(h, dynv, &layer.mlp_conv.base, 0, n, block, hd, CONV_GROUP, None, y, stream)?;
            layer.gu.forward(y, 0, &mut [&mut *g, &mut *u], rows, lin, stream)?;
            unsafe {
                glue::swiglu(cp(g), cp(u), mp(sw), rows * dims.intermediate, stream)?;
            }
            layer.down.forward(sw, o, rows, lin, stream)?;
            dflash2::dynconv(o, dynv, &layer.mlp_conv.base, 1, n, block, hd, CONV_GROUP, Some(x), x2, stream)?;
            std::mem::swap(x, x2);
        }
        unsafe {
            glue::rmsnorm(cp(x), cp(norm), mp(h), rows, hd, eps, stream)?;
            // rows 1..=want of each sequence's block feed the target's head and the selector, packed
            let mut at = 0usize;
            for (s, &want) in wants.iter().enumerate() {
                glue::copy_rows(
                    (h.ptr() as *const u16).add((s * block + 1) * hd) as *const c_void,
                    hd,
                    (sel.ptr() as *mut u16).add(at * hd) as *mut c_void,
                    hd,
                    want,
                    hd,
                    stream,
                )?;
                at += want;
            }
        }
        // a head whose last columns are padding (GLM's 24): the top-k never reads them (`topk_valid`); one without keeps `topk`
        let topk = |logits: &DevBuf, cols: usize, valid: usize, top_ids: &mut DevBuf, top_vals: &mut DevBuf, top_ws: &mut DevBuf| unsafe {
            if valid < cols {
                glue::topk_valid(
                    cp(logits),
                    nd,
                    cols,
                    valid,
                    SELECTOR_TOP_K,
                    mp(top_ids),
                    mp(top_vals),
                    mp(top_ws),
                    top_ws.bytes(),
                    stream,
                )
            } else {
                glue::topk(cp(logits), nd, cols, SELECTOR_TOP_K, mp(top_ids), mp(top_vals), mp(top_ws), top_ws.bytes(), stream)
            }
        };
        let pad = vocab - dims.valid_vocab;
        // the head over every column, or over the restricted set (whose top-k ids are then mapped back to token ids)
        match sub.as_mut().filter(|_| use_sub) {
            None => {
                head.forward(sel, logits, nd, Dtype::BF16, Dtype::BF16, &mut lin.ws, stream)?;
                topk(logits, vocab, dims.valid_vocab, top_ids, top_vals, top_ws)?;
            }
            Some(sub) => {
                sub.w.forward(sel, &mut sub.logits, nd, Dtype::BF16, Dtype::BF16, &mut sub.ws, stream)?;
                // the set's last columns are the head's padding when it keeps the last strip
                let keeps_end = sub.ranges.last().is_some_and(|&(first, len)| first + len >= vocab);
                let valid = if keeps_end { sub.w.n - pad } else { sub.w.n };
                topk(&sub.logits, sub.w.n, valid, top_ids, top_vals, top_ws)?;
                dflash2::remap_ids(top_ids, nd * SELECTOR_TOP_K, &sub.ranges, stream)?;
            }
        }
        unsafe {
            glue::small_linear_f16w(cp(sel), cp(sel_proj), mp(gate), nd, hd, SELECTOR_RANK, stream)?;
        }
        dflash2::selector_walk_ragged(
            anchor_d,
            top_ids,
            top_vals,
            gate,
            GateDtype::Bf16,
            pred,
            succ,
            n,
            nd,
            SELECTOR_TOP_K,
            SELECTOR_RANK,
            out_tokens,
            Some(walk),
            row_off_d,
            Some(conf),
            Some(margin),
            stream,
        )?;
        if capture {
            let edge = edge.as_mut().expect("allocated above");
            dflash2::selector_edges(
                anchor_d,
                top_ids,
                gate,
                GateDtype::Bf16,
                pred,
                succ,
                n,
                nd,
                SELECTOR_TOP_K,
                SELECTOR_RANK,
                row_off_d,
                edge,
                stream,
            )?;
        }
        Ok(())
    }
}

/// The 128-column strips (half-open ranges, ascending, merged) that cover these token-id ranges (`(first id, end id)`) of a head with
/// `n` columns, widened outward to whole strips. Fails on an empty or out-of-range request, more than 4 resulting ranges, or fewer
/// than 64 strips (the decode kernel's K split depends on the strip count below that, so the kept columns would not be bit-identical to
/// the full head's).
pub fn vocab_strips(token_ranges: &[(usize, usize)], n: usize) -> Result<Vec<(usize, usize)>, ModelError> {
    let mut v: Vec<(usize, usize)> = Vec::new();
    for &(lo, hi) in token_ranges {
        if lo >= hi || hi > n {
            return Err(ModelError::Config(format!(
                "draft vocabulary range {lo}..{hi} is empty or past the {n} columns of the head"
            )));
        }
        v.push((lo / 128, hi.div_ceil(128)));
    }
    v.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (lo, hi) in v {
        match merged.last_mut() {
            Some(last) if lo <= last.1 => last.1 = last.1.max(hi),
            _ => merged.push((lo, hi)),
        }
    }
    let strips: usize = merged.iter().map(|&(lo, hi)| hi - lo).sum();
    if merged.is_empty() || merged.len() > 4 || strips < 64 {
        return Err(ModelError::Config(format!(
            "draft vocabulary needs 1 to 4 ranges covering at least 64 strips of 128 columns, got {} ranges and {strips} strips",
            merged.len()
        )));
    }
    Ok(merged)
}

/// The draft vocabulary TensorFold uses for this model family: token ids `[0, 98304)` and the special tokens at the top
/// (`[248032, 248320)`, the strips from 247936).
pub const TF_DRAFT_VOCAB: [(usize, usize); 2] = [(0, 98304), (248032, 248320)];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_ranges_become_merged_strips() {
        // TensorFold's set: 768 strips from 0, and the strip holding 248032 up to the end (1937..1940)
        assert_eq!(vocab_strips(&TF_DRAFT_VOCAB, 248320).unwrap(), vec![(0, 768), (1937, 1940)]);
        // overlapping and touching ranges merge; a range is widened outward to whole strips
        assert_eq!(vocab_strips(&[(0, 100 * 128 + 1), (101 * 128, 200 * 128)], 248320).unwrap(), vec![(0, 200)]);
        assert_eq!(vocab_strips(&[(100, 9000), (50, 70)], 248320).unwrap(), vec![(0, 71)]);
    }

    #[test]
    fn coverage_is_the_share_of_tokens_the_set_can_reach() {
        let set = [(0usize, 100usize), (1000, 10)];
        assert_eq!(coverage_of(&set, &[]), 1.0, "nothing written, nothing out of reach");
        assert_eq!(coverage_of(&set, &[5, 99, 1000, 1009]), 1.0);
        // the first id past each range is outside it
        assert_eq!(coverage_of(&set, &[100, 1010]), 0.0);
        assert!((coverage_of(&set, &[5, 150, 1005]) - 2.0 / 3.0).abs() < 1e-6);
        // 63 of 64 inside is above the adaptive threshold, 62 of 64 is below it
        let mut ids = vec![5u32; 64];
        ids[0] = 5000;
        assert!(coverage_of(&set, &ids) >= 0.98);
        ids[1] = 5000;
        assert!(coverage_of(&set, &ids) < 0.98);
    }

    #[test]
    fn a_vocabulary_the_kernel_could_not_keep_exact_is_refused() {
        assert!(vocab_strips(&[], 248320).is_err());
        assert!(vocab_strips(&[(0, 128 * 63)], 248320).is_err(), "63 strips is below the 64 that keep the K split");
        assert!(vocab_strips(&[(0, 128 * 64)], 248320).is_ok());
        assert!(vocab_strips(&[(10, 10)], 248320).is_err(), "empty");
        assert!(vocab_strips(&[(0, 248321)], 248320).is_err(), "past the head");
        let five: Vec<(usize, usize)> = (0..5).map(|i| (i * 20_000, i * 20_000 + 12_800)).collect();
        assert!(vocab_strips(&five, 248320).is_err(), "more than four ranges");
    }
}
