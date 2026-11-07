//! GLM-5.3-Flash on the family-neutral lockstep seam (`crate::lockstep`, `docs/CLUSTER_FAMILY_PORTING.md`): the family's call and fact
//! types, its runtime (a [`GlmRank`]: one rank's model, `model/tp.rs`), and the boot gate's pack identity. `docs/GLM_TP.md` is the design:
//! the map of every call that touches the routed experts, the determinism argument and what each of MTP, the prefix cache and vision
//! still needs.
//!
//! **Scope of this milestone.** Two ranks ([`MAX_TP_RANKS`]), each holding the WHOLE pack and owning half of the routed experts
//! (`ExpertOwnership::range`, GLM: 0..144 | 144..288); greedy or sampled text (and images on rank 0) with `--prefix-cache off`. With `--spec none`
//! the seven core calls are the whole call stream; with `--spec mtp|auto` the NextN drafter adds the family calls of `tp_mtp` (kinds 32
//! and 33, lane L12); an image prompt on rank 0 (its vision tower) adds `PrefillImage` (kind 40, `tp_vision`, lane L13), and the two
//! combine (`docs/GLM_TP.md` 11).
//!
//! **The prefix cache** (`docs/GLM_TP.md` 4.2, Nemotron's protocol 4 made for GLM): rank 0's engine decides every hit; each rank snapshots and
//! restores ITS OWN state, between two chunks only (a rank of a KDA head split runs each chunk's KDA whole), and nothing but these calls
//! travels: [`GlmCall::EnableSnapshots`] (kind 48, the pool at boot), [`GlmCall::Snapshot`] (49), [`GlmCall::Restore`] (50) and
//! [`GlmCall::Release`] (51). A snapshot's handle is keyed by the sequence number of the call that made it ([`apply_prefix`]), never by a
//! rank-local slot or page id. `Snapshot`, `Restore` and `Release` are cache calls (`lockstep::TpLeader::forward_cache_call`): each rank
//! answers a snapshot with the content digest of what it captured ([`take_snapshot`], `GlmModel::seal_snapshot`) and a restore with the
//! digest of the state it restored (`GlmModel::prefix_state_digest`), rank 0 uses a restore only when every rank's digest equals its own
//! at snapshot time (`lockstep::SnapshotBook`), and anything else falls back to a cold prefill on every rank. The arena is a fact of the
//! Hello ([`GlmFacts::cache_request`] and the three after it).
//!
//! **Determinism** (`NEMOTRON_TP.md` 5.4, made for GLM in `docs/GLM_TP.md` section 2): every call's result is a pure function of the weights
//! and the call's arguments. `begin_slot` / `end_slot` / `commit` / `flush` move host books (the page table, the pending keep counts) and
//! per-slot device state by the call's arguments only; `prefill_chunk` and `window_candidates` run the forward over the batch the call
//! carries (all seven `RowBatch` fields), with the same lazy flush on every rank. A rank runs eagerly on one stream (no graphs, no layer
//! branches: `model/tp.rs`), so its collectives are issued in plan order, one per MoE layer, with a count that is a function of the batch.
//! No clock, RNG or environment read reaches the arithmetic or the stream except those in [`GlmFacts`] and the geometry of the gate.

use super::alloc::{TOKENIZER_VOCAB, WINDOW_ROWS};
use super::model::{DraftAsk, Drafts, GlmModel, ModelOptions, TapBook};
use super::profile_slices::Kind;
use super::tp_mtp::{self, GlmMtpRuntime};
use super::tp_part::{self, PartRuntime, PrefillPart, K_PREFILL_PART};
use super::tp_vision::{self, ImageRuntime, PrefillImage, K_PREFILL_IMAGE};
use crate::error::ModelError;
use crate::forward::Candidates;
use crate::lockstep::{self, fact_differs, BootGate, CallReply, Dec, Enc, FamilyCall, LockstepRuntime, WireError};
use crate::rows::RowBatch;
use gb_cuda::moe::ExpertOwnership;
use gb_engine::RowRequest;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::Path;

/// The largest group this family's runtime, plan and tests cover: two ranks (the configuration lane L2b proves). Not a measurement.
pub const MAX_TP_RANKS: usize = 2;

/// The frame kinds of the prefix cache's calls (`docs/GLM_TP.md` 4.2): a block of their own in the family's range, clear of the draft
/// calls' (32, 33) and the image chunk's (40).
pub const K_ENABLE_SNAPSHOTS: u8 = 48;
pub const K_SNAPSHOT: u8 = 49;
pub const K_RESTORE: u8 = 50;
pub const K_RELEASE: u8 = 51;

/// The family's own commands (kinds `32..=127`): the NextN draft (`tp_mtp`, lane L12: [`GlmCall::MtpDraft`] kind 32,
/// [`GlmCall::MtpDisable`] kind 33), an image chunk ([`PrefillImage`], kind 40: `tp_vision.rs`, lane L13) and the prefix cache's four
/// (kinds 48 to 51, [`apply_prefix`]). A snapshot is taken between two chunks only: an in-chunk cut is never a call. Not `Eq`: the draft
/// call carries sampling settings (`f32`s).
#[derive(Clone, Debug, PartialEq)]
pub enum GlmCall {
    /// `GlmModel::mtp_draft_keyed(asks, keyed)`: rank 0's asks (slot, the sequence's tokens, the depth its arbiter chose) and every ask's
    /// sampler (the keyed-noise seed and settings, or none). Acked with `tp_mtp::drafts_digest` of the drafts.
    MtpDraft { asks: Vec<tp_mtp::AskWire>, keyed: Vec<Option<gb_sampling::Sampling>> },
    /// `GlmModel::mtp_disable_slot(slot)`.
    MtpDisable { slot: u32 },
    /// `GlmModel::dflash_run(reqs)`: a round of the DFlash2 drafter split over the group (`dflash::share_of`), with rank 0's requests
    /// (`GlmModel::dflash_reqs`: each sequence's anchor, position, drafts wanted, restricted-head coverage and sampler). Both ranks meet in
    /// the round's swaps. Acked with `tp_mtp::picks_digest` of the picks, which both ranks hold alike.
    DflashDraft { reqs: Vec<super::model::DflashReq> },
    /// `GlmModel::dflash_prompt_end(slot, end, cut)` on a split drafter's follower (rank 0's word, so both halves absorb the same rows and
    /// keep the same snapshot windows).
    DflashPromptEnd { slot: u32, end: Option<u32>, cut: Option<u32> },
    /// A prefill chunk with the f32 feature rows of the images inside it (rank 0's tower made them; every rank splices these bytes, and
    /// the NextN drafter's prefill of the chunk reads the same rows: `docs/GLM_TP.md` 11).
    PrefillImage(PrefillImage),
    /// A part of a prompt forward of one or several prompts (`tp_part`: `GlmModel::prefill_rows` / `prefill_resume`, kind 41).
    PrefillPart(PrefillPart),
    /// `GlmModel::enable_prefix_cache(slots)`: the snapshot pool on every rank, `slots` of rank 0's plan (a boot-time call: rank 0 settles
    /// it before anything is served). Acked with the bytes of one slot of the pool the rank built (`prefix_snapshot_bytes`), so a rank
    /// whose pool holds other state loses the group's boot naming the call.
    EnableSnapshots { slots: u32 },
    /// `alloc_snapshot`, `prefix_snapshot(slot, snap, len)`, `retain_prefix_pages(slot, len)` and `seal_snapshot` on the rank's OWN state,
    /// all inside the call ([`take_snapshot`]: the retain moves the KV table's references, so it is part of the stream): the state after the
    /// chunk that ended at `len`. A follower keeps its handle by this call's sequence number. A cache call: answered with
    /// `lockstep::taken_digest` of the snapshot's content digest, or `lockstep::SNAPSHOT_SKIPPED` when the rank's pool had no free slot.
    Snapshot { slot: u32, len: u32 },
    /// `restore_prefix(slot, snap, pages, len)` of the snapshot the call numbered `of` made (each rank's own handle of it). A cache call:
    /// answered with `lockstep::taken_digest` of the restored state's digest.
    Restore { slot: u32, len: u32, of: u64 },
    /// `release_prefix(snap, pages)` of the snapshot the call numbered `of` made. A follower never evicts on its own: rank 0's engine
    /// decides, and this call carries the decision. A rank that holds no such snapshot (one it skipped) answers Ok.
    Release { of: u64 },
}

impl GlmCall {
    /// One of the prefix cache's calls ([`apply_prefix`] applies them, with the call's sequence number).
    pub fn is_prefix(&self) -> bool {
        matches!(
            self,
            GlmCall::EnableSnapshots { .. } | GlmCall::Snapshot { .. } | GlmCall::Restore { .. } | GlmCall::Release { .. }
        )
    }
}

/// What two GLM ranks must agree on beyond the gate's geometry (pack, layers, slots, KV pages, context, prefill rows, build, group
/// size, NCCL settings): every switch that shapes the call stream or the arithmetic of a rank.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GlmFacts {
    /// CUDA graphs of the decode windows, the exchange captured inside them (`model::tp::rank_graphs`: `REXL3_GLM_GRAPHS` and
    /// `REXL3_GLM_TP_GRAPHS` with a capturable communicator; lane E Phase 1). Off: every window eager on one stream.
    pub graphs: bool,
    /// With graphs: the layer parts run as side branches (`REXL3_GLM_OVERLAP`, `graphs::PART_*`). On the wire only with graphs.
    pub overlap: u8,
    /// With graphs: shapes captured at their first sighting (`REXL3_GLM_GRAPH_AT=0`). On the wire only with graphs.
    pub graph_at_first: bool,
    /// With graphs and the NextN layer: the draft calls run through graphs of their own (`REXL3_GLM_DRAFT_GRAPHS`, on by default at TP;
    /// lane T 2026-10-18). Bit 2 of the Hello's first byte, set only with graphs on (a Hello without it keeps its bytes).
    pub draft_graphs: bool,
    /// The adaptive prefill slice (lane E Phase 1): rank 0 slices a prompt at this many rows while a sequence is decoding and at the
    /// gate's prefill rows otherwise (`REXL3_GLM_ADAPTIVE_PREFILL`, `tp_boot::adaptive_small`); 0 = one size (the gate's). The slices
    /// are row-invariant, so it changes no bit; a fact so that the ranks of a group are configured alike. On the wire only when set.
    pub prefill_small: u32,
    /// The NextN layer is loaded (and would draft).
    pub mtp: bool,
    /// The deepest draft a call may ask for (`alloc::MTP_MAX_DEPTH`: the chain's buffers and row index sets), 0 without the NextN layer.
    /// On the wire only when `mtp` is on (the Hello of a group without the layer keeps its bytes).
    pub mtp_depth: u32,
    /// Rows of the decode window the model's records are built for.
    pub window_rows: u32,
    /// The draft head's vocabulary (`REXL3_MTP_VOCAB`, effective; `none` without the NextN layer).
    pub mtp_vocab: String,
    /// The digest of the expert ownership table (owner rank of every global expert; [`ownership_digest`]), or of the width split
    /// ([`width_digest`]: every rank holds every expert at its share of the width).
    pub ownership: u64,
    /// The collective policy (`REXL3_TP_COLLECTIVE`, `tp_comm::collective_fingerprint`): ring on one rank and one-shot on another wait
    /// for collectives the other never issues.
    pub collective: String,
    /// This rank holds a vision tower (`docs/GLM_TP.md` 10): a LEADER-ONLY fact. Rank 0 may hold one (it encodes every image and sends
    /// the rows); a follower must not (it never encodes, so a tower there is memory the plan of the group never priced and a sign of a
    /// configuration that expects it to). Appended last to the Hello (2026-10-14, lane L13).
    pub vision: bool,
    /// The head split (lane E Phase 2, `model::tp::TP_SPLIT_ENV`): the bits of what the ranks split besides the experts (`kda`, `lm`).
    /// Flagged by bit 3 of the first byte and appended after the other fields, only when set (a Hello without a split keeps its bytes).
    pub split: u8,
    /// The prefill arithmetic contract (`REXL3_GLM_PREFILL_CONTRACT`, `contract::prefill_fingerprint`: `contract=p1`, or P2 with its dense
    /// kernel's version): two ranks on different contracts would leave different bits in the replicated layers of a prompt chunk. The
    /// LAST field of the Hello, after the split byte (MAIN's layout, 2026-10-19).
    pub prefill: String,
    /// What the prefix cache was asked for (`--prefix-cache`, `--prefix-cache-mib`): `off`, `auto`, `on` or `mib=N`
    /// (`nemotron_h_omni::tp_runtime::cache_request_label`, the text both families compare). Two ranks given different flags plan
    /// different arenas: refused at the Hello, before any weight is read. With the three after it, flagged by bit 4 of the first byte and
    /// appended after the prefill contract only when one of them is not the cache-off default (a Hello with the cache off keeps its bytes).
    pub cache_request: String,
    /// Snapshot slots the rank's plan carved (`CacheBudget::snapshot_slots`; 0: no cache). Rank 0's count is the group's (it forwards
    /// `EnableSnapshots` of it, and every follower builds exactly that pool); a follower must be able to hold at least as many.
    pub cache_slots: u64,
    /// Bytes of one snapshot slot (`CacheBudget::snapshot_bytes`: `prefix::snapshot_bytes` of the rank's shapes, its own KDA heads under a
    /// split): equal, or the ranks' pools hold different state.
    pub cache_slot_bytes: u64,
    /// KV pages the cache may hold on top of admission's (`CacheBudget::page_blocks`); a follower must hold at least rank 0's.
    pub cache_pages: u64,
    /// The MLA latent cache's format (`--kv-dtype`, `gb_cuda::mla::KV_ENV`: `fp16` or `fp8`): two ranks on different formats attend over
    /// different values in the replicated MLA layers. Flagged by bit 5 of the first byte and appended last only when it is not `fp16` (a
    /// Hello of an fp16 group keeps its bytes).
    pub kv: String,
    /// The DFlash2 drafter's tapped layers (`--draft DIR`: its `target_layer_ids`, comma-joined; `none` without a drafter), which both
    /// ranks read from their options: under the sequence-parallel residual of a P2 prompt chunk both ranks relay the taps of the
    /// follower's rows to rank 0 (`model::dflash::TapRelay`), so the ranks must agree on them. Flagged by bit 6 of the first byte and
    /// appended last only when set (a Hello of a group without a drafter keeps its bytes).
    pub taps: String,
    /// The DFlash2 drafter split over the group (`dflash::split_fact`: `off`, or `on:` and a digest of the draft pack's config): both ranks
    /// load a half and run every round together, so both must split, and split the same pack. Flagged by bit 7 of the first byte and
    /// appended last only when on (a Hello without the split keeps its bytes).
    pub draft_split: String,
}

impl GlmFacts {
    /// The facts of a rank of a group of `ranks` that owns the experts by [`ExpertOwnership::range`], without the NextN layer.
    pub fn local(experts: usize, ranks: usize) -> GlmFacts {
        GlmFacts {
            graphs: false,
            overlap: 0,
            graph_at_first: false,
            draft_graphs: false,
            prefill_small: 0,
            mtp: false,
            mtp_depth: 0,
            window_rows: WINDOW_ROWS as u32,
            mtp_vocab: "none".into(),
            ownership: ownership_digest(&range_table(experts, ranks)),
            collective: super::tp_boot::collective_fact(),
            vision: false,
            split: 0,
            prefill: super::contract::prefill_fingerprint(),
            // no prefix cache until `with_cache` says what the plan carved
            cache_request: "off".into(),
            cache_slots: 0,
            cache_slot_bytes: 0,
            cache_pages: 0,
            kv: gb_cuda::mla::configured_kv().unwrap_or_default().name().into(),
            taps: "none".into(),
            draft_split: "off".into(),
        }
    }

    /// These facts with the prefix cache the plan carved: `request` is `nemotron_h_omni::tp_runtime::cache_request_label` of the option
    /// the rank was started with.
    pub fn with_cache(mut self, request: String, budget: &gb_engine::CacheBudget) -> GlmFacts {
        self.cache_request = request;
        self.cache_slots = budget.snapshot_slots as u64;
        self.cache_slot_bytes = budget.snapshot_bytes;
        self.cache_pages = budget.page_blocks as u64;
        self
    }

    /// Whether the cache's facts travel (bit 4 of the first byte): any of them is not the cache-off default, so the Hello round-trips.
    pub fn cache_on_wire(&self) -> bool {
        self.cache_request != "off" || self.cache_slots != 0 || self.cache_slot_bytes != 0 || self.cache_pages != 0
    }

    /// These facts with the NextN layer loaded: the draft depth cap and the draft head's vocabulary (`tp_mtp::mtp_vocab_fact`).
    pub fn with_mtp(mut self, depth_cap: usize, vocab: String) -> GlmFacts {
        self.mtp = true;
        self.mtp_depth = depth_cap as u32;
        self.mtp_vocab = vocab;
        self
    }

    /// These facts with the vision tower held (`true` only on rank 0, with a pack that has one, serving images).
    pub fn with_vision(mut self, vision: bool) -> GlmFacts {
        self.vision = vision;
        self
    }

    /// These facts with the adaptive prefill slice's small size (0, or the gate's own rows: one size).
    pub fn with_prefill_small(mut self, small: usize, chunk: usize) -> GlmFacts {
        self.prefill_small = if small == 0 || small >= chunk { 0 } else { small as u32 };
        self
    }

    /// These facts with the rank's graph settings: `graphs` on, the branch parts and the capture policy (ignored when `graphs` is off:
    /// they change nothing then, and are neither sent nor compared).
    pub fn with_graphs(mut self, graphs: bool, overlap: u8, at_first: bool) -> GlmFacts {
        self.graphs = graphs;
        (self.overlap, self.graph_at_first) = if graphs { (overlap, at_first) } else { (0, false) };
        if !graphs {
            self.draft_graphs = false;
        }
        self
    }

    /// These facts with the NextN draft calls' graphs (meaningful, sent and compared only with graphs on and the NextN layer loaded).
    pub fn with_draft_graphs(mut self, on: bool) -> GlmFacts {
        self.draft_graphs = on && self.graphs && self.mtp;
        self
    }

    /// These facts with the rank's head split (`model::tp::split_from_env`).
    pub fn with_split(mut self, split: u8) -> GlmFacts {
        self.split = split;
        self
    }

    /// These facts for a rank of the expert width split (`tp_boot::ExpertSplit::Width`): the ownership fact is the split's digest.
    pub fn with_width_split(mut self, experts: usize, ranks: usize) -> GlmFacts {
        self.ownership = width_digest(experts, ranks);
        self
    }

    /// These facts with the DFlash2 drafter's tapped layers (`dflash::tap_layers`; none: `none`).
    pub fn with_taps(mut self, layers: &[usize]) -> GlmFacts {
        self.taps = if layers.is_empty() {
            "none".into()
        } else {
            layers.iter().map(|l| l.to_string()).collect::<Vec<_>>().join(",")
        };
        self
    }

    /// These facts with the DFlash2 drafter's split (`dflash::split_fact`).
    pub fn with_draft_split(mut self, fact: String) -> GlmFacts {
        self.draft_split = fact;
        self
    }
}

/// FNV-1a 64 of the expert width split of `experts` experts over `ranks` ranks (every rank every expert, at its share of the width):
/// never a range table's digest.
pub fn width_digest(experts: usize, ranks: usize) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |b: u8| h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
    for b in b"glm-width-v1".iter().copied().chain((experts as u64).to_le_bytes()).chain((ranks as u64).to_le_bytes()) {
        eat(b);
    }
    h
}

/// The owner rank of every global expert under [`ExpertOwnership::range`] at `ranks` ranks.
pub fn range_table(experts: usize, ranks: usize) -> Vec<u32> {
    let mut t = vec![0u32; experts];
    for r in 0..ranks.max(1) {
        if let Ok(o) = ExpertOwnership::range(experts, r, ranks.max(1)) {
            for &g in o.owned() {
                t[g as usize] = r as u32;
            }
        }
    }
    t
}

/// FNV-1a 64 of an ownership table (its length, then each owner as four little-endian bytes).
pub fn ownership_digest(table: &[u32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut eat = |b: u8| h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3);
    for b in b"glm-ownership-v1" {
        eat(*b);
    }
    for b in (table.len() as u64).to_le_bytes() {
        eat(b);
    }
    for &r in table {
        for b in r.to_le_bytes() {
            eat(b);
        }
    }
    h
}

impl FamilyCall for GlmCall {
    type Facts = GlmFacts;

    fn kind(&self) -> u8 {
        match self {
            GlmCall::MtpDraft { .. } => tp_mtp::K_MTP_DRAFT,
            GlmCall::MtpDisable { .. } => tp_mtp::K_MTP_DISABLE,
            GlmCall::DflashDraft { .. } => tp_mtp::K_DFLASH_DRAFT,
            GlmCall::DflashPromptEnd { .. } => tp_mtp::K_DFLASH_PROMPT_END,
            GlmCall::PrefillImage(_) => K_PREFILL_IMAGE,
            GlmCall::PrefillPart(_) => K_PREFILL_PART,
            GlmCall::EnableSnapshots { .. } => K_ENABLE_SNAPSHOTS,
            GlmCall::Snapshot { .. } => K_SNAPSHOT,
            GlmCall::Restore { .. } => K_RESTORE,
            GlmCall::Release { .. } => K_RELEASE,
        }
    }
    fn name(&self) -> &'static str {
        match self {
            GlmCall::MtpDraft { .. } => "mtp_draft",
            GlmCall::MtpDisable { .. } => "mtp_disable_slot",
            GlmCall::DflashDraft { .. } => "dflash_draft",
            GlmCall::DflashPromptEnd { .. } => "dflash_prompt_end",
            GlmCall::PrefillImage(_) => "prefill_image",
            GlmCall::PrefillPart(_) => "prefill_part",
            GlmCall::EnableSnapshots { .. } => "enable_snapshots",
            GlmCall::Snapshot { .. } => "prefix_snapshot",
            GlmCall::Restore { .. } => "restore_prefix",
            GlmCall::Release { .. } => "release_prefix",
        }
    }
    fn encode(&self, e: &mut Enc) {
        match self {
            GlmCall::MtpDraft { asks, keyed } => tp_mtp::encode_draft(e, asks, keyed),
            GlmCall::MtpDisable { slot } => e.u32(*slot),
            GlmCall::DflashDraft { reqs } => tp_mtp::encode_dflash(e, reqs),
            GlmCall::DflashPromptEnd { slot, end, cut } => {
                e.u32(*slot);
                e.opt_u32(*end);
                e.opt_u32(*cut);
            }
            GlmCall::PrefillImage(p) => p.encode(e),
            GlmCall::PrefillPart(p) => p.encode(e),
            GlmCall::EnableSnapshots { slots } => e.u32(*slots),
            GlmCall::Snapshot { slot, len } => {
                e.u32(*slot);
                e.u32(*len);
            }
            GlmCall::Restore { slot, len, of } => {
                e.u32(*slot);
                e.u32(*len);
                e.u64(*of);
            }
            GlmCall::Release { of } => e.u64(*of),
        }
    }
    fn decode(kind: u8, d: &mut Dec<'_>) -> Result<Option<Self>, WireError> {
        Ok(match kind {
            tp_mtp::K_MTP_DRAFT => Some(tp_mtp::decode_draft(d)?),
            tp_mtp::K_MTP_DISABLE => Some(GlmCall::MtpDisable { slot: d.u32("glm disable slot")? }),
            tp_mtp::K_DFLASH_DRAFT => Some(tp_mtp::decode_dflash(d)?),
            tp_mtp::K_DFLASH_PROMPT_END => Some(GlmCall::DflashPromptEnd {
                slot: d.u32("glm dflash prompt slot")?,
                end: d.opt_u32("glm dflash prompt end")?,
                cut: d.opt_u32("glm dflash prompt cut")?,
            }),
            K_PREFILL_IMAGE => Some(GlmCall::PrefillImage(PrefillImage::decode(d)?)),
            K_PREFILL_PART => Some(GlmCall::PrefillPart(PrefillPart::decode(d)?)),
            K_ENABLE_SNAPSHOTS => Some(GlmCall::EnableSnapshots { slots: d.u32("glm snapshot slots")? }),
            K_SNAPSHOT => Some(GlmCall::Snapshot { slot: d.u32("glm snapshot slot")?, len: d.u32("glm snapshot length")? }),
            K_RESTORE => Some(GlmCall::Restore {
                slot: d.u32("glm restore slot")?,
                len: d.u32("glm restore length")?,
                of: d.u64("glm restore handle")?,
            }),
            K_RELEASE => Some(GlmCall::Release { of: d.u64("glm release handle")? }),
            _ => None,
        })
    }
    fn counts_as_prefill(&self) -> bool {
        matches!(self, GlmCall::PrefillImage(_) | GlmCall::PrefillPart(_))
    }

    fn encode_facts(f: &GlmFacts, e: &mut Enc) {
        // the first byte (MAIN's layout): bit 0 graphs (the byte the earlier builds wrote as a boolean), bit 1 the adaptive slice (lane E),
        // bit 2 the NextN draft calls' graphs (lane T), only with graphs on, bit 3 a head split (lane E), bit 4 the prefix cache's arena,
        // bit 5 a latent cache that is not fp16, bit 6 the DFlash2 drafter's taps, bit 7 the drafter's split
        e.u8(u8::from(f.graphs)
            | (u8::from(f.prefill_small != 0) << 1)
            | (u8::from(f.graphs && f.draft_graphs) << 2)
            | (u8::from(f.split != 0) << 3)
            | (u8::from(f.cache_on_wire()) << 4)
            | (u8::from(f.kv != "fp16") << 5)
            | (u8::from(f.taps != "none") << 6)
            | (u8::from(f.draft_split != "off") << 7));
        e.bool(f.mtp);
        if f.mtp {
            e.u32(f.mtp_depth);
        }
        e.u32(f.window_rows);
        e.str(&f.mtp_vocab);
        e.u64(f.ownership);
        e.str(&f.collective);
        e.bool(f.vision);
        // lane E (2026-10-18): the graph settings after the vision fact, only with graphs on (a Hello with graphs off keeps its bytes)
        if f.graphs {
            e.u8(f.overlap);
            e.bool(f.graph_at_first);
        }
        if f.prefill_small != 0 {
            e.u32(f.prefill_small);
        }
        if f.split != 0 {
            e.u8(f.split);
        }
        // Lane P (2026-10-18): the prefill contract (`contract=p1`, or P2's dense / experts / KDA / exchange ids)
        e.str(&f.prefill);
        // the prefix cache's arena after it, only with bit 4 (a Hello with the cache off keeps its bytes)
        if f.cache_on_wire() {
            e.str(&f.cache_request);
            e.u64(f.cache_slots);
            e.u64(f.cache_slot_bytes);
            e.u64(f.cache_pages);
        }
        // the latent cache's format last, only with bit 5 (an fp16 group's Hello keeps its bytes)
        if f.kv != "fp16" {
            e.str(&f.kv);
        }
        // the drafter's taps after it, only with bit 6 (a Hello without a drafter keeps its bytes)
        if f.taps != "none" {
            e.str(&f.taps);
        }
        // the drafter's split last, only with bit 7
        if f.draft_split != "off" {
            e.str(&f.draft_split);
        }
    }

    fn decode_facts(d: &mut Dec<'_>) -> Result<GlmFacts, WireError> {
        let flags = d.u8("glm graphs and adaptive slice")?;
        if flags & 4 != 0 && flags & 1 == 0 {
            return Err(WireError::Invalid(format!(
                "glm graphs, adaptive slice, draft graphs, head split, prefix cache, latent cache format, drafter taps and drafter split: \
                 flags {flags:#04x} (bit 2 only with bit 0)"
            )));
        }
        let (graphs, adaptive, draft_graphs, split) = (flags & 1 != 0, flags & 2 != 0, flags & 4 != 0, flags & 8 != 0);
        let mtp = d.bool("glm mtp")?;
        let mtp_depth = if mtp { d.u32("glm mtp depth cap")? } else { 0 };
        let mut f = GlmFacts {
            graphs,
            overlap: 0,
            graph_at_first: false,
            draft_graphs,
            prefill_small: 0,
            mtp,
            mtp_depth,
            window_rows: d.u32("glm window rows")?,
            mtp_vocab: d.str("glm mtp vocab")?,
            ownership: d.u64("glm ownership")?,
            collective: d.str("glm collective")?,
            vision: d.bool("glm vision tower")?,
            split: 0,
            prefill: String::new(),
            cache_request: "off".into(),
            cache_slots: 0,
            cache_slot_bytes: 0,
            cache_pages: 0,
            kv: "fp16".into(),
            taps: "none".into(),
            draft_split: "off".into(),
        };
        if graphs {
            f.overlap = d.u8("glm graph branches")?;
            f.graph_at_first = d.bool("glm graph capture policy")?;
        }
        if adaptive {
            f.prefill_small = d.u32("glm adaptive prefill slice")?;
        }
        if split {
            f.split = d.u8("glm head split")?;
        }
        f.prefill = d.str("glm prefill contract")?;
        if flags & 16 != 0 {
            f.cache_request = d.str("glm prefix cache request")?;
            f.cache_slots = d.u64("glm prefix cache snapshot slots")?;
            f.cache_slot_bytes = d.u64("glm prefix cache bytes of a snapshot slot")?;
            f.cache_pages = d.u64("glm prefix cache KV pages")?;
        }
        if flags & 32 != 0 {
            f.kv = d.str("glm latent cache format")?;
        }
        if flags & 64 != 0 {
            f.taps = d.str("glm drafter taps")?;
        }
        if flags & 128 != 0 {
            f.draft_split = d.str("glm drafter split")?;
        }
        Ok(f)
    }

    fn check_facts(a: &GlmFacts, b: &GlmFacts, who: u32) -> Vec<String> {
        let onoff = |x: bool| if x { "on" } else { "off" };
        [
            fact_differs("CUDA graphs (REXL3_GLM_GRAPHS)", onoff(a.graphs), onoff(b.graphs), who),
            fact_differs("the graph windows' layer branches (REXL3_GLM_OVERLAP)", a.overlap, b.overlap, who),
            fact_differs("the graph capture policy (REXL3_GLM_GRAPH_AT)", a.graph_at_first, b.graph_at_first, who),
            fact_differs("the NextN draft calls' graphs (REXL3_GLM_DRAFT_GRAPHS)", onoff(a.draft_graphs), onoff(b.draft_graphs), who),
            fact_differs(
                "the adaptive prefill slice (REXL3_GLM_ADAPTIVE_PREFILL; 0 = one size)",
                a.prefill_small,
                b.prefill_small,
                who,
            ),
            fact_differs(
                "the head split (REXL3_GLM_TP_SPLIT)",
                super::model::tp::split_names(a.split),
                super::model::tp::split_names(b.split),
                who,
            ),
            fact_differs("the NextN draft layer (--spec)", onoff(a.mtp), onoff(b.mtp), who),
            fact_differs("the draft depth cap (MTP_MAX_DEPTH)", a.mtp_depth, b.mtp_depth, who),
            fact_differs("decode window rows", a.window_rows, b.window_rows, who),
            fact_differs("the draft head's vocabulary (REXL3_MTP_VOCAB)", &a.mtp_vocab, &b.mtp_vocab, who),
            fact_differs(
                "the expert ownership table (digest)",
                format!("{:016x}", a.ownership),
                format!("{:016x}", b.ownership),
                who,
            ),
            fact_differs(
                "the transport (REXL3_TP_TRANSPORT: auto, roce or nccl)",
                super::tp_boot::transport_of_fact(&a.collective),
                super::tp_boot::transport_of_fact(&b.collective),
                who,
            ),
            fact_differs(
                "the collective policy and transport (REXL3_TP_COLLECTIVE, REXL3_TP_TRANSPORT, REXL3_ROCE_*)",
                &a.collective,
                &b.collective,
                who,
            ),
            fact_differs("the prefill contract (REXL3_GLM_PREFILL_CONTRACT)", &a.prefill, &b.prefill, who),
            fact_differs("the latent cache format (--kv-dtype / REXL3_GLM_KV; pass the same to every rank)", &a.kv, &b.kv, who),
            fact_differs("the DFlash2 drafter's taps (--draft; pass the same to every rank)", &a.taps, &b.taps, who),
            fact_differs(
                "the DFlash2 drafter's split (REXL3_GLM_DRAFT_SPLIT and --draft; pass the same to every rank)",
                &a.draft_split,
                &b.draft_split,
                who,
            ),
            // the prefix cache's arena (Nemotron's protocol 4): rank 0 forwards `EnableSnapshots` of its slots and every follower builds
            // exactly that pool, so the request and a slot's bytes must be equal and a follower's plan must hold at least rank 0's slots
            // and pages (each named, never the generic "KV pages" refusal the cache's pages would otherwise show up as)
            fact_differs(
                "the prefix cache request (--prefix-cache / --prefix-cache-mib; pass the same to every rank)",
                &a.cache_request,
                &b.cache_request,
                who,
            ),
            fact_differs("the prefix cache's bytes of one snapshot slot", a.cache_slot_bytes, b.cache_slot_bytes, who),
            (b.cache_slots < a.cache_slots).then(|| {
                format!(
                    "the prefix cache's snapshot slots: rank 0 planned {}, rank {who} can hold only {} (pass the same \
                     --prefix-cache-mib to every rank, or free memory on rank {who})",
                    a.cache_slots, b.cache_slots
                )
            }),
            (b.cache_pages < a.cache_pages).then(|| {
                format!(
                    "the prefix cache's KV pages: rank 0 planned {}, rank {who} can hold only {} (pass the same --prefix-cache-mib \
                     to every rank, or free memory on rank {who})",
                    a.cache_pages, b.cache_pages
                )
            }),
            // leader-only: rank 0 with or without a tower, a follower never with one (it splices the rows rank 0 sends)
            b.vision.then(|| {
                format!(
                    "the vision tower (--no-vision): rank 0 has {}, rank {who} has one; only rank 0 runs the tower (start rank {who} with --no-vision)",
                    if a.vision { "one" } else { "none" }
                )
            }),
        ]
        .into_iter()
        .flatten()
        .collect()
    }
}

/// An unconditional identity of the pack in `dir` for the boot gate: the SHA-256 (first 16 bytes, hex) of its `config.json` and its
/// safetensors index (every tensor, its shard and place). Header-level, like Nemotron's: it tells packs apart, it does not hash payloads.
pub fn pack_identity(dir: &Path) -> Result<String, String> {
    let mut h = Sha256::new();
    h.update(b"rexl3-glm-pack-v1\0");
    for name in ["config.json", "model.safetensors.index.json"] {
        let path = dir.join(name);
        let bytes = std::fs::read(&path).map_err(|e| format!("pack identity: {}: {e}", path.display()))?;
        h.update(name.as_bytes());
        h.update([0]);
        h.update((bytes.len() as u64).to_le_bytes());
        h.update(&bytes);
    }
    Ok(h.finalize().iter().take(16).map(|b| format!("{b:02x}")).collect())
}

/// The gate of a rank that runs `mo` with prefill chunks of `chunk` rows (placed at its rank by `BootGate::at`).
pub fn gate(pack: String, mo: &ModelOptions, chunk: usize, facts: GlmFacts) -> BootGate<GlmCall> {
    BootGate::new(pack, mo.layers as u32, mo.slots as u32, mo.kv_blocks as u64, mo.max_context as u64, chunk as u32, facts)
}

/// The model options and the prefill chunk a follower loads with: the leader's (it may hold more, never less).
pub fn options_of(g: &BootGate<GlmCall>) -> (ModelOptions, usize) {
    (
        ModelOptions {
            slots: g.slots as usize,
            kv_blocks: g.kv_pages as usize,
            max_context: g.max_context as usize,
            layers: g.layers as usize,
        },
        g.prefill_rows as usize,
    )
}

/// The `(slot, start, tokens)` of a prefill chunk's batch, refused (before anything is launched, so on every rank alike) unless it is
/// exactly the batch `RowBatch::from_requests` makes of one request: one sequence of consecutive positions, what `GlmModel::prefill_chunk`
/// rebuilds from those three.
pub fn chunk_of(b: &RowBatch) -> Result<(usize, usize, Vec<u32>), ModelError> {
    let bad = || ModelError::Request("a prefill chunk is the consecutive rows of one sequence".into());
    if b.n_seq() != 1 || b.rows == 0 || b.ids.len() != b.rows || b.positions.is_empty() {
        return Err(bad());
    }
    let (slot, start) = (b.seq_slot[0], b.positions[0]);
    if slot < 0 || start < 0 || b.ids.iter().any(|&t| t < 0) {
        return Err(bad());
    }
    let tokens: Vec<u32> = b.ids.iter().map(|&t| t as u32).collect();
    let want =
        RowBatch::from_requests(&[RowRequest { slot: slot as usize, tokens: tokens.clone(), first_position: start as u64 }]);
    if want != *b {
        return Err(bad());
    }
    Ok((slot as usize, start as usize, tokens))
}

/// One rank's runtime: the model, as the lockstep module drives it (on rank 0 inside a `TpLeader`, on a follower in `run_follower`), and
/// a follower's handles of the snapshots it took.
pub struct GlmRank {
    pub m: GlmModel,
    /// A follower's own handle of each snapshot it holds, by the sequence number of the `Snapshot` call that made it ([`apply_prefix`]).
    /// Rank 0 keeps its translation in its backend (`tp_backend`), so its table stays empty.
    snaps: SnapHandles,
}

impl GlmRank {
    pub fn new(m: GlmModel) -> GlmRank {
        GlmRank { m, snaps: SnapHandles::new() }
    }
}

impl LockstepRuntime for GlmRank {
    type Call = GlmCall;

    /// The tokenizer's vocabulary (what the backend offers), compared at Ready.
    fn vocab_size(&self) -> usize {
        TOKENIZER_VOCAB
    }

    fn has_mtp(&self) -> bool {
        self.m.has_mtp()
    }

    /// Every rank applies the prefix cache's calls ([`apply_prefix`]); compared at Ready.
    fn snapshots_supported() -> bool {
        true
    }

    fn begin_slot(&mut self, slot: usize) -> Result<(), ModelError> {
        if slot >= self.m.slots {
            return Err(ModelError::Request(format!("slot {slot} of a model of {} slots", self.m.slots)));
        }
        self.m.slices.begin_prompt(slot);
        self.m.begin_slot(slot)
    }

    fn end_slot(&mut self, slot: usize) {
        if slot < self.m.slots {
            self.m.end_slot(slot);
        }
    }

    fn prefill_chunk(&mut self, b: &RowBatch, want: Option<usize>) -> Result<Option<Candidates>, ModelError> {
        let (slot, start, tokens) = chunk_of(b)?;
        let t = self.m.slices.enter(Kind::Prefill, tokens.len(), Some(slot));
        let r = self.m.prefill_chunk(slot, start, &tokens, want);
        self.m.slices.leave(t, r.is_ok());
        r
    }

    fn window_candidates(&mut self, b: &RowBatch, count: usize) -> Result<Candidates, ModelError> {
        if b.parents.is_some() || b.rope3.is_some() {
            return Err(ModelError::Request("a GLM window has no tree and no rope sections".into()));
        }
        let t = self.m.slices.enter(Kind::Window, b.rows, None);
        let r = self.m.window_candidates(b, count);
        self.m.slices.leave(t, r.is_ok());
        let g = self.m.graph_stats();
        *RANK_GRAPHS.lock().unwrap_or_else(|p| p.into_inner()) =
            Some((g.replays, g.captures, g.eager_windows, g.capture_failures));
        r
    }

    fn commit(&mut self, slot: usize, keep: usize) {
        self.m.commit(slot, keep);
    }

    /// The group's flush is a draft round's (`DraftModel::flush`, the only sender of `Msg::Flush` here): lazy on every rank
    /// (`GlmModel::flush_lazy`), the next forward joins its KDA commits.
    fn flush(&mut self) -> Result<(), ModelError> {
        self.m.flush_lazy()
    }

    fn apply_call(&mut self, seq: u64, call: GlmCall) -> Result<CallReply, ModelError> {
        if call.is_prefix() {
            return apply_prefix(self, seq, call);
        }
        apply_family(self, call)
    }
}

/// A follower applies a family call (`GlmRank::apply_call`, and every mock's): an image chunk through `tp_vision`, a prompt forward's part
/// through `tp_part`, a draft call through `tp_mtp`. The prefix cache's calls need the call's sequence number: a runtime that keeps
/// snapshots routes them to [`apply_prefix`] first.
pub fn apply_family<R: GlmMtpRuntime + ImageRuntime + PartRuntime>(r: &mut R, call: GlmCall) -> Result<CallReply, ModelError> {
    match call {
        GlmCall::PrefillImage(p) => tp_vision::apply_image(r, &p),
        GlmCall::PrefillPart(p) => tp_part::apply_part(r, &p),
        other => tp_mtp::apply(r, other),
    }
}

// ------------------------------------------------------------------------------------------------------------------- the prefix cache

/// A follower's handles: the sequence number of the `Snapshot` call that made each snapshot, to the rank's own snapshot slot and the
/// KV pages it retained (rank-local ids: they never travel).
pub type SnapHandles = BTreeMap<u64, (u32, Vec<u32>)>;

/// A snapshot a rank took ([`take_snapshot`]): its slot, the KV pages it holds one reference to each, and whether the drafter's book
/// was sound (`SnapMeta::mtp_ok`: a restore of it keeps the slot drafting).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Taken {
    pub snap: u32,
    pub pages: Vec<u32>,
    pub mtp_ok: bool,
    /// The snapshot's content digest (`GlmModel::seal_snapshot`).
    pub digest: u64,
}

/// A rank's runtime as the prefix cache's calls need it (the real one is [`GlmRank`]; the CPU tests drive a recording mock): the model
/// calls `GlmBackend` makes for the cache, on this rank's own state. Their order in a snapshot is [`take_snapshot`]'s and the handles'
/// keys are [`apply_prefix`]'s, shared by rank 0 and the followers.
pub trait PrefixRuntime: LockstepRuntime<Call = GlmCall> {
    /// `GlmModel::enable_prefix_cache(slots)`; the bytes of one slot of the pool built (`GlmModel::prefix_snapshot_bytes`).
    fn enable_snapshots(&mut self, slots: usize) -> Result<u64, ModelError>;
    /// Snapshot slots no snapshot holds.
    fn free_snapshots(&self) -> usize;
    /// Under P2, the first position of `slot`'s open prefill segment (`GlmModel::p2_anchor`): rank 0 moves its cuts to that grid.
    fn p2_anchor(&self, slot: usize) -> Option<usize>;
    fn alloc_snapshot(&mut self) -> Option<u32>;
    /// `GlmModel::prefix_snapshot`; whether the drafter's book was sound (`SnapMeta::mtp_ok`).
    fn prefix_snapshot(&mut self, slot: usize, snap: u32, len: usize) -> Result<bool, ModelError>;
    fn retain_prefix_pages(&mut self, slot: usize, len: usize) -> Vec<u32>;
    fn restore_prefix(&mut self, slot: usize, snap: u32, pages: &[u32], len: usize) -> Result<(), ModelError>;
    fn release_prefix(&mut self, snap: u32, pages: &[u32]);
    /// `GlmModel::seal_snapshot`: the content digest of a snapshot just taken (its pages retained), recorded with it.
    fn seal_snapshot(&mut self, snap: u32, pages: &[u32]) -> Result<u64, ModelError>;
    /// `GlmModel::prefix_state_digest`: the content digest of `slot`'s live state as a prefix of `len` tokens.
    fn prefix_state_digest(&mut self, slot: usize, len: usize) -> Result<u64, ModelError>;
    /// The KV pool's free pages and the pages each slot's table holds (`KvTables`): a cache that keeps idle pool pages reads rank 0's,
    /// which are every rank's (the calls allocate and release pages in one order on every rank).
    fn kv_pages(&self) -> (usize, Vec<usize>);
    /// A follower's handles ([`apply_prefix`]).
    fn snapshot_handles(&mut self) -> &mut SnapHandles;
}

/// The snapshot of a [`GlmCall::Snapshot`], on any rank (rank 0 runs it as the local half of the call it forwards, a follower in
/// [`apply_prefix`]): a free snapshot slot, the copy of `slot`'s state after `len` tokens, then the references to its pages, as
/// `GlmBackend` takes a snapshot at a slice end. `None`: no free slot (skipped on every rank alike, not an error); the slot is given back
/// if the copy fails.
pub fn take_snapshot<R: PrefixRuntime + ?Sized>(r: &mut R, slot: usize, len: usize) -> Result<Option<Taken>, ModelError> {
    let Some(snap) = r.alloc_snapshot() else { return Ok(None) };
    let mtp_ok = match r.prefix_snapshot(slot, snap, len) {
        Ok(ok) => ok,
        Err(e) => {
            r.release_prefix(snap, &[]);
            return Err(e);
        }
    };
    let pages = r.retain_prefix_pages(slot, len);
    match r.seal_snapshot(snap, &pages) {
        Ok(digest) => Ok(Some(Taken { snap, pages, mtp_ok, digest })),
        Err(e) => {
            r.release_prefix(snap, &pages);
            Err(e)
        }
    }
}

/// A follower applies a prefix-cache call. Its handle of a snapshot is keyed by the sequence number of the `Snapshot` call that made it,
/// and `Restore` / `Release` name that number (`of`). A restore of a handle this rank does not hold is an error by name (rank 0 then
/// prefills cold); a release of one is Ok (a snapshot this rank skipped). The follower never evicts on its own. `Snapshot` and `Restore`
/// answer with this rank's content digest (`lockstep::taken_digest`; a skipped snapshot `lockstep::SNAPSHOT_SKIPPED`).
pub fn apply_prefix<R: PrefixRuntime + ?Sized>(r: &mut R, seq: u64, call: GlmCall) -> Result<CallReply, ModelError> {
    match call {
        GlmCall::EnableSnapshots { slots } => r.enable_snapshots(slots as usize).map(CallReply::digest),
        GlmCall::Snapshot { slot, len } => match take_snapshot(r, slot as usize, len as usize)? {
            Some(t) => {
                let d = lockstep::taken_digest(t.digest);
                r.snapshot_handles().insert(seq, (t.snap, t.pages));
                Ok(CallReply::digest(d))
            }
            None => Ok(CallReply::digest(lockstep::SNAPSHOT_SKIPPED)),
        },
        GlmCall::Restore { slot, len, of } => {
            let (snap, pages) = r.snapshot_handles().get(&of).cloned().ok_or_else(|| {
                ModelError::Request(format!("restore_prefix of the snapshot call {of} made: this rank holds no such snapshot"))
            })?;
            r.restore_prefix(slot as usize, snap, &pages, len as usize)?;
            let d = r.prefix_state_digest(slot as usize, len as usize)?;
            Ok(CallReply::digest(lockstep::taken_digest(d)))
        }
        GlmCall::Release { of } => {
            // a snapshot this rank skipped (or never took) is not held: nothing to release, and not an error
            if let Some((snap, pages)) = r.snapshot_handles().remove(&of) {
                r.release_prefix(snap, &pages);
            }
            Ok(CallReply::NONE)
        }
        other => Err(ModelError::Request(format!("{} is not a prefix-cache call", other.name()))),
    }
}

impl PrefixRuntime for GlmRank {
    fn enable_snapshots(&mut self, slots: usize) -> Result<u64, ModelError> {
        self.m.enable_prefix_cache(slots)?;
        Ok(self.m.prefix_snapshot_bytes())
    }
    fn free_snapshots(&self) -> usize {
        self.m.prefix_free_snapshots()
    }
    fn p2_anchor(&self, slot: usize) -> Option<usize> {
        self.m.p2_anchor(slot)
    }
    fn alloc_snapshot(&mut self) -> Option<u32> {
        self.m.alloc_snapshot()
    }
    fn prefix_snapshot(&mut self, slot: usize, snap: u32, len: usize) -> Result<bool, ModelError> {
        if slot >= self.m.slots {
            return Err(ModelError::Request(format!("a snapshot of slot {slot} of a model of {} slots", self.m.slots)));
        }
        self.m.prefix_snapshot(slot, snap, len)?;
        Ok(self.m.prefix_snapshot_meta(snap).is_some_and(|m| m.mtp_ok))
    }
    fn retain_prefix_pages(&mut self, slot: usize, len: usize) -> Vec<u32> {
        self.m.retain_prefix_pages(slot, len)
    }
    fn restore_prefix(&mut self, slot: usize, snap: u32, pages: &[u32], len: usize) -> Result<(), ModelError> {
        if slot >= self.m.slots {
            return Err(ModelError::Request(format!("a restore into slot {slot} of a model of {} slots", self.m.slots)));
        }
        self.m.restore_prefix(slot, snap, pages, len)
    }
    fn release_prefix(&mut self, snap: u32, pages: &[u32]) {
        self.m.release_prefix(snap, pages)
    }
    fn seal_snapshot(&mut self, snap: u32, pages: &[u32]) -> Result<u64, ModelError> {
        self.m.seal_snapshot(snap, pages)
    }
    fn prefix_state_digest(&mut self, slot: usize, len: usize) -> Result<u64, ModelError> {
        self.m.prefix_state_digest(slot, len)
    }
    fn kv_pages(&self) -> (usize, Vec<usize>) {
        (self.m.kv_free_pages(), (0..self.m.slots).map(|s| self.m.kv_slot_pages(s)).collect())
    }
    fn snapshot_handles(&mut self) -> &mut SnapHandles {
        &mut self.snaps
    }
}

impl GlmMtpRuntime for GlmRank {
    fn mtp_book(&self, slot: usize) -> Option<TapBook> {
        if slot >= self.m.slots {
            return None;
        }
        self.m.mtp_book(slot).cloned()
    }
    fn mtp_draft_keyed(
        &mut self,
        asks: &[DraftAsk<'_>],
        keyed: &[Option<gb_sampling::Sampling>],
    ) -> Result<Vec<Option<Drafts>>, ModelError> {
        if let Some(a) = asks.iter().find(|a| a.slot >= self.m.slots) {
            return Err(ModelError::Request(format!("a draft ask for slot {} of a model of {} slots", a.slot, self.m.slots)));
        }
        let t = self.m.slices.enter(Kind::Draft, asks.len(), None);
        let r = self.m.mtp_draft_keyed(asks, keyed);
        self.m.slices.leave(t, r.is_ok());
        let g = self.m.graph_stats();
        *RANK_DRAFT_GRAPHS.lock().unwrap_or_else(|p| p.into_inner()) =
            Some((g.draft_replays, g.draft_captures, g.draft_eager, g.draft_capture_failures));
        r
    }
    fn mtp_disable_slot(&mut self, slot: usize) {
        if slot < self.m.slots {
            self.m.mtp_disable_slot(slot);
        }
    }
    fn graph_counters(&self) -> Option<gb_engine::GraphCounters> {
        Some(graph_counters_of(&self.m.graph_stats()))
    }
    fn has_dflash(&self) -> bool {
        self.m.has_dflash()
    }
    fn dflash_draft(&mut self, asks: &[super::model::DflashAsk<'_>]) -> Result<Vec<crate::draft::Picked>, ModelError> {
        if let Some(a) = asks.iter().find(|a| a.slot >= self.m.slots) {
            return Err(ModelError::Request(format!("a draft ask for slot {} of a model of {} slots", a.slot, self.m.slots)));
        }
        self.m.dflash_draft(asks)
    }
    fn dflash_split(&self) -> bool {
        self.m.dflash_split()
    }
    fn dflash_reqs(&self, asks: &[super::model::DflashAsk<'_>]) -> Result<Vec<super::model::DflashReq>, ModelError> {
        self.m.dflash_reqs(asks)
    }
    fn dflash_run(&mut self, reqs: &[super::model::DflashReq]) -> Result<Vec<crate::draft::Picked>, ModelError> {
        if let Some(r) = reqs.iter().find(|r| r.slot >= self.m.slots) {
            return Err(ModelError::Request(format!("a draft request for slot {} of a model of {} slots", r.slot, self.m.slots)));
        }
        let t = self.m.slices.enter(Kind::Draft, reqs.len(), None);
        let r = self.m.dflash_run(reqs);
        self.m.slices.leave(t, r.is_ok());
        r
    }
    fn dflash_prompt_end(&mut self, slot: usize, end: Option<usize>, cut: Option<usize>) {
        if slot < self.m.slots {
            self.m.dflash_prompt_end(slot, end, cut);
        }
    }
}

/// The engine's counters of a model's window graph cache.
pub fn graph_counters_of(g: &super::model::GraphStats) -> gb_engine::GraphCounters {
    gb_engine::GraphCounters {
        replays: g.replays,
        captures: g.captures,
        eager_windows: g.eager_windows,
        capture_failures: g.capture_failures,
        evictions: g.evictions,
        capture_ms: g.capture_ms,
        ..Default::default()
    }
}

/// The last window graph counters of this process's rank (`GlmRank::window_candidates` stores them): the follower's stats line reads
/// them (its runtime is owned by the lockstep loop). One rank per process; informational only.
pub static RANK_GRAPHS: std::sync::Mutex<Option<(u64, u64, u64, u64)>> = std::sync::Mutex::new(None);

/// The last NextN draft-call graph counters of this process's rank (`GlmRank::mtp_draft_keyed` stores them; lane T): replays, captures,
/// eager runs, failed captures of the draft parts.
pub static RANK_DRAFT_GRAPHS: std::sync::Mutex<Option<(u64, u64, u64, u64)>> = std::sync::Mutex::new(None);

/// `draft_replays=.. draft_captures=.. draft_eager=.. draft_capture_failures=..` of [`RANK_DRAFT_GRAPHS`] (zeros before the first draft call).
pub fn rank_draft_graphs_line() -> String {
    let (r, c, e, f) = RANK_DRAFT_GRAPHS.lock().unwrap_or_else(|p| p.into_inner()).unwrap_or((0, 0, 0, 0));
    format!("draft_replays={r} draft_captures={c} draft_eager={e} draft_capture_failures={f}")
}

/// `replays=.. captures=.. eager=.. capture_failures=..` of [`RANK_GRAPHS`] (empty before the first window).
pub fn rank_graphs_line() -> String {
    match *RANK_GRAPHS.lock().unwrap_or_else(|p| p.into_inner()) {
        Some((r, c, e, f)) => format!("graph_replays={r} graph_captures={c} graph_eager={e} graph_capture_failures={f}"),
        None => "graph_replays=0 graph_captures=0 graph_eager=0 graph_capture_failures=0".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// w10: the head split is a fact of the Hello. A rank on the default (`kda,lm,mla,sh`) and a rank on `none` are refused by name, in both
    /// directions; two ranks on the same set agree; the set survives the wire (the byte is appended only when set, as before).
    #[test]
    fn two_ranks_on_different_head_splits_are_refused_by_name() {
        use super::super::model::tp::DEFAULT_SPLIT;
        let on = GlmFacts::local(288, 2).with_split(DEFAULT_SPLIT);
        let off = GlmFacts::local(288, 2).with_split(0);
        assert!(GlmCall::check_facts(&on, &on, 1).is_empty());
        assert!(GlmCall::check_facts(&off, &off, 1).is_empty());
        for (a, b) in [(&on, &off), (&off, &on)] {
            let why = GlmCall::check_facts(a, b, 1);
            assert_eq!(why.len(), 1, "{why:?}");
            assert!(
                why[0].contains("REXL3_GLM_TP_SPLIT") && why[0].contains("kda,lm,mla,sh") && why[0].contains("none"),
                "{why:?}"
            );
        }
        let some = GlmFacts::local(288, 2).with_split(super::super::model::tp::SPLIT_KDA | super::super::model::tp::SPLIT_LM);
        assert_eq!(GlmCall::check_facts(&on, &some, 1).len(), 1);
        for f in [&on, &off, &some] {
            let mut e = Enc::new();
            GlmCall::encode_facts(f, &mut e);
            let mut d = Dec::new(&e.0);
            let g = GlmCall::decode_facts(&mut d).unwrap();
            assert_eq!(g.split, f.split);
            assert!(GlmCall::check_facts(f, &g, 1).is_empty());
        }
    }

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    /// The prefix cache's four calls (kinds 48 to 51): their bytes are pinned (length, protocol 5, kind, seq, then the body: `slots`;
    /// `slot`, `len`; `slot`, `len`, `of`; `of`, little-endian), each round-trips, and the kinds are the family's own, clear of the core's,
    /// the draft calls' and the image chunk's.
    #[test]
    fn the_snapshot_calls_have_these_exact_bytes_and_round_trip() {
        use crate::lockstep::Msg;
        let calls = [
            (
                GlmCall::EnableSnapshots { slots: 3 },
                K_ENABLE_SNAPSHOTS,
                "enable_snapshots",
                "0e00000005300b0000000000000003000000",
            ),
            (
                GlmCall::Snapshot { slot: 1, len: 272 },
                K_SNAPSHOT,
                "prefix_snapshot",
                "1200000005310c000000000000000100000010010000",
            ),
            (
                GlmCall::Restore { slot: 2, len: 272, of: 12 },
                K_RESTORE,
                "restore_prefix",
                "1a00000005320d0000000000000002000000100100000c00000000000000",
            ),
            (GlmCall::Release { of: 12 }, K_RELEASE, "release_prefix", "1200000005330e000000000000000c00000000000000"),
        ];
        for (i, (c, kind, name, golden)) in calls.iter().enumerate() {
            let seq = 11 + i as u64;
            let m: Msg<GlmCall> = Msg::Call(c.clone());
            let bytes = m.encode(seq);
            let (back_seq, back) = Msg::<GlmCall>::decode(&bytes[4..]).unwrap();
            assert_eq!((back_seq, &back), (seq, &m), "round trip of {name}");
            assert_eq!(hex(&bytes), *golden, "the {name} frame's bytes moved");
            assert_eq!((c.kind(), c.name(), c.is_prefix()), (*kind, *name, true));
            assert!(!crate::lockstep::CORE_KINDS.contains(kind) && (32..=127).contains(kind));
            assert!(![tp_mtp::K_MTP_DRAFT, tp_mtp::K_MTP_DISABLE, K_PREFILL_IMAGE].contains(kind));
            // a body one byte short is a wire error, never a call
            assert!(Msg::<GlmCall>::decode(&bytes[4..bytes.len() - 1]).is_err(), "{name}");
        }
        assert!(!GlmCall::MtpDisable { slot: 0 }.is_prefix());
    }

    fn cache_facts(o: crate::serve::PrefixCacheOption, slots: usize, pages: usize, bytes: u64) -> GlmFacts {
        let budget =
            gb_engine::CacheBudget { snapshot_slots: slots, page_blocks: pages, snapshot_bytes: bytes, page_bytes: 1 << 20 };
        GlmFacts::local(288, 2).with_cache(crate::families::nemotron_h_omni::tp_runtime::cache_request_label(o), &budget)
    }

    /// The prefix cache's arena is a fact of the Hello (Nemotron's protocol 4): two ranks given another `--prefix-cache-mib` (or the cache on
    /// one rank only) are refused naming the request, a slot of other bytes naming the bytes, and a follower that can hold fewer snapshot
    /// slots or cache pages than rank 0 planned naming which; a follower that holds more is the group (it is built with exactly rank 0's
    /// pool, `EnableSnapshots`). The facts survive the wire, and a whole gate's refusal carries the line.
    #[test]
    fn two_ranks_with_different_prefix_caches_are_refused_by_name() {
        use crate::serve::PrefixCacheOption::{Mib, Off};
        const BYTES: u64 = 73_828_352;
        let on = cache_facts(Mib(3584), 4, 14, BYTES);
        assert!(on.cache_on_wire() && !GlmFacts::local(288, 2).cache_on_wire());
        assert!(GlmCall::check_facts(&on, &on, 1).is_empty());
        let request = |why: &[String], ours: &str, theirs: &str| {
            why.iter().any(|w| {
                w.contains("the prefix cache request (--prefix-cache / --prefix-cache-mib")
                    && w.contains(&format!("rank 0 has {ours}, rank 1 has {theirs}"))
            })
        };
        let why = GlmCall::check_facts(&on, &cache_facts(Mib(4096), 4, 14, BYTES), 1);
        assert_eq!(why.len(), 1, "{why:?}");
        assert!(request(&why, "mib=3584", "mib=4096"), "{why:?}");
        for off in [cache_facts(Off, 0, 0, 0), GlmFacts::local(288, 2)] {
            let why = GlmCall::check_facts(&on, &off, 1);
            assert!(request(&why, "mib=3584", "off"), "{why:?}");
            let why = GlmCall::check_facts(&off, &on, 1);
            assert!(request(&why, "off", "mib=3584"), "{why:?}");
        }
        let why = GlmCall::check_facts(&on, &cache_facts(Mib(3584), 3, 14, BYTES), 1);
        assert_eq!(why.len(), 1, "{why:?}");
        assert!(why[0].starts_with("the prefix cache's snapshot slots: rank 0 planned 4, rank 1 can hold only 3"), "{why:?}");
        let why = GlmCall::check_facts(&on, &cache_facts(Mib(3584), 4, 13, BYTES), 1);
        assert_eq!(why.len(), 1, "{why:?}");
        assert!(why[0].starts_with("the prefix cache's KV pages: rank 0 planned 14, rank 1 can hold only 13"), "{why:?}");
        let why = GlmCall::check_facts(&on, &cache_facts(Mib(3584), 4, 14, BYTES / 2), 1);
        assert_eq!(why.len(), 1, "{why:?}");
        assert!(
            why[0].starts_with("the prefix cache's bytes of one snapshot slot: rank 0 has 73828352, rank 1 has 36914176"),
            "{why:?}"
        );
        assert!(GlmCall::check_facts(&on, &cache_facts(Mib(3584), 5, 20, BYTES), 1).is_empty(), "a follower may hold more");
        // the wire carries every one of them
        for f in [&on, &cache_facts(Mib(4096), 3, 13, BYTES / 2), &cache_facts(Off, 0, 0, 0)] {
            let mut e = Enc::new();
            GlmCall::encode_facts(f, &mut e);
            let back = GlmCall::decode_facts(&mut Dec::new(&e.0)).unwrap();
            assert_eq!(&back, f);
        }
        let mo = ModelOptions { slots: 2, kv_blocks: 64, max_context: 8192, layers: 45 };
        let g0 = gate("p".into(), &mo, 512, on.clone()).at(0, 2);
        assert!(BootGate::check(&g0, &gate("p".into(), &mo, 512, on.clone()).at(1, 2)).is_ok());
        let e = BootGate::check(&g0, &gate("p".into(), &mo, 512, cache_facts(Mib(3584), 3, 14, BYTES)).at(1, 2)).unwrap_err();
        assert!(e.contains("the prefix cache's snapshot slots: rank 0 planned 4, rank 1 can hold only 3"), "{e}");
    }

    /// A follower's handle is keyed by the sequence number of the `Snapshot` call that made it: a restore naming a call that made none (or
    /// a handle already released) is refused by name on that rank; a release of one is Ok and changes nothing (a snapshot this rank
    /// skipped); its table holds exactly the snapshots rank 0 has not released. The snapshot's answer is the rank's content digest (forced
    /// odd), and the restore's the digest of what it restored, equal to it. In a group the restore's refusal comes back to rank 0 as that
    /// rank's answer to a cache call (rank 0 then prefills cold) and the group stays up.
    #[test]
    fn an_unknown_snapshot_handle_is_refused_by_name_and_answered_to_rank_0() {
        use crate::families::glm::tp_backend::tests::{t, Log, Rec};
        use crate::lockstep::{run_follower, FollowerConfig, Link, RankAnswer, TpLeader};
        use gb_engine::RowRequest;
        let mut r = Rec::new(Log::default());
        r.snap_base = 100;
        let apply = |r: &mut Rec, seq: u64, c: GlmCall| r.apply_call(seq, c);
        assert_eq!(apply(&mut r, 1, GlmCall::EnableSnapshots { slots: 2 }).unwrap().digest, 4096);
        r.begin_slot(0).unwrap();
        r.prefill_chunk(&RowBatch::from_requests(&[RowRequest { slot: 0, tokens: vec![5; 300], first_position: 0 }]), None)
            .unwrap();
        let d = apply(&mut r, 2, GlmCall::Snapshot { slot: 0, len: 300 }).unwrap().digest;
        assert_eq!(d, lockstep::taken_digest(r.seal_snapshot(100, &[0, 1]).unwrap()), "the content digest of the snapshot taken");
        assert_eq!(d & 1, 1, "never SNAPSHOT_SKIPPED");
        assert_eq!(r.handles.get(&2), Some(&(100, vec![0, 1])), "keyed by the call, holding this rank's own slot");
        assert_eq!(
            apply(&mut r, 3, GlmCall::Restore { slot: 1, len: 300, of: 2 }).unwrap().digest,
            d,
            "restored what it captured"
        );
        assert_eq!(r.handles.len(), 1, "a restore keeps the snapshot");
        let e = apply(&mut r, 4, GlmCall::Restore { slot: 1, len: 300, of: 3 }).unwrap_err();
        assert!(
            matches!(&e, ModelError::Request(m) if m.contains("restore_prefix of the snapshot call 3 made") && m.contains("holds no such snapshot")),
            "{e}"
        );
        assert_eq!(apply(&mut r, 5, GlmCall::Release { of: 100 }).unwrap().digest, 0, "a release of a snapshot not held is Ok");
        assert_eq!(r.handles.len(), 1, "and changes nothing");
        apply(&mut r, 6, GlmCall::Release { of: 2 }).unwrap();
        assert!(r.handles.is_empty());
        apply(&mut r, 7, GlmCall::Release { of: 2 }).unwrap();
        assert!(r.handles.is_empty(), "released: a second release changes nothing (never released twice)");
        // the group: rank 0 sends a restore of a call that made no snapshot; the follower's refusal is its answer, the group stays up
        let (a, b) = Link::<GlmCall>::pair(std::time::Duration::from_secs(5)).unwrap();
        let h = std::thread::spawn(move || run_follower(Rec::new(Log::default()), b, FollowerConfig::new(t())));
        let mut leader = TpLeader::lead(Rec::new(Log::default()), a, t());
        let out = leader.forward_cache_call(GlmCall::Restore { slot: 0, len: 16, of: 77 }, |_, _| Ok(()), true).unwrap();
        assert!(matches!(&out.followers[0], RankAnswer::Err(m) if m.contains("call 77 made")), "{out:?}");
        leader.barrier().unwrap();
        assert_eq!(leader.failure(), None);
        drop(leader);
        let _ = h.join();
    }

    #[test]
    fn the_range_table_puts_each_expert_on_its_rank_and_the_digest_tells_tables_apart() {
        let t = range_table(288, 2);
        assert!(t[..144].iter().all(|&r| r == 0) && t[144..].iter().all(|&r| r == 1));
        assert_eq!(range_table(288, 1), vec![0; 288]);
        let (d1, d2, d3) = (ownership_digest(&range_table(288, 1)), ownership_digest(&t), ownership_digest(&range_table(288, 3)));
        assert!(d1 != d2 && d2 != d3 && d1 != d3);
        let mut moved = t.clone();
        moved.swap(143, 144);
        assert_ne!(ownership_digest(&moved), d2);
        assert_eq!(ownership_digest(&range_table(288, 2)), d2, "a pure function of the table");
    }

    #[test]
    fn a_prefill_batch_must_be_one_sequence_of_consecutive_rows() {
        let one = RowBatch::from_requests(&[RowRequest { slot: 2, tokens: vec![5, 6, 7], first_position: 10 }]);
        assert_eq!(chunk_of(&one).unwrap(), (2, 10, vec![5, 6, 7]));
        let two = RowBatch::from_requests(&[
            RowRequest { slot: 0, tokens: vec![1], first_position: 0 },
            RowRequest { slot: 1, tokens: vec![2], first_position: 0 },
        ]);
        assert!(chunk_of(&two).is_err());
        let mut gap = one.clone();
        gap.positions[1] += 1;
        assert!(chunk_of(&gap).is_err());
        let mut tree = one.clone();
        tree.parents = Some(vec![-1, 0, 1]);
        assert!(chunk_of(&tree).is_err());
        let mut neg = one.clone();
        neg.ids[0] = -1;
        assert!(chunk_of(&neg).is_err());
        assert!(chunk_of(&RowBatch::from_requests(&[])).is_err());
    }
}
