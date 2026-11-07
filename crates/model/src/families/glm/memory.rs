//! The GLM-5.3-Flash memory model: an MLA latent plus a pooled indexer key per token on 11 layers (a 12th for the MTP head), Kimi delta
//! attention state and the DSA raw-row rings per slot, a replay record per window row, and a vision tower. Every figure is the arithmetic
//! of `alloc.rs`, which is built from the kernel crates' own size functions, so what admission is told is what the backend allocates (a
//! GPU test sums the backend's real allocations against it).

use super::alloc::{self, AllocPlan};
use super::config::GlmConfig;
use crate::family::MemorySource;
use crate::kvtables::BLOCK_TOKENS;
use crate::memory::MemoryOptions;
use gb_sched::{KvLayout, MemoryError, MemoryModel, MemoryProfile, StepShape, VisionMemory};

/// The GLM blanket: the same 5 GiB the Qwen engine has run on. The modelled buffers (scratch plus a full window of records, about 0.9 GiB
/// at 128 rows) are far under it, so it stays the reserve; the planner takes the larger of the two.
pub const ACTIVATION_RESERVE: u64 = 5 << 30;

/// The memory model of a GLM pack, ready to be evaluated at any serving settings.
pub struct GlmMemorySource(pub GlmConfig);

impl MemorySource for GlmMemorySource {
    fn profile(&self, opts: &MemoryOptions) -> Result<MemoryProfile, MemoryError> {
        memory_profile(&self.0, opts)
    }
}

/// The settings the GLM backend is evaluated at, from the generic serving options: a window is the rows of one decode or verify step across
/// all sequences (`alloc::WINDOW_ROWS`), a prefill chunk is `--prefill-chunk` rows (`alloc::prefill_chunk_for`: 512 unless asked, at most
/// 2,048; above `alloc::MAX_ROWS` rows the large-row kernels run and their scratch is priced). Context and the MTP and vision choices are the caller's.
pub fn engine_options(opts: &MemoryOptions) -> MemoryOptions {
    MemoryOptions { window_rows: alloc::WINDOW_ROWS, prefill_rows: alloc::prefill_chunk_for(opts.prefill_rows), ..*opts }
}

/// The validated profile of `cfg` at `opts`. Speculation with the pack's MTP head is refused for a pack that has none.
pub fn memory_profile(cfg: &GlmConfig, opts: &MemoryOptions) -> Result<MemoryProfile, MemoryError> {
    if opts.mtp && !cfg.has_mtp() {
        return Err(MemoryError::Inconsistent(
            "glm5_next: speculation with the pack's MTP head was asked for but the pack has no NextN layer".into(),
        ));
    }
    let window = opts.window_rows as u64;
    let opts = &MemoryOptions { prefill_rows: alloc::prefill_chunk_for(opts.prefill_rows), ..*opts };
    let shape =
        StepShape { page_tokens: BLOCK_TOKENS as u64, window_rows: window, scratch_rows: (opts.prefill_rows as u64).max(window) };
    MemoryProfile::new(&GlmMemory { cfg, opts: *opts }, shape)
}

/// The GLM-5.3-Flash memory model.
pub struct GlmMemory<'a> {
    pub cfg: &'a GlmConfig,
    pub opts: MemoryOptions,
}

impl GlmMemory<'_> {
    /// What one slot holds for the target model alone (`mtp` off), or for the NextN drafter on top of it.
    fn slot_bytes(&self, mtp: bool) -> u64 {
        AllocPlan::slot_items(self.cfg, self.opts.max_context, mtp).map_or(0, |v| v.iter().map(|i| i.bytes).sum())
    }
}

impl MemoryModel for GlmMemory<'_> {
    fn family(&self) -> &str {
        "glm5_next"
    }
    fn kv_layout(&self) -> KvLayout {
        self.cfg.kv_layout(self.opts.mtp)
    }
    fn recurrent_layers(&self) -> u64 {
        self.cfg.kda_layers().len() as u64
    }
    /// A slot's state: the KDA pools of the 34 layers (two fp32 state buffers, the conv history and the parity word each: the double
    /// buffer is what makes a fully accepted window free of replay), the 11 DSA rings and the slot's block-table row. All of it is
    /// allocated per slot up front, whatever the sequence's length.
    fn state_bytes_per_seq(&self) -> u64 {
        self.slot_bytes(false)
    }
    fn window_record_bytes_per_row(&self) -> u64 {
        self.cfg.kda_layers().len() as u64 * self.cfg.kda_record_bytes_per_row_per_layer()
    }
    /// The scratch of a forward: the window-scale buffers at the decode kernels' 128 rows (fewer for a smaller `rows`) and the buffers that scale with
    /// a prefill chunk at `rows` (up to the plan's widest chunk).
    fn scratch_bytes(&self, rows: u64) -> u64 {
        let window = (rows as usize).clamp(1, alloc::MAX_ROWS);
        let chunk = (rows as usize).clamp(window, alloc::max_prefill_chunk());
        AllocPlan::scratch_items_chunked(self.cfg, window, chunk, self.opts.max_context, self.opts.mtp)
            .map_or(0, |v| v.iter().map(|i| i.bytes).sum())
    }
    fn activation_reserve(&self) -> u64 {
        ACTIVATION_RESERVE
    }
    /// The NextN head's per-slot state when it drafts: its DSA raw-row ring (the 12th layer's, 128 KiB) and the store of pending tapped hidden
    /// rows (`alloc::MTP_TAP_CAP` fp16 rows of 4,096). Its latent cache and pooled keys are per token and charged through the KV layout
    /// (a 12th layer in every page); its buffers for a call are in the scratch. Zero without MTP.
    fn drafter_state_bytes_per_slot(&self) -> u64 {
        if self.opts.mtp {
            self.slot_bytes(true) - self.slot_bytes(false)
        } else {
            0
        }
    }
    fn vision(&self) -> Option<VisionMemory> {
        let (v, o) = (self.cfg.vision.as_ref()?, self.opts.vision?);
        // the processor never produces an image above its own cap, whatever the server would accept
        let tokens = super::vision::image_token_cap(self.cfg, o.max_image_tokens);
        Some(VisionMemory {
            weight_bytes: Some(super::vision::weight_region_bytes(v)),
            encoder_scratch_bytes: v.encoder_scratch_bytes(tokens),
            feature_bytes_per_image: v.feature_bytes(tokens),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{VisionOptions, ENCODERS_IN_FLIGHT};
    use serde_json::Value;

    const GLM_CONFIG: &str = include_str!("../../../../../testdata/glm53/config.json");
    const GIB: u64 = 1 << 30;

    fn real() -> GlmConfig {
        GlmConfig::from_json(&serde_json::from_str(GLM_CONFIG).unwrap()).unwrap()
    }

    fn with_processor() -> GlmConfig {
        let mut c = real();
        let p: Value = serde_json::from_str(include_str!("../../../../../testdata/glm53/processor_config.json")).unwrap();
        c.processor = Some(GlmConfig::processor_from_json(&p).unwrap());
        c
    }

    /// The options of a run with a 128-row chunk (the decode kernels' window: no large-row scratch).
    fn opts() -> MemoryOptions {
        engine_options(&MemoryOptions { prefill_rows: 128, ..MemoryOptions::serving_defaults() })
    }

    #[test]
    fn the_config_has_head_dim_zero_and_still_prices_every_token() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        let v: Value = serde_json::from_str(GLM_CONFIG).unwrap();
        assert_eq!(v["text_config"]["head_dim"], 0, "the real pack config says head_dim 0");
        assert!(memory_profile(&real(), &opts()).unwrap().kv_bytes_per_token() > 0);
    }

    #[test]
    fn kv_per_token_by_hand() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        let c = real();
        let p = memory_profile(&c, &opts()).unwrap();
        // one MLA layer with its own indexer, per token, fp16: latent 512 x 2 = 1,024 B; the pooled indexer key, 128 x 2 B per 4 tokens, 64 B.
        // There is NO per-token indexer plane (the oracle keeps [k | gate], 512 B a token and layer): the raw rows are a ring per slot.
        // Eleven such layers (3, 7, ..., 43):
        assert_eq!(p.kv_bytes_per_token(), 11 * (1024 + 64));
        assert_eq!(p.kv_bytes_per_token(), 11_968);
        // a 256-token page per layer: 256 x 1,024 + 64 pools x 256 B = 262,144 + 16,384 = 278,528; eleven layers
        assert_eq!(p.kv_page_bytes(), 11 * 278_528);
        // docs/GLM53.md: "KV about 4.3 GiB at 262,144 tokens" counted the per-token plane; without it it is 2.92 GiB
        let at_262144 = p.kv_bytes_per_token() * 262_144;
        assert_eq!(at_262144, 3_137_339_392);
        assert!((at_262144 as f64 / GIB as f64 - 2.92).abs() < 0.01);
        // with the MTP head's own cache (a 12th layer at the same positions)
        let with_mtp = memory_profile(&c, &MemoryOptions { mtp: true, ..opts() }).unwrap();
        assert_eq!(with_mtp.kv_bytes_per_token(), 12 * 1088);
        assert_eq!(with_mtp.kv_page_bytes(), 12 * 278_528);
        // an fp8 latent (`--kv-dtype fp8`): 512 E4M3 bytes and one fp32 scale a token and layer, the pool plane unchanged: 516 + 64 B, so a
        // fixed budget holds 11,968 / 6,380 = 1.876x the tokens; the page is the device's (`kv8::PAGE_BYTES` + the pools)
        let c8 = GlmConfig { latent_kv: gb_cuda::mla::LatentKv::Fp8, ..real() };
        let p8 = memory_profile(&c8, &opts()).unwrap();
        assert_eq!(p8.kv_bytes_per_token(), 11 * (516 + 64));
        assert_eq!(p8.kv_bytes_per_token(), 6_380);
        assert_eq!(p8.kv_page_bytes(), 11 * (gb_cuda::mla::kv8::PAGE_BYTES as u64 + 16_384));
        let a8 = super::super::alloc::AllocPlan::new(
            &c8,
            &super::super::alloc::AllocParams { slots: 1, kv_blocks: 7, max_context: 4096, max_rows: 128, window_rows: 64, mtp: false },
        )
        .unwrap();
        assert_eq!(a8.get("mla_latent") + a8.get("dsa_pools"), 7 * p8.kv_page_bytes(), "the plan's pages are the profile's");
    }

    #[test]
    fn state_and_records_by_hand() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        let c = real();
        // delta-rule state: 64 heads x 128 x 128 x 4 B = 4 MiB per layer and buffer; 34 layers = 136 MiB per buffer
        assert_eq!(c.kda_delta_state_bytes_per_layer(), 4 << 20);
        assert_eq!(34 * c.kda_delta_state_bytes_per_layer(), 136 << 20);
        // conv history: (4 - 1) taps x 3 (q, k, v) x 64 x 128 channels x 2 B = 147,456 per layer
        assert_eq!(c.kda_conv_history_bytes_per_layer(), 147_456);
        // a record: the raw bf16 q|k|v row 3 x 8,192 x 2 = 49,152 + decay fp32 8,192 x 4 = 32,768 + beta fp32 64 x 4 = 256 = 82,176;
        // over 34 layers 2,793,984 B: the "about 2.7 MB" of docs/GLM53.md (2.66 MiB). The conv shift reads the same raw rows.
        assert_eq!(c.kda_record_bytes_per_row_per_layer(), 82_176);
        assert_eq!(34 * 82_176, 2_793_984);

        let p = memory_profile(&c, &opts()).unwrap();
        assert_eq!(p.recurrent_layers(), 34);
        // a slot: 34 x (two state buffers 8 MiB + the conv history once + the parity word) + 11 rings of 256 x 512 B + the block-table row
        // (1,024 pages x 4 B at 262,144 tokens)
        let layer = 2 * (4 << 20) + 147_456 + 4;
        assert_eq!(layer, gb_cuda::kda::Dims::glm().slot_bytes());
        assert_eq!(p.state_bytes_per_seq(), 34 * layer as u64 + 11 * 256 * 512 + 4096);
        assert_eq!(p.state_bytes_per_seq(), 291_672_200);
        assert_eq!(p.state_bytes_per_slot(), p.state_bytes_per_seq());
        assert_eq!(p.window_record_bytes_per_row(), 2_793_984);
        assert_eq!(p.window_record_bytes(), 64 * 2_793_984);
        assert_eq!(p.drafter_state_bytes_per_slot(), 0, "no drafter, no drafter state");
    }

    #[test]
    fn scratch_is_the_sum_of_what_the_backend_allocates() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        let c = real();
        let p = memory_profile(&c, &opts()).unwrap();
        let items = AllocPlan::scratch_items(&c, 128, 262_144, false).unwrap();
        assert_eq!(p.scratch_bytes(), items.iter().map(|i| i.bytes).sum::<u64>());
        // under a gigabyte of scratch at 128 rows (the logits alone are 38 MB, the MoE scratch 100 MiB, the MLA workspace 129 MiB)
        assert!(p.scratch_bytes() < GIB, "{} MiB", p.scratch_bytes() >> 20);
    }

    /// The server's default `--prefill-chunk` (4,096, shared by every family) is this family's default chunk; any other value is taken as asked
    /// (128..=2,048), and the scratch of a wider chunk is priced: the items that scale with a chunk's rows at the chunk, the large-row path's own.
    #[test]
    fn the_chunk_is_priced_and_the_server_default_means_the_family_default() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        let c = real();
        let at = |rows: usize| {
            memory_profile(&c, &engine_options(&MemoryOptions { prefill_rows: rows, ..MemoryOptions::serving_defaults() }))
                .unwrap()
        };
        let generic = memory_profile(&c, &MemoryOptions::serving_defaults()).unwrap();
        // the family's default chunk and its widest, by the process's contract: P1 512 / 2,048; P2 (the default since 2026-10-19) 4,096 / 4,096
        let (family_default, top) = match crate::families::glm::contract::current() {
            crate::families::glm::contract::PrefillContract::P1 => (alloc::DEFAULT_PREFILL_CHUNK, alloc::MAX_PREFILL_CHUNK),
            crate::families::glm::contract::PrefillContract::P2 => (alloc::P2_MAX_PREFILL_CHUNK, alloc::P2_MAX_PREFILL_CHUNK),
        };
        assert_eq!(
            generic.scratch_bytes(),
            at(family_default).scratch_bytes(),
            "4,096 is the server default: the family's default chunk under the contract ({family_default})"
        );
        assert_eq!(at(4096).scratch_bytes(), at(family_default).scratch_bytes());
        // asked for explicitly: clamped to 128..=the widest
        assert_eq!(at(1).scratch_bytes(), at(128).scratch_bytes());
        assert_eq!(at(top * 3 / 2).scratch_bytes(), at(top).scratch_bytes());
        // each wider chunk costs more, by the arithmetic of the plan
        assert!(at(128).scratch_bytes() < at(512).scratch_bytes() && at(512).scratch_bytes() < at(1024).scratch_bytes());
        assert!(at(1024).scratch_bytes() < at(2048).scratch_bytes());
        for rows in [128usize, 512, 1024, 2048] {
            let items = AllocPlan::scratch_items_chunked(&c, 128, rows, 262_144, false).unwrap();
            assert_eq!(at(rows).scratch_bytes(), items.iter().map(|i| i.bytes).sum::<u64>(), "chunk {rows}");
        }
        assert!(at(2048).scratch_bytes() < 3 * GIB, "{} MiB", at(2048).scratch_bytes() >> 20);
    }

    #[test]
    fn the_plan_and_the_profile_agree_on_every_total() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        let c = real();
        for mtp in [false, true] {
            let o = MemoryOptions { mtp, ..opts() };
            let p = memory_profile(&c, &o).unwrap();
            let (slots, blocks) = (8usize, 1032usize);
            let a = AllocPlan::new(
                &c,
                &alloc::AllocParams {
                    slots,
                    kv_blocks: blocks,
                    max_context: o.max_context,
                    max_rows: alloc::MAX_ROWS,
                    window_rows: o.window_rows,
                    mtp,
                },
            )
            .unwrap();
            // per slot, per page, per window row and for scratch the profile's numbers are the plan's items
            assert_eq!(a.sum_of(&alloc::PER_SLOT), slots as u64 * p.state_bytes_per_slot(), "mtp {mtp}");
            assert_eq!(a.sum_of(&alloc::PER_PAGE), blocks as u64 * p.kv_page_bytes(), "mtp {mtp}");
            assert_eq!(a.sum_of(&alloc::WINDOW), p.window_record_bytes(), "mtp {mtp}");
            let scratch: u64 = AllocPlan::scratch_items(&c, 128, o.max_context, mtp).unwrap().iter().map(|i| i.bytes).sum();
            assert_eq!(
                a.total(),
                slots as u64 * p.state_bytes_per_slot() + blocks as u64 * p.kv_page_bytes() + p.window_record_bytes() + scratch,
                "mtp {mtp}"
            );
            assert_eq!(scratch, p.scratch_bytes(), "mtp {mtp}");
        }
    }

    #[test]
    fn the_nextn_drafter_is_charged_to_admission_per_token_per_slot_and_in_the_scratch() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        let c = real();
        let off = memory_profile(&c, &opts()).unwrap();
        let on = memory_profile(&c, &MemoryOptions { mtp: true, ..opts() }).unwrap();
        // per token: a 12th layer of latent and pooled key; per page the same
        assert_eq!(on.kv_bytes_per_token() - off.kv_bytes_per_token(), 1024 + 64);
        assert_eq!(on.kv_page_bytes() - off.kv_page_bytes(), 278_528);
        // per slot: the 12th ring (256 raw rows of 512 B) and the tap store (32 fp16 rows of 4,096), on top of the target's unchanged state
        assert_eq!(off.drafter_state_bytes_per_slot(), 0);
        assert_eq!(on.drafter_state_bytes_per_slot(), 256 * 512 + 32 * 4096 * 2);
        assert_eq!(on.state_bytes_per_seq(), off.state_bytes_per_seq());
        assert_eq!(on.state_bytes_per_slot(), off.state_bytes_per_slot() + 131_072 + 262_144);
        // scratch: the drafter's buffers at 128 rows, a few MiB; the window records do not move
        assert!(on.scratch_bytes() > off.scratch_bytes() && on.scratch_bytes() - off.scratch_bytes() < 12 << 20);
        assert_eq!(on.window_record_bytes(), off.window_record_bytes());
        // what the server's defaults charge in all: 262,144 tokens of 1,088 B, 8 slots of 384 KiB, 8 MiB of scratch
        let extra = |p: &MemoryProfile| p.kv_bytes_per_token() * 262_144 + 8 * p.state_bytes_per_slot() + p.scratch_bytes();
        assert!(extra(&on) - extra(&off) < 300 << 20, "{} MiB", (extra(&on) - extra(&off)) >> 20);
    }

    #[test]
    fn the_reserve_is_the_blanket_with_room_to_spare() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        let p = memory_profile(&real(), &opts()).unwrap();
        assert_eq!(p.engine_reserve_bytes(), 5 * GIB);
        assert!(p.scratch_bytes() + p.window_record_bytes() < 5 * GIB / 2, "the modelled buffers are about 1 GiB");
    }

    #[test]
    fn vision_reserve_has_two_distinct_terms() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        let p = memory_profile(&with_processor(), &opts()).unwrap();
        let v = p.vision().unwrap();
        // tower weights: the pack's 347 plain bf16 tensors in the plan's one region (1,127,254,016 bytes of data, each tensor on a 256-byte boundary)
        assert_eq!(v.weight_bytes, Some(1_127_260_160));
        // 4096 tokens x 4096 x 4 B = 16 KiB per token: the features are the merger's fp32 output, spliced as it is
        assert_eq!(v.feature_bytes_per_image, 4096 * 4096 * 4);
        // 4096 tokens = 16,384 patches of 33,952 B each (patches 1,184 + residual 4,096 + normed 2,048 + qkv 6,144 + attention 2,048 + sublayer
        // 2,048 + gate|up 16,384), the rope's 64 B of frequencies, a 4 B counter and the 32 MiB cuBLAS workspace: what the tower allocates
        assert_eq!(v.encoder_scratch_bytes, 16_384 * 33_952 + 64 + 4 + (32 << 20));
        assert_eq!(v.encoder_scratch_bytes, 589_824_068);
        // below the Qwen tower's 0.85 GiB at the same cap (head dim 64, no EXL3 prefill scratch)
        assert!((v.encoder_scratch_bytes as f64 / GIB as f64 - 0.55).abs() < 0.01);
        // 8 slots x 4 images of features, and one encoder's scratch: 32 x 64 MiB + 562 MiB
        let reserve = p.vision_reserve_bytes(ENCODERS_IN_FLIGHT, VisionOptions::SERVER_DEFAULT.concurrent_images(8)).unwrap();
        assert_eq!(reserve, 589_824_068 + 32 * 67_108_864);
        assert!((reserve as f64 / GIB as f64 - 2.55).abs() < 0.01);
    }

    #[test]
    fn the_processors_image_cap_bounds_the_vision_reserve() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        let big =
            MemoryOptions { vision: Some(VisionOptions { max_image_tokens: 100_000, max_images_per_request: 1 }), ..opts() };
        let v = memory_profile(&with_processor(), &big).unwrap().vision().unwrap();
        assert_eq!(v.feature_bytes_per_image, 8000 * 4096 * 4, "processor_config.json: max_image_tokens 8000");
        // 8000 tokens = 32,000 patches
        assert_eq!(v.encoder_scratch_bytes, 1_120_018_500);
        // without a processor config the server's limit is all there is to go on
        let v = memory_profile(&real(), &big).unwrap().vision().unwrap();
        assert_eq!(v.feature_bytes_per_image, 100_000 * 4096 * 4);
    }

    #[test]
    fn text_only_serving_reserves_nothing_for_vision() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        let p = memory_profile(&real(), &MemoryOptions { vision: None, ..opts() }).unwrap();
        assert!(p.vision().is_none());
        assert_eq!(p.vision_reserve_bytes(1, 32), Ok(0));
    }

    #[test]
    fn mtp_speculation_needs_an_mtp_layer() {
        if !gb_cuda::rt::kernels_built() {
            eprintln!("not run: this build has no CUDA kernels (no nvcc), so the kernels' workspace sizes are not known");
            return;
        }
        let mut v: Value = serde_json::from_str(GLM_CONFIG).unwrap();
        v["text_config"]["num_nextn_predict_layers"] = serde_json::json!(0);
        let c = GlmConfig::from_json(&v).unwrap();
        assert!(memory_profile(&c, &opts()).is_ok());
        let e = memory_profile(&c, &MemoryOptions { mtp: true, ..opts() }).unwrap_err();
        assert!(matches!(e, MemoryError::Inconsistent(m) if m.contains("no NextN layer")));
    }

    #[test]
    fn a_zero_kv_cost_is_refused_not_admitted_free() {
        // the fail-closed rules still hold for the real model: no plane may have a zero extent, and a profile that prices a token at
        // nothing cannot be built (MemoryProfile::new), whatever the config says
        let mut c = real();
        c.mla.kv_lora_rank = 0;
        assert!(matches!(memory_profile(&c, &opts()), Err(MemoryError::Zero(_))), "a latent of zero width");
        let mut c = real();
        c.indexer.head_dim = 0;
        assert!(matches!(memory_profile(&c, &opts()), Err(MemoryError::Zero(_))), "a pooled key of zero width");
        // with no attention layer at all the layout is empty, which the profile refuses too
        let mut c = real();
        c.layers.retain(|l| !matches!(l.mixer, crate::family::Mixer::MlaDsa { .. }));
        assert!(memory_profile(&c, &opts()).is_err(), "no layer caches context: refused");
    }
}
