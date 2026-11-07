//! The weight map and the load plan, tested on the real pack's tensor inventory (`testdata/glm53/tensor_inventory.tsv`, checked against
//! the real pack by `tests/glm_pack.rs`) with the real config, and on edited copies of it: every refusal below is a mutation that the
//! mapper must name.

use super::bind::{LayoutError, Problem};
use super::config::GlmConfig;
use super::inventory::GroupInfo;
use super::inventory::PackInventory;
use super::plan::{bytes_of, ItemPart, RegionKind, MARKER_BYTES};
use super::weights::*;
use gb_exl3::budget::{categorize, Bytes, Category};
use gb_exl3::Bits;
use std::sync::OnceLock;

const TABLE: &str = include_str!("../../../../../testdata/glm53/tensor_inventory.tsv");
const CONFIG: &str = include_str!("../../../../../testdata/glm53/config.json");
const LAYER_PREFIX: &str = "model.language_model.layers.";

fn config() -> GlmConfig {
    GlmConfig::from_json(&serde_json::from_str(CONFIG).unwrap()).unwrap()
}

fn inventory() -> PackInventory {
    static INV: OnceLock<PackInventory> = OnceLock::new();
    INV.get_or_init(|| PackInventory::from_table(TABLE).unwrap()).clone()
}

fn weights() -> &'static GlmWeights {
    static W: OnceLock<GlmWeights> = OnceLock::new();
    W.get_or_init(|| GlmWeights::map(&inventory(), &config()).unwrap())
}

fn layer_name(i: usize, rest: &str) -> String {
    format!("{LAYER_PREFIX}{i}.{rest}")
}

/// The problems of mapping an edited inventory (it must not map).
fn refused(inv: &PackInventory) -> Vec<Problem> {
    GlmWeights::map(inv, &config()).expect_err("this mutation must be refused").problems
}

// ----------------------------------------------------------------------------------------------------------- the table

#[test]
fn the_committed_table_is_the_pack_inventory() {
    let inv = inventory();
    assert_eq!((inv.groups.len(), inv.plain.len()), (37_420, 1_214));
    assert_eq!(inv.tensor_count(), 150_894);
    assert!(inv.refused.is_empty(), "the scanner's own group parser accepts every row: {:?}", inv.refused.iter().next());
}

#[test]
fn a_malformed_table_is_an_error() {
    let ok = "experts 2\nlm_head.suh\tF16\t4\t-\n";
    assert!(PackInventory::from_table(ok).is_ok());
    for (bad, why) in [
        ("lm_head.suh\tF16\t4\n", "columns"),
        ("lm_head.suh\tF16\tx\t-\n", "bad shape"),
        ("lm_head.suh\tF16\t4\t-\nlm_head.suh\tF16\t4\t-\n", "twice"),
        ("a.{E}.suh\tF16\t4\t-\n", "before the `experts` line"),
        ("lm_head.suh\tF99\t4\t-\n", "unknown dtype"),
        ("a.{L}.w\tF16\t4\t3-x\n", "bad layers"),
    ] {
        let e = PackInventory::from_table(bad).expect_err(why);
        assert!(e.contains(why), "{e}");
    }
}

// ----------------------------------------------------------------------------------------------------------- the mapping

#[test]
fn every_tensor_of_the_real_inventory_lands_in_exactly_one_slot() {
    let w = weights();
    assert_eq!(w.counts(), (37_420, 1_214), "every group and every plain tensor is claimed once");
    // and the claims are by name: the leaves' names are the inventory's, with no repeats
    let mut names = std::collections::BTreeSet::new();
    w.visit(&mut |l| {
        let n = match l {
            Leaf::Plain(t) => t.name.clone(),
            Leaf::Exl3(g) => g.prefix.clone(),
        };
        assert!(names.insert(n.clone()), "{n} is in two slots");
    });
    assert_eq!(names.len(), 37_420 + 1_214);
}

#[test]
fn the_layer_recipe_decides_the_slots() {
    let w = weights();
    assert_eq!((w.layers.len(), w.mtp.len()), (45, 1));
    for (i, l) in w.layers.iter().enumerate() {
        assert_eq!(l.index, i);
        assert!(l.mhc.is_some(), "layer {i}: every trunk layer has hyper-connection tensors");
        let kda = i % 4 != 3;
        assert_eq!(matches!(l.mixer, MixerWeights::Kda(_)), kda, "layer {i}");
        assert_eq!(matches!(l.ffn, FfnWeights::Dense(_)), i < 3, "layer {i}");
        if let MixerWeights::Mla(m) = &l.mixer {
            assert!(m.indexer.as_ref().is_some_and(|ix| ix.kpool.is_some()), "layer {i}: MLA layers carry their own indexer");
        }
    }
    let m = &w.mtp[0];
    assert_eq!(m.layer.index, 45);
    assert!(m.layer.mhc.is_none(), "the MTP layer has a plain residual: no hc tensors");
    assert!(matches!(m.layer.mixer, MixerWeights::Mla(_)) && matches!(m.layer.ffn, FfnWeights::Moe(_)));
    assert_eq!((m.eh_proj.k, m.eh_proj.n), (8192, 4096));
}

#[test]
fn shapes_and_dtypes_come_from_the_config_and_match_the_pack() {
    let w = weights();
    let FfnWeights::Moe(moe) = &w.layers[3].ffn else { panic!("layer 3 is MoE") };
    assert_eq!((moe.router.dtype, moe.router.shape.clone()), (PlainDtype::F16, vec![288, 4096]));
    assert_eq!((moe.e_score_correction_bias.dtype, moe.e_score_correction_bias.shape.clone()), (PlainDtype::F32, vec![288]));
    let MixerWeights::Mla(mla) = &w.layers[3].mixer else { panic!("layer 3 is MLA") };
    // plain fp16 (not bf16), and not EXL3
    assert_eq!((mla.kv_b_proj.dtype, mla.kv_b_proj.shape.clone()), (PlainDtype::F16, vec![32768, 512]));
    assert_eq!((mla.q_b_proj.k, mla.q_b_proj.n, mla.o_proj.k), (1536, 16384, 16384));
    let ix = mla.indexer.as_ref().unwrap();
    let wq_b = ix.wq_b.exl3().expect("the 2.25bpw pack's wq_b is EXL3");
    assert_eq!((wq_b.k, wq_b.n), (1536, 4096));
    assert_eq!((ix.k_norm_bias.dtype, ix.k_norm_bias.shape.clone()), (PlainDtype::F16, vec![128]));
    assert_eq!(ix.kpool.as_ref().unwrap().compress_ape.shape, vec![4, 128]);
    let MixerWeights::Kda(kda) = &w.layers[0].mixer else { panic!("layer 0 is KDA") };
    assert_eq!((kda.qkv_proj.k, kda.qkv_proj.n, kda.o_proj.k), (4096, 24576, 8192));
    assert_eq!(kda.conv1d_weight.shape, vec![24576, 1, 4]);
    assert_eq!((kda.a_log.shape.clone(), kda.dt_bias.shape.clone()), (vec![64], vec![8192]), "per head and per channel");
    assert_eq!(kda.f_a_proj.shape, vec![128, 4096]);
    let hc = &w.layers[0].mhc.as_ref().unwrap().attn;
    assert_eq!((hc.fn_.shape.clone(), hc.base.shape.clone(), hc.scale.shape.clone()), (vec![24, 16384], vec![24], vec![3]));
    let FfnWeights::Dense(d) = &w.layers[0].ffn else { panic!("layer 0 is dense") };
    assert_eq!((d.gate_proj.k, d.gate_proj.n, d.down_proj.k), (4096, 12288, 12288));
    assert_eq!((w.lm_head.k, w.lm_head.n, w.lm_head.bits.as_f64()), (4096, 154_880, 6.0));
    assert_eq!(w.embed_tokens.shape, vec![154_880, 4096]);
}

#[test]
fn experts_are_slotted_by_id_not_by_file_order() {
    let w = weights();
    let inv = inventory();
    // the shards list experts as 0, 1, 10, 100, 101, ...: slot 100 must hold expert 100
    let FfnWeights::Moe(moe) = &w.layers[7].ffn else { panic!("layer 7 is MoE") };
    let bank = &moe.experts;
    assert_eq!(bank.experts.len(), 288);
    for e in [0usize, 1, 9, 10, 99, 100, 101, 287] {
        let x = &bank.experts[e];
        for (proj, g, lay) in [
            ("gate_proj", &x.gate_proj, &bank.layout.gate),
            ("up_proj", &x.up_proj, &bank.layout.up),
            ("down_proj", &x.down_proj, &bank.layout.down),
        ] {
            let prefix = layer_name(7, &format!("mlp.experts.{e}.{proj}"));
            assert_eq!(g.prefix, prefix);
            let src = &inv.groups[&prefix];
            assert_eq!(
                (g.trellis.src.offset, g.suh.src.offset, g.svh.src.offset),
                (src.trellis.offset, src.in_scale.offset, src.out_scale.offset)
            );
            // destinations follow the id: expert e at e x stride, scales stacked
            assert_eq!(g.trellis.dst, Dst { region: bank.region, offset: lay.trellis + e as u64 * g.trellis.bytes });
            assert_eq!(g.suh.dst.offset, lay.suh + e as u64 * g.suh.bytes);
            assert_eq!(g.svh.dst.offset, lay.svh + e as u64 * g.svh.bytes);
        }
    }
    // the layout of a 2-bit bank: gate/up trellises 2 MiB each, scales stacked [288, K] and [288, N]
    let (g, d) = (&bank.experts[0].gate_proj, &bank.experts[0].down_proj);
    assert_eq!((bank.bits_gate_up.as_f64(), bank.bits_down.as_f64()), (2.0, 2.0));
    assert_eq!((g.trellis.bytes, g.suh.bytes, g.svh.bytes), (2 << 20, 8192, 4096));
    assert_eq!((d.trellis.bytes, d.suh.bytes, d.svh.bytes), (2 << 20, 4096, 8192));
    assert_eq!(bank.layout.gate.trellis_stride, 2 << 20);
    assert_eq!(bank.layout.gate.suh_stride, 8192);
}

#[test]
fn bits_come_from_the_trellis_shapes() {
    let w = weights();
    let three_bit = [3usize, 4, 5, 14, 24, 33, 39, 40, 41, 42, 43, 44];
    for b in w.banks() {
        let want = match b.layer {
            45 => 4.0,
            l if three_bit.contains(&l) => 3.0,
            _ => 2.0,
        };
        assert_eq!((b.bits_gate_up.as_f64(), b.bits_down.as_f64()), (want, want), "layer {}", b.layer);
    }
    assert_eq!(w.banks().count(), 43);
}

// ------------------------------------------------------------------------------------------------------------ fail closed

#[test]
fn a_missing_tensor_is_named() {
    let mut inv = inventory();
    let name = layer_name(17, "hc_ffn_fn");
    inv.plain.remove(&name).unwrap();
    assert_eq!(refused(&inv), vec![Problem::Missing { name, group: false }]);
}

#[test]
fn a_missing_group_is_named_and_its_bank_reports_only_that() {
    let mut inv = inventory();
    let prefix = layer_name(20, "mlp.experts.200.up_proj");
    inv.groups.remove(&prefix).unwrap();
    assert_eq!(refused(&inv), vec![Problem::Missing { name: prefix, group: true }]);
}

#[test]
fn a_tensor_no_slot_takes_is_an_error() {
    // a KDA tensor on an MLA layer, a tensor nobody has heard of, and an extra expert id
    let mut inv = inventory();
    let mut dup = inv.plain[&layer_name(2, "self_attn.A_log")].clone();
    let kda_on_mla = layer_name(3, "self_attn.A_log");
    inv.plain.insert(kda_on_mla.clone(), dup.clone());
    dup.shape = vec![1];
    inv.plain.insert("model.language_model.mystery.weight".into(), dup);
    let extra = layer_name(5, "mlp.experts.288.gate_proj");
    let g = inv.groups[&layer_name(5, "mlp.experts.287.gate_proj")].clone();
    inv.groups.insert(extra.clone(), g);
    let p = refused(&inv);
    assert!(p.contains(&Problem::Unexpected { name: kda_on_mla, group: false }), "{p:?}");
    assert!(p.contains(&Problem::Unexpected { name: "model.language_model.mystery.weight".into(), group: false }), "{p:?}");
    assert!(p.contains(&Problem::Unexpected { name: extra, group: true }), "{p:?}");
    assert_eq!(p.len(), 3);
}

#[test]
fn a_wrong_dtype_or_shape_is_named() {
    let mut inv = inventory();
    // kv_b_proj is fp16 in this pack: bf16 is refused, not accepted as "the same thing"
    let kv = layer_name(11, "self_attn.kv_b_proj.weight");
    inv.plain.get_mut(&kv).unwrap().dtype = "BF16".into();
    let hc = layer_name(30, "hc_attn_fn");
    inv.plain.get_mut(&hc).unwrap().shape = vec![24, 8192];
    let p = refused(&inv);
    assert!(p.contains(&Problem::Dtype { name: kv, want: "F16", got: "BF16".into() }), "{p:?}");
    assert!(p.contains(&Problem::Shape { name: hc, want: vec![24, 16384], got: vec![24, 8192] }), "{p:?}");
    assert_eq!(p.len(), 2);
}

#[test]
fn a_group_of_the_wrong_shape_is_named() {
    let mut inv = inventory();
    let prefix = layer_name(9, "self_attn.o_proj");
    inv.groups.get_mut(&prefix).unwrap().k = 4096;
    assert_eq!(refused(&inv), vec![Problem::GroupShape { prefix, want: (8192, 4096), got: (4096, 4096) }]);
}

#[test]
fn groups_the_loader_cannot_take_are_refused_not_converted() {
    for (what, edit) in [
        ("fractional", Box::new(|g: &mut GroupInfo| g.bits = Bits::from_twice(5).unwrap()) as Box<dyn Fn(&mut GroupInfo)>),
        ("packed signs", Box::new(|g: &mut GroupInfo| g.in_scales = "su")),
        ("bias", Box::new(|g: &mut GroupInfo| g.bias = Some(g.in_scale.clone()))),
    ] {
        let mut inv = inventory();
        let prefix = layer_name(1, "self_attn.qkv_proj");
        edit(inv.groups.get_mut(&prefix).unwrap());
        let p = refused(&inv);
        assert!(matches!(&p[..], [Problem::Unsupported { prefix: q, .. }] if *q == prefix), "{what}: {p:?}");
    }
}

#[test]
fn a_refused_group_and_a_header_issue_stop_the_mapping() {
    let mut inv = inventory();
    inv.refused.insert("model.language_model.layers.4.mlp.experts.1.up_proj".into(), "no .trellis".into());
    // the duplicate-resolved-by-the-index note is benign; any other issue is not
    inv.issues.push("tensor x is also stored in a; using b as the index says".into());
    inv.issues.push("tensor y: data_offsets end 5 is before start 9".into());
    let p = refused(&inv);
    assert!(matches!(&p[0], Problem::Refused { .. }), "{p:?}");
    assert!(p.contains(&Problem::HeaderIssue("tensor y: data_offsets end 5 is before start 9".into())), "{p:?}");
    assert_eq!(p.len(), 2, "the benign note is not a problem: {p:?}");
}

#[test]
fn a_mixed_width_bank_maps_into_subbanks_and_a_bad_one_is_refused() {
    use gb_cuda::moe::SubbankKey;
    let key = |g: u8, u: u8, d: u8| SubbankKey { bits_gate: g, bits_up: u, bits_down: d, codebook: gb_exl3::Codebook::Mul1 };
    // set a group's width the way a pack stores it: bits from the trellis' last dimension, the trellis' bytes follow
    let set_bits = |inv: &mut PackInventory, prefix: &str, bits: u8| {
        let g = inv.groups.get_mut(prefix).unwrap();
        g.trellis.bytes = (g.k / 16 * g.n / 16 * 16 * bits as usize * 2) as u64;
        g.bits = Bits::from_twice(2 * bits).unwrap();
    };
    // one expert of layer 20 (2-bit) with a 3-bit down, one with a 3-bit up (gate != up), two 4-bit experts
    let mut inv = inventory();
    set_bits(&mut inv, &layer_name(20, "mlp.experts.7.down_proj"), 3);
    set_bits(&mut inv, &layer_name(20, "mlp.experts.9.up_proj"), 3);
    for e in [100, 3] {
        for p in ["gate_proj", "up_proj", "down_proj"] {
            set_bits(&mut inv, &layer_name(20, &format!("mlp.experts.{e}.{p}")), 4);
        }
    }
    let w = GlmWeights::map(&inv, &config()).unwrap();
    let FfnWeights::Moe(moe) = &w.layers[20].ffn else { panic!("layer 20 is MoE") };
    let bank = &moe.experts;
    assert!(!bank.is_single());
    let got: Vec<(SubbankKey, Vec<u32>)> = bank.subbanks.iter().map(|s| (s.key, s.ids.clone())).collect();
    let rest: Vec<u32> = (0..288).filter(|e| ![3, 7, 9, 100].contains(e)).collect();
    assert_eq!(
        got,
        vec![(key(2, 2, 2), rest), (key(2, 2, 3), vec![7]), (key(2, 3, 2), vec![9]), (key(4, 4, 4), vec![3, 100])],
        "subbanks in key order, ids ascending"
    );
    // every placed byte inside the region, no two blocks overlapping, every expert placed in its subbank's slot
    let region = w.regions[bank.region as usize].bytes;
    let mut spans: Vec<(u64, u64)> = Vec::new();
    for sb in &bank.subbanks {
        for (local, &g) in sb.ids.iter().enumerate() {
            let x = &bank.experts[g as usize];
            for (gr, lay) in [(&x.gate_proj, &sb.layout.gate), (&x.up_proj, &sb.layout.up), (&x.down_proj, &sb.layout.down)] {
                assert_eq!(gr.trellis.dst, Dst { region: bank.region, offset: lay.trellis_at(local) });
                assert_eq!((gr.suh.dst.offset, gr.svh.dst.offset), (lay.suh_at(local), lay.svh_at(local)));
                for part in [&gr.trellis, &gr.suh, &gr.svh] {
                    spans.push((part.dst.offset, part.dst.offset + part.bytes));
                }
            }
        }
    }
    spans.sort();
    assert!(spans.windows(2).all(|w| w[0].1 <= w[1].0) && spans.last().unwrap().1 <= region, "no overlap, inside the region");
    // the mapper's layout is `SubbankRegion::stacked` (what the per-rank byte arithmetic uses), subbank after subbank
    let mut cursor = 0usize;
    for sb in &bank.subbanks {
        let (lay, end) = gb_cuda::moe::SubbankRegion::stacked(sb.key, sb.ids.len(), 4096, 2048, cursor, ALIGN as usize);
        for (a, b) in [(&lay.gate, &sb.layout.gate), (&lay.up, &sb.layout.up), (&lay.down, &sb.layout.down)] {
            assert_eq!(
                (a.trellis as u64, a.trellis_stride as u64, a.suh as u64, a.svh as u64),
                (b.trellis, b.trellis_stride, b.suh, b.svh),
                "{:?}",
                sb.key
            );
        }
        cursor = end;
    }
    assert_eq!(align_up(cursor as u64), region);
    // the other layers are untouched: one subbank, the single bank
    for b in w.banks().filter(|b| b.layer != 20) {
        assert!(b.is_single(), "layer {}", b.layer);
    }
    // (no load plan here: the edited widths grow trellises the inventory's source offsets do not move, so the shard reads overlap)
    // a projection of an expert in another codebook is refused (one expert, one codebook)
    let mut inv = inventory();
    inv.groups.get_mut(&layer_name(21, "mlp.experts.3.gate_proj")).unwrap().codebook = gb_exl3::Codebook::Mcg;
    let p = refused(&inv);
    assert!(matches!(&p[..], [Problem::NonUniform { layer: 21, what }] if what.contains("codebooks")), "{p:?}");
    // a member of a subbank whose bytes differ from the subbank's first member's is refused (the strides are constants)
    let mut inv = inventory();
    inv.groups.get_mut(&layer_name(20, "mlp.experts.11.down_proj")).unwrap().out_scale.bytes += 2;
    let p = refused(&inv);
    assert!(matches!(&p[..], [Problem::NonUniform { layer: 20, what }] if what.contains("expert 11 down_proj")), "{p:?}");
}

#[test]
fn a_rank_places_and_plans_only_its_owned_experts() {
    use gb_cuda::moe::ExpertOwnership;
    let whole = plan();
    let mut sum_routed = 0u64;
    let mut seen = std::collections::BTreeMap::<String, usize>::new();
    for r in 0..2 {
        let own = ExpertOwnership::range(288, r, 2).unwrap();
        let w = GlmWeights::map_owned(&inventory(), &config(), &own).unwrap();
        let p = w.load_plan().unwrap();
        for b in w.banks() {
            assert_eq!(b.subbanks.len(), 1, "a uniform layer is one subbank on a rank too");
            assert_eq!(b.subbanks[0].ids, own.owned(), "layer {}", b.layer);
            assert!(!b.is_single());
            for e in 0..288 {
                let placed = b.experts[e].down_proj.trellis.dst != Dst::NONE;
                assert_eq!(placed, own.owns(e), "layer {} expert {e}", b.layer);
            }
        }
        // every routed item is an owned expert's, every non-routed item is the single node's
        // routed experts by name: the NextN layer's are categorised `Mtp`
        let routed = |i: &super::plan::LoadItem| i.name.contains(".mlp.experts.");
        for it in &p.items {
            if routed(it) {
                let e: usize = it.name.split(".experts.").nth(1).unwrap().split('.').next().unwrap().parse().unwrap();
                assert!(own.owns(e), "{}", it.name);
                sum_routed += it.bytes;
                *seen.entry(it.name.clone()).or_default() += 1;
            }
        }
        let non_routed = |p: &super::plan::LoadPlan| -> Vec<(String, u64)> {
            p.items.iter().filter(|i| !routed(i)).map(|i| (i.name.clone(), i.bytes)).collect()
        };
        assert_eq!(non_routed(&p), non_routed(whole), "rank {r}: the replicated items are the single node's");
        // the expert region of a rank is about half the single node's
        for (a, b) in p.regions.iter().zip(&whole.regions) {
            if matches!(a.kind, RegionKind::Experts(_)) {
                assert!(a.bytes * 2 <= b.bytes + 8 * ALIGN && a.bytes * 2 + 8 * ALIGN >= b.bytes, "{:?}", a.kind);
            } else {
                assert_eq!(a, b);
            }
        }
    }
    // the two ranks read every routed item of the single node exactly once
    let whole_routed: Vec<&super::plan::LoadItem> = whole.items.iter().filter(|i| i.name.contains(".mlp.experts.")).collect();
    assert_eq!(sum_routed, whole_routed.iter().map(|i| i.bytes).sum::<u64>());
    assert_eq!(seen.len(), whole_routed.len());
    assert!(seen.values().all(|&n| n == 1));
}

/// Lane E, the rank strips of the KDA head split (`KdaConfig::rank`): a rank maps every tensor as the single node does but places and plans
/// only its heads' `qkv_proj` strips (3 ranges a k-tile row, `svh` 3 ranges, `suh` whole). The two ranks' runs read every byte of every
/// KDA qkv exactly once, each rank's runs fill its cut contiguously, every other item is the single node's, and each KDA layer's trunk
/// region shrinks by exactly the half it does not hold; the rank's state follows its heads.
#[test]
fn a_kda_split_rank_places_and_plans_only_its_qkv_strips() {
    let whole_w = weights();
    let whole = plan();
    let qkv = |n: &str| n.ends_with(".self_attn.qkv_proj.trellis") || n.ends_with(".self_attn.qkv_proj.svh");
    // per stored part (name), the source byte ranges both ranks read
    let mut read: std::collections::BTreeMap<String, Vec<(u64, u64)>> = Default::default();
    for r in 0..2 {
        let cfg = config().with_kda_rank(r, 2).unwrap();
        assert_eq!(cfg.kda.rank, Some((32 * r, 32 * (r + 1))));
        assert!(cfg.with_kda_rank(r, 2).is_err(), "a rank's configuration is not split again");
        let w = GlmWeights::map(&inventory(), &cfg).unwrap();
        assert_eq!(w.counts(), whole_w.counts(), "rank {r}: every tensor is still claimed");
        let p = w.load_plan().unwrap();
        let mut saved = std::collections::BTreeMap::<u32, u64>::new();
        for l in cfg.kda_layers() {
            let (MixerWeights::Kda(a), MixerWeights::Kda(b)) = (&w.layer(l).unwrap().mixer, &whole_w.layer(l).unwrap().mixer)
            else {
                panic!("layer {l} is a KDA layer")
            };
            let (g, f) = (&a.qkv_proj, &b.qkv_proj);
            let ranges = kda_qkv_strips(64, 128, 32 * r, 32 * (r + 1));
            assert_eq!(ranges, vec![(32 * r, 32 * r + 32), (64 + 32 * r, 96 + 32 * r), (128 + 32 * r, 160 + 32 * r)]);
            assert_eq!(g.cut.as_ref().map(|c| &c.ranges), Some(&ranges), "layer {l}");
            assert_eq!((g.k, g.n, g.placed_n()), (4096, 24576, 12288));
            assert_eq!((g.trellis.bytes * 2, g.svh.bytes * 2, g.suh.bytes), (f.trellis.bytes, f.svh.bytes, f.suh.bytes));
            assert_eq!((g.trellis.src, g.svh.src, g.suh.src), (f.trellis.src, f.svh.src, f.suh.src));
            assert!(f.cut.is_none() && a.o_proj.cut.is_none());
            *saved.entry(g.trellis.dst.region).or_default() += g.trellis.bytes + g.svh.bytes;
            // the cut's runs: 256 k-tile rows x 3 ranges of the trellis, 3 of svh, each landing right after the previous one
            for (suffix, part, runs) in [("trellis", &g.trellis, 256 * 3), ("svh", &g.svh, 3)] {
                let name = format!("{}.{suffix}", g.prefix);
                let mut items: Vec<_> = p.items.iter().filter(|i| i.name == name).collect();
                assert_eq!(items.len(), runs, "{name}");
                items.sort_by_key(|i| i.dst);
                let mut at = part.dst.offset;
                for i in &items {
                    assert_eq!((i.region, i.dst, i.bytes, i.dst_bytes), (part.dst.region, at, i.bytes, i.bytes), "{name}");
                    at += i.bytes;
                    read.entry(name.clone()).or_default().push((i.offset, i.offset + i.bytes));
                }
                assert_eq!(at, part.dst.offset + part.bytes, "{name}: the runs fill the cut");
            }
        }
        // every other item is the single node's (name, part, source and size)
        let others = |p: &super::plan::LoadPlan| -> Vec<(String, ItemPart, u16, u64, u64)> {
            p.items.iter().filter(|i| !qkv(&i.name)).map(|i| (i.name.clone(), i.part, i.file, i.offset, i.bytes)).collect()
        };
        assert_eq!(others(&p), others(whole), "rank {r}");
        assert_eq!(p.markers, whole.markers, "rank {r}: every codebook marker is still checked");
        for (a, b) in p.regions.iter().zip(&whole.regions) {
            assert_eq!((a.id, a.kind), (b.id, b.kind));
            assert_eq!(
                a.bytes + saved.get(&a.id).copied().unwrap_or(0),
                b.bytes,
                "rank {r} {:?}: the region drops the other half exactly",
                a.kind
            );
        }
        let (tw, tr) = (whole.totals().uploaded_total(), p.totals().uploaded_total());
        let half: u64 = saved.values().sum();
        assert_eq!(tw - tr, half, "rank {r}");
        // 34 layers x (half of 4096 x 24576 at 8 bits + half of svh) on the 2.25bpw pack's qkv width
        eprintln!(
            "rank {r}: {:.3} GiB of qkv strips not placed ({} KDA layers)",
            half as f64 / (1u64 << 30) as f64,
            cfg.kda_layers().len()
        );
        // the rank's state: its 32 heads' pools, records and workspace
        let d = super::alloc::dims_of(&cfg).unwrap();
        assert_eq!(d, gb_cuda::kda::Dims { heads: 32, hidden: 4096 });
        assert_eq!(cfg.kda_delta_state_bytes_per_layer() * 2, config().kda_delta_state_bytes_per_layer());
        assert_eq!(cfg.kda_record_bytes_per_row_per_layer() as usize, d.record_bytes_per_row());
        let pools = |c: &GlmConfig| {
            super::alloc::AllocPlan::slot_items(c, 8192, false).unwrap().iter().find(|i| i.name == "kda_pools").unwrap().bytes
        };
        assert_eq!(pools(&cfg), 34 * d.slot_bytes() as u64);
        assert_eq!(pools(&config()), 34 * gb_cuda::kda::Dims::glm().slot_bytes() as u64);
    }
    // the two ranks together read every stored byte of every KDA qkv trellis and svh exactly once
    let stored: std::collections::BTreeMap<String, (u64, u64)> =
        whole.items.iter().filter(|i| qkv(&i.name)).map(|i| (i.name.clone(), (i.offset, i.offset + i.bytes))).collect();
    assert_eq!(stored.len(), 34 * 2);
    assert_eq!(read.keys().collect::<Vec<_>>(), stored.keys().collect::<Vec<_>>());
    for (name, mut v) in read {
        v.sort();
        let (lo, hi) = stored[&name];
        assert_eq!(v.first().unwrap().0, lo, "{name}");
        assert_eq!(v.last().unwrap().1, hi, "{name}");
        assert!(v.windows(2).all(|w| w[0].1 == w[1].0), "{name}: the two ranks' runs tile the stored part");
    }
    for bad in [(2, 2), (0, 3), (1, 1)] {
        assert!(config().with_kda_rank(bad.0, bad.1).is_err(), "{bad:?}");
    }
}

#[test]
fn the_tower_is_checked_and_absent_when_the_config_has_none() {
    let mut inv = inventory();
    let name = "model.visual.blocks.5.attn.qkv.bias".to_string();
    inv.plain.remove(&name).unwrap();
    assert_eq!(refused(&inv), vec![Problem::Missing { name, group: false }]);
    // a config without a vision section: the 347 tower tensors in the pack are all unexpected
    let mut v: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
    v.as_object_mut().unwrap().remove("vision_config");
    let cfg = GlmConfig::from_json(&v).unwrap();
    let p = GlmWeights::map(&inventory(), &cfg).unwrap_err().problems;
    assert_eq!(p.len(), 347);
    assert!(p.iter().all(|p| matches!(p, Problem::Unexpected { name, .. } if name.starts_with("model.visual."))));
}

#[test]
fn the_mtp_layer_exists_only_when_the_config_says_so() {
    let mut v: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
    v["text_config"]["num_nextn_predict_layers"] = serde_json::json!(0);
    let cfg = GlmConfig::from_json(&v).unwrap();
    let p = GlmWeights::map(&inventory(), &cfg).unwrap_err().problems;
    // every layer-45 tensor and group is unexpected (16 plain, 873 groups)
    assert_eq!(p.len(), 16 + 873, "{}", p.len());
    assert!(p.iter().all(|p| matches!(p, Problem::Unexpected { name, .. } if name.starts_with(&layer_name(45, "")))));
}

#[test]
fn a_shared_indexer_layer_has_none_and_one_is_unexpected() {
    // layer 7 shares layer 3's selection: it must not carry indexer tensors, and the pack's are then unexpected
    let mut v: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
    v["text_config"]["indexer_types"][7] = serde_json::json!("shared");
    let cfg = GlmConfig::from_json(&v).unwrap();
    let p = GlmWeights::map(&inventory(), &cfg).unwrap_err().problems;
    assert_eq!(p.len(), 6 + 1, "six plain indexer tensors (wk, weights_proj, the k norm's two, the k-pool pair) and wq_b: {p:?}");
    assert!(
        p.iter().all(|p| matches!(p, Problem::Unexpected { name, .. } if name.contains("layers.7.self_attn.indexer"))),
        "{p:?}"
    );
}

#[test]
fn all_problems_are_reported_together() {
    let mut inv = inventory();
    for i in 0..30 {
        inv.plain.remove(&layer_name(i, "hc_attn_scale")).unwrap();
    }
    let e: LayoutError = GlmWeights::map(&inv, &config()).unwrap_err();
    assert_eq!(e.problems.len(), 30);
    let text = e.to_string();
    assert!(text.contains("30 problem(s)") && text.contains("layers.0.hc_attn_scale") && text.contains("and 18 more"), "{text}");
}

// --------------------------------------------------------------------------------------------------------------- the plan

fn plan() -> &'static super::plan::LoadPlan {
    static PLAN: OnceLock<super::plan::LoadPlan> = OnceLock::new();
    PLAN.get_or_init(|| weights().load_plan().unwrap())
}

#[test]
fn the_plan_has_one_item_per_uploaded_part() {
    let p = plan();
    // 37,420 groups x (trellis, suh, svh) and 1,214 plain tensors; the 37,420 markers are not uploaded
    assert_eq!(p.items.len(), 37_420 * 3 + 1_214);
    assert_eq!(p.items.iter().filter(|i| i.part == ItemPart::Trellis).count(), 37_420);
    assert_eq!(p.totals().skipped_total(), 37_420 * 4);
    // regions: global, 46 layer trunks, 43 expert banks, the tower
    let kinds = |f: fn(&RegionKind) -> bool| p.regions.iter().filter(|r| f(&r.kind)).count();
    assert_eq!(p.regions.len(), 1 + 46 + 43 + 1);
    assert_eq!(kinds(|k| matches!(k, RegionKind::Experts(_))), 43);
    assert_eq!(kinds(|k| matches!(k, RegionKind::Vision)), 1);
    assert!(p.regions.iter().all(|r| r.bytes > 0 && r.bytes % ALIGN == 0));
}

#[test]
fn the_plans_categories_agree_with_exl3_inspects() {
    // two independent classifications: the mapper places every leaf by its slot, `budget::categorize` reads the name
    let p = plan();
    let inv = inventory();
    let mut want_on_disk = Bytes::default();
    for (name, g) in &inv.groups {
        want_on_disk.add(
            categorize(name, Some(45)),
            g.trellis.bytes + g.in_scale.bytes + g.out_scale.bytes + g.marker.as_ref().map_or(0, |m| m.bytes),
        );
    }
    for (name, t) in &inv.plain {
        want_on_disk.add(categorize(name, Some(45)), t.bytes);
    }
    for i in &p.items {
        assert_eq!(i.category, categorize(&i.name, Some(45)), "{}", i.name);
    }
    let t = p.totals();
    let mut got = t.uploaded;
    for c in [
        Category::RoutedExpert,
        Category::Decoder,
        Category::Head,
        Category::Other,
        Category::Mtp,
        Category::Embedding,
        Category::Vision,
    ] {
        got.add(c, bytes_of(&t.skipped, c));
    }
    assert_eq!(got, want_on_disk, "uploaded + markers == bytes on disk, category by category");
    // the figures `exl3-inspect` prints for this pack
    let gb = |b: u64| (b as f64 / 1e9 * 1000.0).round() / 1000.0;
    assert_eq!(
        (gb(got.routed), gb(got.decoder), gb(got.head), gb(got.mtp), gb(got.embedding), gb(got.vision)),
        (87.419, 4.351, 0.476, 3.756, 1.269, 1.127)
    );
    assert_eq!(gb(got.always_read()), 4.827);
}

#[test]
fn the_plan_reads_each_shard_front_to_back_and_places_without_overlap() {
    let p = plan();
    p.validate().unwrap();
    // every expert region is exactly the bank: three trellis blocks and six stacked scale blocks
    for r in p.regions.iter().filter(|r| matches!(r.kind, RegionKind::Experts(_))) {
        let RegionKind::Experts(l) = r.kind else { unreachable!() };
        let moe_bytes: u64 = p.items.iter().filter(|i| i.region == r.id).map(|i| i.bytes).sum();
        assert_eq!(r.bytes, moe_bytes, "layer {l}'s bank has no padding between blocks (every block is a multiple of {ALIGN})");
    }
}

#[test]
fn the_plan_validator_can_fail() {
    let good = plan().clone();
    // overlap in a region
    let mut p = good.clone();
    let victim = p.items.iter().position(|i| i.region == p.regions[5].id && i.dst > 0).unwrap();
    p.items[victim].dst = 0;
    assert!(p.validate().unwrap_err().contains("overlap"), "destinations overlap");
    // a destination off the alignment
    let mut p = good.clone();
    p.items[10].dst += 4;
    assert!(p.validate().unwrap_err().contains("not aligned"));
    // a destination past its region
    let mut p = good.clone();
    let r = p.items[20].region;
    p.regions.iter_mut().find(|x| x.id == r).unwrap().bytes = 256;
    assert!(p.validate().unwrap_err().contains("runs past"));
    // unsorted source order, and overlapping sources
    let mut p = good.clone();
    p.items.swap(30, 31);
    assert!(p.validate().is_err());
    let mut p = good.clone();
    p.items[40].offset = p.items[39].offset;
    assert!(p.validate().unwrap_err().contains("overlap"));
    // an unknown region
    let mut p = good.clone();
    p.items[50].region = 9999;
    assert!(p.validate().unwrap_err().contains("region 9999"));
}

#[test]
fn select_drops_whole_components() {
    let p = plan();
    let text = p.select(|c| c != Category::Vision && c != Category::Mtp);
    text.validate().unwrap();
    let t = text.totals();
    assert_eq!((t.uploaded.vision, t.uploaded.mtp, t.skipped.mtp, t.skipped.vision), (0, 0, 0, 0));
    assert_eq!(t.uploaded.routed, p.totals().uploaded.routed);
    // the tower's region and both of the MTP layer's go with it
    assert_eq!(text.regions.len(), p.regions.len() - 3);
    assert!(text.regions.iter().all(|r| !matches!(r.kind, RegionKind::Vision | RegionKind::Trunk(45) | RegionKind::Experts(45))));
}

#[test]
fn extents_merge_contiguous_reads_and_respect_the_cap() {
    let p = plan();
    let n = p.items.len();
    // the synthetic shard puts a group's suh, svh, marker and trellis together; merging over the 4-byte marker joins them
    let merged = p.extents(4, u64::MAX);
    let split = p.extents(0, u64::MAX);
    assert!(merged.len() < split.len() && split.len() <= n);
    assert_eq!(merged.len(), 1, "the synthetic shard is contiguous but for the 4-byte markers");
    // a cap bounds an extent unless one item is bigger than it
    let cap = 64 << 20;
    let capped = p.extents(4, cap);
    assert!(
        capped.iter().all(|e| e.bytes <= cap + 4 || e.items.len() == 1),
        "extents respect the cap, to within a trailing marker"
    );
    // extents partition the items in order
    let mut next = 0;
    for e in &capped {
        assert_eq!(e.items.start, next);
        let first = &p.items[e.items.start];
        let last = &p.items[e.items.end - 1];
        assert_eq!((e.file, e.offset), (first.file, first.offset));
        let end = e.offset + e.bytes;
        assert!(
            end >= last.offset + last.bytes && end <= last.offset + last.bytes + 4,
            "an extent ends at its last item or its marker"
        );
        next = e.items.end;
    }
    assert_eq!(next, n);
}

/// The inventory with every unit (a plain tensor, or a group with its parts in the real order) moved to a hashed position in one of two
/// shards, so that file order is neither name order nor one shard.
fn shuffled(inv: &PackInventory) -> PackInventory {
    let mut inv = inv.clone();
    inv.files = vec!["a.safetensors".into(), "b.safetensors".into()];
    let h = |s: &str| gb_exl3::fnv1a(s.bytes());
    let mut units: Vec<(u64, String, bool)> = inv.plain.keys().map(|n| (h(n), n.clone(), false)).collect();
    units.extend(inv.groups.keys().map(|n| (h(n), n.clone(), true)));
    units.sort();
    let mut cursor = [0u64; 2];
    for (hash, name, is_group) in units {
        let f = ((hash >> 7) & 1) as usize;
        let mut put = |t: &mut super::inventory::TensorInfo| {
            t.file = f as u16;
            t.offset = cursor[f];
            cursor[f] += t.bytes;
        };
        if is_group {
            let g = inv.groups.get_mut(&name).unwrap();
            put(&mut g.in_scale);
            put(&mut g.out_scale);
            if let Some(m) = &mut g.marker {
                put(m);
            }
            put(&mut g.trellis);
        } else {
            put(inv.plain.get_mut(&name).unwrap());
        }
    }
    inv
}

#[test]
fn the_plan_follows_the_shards_not_the_names() {
    let inv = shuffled(&inventory());
    let w = GlmWeights::map(&inv, &config()).unwrap();
    let p = w.load_plan().unwrap();
    assert!(
        p.items.windows(2).all(|w| (w[0].file, w[0].offset) < (w[1].file, w[1].offset)),
        "items are in (shard, offset) order"
    );
    assert!(!p.items.windows(2).all(|w| w[0].name <= w[1].name), "...which here is not name order");
    assert!(p.items.iter().any(|i| i.file == 0) && p.items.iter().any(|i| i.file == 1), "both shards are used");
    // every leaf still points at where its tensor now is
    let name = layer_name(31, "mlp.experts.77.down_proj");
    let FfnWeights::Moe(moe) = &w.layers[31].ffn else { panic!("layer 31 is MoE") };
    let src = &inv.groups[&name];
    let leaf = &moe.experts.experts[77].down_proj;
    assert_eq!((leaf.trellis.src.file, leaf.trellis.src.offset), (src.trellis.file, src.trellis.offset));
    let item = p.items.iter().find(|i| i.name == format!("{name}.trellis")).unwrap();
    assert_eq!((item.file, item.offset), (src.trellis.file, src.trellis.offset));
    // the reads and the marker checks follow the same order
    let ex = p.extents(4, 64 << 20);
    assert!(p.uncovered_markers(&ex).is_empty());
    assert!(ex.windows(2).all(|w| (w[0].file, w[0].offset) < (w[1].file, w[1].offset)));
    assert!(ex.len() > 1000, "the shuffle splits the long runs, but a unit is still one read: {}", ex.len());
}

#[test]
fn every_codebook_marker_is_a_check_the_plan_owes() {
    let p = plan();
    let inv = inventory();
    assert_eq!(p.markers.len(), 37_420);
    assert!(p.markers.windows(2).all(|w| (w[0].file, w[0].offset) < (w[1].file, w[1].offset)));
    assert!(p.markers.iter().all(|m| m.want == gb_exl3::MARKER_MUL1), "this pack is all mul1");
    let m = p.markers.iter().find(|m| m.name == layer_name(9, "mlp.experts.100.up_proj.mul1")).unwrap();
    assert_eq!(m.offset, inv.groups[&layer_name(9, "mlp.experts.100.up_proj")].marker.as_ref().unwrap().offset);
    assert_eq!(m.category, Category::RoutedExpert);
    // the markers are inside the reads the executor makes anyway, whatever the cap: a split between a group's scales and its
    // trellis would strand one, so the extent that ends at the scales takes the marker with it
    for cap in [12_288u64, 13_000, 1 << 20, 3 << 20, 5 << 20, 64 << 20] {
        let ex = p.extents(4, cap);
        assert!(p.uncovered_markers(&ex).is_empty(), "cap {cap}");
        let covered: usize = ex.iter().map(|e| p.markers_in(e).len()).sum();
        assert_eq!(covered, 37_420, "each marker is inside exactly one extent (cap {cap})");
        for e in &ex {
            assert!(e.bytes <= cap.max(1 << 21) + 4 || e.items.len() == 1, "an extent exceeds its cap by at most the marker");
            for m in p.markers_in(e) {
                assert!(m.offset >= e.offset && m.offset + MARKER_BYTES <= e.offset + e.bytes && m.file == e.file);
            }
        }
    }
    // a marker too far from the reads (here 100 bytes from the next item, with max_gap 16) is left to a read of its own
    let item = |name: &str, offset, bytes| super::plan::LoadItem {
        name: name.into(),
        part: ItemPart::Tensor,
        category: Category::Decoder,
        file: 0,
        offset,
        bytes,
        xform: super::weights::Xform::None,
        dst_bytes: bytes,
        region: 0,
        dst: offset,
        pitch: None,
    };
    let far = super::plan::LoadPlan {
        files: vec!["f".into()],
        regions: vec![],
        items: vec![item("a", 0, 100), item("b", 300, 100)],
        skipped: Bytes::default(),
        markers: vec![super::plan::MarkerCheck { name: "m".into(), category: Category::Decoder, file: 0, offset: 150, want: 7 }],
    };
    let ex = far.extents(16, u64::MAX);
    assert_eq!(ex.len(), 2);
    assert_eq!(far.uncovered_markers(&ex).len(), 1);
    assert_eq!(far.markers_in(&ex[0]).len() + far.markers_in(&ex[1]).len(), 0);
    let ex = far.extents(1000, u64::MAX);
    assert_eq!(
        (ex.len(), far.uncovered_markers(&ex).len(), far.markers_in(&ex[0]).len()),
        (1, 0, 1),
        "a long enough gap is read through"
    );
    // the validator checks them too: unsorted, and a marker that overlaps an item
    let mut bad = p.clone();
    bad.markers.swap(5, 6);
    assert!(bad.validate().unwrap_err().contains("markers are not sorted"));
    let mut bad = p.clone();
    bad.markers[7].offset = bad.items[100].offset + 1;
    bad.markers.sort_by_key(|m| (m.file, m.offset));
    assert!(bad.validate().unwrap_err().contains("overlaps an item"));
    // selecting a component takes its markers with it
    let text = p.select(|c| c != Category::Mtp);
    assert_eq!(text.markers.len(), 37_420 - 873);
}

// ------------------------------------------------------------------------------------------------- a real scan, in miniature

/// Write a one-shard pack: a plain fp32 tensor and a 128 x 128 2-bit EXL3 group, each part filled with a byte of its own, in the
/// order the real shards use (`suh svh mul1 trellis`). Returns the directory.
fn tiny_pack(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("rexl3-glm-inv-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let marker = gb_exl3::MARKER_MUL1.to_le_bytes();
    let parts: [(&str, &str, &str, Vec<u8>); 5] = [
        ("p.weight", "F32", "[4]", vec![0x33; 16]),
        ("x.suh", "F16", "[128]", vec![0x11; 256]),
        ("x.svh", "F16", "[128]", vec![0x22; 256]),
        ("x.mul1", "I32", "[]", marker.to_vec()),
        ("x.trellis", "I16", "[8,8,32]", (0..4096).map(|i| (i % 251) as u8).collect()),
    ];
    let (mut header, mut data) = (String::from("{"), Vec::new());
    for (name, dtype, shape, bytes) in &parts {
        let (a, b) = (data.len(), data.len() + bytes.len());
        header += &format!("\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":{shape},\"data_offsets\":[{a},{b}]}},");
        data.extend_from_slice(bytes);
    }
    header.pop();
    header += "}";
    let mut shard = (header.len() as u64).to_le_bytes().to_vec();
    shard.extend_from_slice(header.as_bytes());
    shard.extend_from_slice(&data);
    std::fs::write(dir.join("m.safetensors"), shard).unwrap();
    dir
}

#[test]
fn a_scanned_checkpoints_inventory_points_at_the_real_bytes() {
    let dir = tiny_pack("offsets");
    let ck = gb_exl3::checkpoint::scan(&dir, true).unwrap();
    let inv = PackInventory::from_checkpoint(&ck).unwrap();
    assert_eq!(inv.files, vec!["m.safetensors".to_string()]);
    assert_eq!((inv.plain.len(), inv.groups.len(), inv.refused.len()), (1, 1, 0));
    let file = std::fs::read(dir.join("m.safetensors")).unwrap();
    let at = |t: &super::inventory::TensorInfo| &file[t.offset as usize..(t.offset + t.bytes) as usize];
    assert!(at(&inv.plain["p.weight"]).iter().all(|&b| b == 0x33));
    let g = &inv.groups["x"];
    assert!(at(&g.in_scale).iter().all(|&b| b == 0x11) && g.in_scale.bytes == 256);
    assert!(at(&g.out_scale).iter().all(|&b| b == 0x22));
    assert_eq!(at(g.marker.as_ref().unwrap()), gb_exl3::MARKER_MUL1.to_le_bytes());
    assert_eq!(g.trellis.bytes, 4096);
    assert!(at(&g.trellis).iter().enumerate().all(|(i, &b)| b == (i % 251) as u8), "the trellis offset is the absolute one");
    assert_eq!((g.k, g.n, g.bits.as_f64(), g.in_scales, g.out_scales), (128, 128, 2.0, "suh", "svh"));
    std::fs::remove_dir_all(dir).unwrap();
}

/// FNV-1a 64 over the plan as data (regions, items, markers, skipped bytes): a single-node plan regression guard.
fn plan_digest(p: &super::plan::LoadPlan) -> u64 {
    let text = format!("{:?}|{:?}|{:?}|{:?}", p.regions, p.items, p.markers, p.skipped);
    text.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3))
}

#[test]
fn the_single_node_plan_is_unchanged_by_subbanks() {
    // the digest of the plan the mapper made before expert subbanks and ownership existed (22a21aa, the same inventory and config):
    // a uniform pack on one node is laid out, read and placed exactly as before
    assert_eq!(plan_digest(plan()), 0x8f6c_6195_12f3_b20e, "items {}", plan().items.len());
    let all = GlmWeights::map_owned(&inventory(), &config(), &gb_cuda::moe::ExpertOwnership::all(288)).unwrap();
    assert_eq!(plan_digest(&all.load_plan().unwrap()), 0x8f6c_6195_12f3_b20e);
    assert!(weights().banks().all(|b| b.is_single() && b.width.is_none()));
    assert!(plan().items.iter().all(|i| i.pitch.is_none()), "no strided item without a width share");
}

/// The expert WIDTH split (`GlmConfig::with_expert_width`, `gb_cuda::moe_width`): rank r of 2 maps every expert of every bank (the NextN
/// layer's too) at the intermediate columns `[1024 r, 1024 (r + 1))`. Per subbank the layout is `SubbankRegion::stacked` at width 1024;
/// gate/up read their column half as ONE strided item a projection, down its row half as one contiguous item; the two ranks read every
/// stored trellis byte exactly once, the scale halves once and the model-width scales on both; every other item and every marker is
/// the single node's. Ownership other than `all` is refused.
fn check_width_split(inv: &PackInventory, cfg: &GlmConfig) {
    use gb_cuda::moe::{ExpertOwnership, SubbankRegion, WidthShare};
    use std::collections::BTreeMap;
    let whole = GlmWeights::map(inv, cfg).unwrap().load_plan().unwrap();
    let routed = |i: &super::plan::LoadItem| i.name.contains(".mlp.experts.");
    let by_name: BTreeMap<&str, &super::plan::LoadItem> =
        whole.items.iter().filter(|i| routed(i)).map(|i| (i.name.as_str(), i)).collect();
    // per stored part: the copied bytes of both ranks, and the source ranges they read
    let mut copied: BTreeMap<String, u64> = BTreeMap::new();
    let mut reads: BTreeMap<String, Vec<(u64, u64)>> = BTreeMap::new();
    for r in 0..2 {
        let wcfg = cfg.with_expert_width(r, 2).unwrap();
        let share = WidthShare::new(r, 2).unwrap();
        assert_eq!(wcfg.ffn.width, Some(share));
        let own = ExpertOwnership::range(288, r, 2).unwrap();
        let e = GlmWeights::map_owned(inv, &wcfg, &own).expect_err("a width share takes every expert");
        assert!(e.problems.iter().all(|p| matches!(p, Problem::NonUniform { what, .. } if what.contains("width share"))), "{e:?}");
        let w = GlmWeights::map_owned(inv, &wcfg, &ExpertOwnership::all(288)).unwrap();
        let p = w.load_plan().unwrap(); // validated, strided items included
        assert!(w.banks().count() > 0);
        for b in w.banks() {
            assert_eq!(b.width, Some(share), "layer {}", b.layer);
            assert!(!b.is_single() && b.own.is_all() && b.inter == 2048);
            for (x, ex) in b.experts.iter().enumerate() {
                for g in [&ex.gate_proj, &ex.up_proj, &ex.down_proj] {
                    assert!(g.trellis.dst != Dst::NONE && g.suh.dst != Dst::NONE && g.svh.dst != Dst::NONE, "layer {} expert {x}", b.layer);
                }
                assert_eq!(ex.gate_proj.cut, Some(StripCut { ranges: vec![(8 * r, 8 * r + 8)] }));
                assert_eq!((ex.down_proj.rows, ex.down_proj.cut.is_none()), (Some((1024 * r, 1024 * (r + 1))), true));
            }
            let mut cursor = 0usize;
            for sb in &b.subbanks {
                let (lay, end) = SubbankRegion::stacked(sb.key, sb.ids.len(), 4096, 1024, cursor, ALIGN as usize);
                for (a, l) in [(&lay.gate, &sb.layout.gate), (&lay.up, &sb.layout.up), (&lay.down, &sb.layout.down)] {
                    assert_eq!(
                        (a.trellis as u64, a.trellis_stride as u64, a.suh as u64, a.svh as u64),
                        (l.trellis, l.trellis_stride, l.suh, l.svh),
                        "layer {} {:?}",
                        b.layer,
                        sb.key
                    );
                }
                cursor = end;
            }
            assert_eq!(b.subbanks.iter().map(|s| s.ids.len()).sum::<usize>(), 288);
            assert_eq!(align_up(cursor as u64), w.regions[b.region as usize].bytes, "layer {}", b.layer);
        }
        // every non-routed item and every marker is the single node's
        let non_routed = |p: &super::plan::LoadPlan| p.items.iter().filter(|i| !routed(i)).cloned().collect::<Vec<_>>();
        assert_eq!(non_routed(&p), non_routed(&whole), "rank {r}");
        assert_eq!(p.markers, whole.markers, "rank {r}");
        // routed: one item a stored part (9 an expert); every read inside the stored part; a strided item is a gate/up trellis half
        assert_eq!(p.items.iter().filter(|i| routed(i)).count(), by_name.len(), "rank {r}");
        for it in p.items.iter().filter(|i| routed(i)) {
            let st = by_name[it.name.as_str()];
            assert!(it.offset >= st.offset && it.offset + it.bytes <= st.offset + st.bytes, "{} reads outside its part", it.name);
            let gate_up = it.name.contains(".gate_proj.") || it.name.contains(".up_proj.");
            assert_eq!(it.pitch.is_some(), gate_up && it.part == ItemPart::Trellis, "{}", it.name);
            if let Some(pt) = it.pitch {
                assert_eq!((pt.src_pitch, it.dst_bytes * 2), (pt.width * 2, st.bytes), "{}", it.name);
                let rows = it.dst_bytes / pt.width;
                // the half's tiles of every k-tile row, one row's bytes into each source row
                assert_eq!(it.offset, st.offset + r as u64 * pt.width, "{}", it.name);
                for k in 0..rows {
                    reads.entry(it.name.clone()).or_default().push((it.offset + k * pt.src_pitch, pt.width));
                }
            } else {
                reads.entry(it.name.clone()).or_default().push((it.offset, it.bytes));
            }
            *copied.entry(it.name.clone()).or_default() += it.copied_bytes();
        }
    }
    // both ranks: every trellis byte exactly once, the halves' scales once, the model-width scales (gate/up suh, down svh) twice
    for (name, st) in &by_name {
        let twice = name.ends_with(".down_proj.svh")
            || (name.ends_with(".suh") && (name.contains(".gate_proj.") || name.contains(".up_proj.")));
        assert_eq!(copied[*name], if twice { 2 * st.bytes } else { st.bytes }, "{name}");
        if !twice {
            let mut v = reads[*name].clone();
            v.sort();
            let mut at = st.offset;
            for (o, n) in v {
                assert_eq!(o, at, "{name}: a gap or an overlap");
                at += n;
            }
            assert_eq!(at, st.offset + st.bytes, "{name}");
        }
    }
}

#[test]
fn a_width_share_places_every_expert_at_its_columns() {
    check_width_split(&inventory(), &config());
    let (inv, cfg) = sc410().text_side();
    check_width_split(&inv, &cfg);
}

// ------------------------------------------------------------------------------------- every pinned pack: 2.25bpw and sc_4.10bpw
//
// `turboderp/GLM-5.3-Flash-exl3` branch `sc_4.10bpw` @c6b8ac3f (exllamav3 1.6.2, 156.19 GiB, served at TP = 2 only): its inventory
// (`testdata/glm53-sc410/tensor_inventory.tsv`, `tools/gen_glm_inventory.py` over the 23 shard headers, checked against the real pack by
// `tests/glm_pack.rs`) and its config. The tests above pin the 2.25bpw pack; the ones below run over both fixtures (their pins side by
// side) and pin what the sc_4.10bpw pack adds: mixed widths by class (K4/K5 experts, K8 trunk attention and shared experts, K6 NextN
// layer, dense MLP and `lm_head`, K5 `eh_proj`) and an EXL3 vision tower (K6 groups with F16 biases). The tower is lane G-V's: these tests
// map the TEXT side (the tower's tensors and `vision_config` removed) and accept, for the whole pack, only refusals that name the tower.

const SC410_TABLE: &str = include_str!("../../../../../testdata/glm53-sc410/tensor_inventory.tsv");
const SC410_CONFIG: &str = include_str!("../../../../../testdata/glm53-sc410/config.json");
const TOWER: &str = "model.visual.";

/// One committed pack fixture and the counts it is pinned on.
struct Fixture {
    name: &'static str,
    table: &'static str,
    config: &'static str,
    /// (EXL3 groups, plain tensors, tensors) of the whole inventory
    counts: (usize, usize, usize),
    /// (EXL3 groups, plain tensors) of the tower (`model.visual.*`)
    tower: (usize, usize),
    /// the routed experts' width of a MoE layer (3..=45; gate, up and down share it in both packs)
    expert_bits: fn(usize) -> u8,
}

const FIXTURES: [Fixture; 2] = [
    Fixture {
        name: "2.25bpw",
        table: TABLE,
        config: CONFIG,
        counts: (37_420, 1_214, 150_894),
        tower: (0, 347),
        expert_bits: |l| match l {
            45 => 4,
            3..=5 | 14 | 24 | 33 | 39..=44 => 3,
            _ => 2,
        },
    },
    Fixture {
        name: "sc_4.10bpw",
        table: SC410_TABLE,
        config: SC410_CONFIG,
        counts: (37_592, 1_018, 151_554),
        tower: (172, 151),
        expert_bits: |l| if l == 3 { 5 } else { 4 },
    },
];

impl Fixture {
    fn inventory(&self) -> PackInventory {
        PackInventory::from_table(self.table).unwrap()
    }
    fn config_json(&self) -> serde_json::Value {
        serde_json::from_str(self.config).unwrap()
    }
    /// The text side: the inventory without the tower's tensors and the config without `vision_config`.
    fn text_side(&self) -> (PackInventory, GlmConfig) {
        let mut inv = self.inventory();
        inv.groups.retain(|p, _| !p.starts_with(TOWER));
        inv.plain.retain(|n, _| !n.starts_with(TOWER));
        let mut v = self.config_json();
        v.as_object_mut().unwrap().remove("vision_config");
        (inv, GlmConfig::from_json(&v).unwrap())
    }
}

fn sc410() -> &'static Fixture {
    &FIXTURES[1]
}

#[test]
fn every_fixture_has_its_pinned_counts_and_its_tower() {
    for f in &FIXTURES {
        let inv = f.inventory();
        assert_eq!((inv.groups.len(), inv.plain.len(), inv.tensor_count()), f.counts, "{}", f.name);
        assert!(inv.refused.is_empty() && inv.issues.is_empty(), "{}: {:?}", f.name, inv.refused.iter().next());
        let tower = (
            inv.groups.keys().filter(|p| p.starts_with(TOWER)).count(),
            inv.plain.keys().filter(|n| n.starts_with(TOWER)).count(),
        );
        assert_eq!(tower, f.tower, "{}", f.name);
        // the config parses with its tower (the tower's form is the mapper's business, not the config's)
        let cfg = GlmConfig::from_json(&f.config_json()).unwrap();
        assert!(cfg.vision.is_some() && cfg.has_mtp(), "{}", f.name);
    }
}

#[test]
fn the_text_side_of_every_fixture_maps_and_its_bits_come_from_the_trellis_shapes() {
    for f in &FIXTURES {
        let (inv, cfg) = f.text_side();
        let w = GlmWeights::map(&inv, &cfg).unwrap_or_else(|e| panic!("{}: the text side must map: {e}", f.name));
        // the text sides of the two packs have the same tensors: 37,420 groups and 867 plain tensors
        assert_eq!(w.counts(), (f.counts.0 - f.tower.0, f.counts.1 - f.tower.1), "{}", f.name);
        assert_eq!(w.counts(), (37_420, 867), "{}", f.name);
        assert!(w.vision.is_none(), "{}", f.name);
        assert_eq!(w.banks().count(), 43, "{}", f.name);
        for b in w.banks() {
            let want = (f.expert_bits)(b.layer) as f64;
            assert_eq!((b.bits_gate_up.as_f64(), b.bits_down.as_f64()), (want, want), "{} layer {}", f.name, b.layer);
            assert!(
                b.is_single() && b.subbanks.len() == 1,
                "{} layer {}: a uniform layer is one bank on one node",
                f.name,
                b.layer
            );
        }
        w.load_plan().unwrap_or_else(|e| panic!("{}: {e}", f.name)).validate().unwrap();
    }
}

#[test]
fn the_sc410_text_side_is_the_old_packs_but_for_the_trellis_widths() {
    // names, dtypes and shapes of every non-tower tensor are the 2.25bpw pack's; only a trellis' last dimension (16 x bits) moves
    let (old, _) = FIXTURES[0].text_side();
    let (new, _) = sc410().text_side();
    // (dtype, shape): the fixtures' offsets follow the bytes before them, which the widths move
    let ds = |t: &super::inventory::TensorInfo| (t.dtype.clone(), t.shape.clone());
    let plain = |inv: &PackInventory| inv.plain.iter().map(|(n, t)| (n.clone(), ds(t))).collect::<Vec<_>>();
    assert_eq!(plain(&old), plain(&new), "the plain tensors of the text side are unchanged");
    assert_eq!(old.groups.keys().collect::<Vec<_>>(), new.groups.keys().collect::<Vec<_>>());
    let mut moved = 0;
    for (p, a) in &old.groups {
        let b = &new.groups[p];
        assert_eq!((a.k, a.n, a.codebook, a.in_scales, a.out_scales), (b.k, b.n, b.codebook, b.in_scales, b.out_scales), "{p}");
        assert_eq!(
            (ds(&a.in_scale), ds(&a.out_scale), a.bias.is_some()),
            (ds(&b.in_scale), ds(&b.out_scale), b.bias.is_some()),
            "{p}"
        );
        let (ta, tb) = (&a.trellis.shape, &b.trellis.shape);
        assert_eq!((ta.len(), &ta[..2], a.trellis.dtype.as_str()), (3, &tb[..2], "I16"), "{p}");
        assert_eq!(tb[2], b.bits.twice() as usize * 8, "{p}: the width is the trellis' last dimension / 16");
        moved += (a.bits != b.bits) as usize;
    }
    // every text group changed width but `lm_head` (K6 in both) and the NextN layer's 864 experts (K4 in both)
    assert_eq!(moved, 37_420 - 1 - 864, "every text group but lm_head and the NextN experts changed width");
}

/// The class of a text group, for the width histogram.
fn class_of(prefix: &str) -> String {
    let layer = prefix.strip_prefix(LAYER_PREFIX).and_then(|r| r.split('.').next()).and_then(|l| l.parse::<usize>().ok());
    let where_ = match layer {
        Some(45) => "nextn",
        Some(_) => "trunk",
        None => "global",
    };
    let what = if prefix.starts_with(TOWER) {
        return "vision".into();
    } else if prefix.contains(".mlp.experts.") {
        return format!("experts L{}", layer.unwrap());
    } else if prefix.contains("shared_experts") {
        "shared"
    } else if prefix.contains(".mlp.") {
        "dense mlp"
    } else if prefix.contains("eh_proj") {
        "eh_proj"
    } else if prefix == "lm_head" {
        "lm_head"
    } else {
        "attention"
    };
    format!("{what} {where_}")
}

#[test]
fn the_sc410_widths_by_class() {
    // exllamav3 1.6.2's `sc_4.10bpw` recipe as stored: (class, bits) -> groups
    let inv = sc410().inventory();
    let mut hist: std::collections::BTreeMap<(String, u8), usize> = Default::default();
    for (p, g) in &inv.groups {
        assert!(g.bits.is_integer() && g.codebook == gb_exl3::Codebook::Mul1, "{p}");
        assert_eq!((g.in_scales, g.out_scales), ("suh", "svh"), "{p}");
        *hist.entry((class_of(p), g.bits.twice() / 2)).or_default() += 1;
    }
    let mut want: std::collections::BTreeMap<(String, u8), usize> = [
        (("attention trunk", 8), 123),
        (("attention nextn", 6), 5),
        (("shared trunk", 8), 126),
        (("shared nextn", 6), 3),
        (("dense mlp trunk", 6), 9),
        (("eh_proj nextn", 5), 1),
        (("lm_head global", 6), 1),
        (("vision", 6), 172),
        (("experts L3", 5), 864),
    ]
    .into_iter()
    .map(|((c, b), n)| ((c.to_string(), b), n))
    .collect();
    for l in 4..=45 {
        want.insert((format!("experts L{l}"), 4), 864);
    }
    assert_eq!(hist, want);
    assert_eq!(hist.values().sum::<usize>(), 37_592);
}

#[test]
fn the_sc410_tower_is_exl3_k6_with_f16_biases_beside_the_bf16_fused_qkv() {
    // 24 blocks x 7 biased linears (q, k, v, proj; gate, up, down) + the merger's 4 unbiased ones; the fused `attn.qkv` stays BF16 plain
    let inv = sc410().inventory();
    let tower: Vec<(&String, &GroupInfo)> = inv.groups.iter().filter(|(p, _)| p.starts_with(TOWER)).collect();
    assert_eq!(tower.len(), 24 * 7 + 4);
    let biased: Vec<_> = tower.iter().filter(|(_, g)| g.bias.is_some()).collect();
    assert_eq!(biased.len(), 168);
    for (p, g) in &biased {
        let b = g.bias.as_ref().unwrap();
        assert_eq!((b.dtype.as_str(), b.shape.clone()), ("F16", vec![g.n]), "{p}");
        assert!(p.starts_with("model.visual.blocks."), "{p}");
    }
    for m in ["down_proj", "gate_proj", "up_proj", "proj"] {
        let g = &inv.groups[&format!("{TOWER}merger.{m}")];
        assert!(g.bias.is_none() && g.bits.as_f64() == 6.0, "merger.{m}");
    }
    let plain = |pred: &dyn Fn(&str) -> bool| inv.plain.iter().filter(|(n, _)| n.starts_with(TOWER) && pred(n)).count();
    assert_eq!(plain(&|n| n.contains(".attn.qkv.")), 48, "the fused qkv weight and bias of 24 blocks");
    assert!(inv.plain.iter().filter(|(n, _)| n.contains(".attn.qkv.")).all(|(_, t)| t.dtype == "BF16"));
    let f16: Vec<&str> =
        inv.plain.iter().filter(|(n, t)| n.starts_with(TOWER) && t.dtype == "F16").map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        f16,
        [
            "model.visual.downsample.bias",
            "model.visual.downsample.weight",
            "model.visual.merger.post_projection_norm.bias",
            "model.visual.merger.post_projection_norm.weight",
            "model.visual.patch_embed.proj.bias",
            "model.visual.patch_embed.proj.weight"
        ]
    );
    assert_eq!(plain(&|_| true), 151);
}

#[test]
fn the_sc410_whole_inventory_maps_or_refuses_only_the_tower() {
    // until lane G-V lands the EXL3 tower the mapper refuses it; nothing outside `model.visual.*` may be named (this stays true after)
    let f = sc410();
    let cfg = GlmConfig::from_json(&f.config_json()).unwrap();
    match GlmWeights::map(&f.inventory(), &cfg) {
        // the EXL3 tower mapped (lane G-V): its 24 x 7 fp16 group biases are plain leaves of the tree (the inventory counts them inside
        // their groups) and the 48 fused bf16 `attn.qkv` tensors are claimed and skipped, so every tensor is claimed exactly once
        Ok(w) => {
            assert_eq!(w.skipped.len(), 48, "the fused qkv weight and bias of 24 blocks");
            assert!(w.skipped.iter().all(|s| s.name.starts_with(TOWER) && s.name.contains(".attn.qkv.")));
            assert_eq!(w.counts(), (f.counts.0, f.counts.1 + 24 * 7 - w.skipped.len()));
        }
        Err(e) => {
            let outside: Vec<String> =
                e.problems.iter().map(|p| p.to_string()).filter(|s| !s.contains(&format!("`{TOWER}"))).take(5).collect();
            assert!(outside.is_empty(), "{} problem(s), outside the tower: {outside:?}", e.problems.len());
        }
    }
}

#[test]
fn a_sc410_rank_places_and_plans_only_its_owned_experts() {
    use gb_cuda::moe::ExpertOwnership;
    let (inv, cfg) = sc410().text_side();
    let whole = super::GlmPack::from_inventory(&inv, cfg.clone()).unwrap();
    let gib = |b: u64| (b as f64 / (1u64 << 30) as f64 * 100.0).round() / 100.0;
    let kinds = |p: &super::plan::LoadPlan| -> (u64, u64, u64, u64) {
        let (mut g, mut t, mut e, mut m) = (0, 0, 0, 0);
        for r in &p.regions {
            match r.kind {
                RegionKind::Global => g += r.bytes,
                RegionKind::Trunk(l) if l < 45 => t += r.bytes,
                RegionKind::Experts(l) if l < 45 => e += r.bytes,
                RegionKind::Trunk(_) | RegionKind::Experts(_) => m += r.bytes,
                _ => {}
            }
        }
        (g, t, e, m)
    };
    // one node (every expert; text, the NextN layer apart): global, trunk, routed experts, the NextN layer
    let w = kinds(&whole.plan);
    assert_eq!((gib(w.0), gib(w.1), gib(w.2), gib(w.3)), (1.63, 7.4, 143.01, 3.53), "one node: {w:?}");
    assert_eq!(super::tp_boot::rank_weight_bytes(&whole), w.0 + w.1 + w.2, "the NextN layer is not counted");
    let mut experts = 0;
    for r in 0..2 {
        let own = ExpertOwnership::range(288, r, 2).unwrap();
        let p = super::GlmPack::from_inventory_owned(&inv, cfg.clone(), &own).unwrap();
        for b in p.weights.banks() {
            assert_eq!(b.subbanks.len(), 1, "layer {}: a uniform layer is one subbank on a rank", b.layer);
            assert_eq!(b.subbanks[0].ids, own.owned(), "layer {}", b.layer);
        }
        let k = kinds(&p.plan);
        assert_eq!((k.0, k.1), (w.0, w.1), "rank {r}: the trunk is replicated");
        experts += k.2;
        let bytes = super::tp_boot::rank_weight_bytes(&p);
        assert_eq!(bytes, k.0 + k.1 + k.2);
        assert_eq!(gib(bytes), 80.53, "rank {r}: replicated {} + experts {} GiB", gib(k.0 + k.1), gib(k.2));
        assert_eq!(gib(super::tp_boot::rank_weight_bytes_with(&p, true)), 82.37, "rank {r} with its NextN share");
        assert!(
            p.plan
                .items
                .iter()
                .filter(|i| i.name.contains(".mlp.experts."))
                .all(|i| { own.owns(i.name.split(".mlp.experts.").nth(1).unwrap().split('.').next().unwrap().parse().unwrap()) }),
            "rank {r} reads an expert it does not own"
        );
    }
    assert_eq!(experts, w.2, "the ranks' expert regions are the single node's, split");
}

#[test]
fn the_sc410_text_plan_is_pinned() {
    // the single-node text-side plan as data, and the bytes on disk by `exl3-inspect`'s categories (a regression guard for G-T's merge)
    let (inv, cfg) = sc410().text_side();
    let p = GlmWeights::map(&inv, &cfg).unwrap().load_plan().unwrap();
    assert_eq!(p.items.len(), 37_420 * 3 + 867);
    let t = p.totals();
    let mut got = t.uploaded;
    for c in [Category::RoutedExpert, Category::Decoder, Category::Head, Category::Other, Category::Mtp, Category::Embedding] {
        got.add(c, bytes_of(&t.skipped, c));
    }
    let gb = |b: u64| (b as f64 / 1e9 * 1000.0).round() / 1000.0;
    assert_eq!(
        (gb(got.routed), gb(got.decoder), gb(got.head), gb(got.mtp), gb(got.embedding), gb(got.vision)),
        (153.555, 8.019, 0.476, 3.793, 1.269, 0.0)
    );
    // the tower's bytes on disk, from the inventory (the plan of the whole pack is G-V's)
    let full = sc410().inventory();
    let tower: u64 = full
        .groups
        .iter()
        .filter(|(n, _)| n.starts_with(TOWER))
        .map(|(_, g)| {
            g.trellis.bytes
                + g.in_scale.bytes
                + g.out_scale.bytes
                + g.bias.as_ref().map_or(0, |b| b.bytes)
                + g.marker.as_ref().map_or(0, |m| m.bytes)
        })
        .sum::<u64>()
        + full.plain.iter().filter(|(n, _)| n.starts_with(TOWER)).map(|(_, t)| t.bytes).sum::<u64>();
    assert_eq!(gb(tower), 0.598);
    assert_eq!(plan_digest(&p), 0x34ab_c92d_00e1_b211, "items {}", p.items.len());
}
