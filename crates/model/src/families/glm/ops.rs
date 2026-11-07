//! The schedule of one GLM-5.3-Flash forward as DATA: the list of operations the executor (`model.rs`) walks, each naming the layer and the
//! weights it uses. Building the list is a pure function of the config, so the invariants the numbers depend on are checked on the CPU, and
//! mutated, without a GPU:
//!
//! * **mHC site order.** The fused kernel `apply_partials` finishes one site's update and, in the same pass over the streams, takes the partials of
//!   the NEXT site: the model must pass that site's weights, so every `ApplyPartials` names the site of the `Finalize` that follows it (attention
//!   site of layer `l` -> the FFN site of layer `l` -> the attention site of layer `l + 1`; the last site has no next and uses the plain `Apply`).
//! * **Layer types.** KDA on every layer but 3, 7, ..., 43 (MLA with the DSA indexer); dense gated MLP on layers 0-2, mixture of experts on 3-44.
//! * **The shared expert is added last**: it is computed before the routed experts and handed to their combine, which adds it after the eight
//!   slots in rank order.
//! * **The head** is the unweighted mean of the four streams followed by the final norm (one pass), then the `lm_head`.
//!
//! ```text
//! Embed, FirstPartials(attn 0)
//! per layer l:  Finalize(attn l)  Mixer(l)  ApplyPartials(ffn l)  Finalize(ffn l)  [DenseFfn | MoeShared MoeRoute MoeExperts]  ApplyPartials(attn l+1) | Apply
//! HeadMean, LmHead
//! ```
//!
//! A prompt forward whose last layer is KDA may run that layer's tail on the head's rows alone ([`plan_forward_head_rows`]): everything
//! after its recurrence feeds only the head.

use super::config::GlmConfig;
use crate::family::{Ffn, Mixer};

/// Which of a layer's two mHC sites.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SiteKind {
    /// Around the mixer (`hc_attn_*`, normed by `input_layernorm`).
    Attn,
    /// Around the feed-forward block (`hc_ffn_*`, normed by `post_attention_layernorm`).
    Ffn,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Site {
    pub layer: usize,
    pub kind: SiteKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixerKind {
    Kda,
    MlaDsa,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    /// Token ids -> the four fp32 streams (the embedding of each row, broadcast).
    Embed,
    /// The partials of the first site (every later site's come out of the preceding `ApplyPartials`).
    FirstPartials(Site),
    /// Reduce the site's partials, derive its mixing (Sinkhorn), collapse the streams and RMSNorm the result into the hidden row.
    Finalize(Site),
    /// The mixer of layer `layer`: reads the hidden row, writes the sublayer output.
    Mixer { layer: usize, kind: MixerKind },
    /// Layers 0-2: the gated MLP.
    DenseFfn { layer: usize },
    /// The shared expert of a MoE layer (before the routed experts).
    MoeShared { layer: usize },
    /// The router: logits and the eight picks with their gate weights.
    MoeRoute { layer: usize },
    /// The routed experts on the picks; `with_shared`: the shared expert's output is added last, in the combine.
    MoeExperts { layer: usize, with_shared: bool },
    /// Update the streams with the sublayer output and take the partials of `next` (that site's weights).
    ApplyPartials { next: Site },
    /// The last site's update (no next site).
    Apply,
    /// The mean of the four streams and the final RMSNorm, one pass.
    HeadMean,
    /// The `lm_head` on the head rows.
    LmHead,
    /// The last layer's KDA mixer up to its core output ([`plan_forward_head_rows`]): the qkv projection, the recurrence over every row
    /// and its state commit; the core output stays for [`Op::MixerOut`].
    MixerCore { layer: usize },
    /// Each sequence's last row (the head's) gathered to the front: every later op runs on those rows alone.
    HeadRows,
    /// The `o_proj` of [`Op::MixerCore`]'s output, on the head rows.
    MixerOut { layer: usize },
}

/// The kinds of layer `l`, from the config's recipe.
pub fn mixer_kind(cfg: &GlmConfig, l: usize) -> MixerKind {
    match cfg.layers[l].mixer {
        Mixer::Kda => MixerKind::Kda,
        _ => MixerKind::MlaDsa,
    }
}

pub fn is_dense(cfg: &GlmConfig, l: usize) -> bool {
    cfg.layers[l].ffn == Ffn::Dense
}

/// The forward of the first `layers` layers of `cfg` (`head`: followed by the mean, the final norm and the `lm_head`).
pub fn plan_forward(cfg: &GlmConfig, layers: usize, head: bool) -> Vec<Op> {
    build(cfg, layers, head, None)
}

/// The forward of the first `layers` layers followed by the head's mean and final norm but NOT the `lm_head`: what a prompt chunk that is not
/// the last runs when the NextN drafter taps every row's final-normed hidden state (the tap is `HeadMean`'s output).
pub fn plan_forward_tap(cfg: &GlmConfig, layers: usize) -> Vec<Op> {
    let mut ops = build(cfg, layers, true, None);
    assert_eq!(ops.pop(), Some(Op::LmHead));
    ops
}

/// [`plan_forward`] of a prompt forward whose last layer is KDA, with that layer's tail on the head rows alone. The recurrence (and its
/// state commit) runs on every row ([`Op::MixerCore`]); everything after it (the `o_proj`, the FFN site, the MoE block, the last apply)
/// feeds only the head, which reads each sequence's last row. With a head that tail runs after [`Op::HeadRows`] on those rows; without
/// one it does not run. `None` when the last layer is not KDA. The forward's other readers of the tail's rows (a drafter's tap of the
/// last layer, the NextN tap, a capture) are the caller's to rule out.
pub fn plan_forward_head_rows(cfg: &GlmConfig, layers: usize, head: bool) -> Option<Vec<Op>> {
    let n = layers.min(cfg.layers.len());
    if n == 0 || mixer_kind(cfg, n - 1) != MixerKind::Kda {
        return None;
    }
    let layer = n - 1;
    let mut ops = build(cfg, layers, head, None);
    let at = ops.iter().position(|o| *o == Op::Mixer { layer, kind: MixerKind::Kda })?;
    if head {
        ops.splice(at..=at, [Op::MixerCore { layer }, Op::HeadRows, Op::MixerOut { layer }]);
    } else {
        ops.truncate(at);
        ops.push(Op::MixerCore { layer });
    }
    Some(ops)
}

/// A deliberate bug in the schedule, so the tests can show that the validator ([`check_plan`]) catches each one. Production passes none.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mutant {
    /// `ApplyPartials` takes the partials of the site it just updated instead of the next one.
    SameSite,
    /// The attention layers are 2, 6, ... instead of 3, 7, ....
    MlaShifted,
    /// Layer 3 is dense too.
    FourDense,
    /// The shared expert is not handed to the combine.
    SharedDropped,
    /// The shared expert is computed after the routed experts.
    SharedAfter,
    /// The head skips the mean.
    NoHeadMean,
    /// Not a schedule bug: the head's top-k also offers the 24 padding ids at and past the tokenizer's vocabulary (the model's mutation hook
    /// only; [`plan_forward_mutated`] schedules it like production).
    UnmaskedVocab,
}

/// [`plan_forward`] with a deliberate bug (for tests).
pub fn plan_forward_mutated(cfg: &GlmConfig, layers: usize, head: bool, m: Mutant) -> Vec<Op> {
    build(cfg, layers, head, Some(m))
}

fn build(cfg: &GlmConfig, layers: usize, head: bool, m: Option<Mutant>) -> Vec<Op> {
    let is = |x: Mutant| m == Some(x);
    let n = layers.min(cfg.layers.len());
    let attn = |layer| Site { layer, kind: SiteKind::Attn };
    let ffn = |layer| Site { layer, kind: SiteKind::Ffn };
    let mut ops = vec![Op::Embed, Op::FirstPartials(attn(0))];
    for l in 0..n {
        let kind = if is(Mutant::MlaShifted) {
            if l % 4 == 2 {
                MixerKind::MlaDsa
            } else {
                MixerKind::Kda
            }
        } else {
            mixer_kind(cfg, l)
        };
        let dense = if is(Mutant::FourDense) { l < 4 } else { is_dense(cfg, l) };
        ops.push(Op::Finalize(attn(l)));
        ops.push(Op::Mixer { layer: l, kind });
        ops.push(Op::ApplyPartials { next: if is(Mutant::SameSite) { attn(l) } else { ffn(l) } });
        ops.push(Op::Finalize(ffn(l)));
        if dense {
            ops.push(Op::DenseFfn { layer: l });
        } else if is(Mutant::SharedAfter) {
            ops.extend([Op::MoeRoute { layer: l }, Op::MoeExperts { layer: l, with_shared: true }, Op::MoeShared { layer: l }]);
        } else {
            ops.extend([
                Op::MoeShared { layer: l },
                Op::MoeRoute { layer: l },
                Op::MoeExperts { layer: l, with_shared: !is(Mutant::SharedDropped) },
            ]);
        }
        if l + 1 < n {
            ops.push(Op::ApplyPartials { next: if is(Mutant::SameSite) { ffn(l) } else { attn(l + 1) } });
        } else {
            ops.push(Op::Apply);
        }
    }
    if head {
        if !is(Mutant::NoHeadMean) {
            ops.push(Op::HeadMean);
        }
        ops.push(Op::LmHead);
    }
    ops
}

/// The invariants of a schedule, written independently of the builder: `Err` names the first violated one.
pub fn check_plan(cfg: &GlmConfig, layers: usize, head: bool, ops: &[Op]) -> Result<(), String> {
    let n = layers.min(cfg.layers.len());
    if ops.first() != Some(&Op::Embed) {
        return Err("the forward must start with the embedding".into());
    }
    if ops.iter().filter(|o| **o == Op::Embed).count() != 1 {
        return Err("exactly one embedding".into());
    }
    // site order: every ApplyPartials names the site the next Finalize works on; the first Finalize is the first partials' site
    let mut pending: Option<Site> = match ops.get(1) {
        Some(Op::FirstPartials(s)) => Some(*s),
        _ => return Err("the first site's partials come right after the embedding".into()),
    };
    let mut expect_site = Site { layer: 0, kind: SiteKind::Attn };
    if pending != Some(expect_site) {
        return Err(format!("the first partials are of {:?}, not of layer 0's attention site", pending));
    }
    let mut seen_sites = 0usize;
    let mut applied_since_site = true;
    let mut mixers = Vec::new();
    let mut last_finalized: Option<Site> = None;
    for (i, op) in ops.iter().enumerate() {
        match *op {
            Op::Finalize(s) => {
                if pending != Some(s) {
                    return Err(format!("op {i}: Finalize({s:?}) but the partials in hand are of {pending:?}: the previous apply used the wrong site's weights"));
                }
                if s != expect_site {
                    return Err(format!("op {i}: site {s:?} out of order, expected {expect_site:?}"));
                }
                expect_site = match s.kind {
                    SiteKind::Attn => Site { layer: s.layer, kind: SiteKind::Ffn },
                    SiteKind::Ffn => Site { layer: s.layer + 1, kind: SiteKind::Attn },
                };
                pending = None;
                last_finalized = Some(s);
                applied_since_site = false;
                seen_sites += 1;
            }
            Op::ApplyPartials { next } => {
                if applied_since_site || pending.is_some() {
                    return Err(format!("op {i}: ApplyPartials without a finalized site before it"));
                }
                pending = Some(next);
                applied_since_site = true;
            }
            Op::Apply => {
                if applied_since_site || pending.is_some() {
                    return Err(format!("op {i}: Apply without a finalized site before it"));
                }
                applied_since_site = true;
            }
            Op::Mixer { layer, kind } => {
                if last_finalized != Some(Site { layer, kind: SiteKind::Attn }) {
                    return Err(format!("op {i}: the mixer of layer {layer} must follow its attention site"));
                }
                mixers.push((layer, kind));
            }
            _ => {}
        }
    }
    if seen_sites != 2 * n {
        return Err(format!("{seen_sites} sites for {n} layers (two each)"));
    }
    if !applied_since_site || pending.is_some() {
        return Err("the last site must be applied and leave no partials behind".into());
    }
    if ops.iter().filter(|o| matches!(o, Op::Apply)).count() != 1 {
        return Err("exactly one plain Apply, for the last site".into());
    }
    // layer types
    let want: Vec<(usize, MixerKind)> =
        (0..n).map(|l| (l, if l % 4 == 3 { MixerKind::MlaDsa } else { MixerKind::Kda })).collect();
    if mixers != want {
        return Err("layers 3, 7, ..., 43 are the MLA+DSA layers and every other one is KDA".into());
    }
    if mixer_kind_config_disagrees(cfg, n) {
        return Err("the config's own layer types disagree with 3, 7, ..., 43".into());
    }
    // feed-forward blocks and the shared expert's place
    for l in 0..n {
        let at = |pred: &dyn Fn(&Op) -> bool| ops.iter().position(pred);
        let dense = at(&|o| *o == Op::DenseFfn { layer: l });
        let shared = at(&|o| *o == Op::MoeShared { layer: l });
        let route = at(&|o| *o == Op::MoeRoute { layer: l });
        let experts = at(&|o| matches!(o, Op::MoeExperts { layer, .. } if *layer == l));
        if l < 3 {
            if dense.is_none() || shared.is_some() || route.is_some() || experts.is_some() {
                return Err(format!("layer {l} is dense: one DenseFfn and no MoE ops"));
            }
        } else {
            let (Some(s), Some(r), Some(e)) = (shared, route, experts) else {
                return Err(format!("layer {l} is a MoE layer: a shared expert, the router and the routed experts"));
            };
            if dense.is_some() {
                return Err(format!("layer {l} is a MoE layer and has a dense MLP too"));
            }
            if !(s < r && r < e) {
                return Err(format!(
                    "layer {l}: the shared expert is computed first, then the router, then the experts (added last)"
                ));
            }
            if ops[e] != (Op::MoeExperts { layer: l, with_shared: true }) {
                return Err(format!("layer {l}: the shared expert's output must be added last by the routed experts' combine"));
            }
        }
    }
    // the head
    let hm = ops.iter().filter(|o| matches!(o, Op::HeadMean)).count();
    let lm = ops.iter().filter(|o| matches!(o, Op::LmHead)).count();
    if head {
        if hm != 1 || lm != 1 || ops[ops.len() - 2..] != [Op::HeadMean, Op::LmHead] {
            return Err("the head is one mean+norm then the lm_head, last".into());
        }
    } else if hm != 0 || lm != 0 {
        return Err("no head requested but head ops are present".into());
    }
    Ok(())
}

fn mixer_kind_config_disagrees(cfg: &GlmConfig, n: usize) -> bool {
    (0..n).any(|l| (mixer_kind(cfg, l) == MixerKind::MlaDsa) != (l % 4 == 3))
        || (0..n.min(3)).any(|l| !is_dense(cfg, l))
        || (3..n).any(|l| is_dense(cfg, l))
}

#[cfg(test)]
mod tests {
    use super::*;

    const GLM_CONFIG: &str = include_str!("../../../../../testdata/glm53/config.json");

    fn real() -> GlmConfig {
        GlmConfig::from_json(&serde_json::from_str(GLM_CONFIG).unwrap()).unwrap()
    }

    #[test]
    fn the_full_schedule_by_hand() {
        let c = real();
        let ops = plan_forward(&c, 45, true);
        // embed, first partials, then per layer: finalize, mixer, apply_partials, finalize, ffn ops (1 dense, 3 MoE), apply_partials / apply;
        // then the head's two ops
        let ffn_ops = 3 + 42 * 3;
        assert_eq!(ops.len(), 2 + 45 * 5 + ffn_ops + 2);
        assert_eq!(&ops[..2], &[Op::Embed, Op::FirstPartials(Site { layer: 0, kind: SiteKind::Attn })]);
        assert_eq!(&ops[ops.len() - 2..], &[Op::HeadMean, Op::LmHead]);
        // layer 0: KDA + dense; layer 3: MLA + MoE
        assert_eq!(ops[3], Op::Mixer { layer: 0, kind: MixerKind::Kda });
        let l3 = ops.iter().position(|o| *o == Op::Mixer { layer: 3, kind: MixerKind::MlaDsa }).unwrap();
        assert_eq!(ops[l3 + 1], Op::ApplyPartials { next: Site { layer: 3, kind: SiteKind::Ffn } });
        assert_eq!(ops[l3 + 3], Op::MoeShared { layer: 3 });
        assert_eq!(ops[l3 + 5], Op::MoeExperts { layer: 3, with_shared: true });
        assert_eq!(ops[l3 + 6], Op::ApplyPartials { next: Site { layer: 4, kind: SiteKind::Attn } });
        check_plan(&c, 45, true, &ops).unwrap();
    }

    #[test]
    fn the_attention_layers_are_3_7_through_43_and_the_dense_ones_are_0_to_2() {
        let c = real();
        let ops = plan_forward(&c, 45, false);
        let mla: Vec<usize> = ops
            .iter()
            .filter_map(|o| match o {
                Op::Mixer { layer, kind: MixerKind::MlaDsa } => Some(*layer),
                _ => None,
            })
            .collect();
        assert_eq!(mla, (0..11).map(|k| 3 + 4 * k).collect::<Vec<_>>());
        let dense: Vec<usize> = ops
            .iter()
            .filter_map(|o| match o {
                Op::DenseFfn { layer } => Some(*layer),
                _ => None,
            })
            .collect();
        assert_eq!(dense, vec![0, 1, 2]);
        let moe = ops.iter().filter(|o| matches!(o, Op::MoeRoute { .. })).count();
        assert_eq!(moe, 42);
    }

    #[test]
    fn every_prefix_of_the_model_is_a_valid_schedule() {
        let c = real();
        for n in 1..=45 {
            for head in [false, true] {
                let ops = plan_forward(&c, n, head);
                check_plan(&c, n, head, &ops).unwrap_or_else(|e| panic!("{n} layers, head {head}: {e}"));
                // the last site is applied alone
                let last_apply = if head { ops.len() - 3 } else { ops.len() - 1 };
                assert_eq!(ops[last_apply], Op::Apply, "{n} layers");
            }
        }
    }

    #[test]
    fn each_mutant_breaks_an_invariant() {
        let c = real();
        for m in [
            Mutant::SameSite,
            Mutant::MlaShifted,
            Mutant::FourDense,
            Mutant::SharedDropped,
            Mutant::SharedAfter,
            Mutant::NoHeadMean,
        ] {
            let ops = plan_forward_mutated(&c, 45, true, m);
            let e = check_plan(&c, 45, true, &ops).err().unwrap_or_else(|| panic!("{m:?} was not caught"));
            eprintln!("{m:?}: caught: {e}");
        }
    }

    #[test]
    fn the_tap_plan_is_the_headed_plan_without_the_lm_head() {
        // the NextN drafter taps HeadMean's output (mean of the streams through the final norm) for every row of a chunk that wants no logits
        let c = real();
        for n in [1, 3, 45] {
            let (tap, full) = (plan_forward_tap(&c, n), plan_forward(&c, n, true));
            assert_eq!(tap[..], full[..full.len() - 1], "{n} layers");
            assert_eq!(tap.last(), Some(&Op::HeadMean));
            assert!(!tap.contains(&Op::LmHead));
            // everything before the head is the plain plan's, so the tap sees the same hidden state the logits would have been made from
            assert_eq!(tap[..tap.len() - 1], plan_forward(&c, n, false)[..]);
        }
    }

    #[test]
    fn the_vocabulary_mutant_leaves_the_schedule_alone() {
        let c = real();
        assert_eq!(plan_forward_mutated(&c, 45, true, Mutant::UnmaskedVocab), plan_forward(&c, 45, true));
    }

    #[test]
    fn the_head_rows_plan_is_the_plain_plan_with_the_last_kda_layer_cut_at_its_recurrence() {
        let c = real();
        for n in 1..=45 {
            let last_kda = mixer_kind(&c, n - 1) == MixerKind::Kda;
            for head in [false, true] {
                let full = plan_forward(&c, n, head);
                let Some(ops) = plan_forward_head_rows(&c, n, head) else {
                    assert!(!last_kda, "{n} layers: the last layer is KDA");
                    continue;
                };
                assert!(last_kda, "{n} layers: the last layer is MLA");
                let at = full.iter().position(|o| *o == Op::Mixer { layer: n - 1, kind: MixerKind::Kda }).unwrap();
                // everything before the last mixer is the plain plan's
                assert_eq!(ops[..at], full[..at], "{n} layers, head {head}");
                if head {
                    // the mixer cut in two around the gather, the rest (FFN site, block, apply, head) the plain plan's on the head rows
                    assert_eq!(ops[at..at + 3], [Op::MixerCore { layer: n - 1 }, Op::HeadRows, Op::MixerOut { layer: n - 1 }]);
                    assert_eq!(ops[at + 3..], full[at + 1..], "{n} layers");
                    let joined: Vec<Op> = ops
                        .iter()
                        .filter(|o| !matches!(o, Op::HeadRows | Op::MixerOut { .. }))
                        .map(|o| match *o {
                            Op::MixerCore { layer } => Op::Mixer { layer, kind: MixerKind::Kda },
                            o => o,
                        })
                        .collect();
                    check_plan(&c, n, true, &joined).unwrap_or_else(|e| panic!("{n} layers: {e}"));
                } else {
                    // no head: the forward ends at the last recurrence
                    assert_eq!(ops[at..], [Op::MixerCore { layer: n - 1 }], "{n} layers");
                }
            }
        }
        assert!(plan_forward_head_rows(&c, 45, true).is_some(), "layer 44 is KDA");
    }

    #[test]
    fn the_mixer_assignment_follows_the_config_not_a_literal() {
        // a config whose layer list says layer 7 is KDA cannot be scheduled as the pack's: check_plan notices the config disagrees
        let mut c = real();
        c.layers[7].mixer = Mixer::Kda;
        let ops = plan_forward(&c, 45, true);
        assert!(check_plan(&c, 45, true, &ops).is_err());
    }
}
