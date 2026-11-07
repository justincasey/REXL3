//! A bump arena over a few large `cudaMalloc` blocks.
//!
//! The weights of a model are about 3,000 separate tensors; one `cudaMalloc` each costs a driver call and a page-table update
//! apiece, and leaves 3,000 allocations to free. The arena carves [`DevBuf`] views out of blocks of `block_bytes` (a tensor
//! larger than half a block gets a block of its own), so the target, draft and vision weights together take about twenty
//! allocations. A view keeps its block alive; nothing is freed until the arena and every view are gone.
//!
//! Every carve starts on a [`ALIGN`]-byte boundary, the alignment `cudaMalloc` itself guarantees, so kernels see the same
//! alignment as before. Requests are placed first-fit across the open blocks, so the small tensors that follow a block-sized
//! one fill what it left.

use crate::rt::{Block, CudaError, DevBuf};
use std::sync::{Arc, Mutex};

/// Alignment of every carve, in bytes (the alignment `cudaMalloc` guarantees).
pub const ALIGN: usize = 256;

struct Open {
    block: Arc<Block>,
    used: usize,
}

/// Where an arena's blocks live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockKind {
    Device,
    /// Pinned host memory mapped into the device's address space (`cudaHostAllocMapped`).
    MappedHost,
    /// Ordinary host memory the GPU reads in place (no allocation cost; slower to read, see `bench_decode_gbps_mapped`).
    PageableHost,
}

pub struct Arena {
    block_bytes: usize,
    /// What the blocks are made of (device memory unless a measurement asks for host memory).
    kind: BlockKind,
    /// The first block is `block_bytes` (the load's whole size, known up front) and any carve past it gets a block of exactly its size:
    /// no block is ever bigger than what is carved from it ([`Arena::exact`]).
    exact: bool,
    blocks: Mutex<Vec<Open>>,
}

fn round_up(n: usize, to: usize) -> usize {
    n.div_ceil(to) * to
}

impl Arena {
    /// An arena that allocates `block_bytes` at a time (at least [`ALIGN`]).
    pub fn new(block_bytes: usize) -> Arc<Self> {
        Self::with_kind(block_bytes, BlockKind::Device)
    }

    /// An arena whose blocks are mapped pinned host memory, which the GPU reads in place (zero-copy). Used to measure
    /// whether decode loses bandwidth reading weights from it (`bench_decode_gbps_mapped`).
    pub fn new_mapped_host(block_bytes: usize) -> Arc<Self> {
        Self::with_kind(block_bytes, BlockKind::MappedHost)
    }

    /// An arena of the given kind (measurement code picks host memory; the loader uses [`BlockKind::Device`]).
    pub fn with_kind(block_bytes: usize, kind: BlockKind) -> Arc<Self> {
        Arc::new(Self { block_bytes: block_bytes.max(ALIGN), kind, exact: false, blocks: Mutex::new(Vec::new()) })
    }

    /// A device arena for a load whose size is known before its first carve (`bytes`: the sum of its carves, each rounded up to [`ALIGN`]): one
    /// block of `bytes`, and a carve that does not fit what is left gets a block of exactly its own size. A fixed-size arena strands the unused
    /// tail of its last block for the process's life (a 1 GiB block under a 0.8 GiB drafter kept 0.22 GiB nobody could use).
    pub fn exact(bytes: usize) -> Arc<Self> {
        Arc::new(Self { block_bytes: bytes.max(ALIGN), kind: BlockKind::Device, exact: true, blocks: Mutex::new(Vec::new()) })
    }

    /// A view of `bytes` bytes (one byte for an empty request, as `DevBuf::alloc` does). Contents are uninitialised.
    pub fn alloc(&self, bytes: usize) -> Result<DevBuf, CudaError> {
        let need = round_up(bytes.max(1), ALIGN);
        let mut blocks = self.blocks.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(open) = blocks.iter_mut().find(|o| o.block.bytes() - o.used >= need) {
            let at = open.used;
            open.used += need;
            return Ok(DevBuf::view(&open.block, at, bytes));
        }
        // a request over half a block would strand most of a shared block: give it exactly what it needs (an exact arena: its one block first, then
        // exactly what each carve past it needs)
        let size = if self.exact {
            if blocks.is_empty() {
                self.block_bytes.max(need)
            } else {
                need
            }
        } else if need > self.block_bytes / 2 {
            need
        } else {
            self.block_bytes
        };
        let block = Arc::new(match self.kind {
            BlockKind::Device => Block::alloc(size)?,
            BlockKind::MappedHost => Block::alloc_mapped_host(size)?,
            BlockKind::PageableHost => Block::alloc_pageable_host(size)?,
        });
        let view = DevBuf::view(&block, 0, bytes);
        blocks.push(Open { block, used: need });
        Ok(view)
    }

    /// Bytes `cudaMalloc`ed so far.
    pub fn reserved_bytes(&self) -> usize {
        self.blocks.lock().unwrap_or_else(|e| e.into_inner()).iter().map(|o| o.block.bytes()).sum()
    }

    /// Bytes carved so far, alignment padding included.
    pub fn used_bytes(&self) -> usize {
        self.blocks.lock().unwrap_or_else(|e| e.into_inner()).iter().map(|o| o.used).sum()
    }

    /// Blocks allocated so far (the number of `cudaMalloc` calls).
    pub fn block_count(&self) -> usize {
        self.blocks.lock().unwrap_or_else(|e| e.into_inner()).len()
    }
}

#[cfg(all(test, not(gb_no_cuda)))]
mod tests {
    use super::*;

    fn gpu() -> bool {
        if crate::device_available() {
            true
        } else {
            eprintln!("no CUDA device: skipping");
            false
        }
    }

    #[test]
    fn carves_are_aligned_disjoint_and_share_blocks() {
        if !gpu() {
            return;
        }
        let a = Arena::new(1 << 20);
        let sizes = [1usize, 255, 256, 257, 4096, 100_000, 0, 7, 600_000, 600_000, 3];
        let bufs: Vec<DevBuf> = sizes.iter().map(|&n| a.alloc(n).unwrap()).collect();
        let mut spans: Vec<(usize, usize)> = Vec::new();
        for (b, &n) in bufs.iter().zip(&sizes) {
            assert_eq!(b.bytes(), n);
            assert_eq!(b.ptr() as usize % ALIGN, 0, "{n}-byte view is not {ALIGN}-aligned");
            let (lo, hi) = (b.ptr() as usize, b.ptr() as usize + n.max(1));
            assert!(spans.iter().all(|&(l, h)| hi <= l || h <= lo), "views overlap");
            spans.push((lo, hi));
        }
        // the 600 KB requests exceed half of a 1 MiB block, so they get blocks of their own; the small ones share one
        assert!(a.block_count() <= 4, "{} blocks for {} views", a.block_count(), bufs.len());
        assert!(a.used_bytes() <= a.reserved_bytes());
        // a view stays valid after the arena handle is gone, and writes through it read back
        let mut v = a.alloc(64).unwrap();
        drop(a);
        v.upload(&[7u8; 64]).unwrap();
        let back: Vec<u8> = v.to_vec().unwrap();
        assert_eq!(back, vec![7u8; 64]);
    }

    #[test]
    fn an_exact_arena_allocates_what_it_is_asked_for_and_no_more() {
        if !gpu() {
            return;
        }
        let sizes = [1usize, 255, 256, 600_000, 4096, 0, 100_000];
        let total: usize = sizes.iter().map(|&n| round_up(n.max(1), ALIGN)).sum();
        let a = Arena::exact(total);
        let bufs: Vec<DevBuf> = sizes.iter().map(|&n| a.alloc(n).unwrap()).collect();
        assert_eq!((a.block_count(), a.reserved_bytes(), a.used_bytes()), (1, total, total), "one block, all of it carved");
        // past the size it was given: a block of exactly the carve
        let extra = a.alloc(1000).unwrap();
        assert_eq!((a.block_count(), a.reserved_bytes()), (2, total + round_up(1000, ALIGN)));
        for b in bufs.iter().chain([&extra]) {
            assert_eq!(b.ptr() as usize % ALIGN, 0);
        }
    }
}
