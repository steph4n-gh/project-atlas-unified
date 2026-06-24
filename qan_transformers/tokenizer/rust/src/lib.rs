//! microtok — focused experiments in micro-optimizing LLM tokenizers.
//!
//! Start here for the core BPE logic, fast paths, and data structures we are
//! trying to squeeze.

pub mod bpe;

/// Re-export common types for experiments.
pub use bpe::SimpleBpe;
