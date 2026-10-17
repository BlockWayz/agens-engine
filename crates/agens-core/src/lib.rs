//! Host-side core of the Agens engine: config, layer map, checkpoint loading, CPU reference math.
//! Everything here builds and tests without a GPU.

pub mod config;
pub mod reference;
pub mod safetensors;
pub mod weights;

pub use config::{LayerSpec, Mixer, VolundrConfig};
pub use weights::{Checkpoint, LinearFormat};
