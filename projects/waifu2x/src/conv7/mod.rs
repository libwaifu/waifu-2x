use std::path::Path;
use std::sync::Arc;

use image::{DynamicImage, Rgb32FImage};
use image::imageops::FilterType;
use ort::{Environment, OrtResult, Session};

use crate::DenoiseLevel;
use crate::utils::make_session;

#[derive(Debug)]
pub struct Convolution7 {
    off: Session,
    low: Session,
    medium: Session,
    high: Session,
    extreme: Session,
}

impl Convolution7 {
    pub fn new(runtime: &Arc<Environment>, model: &Path) -> OrtResult<Self> {
        if !model.is_dir() {
            panic!("Model path is not a directory")
        }
        Ok(Self {
            off: make_session(runtime, model.join("scale2.0x_model.onnx"))?,
            low: make_session(runtime, model.join("noise0_model.onnx"))?,
            medium: make_session(runtime, model.join("noise1_model.onnx"))?,
            high: make_session(runtime, model.join("noise2_model.onnx"))?,
            extreme: make_session(runtime, model.join("noise3_model.onnx"))?,
        })
    }
    // 142 -> 128
    pub fn upscale_2x(&mut self, image: &DynamicImage, level: DenoiseLevel) -> DynamicImage {
        let upscale = image.resize_exact(image.width() * 2 + 14, image.height() * 2 + 14, FilterType::CatmullRom)?;
        let model = self.select_net(level);
        println!("{:?}", model.inputs);
        // let array = upscale;
        // let out = model.run(&[array])?;
        todo!()
    }
    fn select_net(&mut self, level: DenoiseLevel) -> &mut Session {
        match level {
            DenoiseLevel::Off => {
                &mut self.off
            }
            DenoiseLevel::Low => {
                &mut self.low
            }
            DenoiseLevel::Medium => {
                &mut self.medium
            }
            DenoiseLevel::High => {
                &mut self.high
            }
            DenoiseLevel::Extreme => {
                &mut self.extreme
            }
        }
    }

    pub fn upscale_2x_batch(&self, image: &[Rgb32FImage], level: DenoiseLevel) -> OrtResult<Vec<DynamicImage>> {
        todo!()
    }
}
