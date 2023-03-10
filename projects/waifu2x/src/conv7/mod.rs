use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use image::{imageops::FilterType, DynamicImage, Rgb32FImage};
use ndarray::ArrayD;
use ort::{tensor::InputTensor, Environment, OrtResult, Session};

use crate::{
    utils::{cancel_dimension, make_session},
    DenoiseLevel,
};

#[derive(Debug)]
pub struct Convolution7 {
    session: Session,
}

impl Convolution7 {
    pub fn new(runtime: &Arc<Environment>, models: &Path, level: DenoiseLevel) -> OrtResult<Self> {
        let mut session = make_session(runtime, &select_net(models, level))?;
        match session.inputs.get_mut(0) {
            Some(s) => cancel_dimension(s, &[2, 3]),
            None => {
                panic!("")
            }
        };
        Ok(Self { session })
    }
    // 142 -> 128
    pub fn upscale_2x(&mut self, image: &DynamicImage) -> OrtResult<DynamicImage> {
        let upscale = image.resize_exact(image.width() * 2 + 14, image.height() * 2 + 14, FilterType::CatmullRom);
        let tensor = one_rgb_to_tensor(upscale.to_rgb32f());
        println!("{:?}", self.session.inputs);
        let out = self.session.run(&[tensor])?;

        // let array = upscale;
        // let out = model.run(&[array])?;
        todo!()
    }

    pub fn upscale_2x_batch(&mut self, image: &[Rgb32FImage], level: DenoiseLevel) -> OrtResult<Vec<DynamicImage>> {
        todo!()
    }
}

fn select_net(path: &Path, level: DenoiseLevel) -> PathBuf {
    let model = match level {
        DenoiseLevel::Off => path.join("scale2.0x_model.onnx"),
        DenoiseLevel::Low => path.join("noise0_model.onnx"),
        DenoiseLevel::Medium => path.join("noise1_model.onnx"),
        DenoiseLevel::High => path.join("noise2_model.onnx"),
        DenoiseLevel::Extreme => path.join("noise3_model.onnx"),
    };
    model
}

fn one_rgb_to_tensor(image: Rgb32FImage) -> InputTensor {
    let shape = vec![1, image.width() as usize, image.height() as usize, 3];
    let array = ArrayD::from_shape_vec(shape, image.as_raw().to_vec()).unwrap();
    InputTensor::FloatTensor(array)
}
