use std::fmt::Debug;

use image::{DynamicImage, EncodableLayout, Rgb32FImage, Rgba32FImage};
use image::imageops::FilterType;
use tch::{ Kind,  TchError, Tensor};
use tch::nn::{Conv2D, conv2d, Module, VarStore};

#[derive(Debug)]
pub struct SRCNN {
    conv1: Conv2D,
    conv2: Conv2D,
    conv3: Conv2D,
    conv4: Conv2D,
    conv5: Conv2D,
    conv6: Conv2D,
    conv7: Conv2D,
}

impl SRCNN {
    pub fn new(weights: &VarStore) -> SRCNN {
        let vs = &weights.root();
        SRCNN {
            // 1 * 3 * 142 * 142
            conv1: conv2d(vs / "noise1", 3, 32, 3, Default::default()),
            // 32 * 16 * 140 * 140
            conv2: conv2d(vs / "noise2", 32, 32, 3, Default::default()),
            // 64 * 32 * 138 * 138
            conv3: conv2d(vs / "noise3", 32, 64, 3, Default::default()),
            // 128 * 64 * 136 * 136
            conv4: conv2d(vs / "noise4", 64, 64, 3, Default::default()),
            // 256 * 128 * 134 * 134
            conv5: conv2d(vs / "noise5", 64, 128, 3, Default::default()),
            // 256 * 128 * 132 * 132
            conv6: conv2d(vs / "noise6", 128, 128, 3, Default::default()),
            // 512 * 256 * 130 * 130
            conv7: conv2d(vs / "noise7", 128, 3, 3, Default::default()),
            // 512 * 256 * 128 * 128
        }
    }
    /// Low-level api, used to enlarge images in batches
    ///
    /// **Attention**: `target = (N, 3, w + 6, h + 6)`
    pub fn resize(&self, target: &Tensor) -> Tensor {
        self.forward(target)
    }
    /// Enlarge the picture to twice, support transparent access
    pub fn resize_image2x(&self, image: &DynamicImage) -> Result<Rgba32FImage, TchError> {
        let w = (image.width() * 2 + 14) as u32;
        let h = (image.height() * 2 + 14) as u32;
        let rgb = image.resize(w, h, FilterType::CatmullRom).to_rgb32f();
        let tensor = Tensor::f_of_data_size(rgb.as_bytes(), &[1, 3, rgb.height() as i64, rgb.width() as i64], Kind::Float)?;
        let count = (image.width() * 2 * image.height() * 2 * 3) as usize;
        let mut out = vec![0.0; count];
        let out_tensor = self.forward(&tensor);
        println!("Tensor: {:?}, {}", out_tensor.size(),  out_tensor.numel());
        println!("Image: {}", image.width() * 2 * image.height() * 2 * 3);
        out_tensor.f_copy_data(&mut out, out_tensor.numel())?;
        let rgb = match Rgb32FImage::from_raw(image.width() * 2, image.height() * 2, out) {
            Some(s) => {
                s
            }
            None => { panic!("Failed to convert to rgb image"); }
        };
        let mut rgba = image.resize(image.width() * 2, image.height() * 2, FilterType::CatmullRom).to_rgba32f();
        for (x, y, pixel) in rgb.enumerate_pixels() {
            let target = rgba.get_pixel_mut(x, y);
            target.0[0] = pixel.0[0];
            target.0[1] = pixel.0[1];
            target.0[2] = pixel.0[2];
        }
        Ok(rgba)
    }
}

impl Module for SRCNN {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.apply(&self.conv1)
            .leaky_relu()
            .apply(&self.conv2)
            .leaky_relu()
            .apply(&self.conv3)
            .leaky_relu()
            .apply(&self.conv4)
            .leaky_relu()
            .apply(&self.conv5)
            .leaky_relu()
            .apply(&self.conv6)
            .leaky_relu()
            .apply(&self.conv7)
            .leaky_relu()
    }
}



