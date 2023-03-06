// CNN model. This should rearch 99.1% accuracy.

use std::path::Path;
use nn::{Conv2D, conv2d};
use tch::{Device, nn, nn::ModuleT, nn::OptimizerConfig, Tensor};
use tch::nn::{conv_transpose2d, ConvConfigND, VarStore};

use anyhow::Result;

#[derive(Debug)]
struct SRCNN {
    conv1: Conv2D,
    conv2: Conv2D,
    conv3: Conv2D,
    conv4: Conv2D,
    conv5: Conv2D,
    conv6: Conv2D,
    conv7: Conv2D,
}

impl SRCNN {
    fn new(vs: &VarStore) -> SRCNN {
        let vs = &vs.root();
        SRCNN {
            // 1 * 3 * 142 * 142
            conv1: conv2d(vs, 3, 32, 3, Default::default()),
            // 32 * 16 * 140 * 140
            conv2: conv2d(vs, 32, 32, 3, Default::default()),
            // 64 * 32 * 138 * 138
            conv3: conv2d(vs, 32, 64, 3, Default::default()),
            // 128 * 64 * 136 * 136
            conv4: conv2d(vs, 64, 64, 3, Default::default()),
            // 256 * 128 * 134 * 134
            conv5: conv2d(vs, 64, 128, 3, Default::default()),
            // 256 * 128 * 132 * 132
            conv6: conv2d(vs, 128, 128, 3, Default::default()),
            // 512 * 256 * 130 * 130
            conv7: conv2d(vs, 128, 3, 3, Default::default()),
            // 512 * 256 * 128 * 128
        }
    }
}

impl ModuleT for SRCNN {
    fn forward_t(&self, xs: &Tensor, _: bool) -> Tensor {
        xs.view([-1, 1, 142, 142])
            .apply(&self.conv1)
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

#[test]
pub fn run() {
    // let m = tch::vision::mnist::load_dir("data")?;
    let vs = VarStore::new(Device::cuda_if_available());
    let net = SRCNN::new(&vs);
    for variable in vs.variables() {
        println!("{:?}", variable);
    }
}