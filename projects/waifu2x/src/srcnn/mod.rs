use std::collections::BTreeMap;
use std::fmt::Debug;

use tch::{Device, Kind, nn::ModuleT, Tensor};
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


#[test]
pub fn run() {
    // let m = tch::vision::mnist::load_dir("data")?;
    let vs = VarStore::new(Device::cuda_if_available());
    let net = SRCNN::new(&vs);
    for variable in vs.variables() {
        println!("{:?}", variable);
    }
    let result = net.forward_t(&Tensor::zeros(&[2, 3, 128, 128], (Kind::Float, Device::Cpu)), false);
    println!("{:?}", result);
}


fn map() {
    let mut map: BTreeMap<String, String> = BTreeMap::new();
    map.insert("convolution_W".to_string(), "layer1.weight".to_string());
    map.insert("convolution_B".to_string(), "layer1.bias".to_string());
    for i in 1..=5 {
        map.insert(format!("convolution{}_W", i), format!("layer{}.weight", i));
        map.insert(format!("convolution{}_B", i), format!("layer{}.bias", i));
    }
}