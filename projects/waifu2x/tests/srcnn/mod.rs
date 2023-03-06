use image::DynamicImage;
use tch::Device;
use tch::nn::VarStore;

use sub_projects::SRCNN;

#[test]
pub fn run() {
    // let m = tch::vision::mnist::load_dir("data")?;
    let vs = VarStore::new(Device::cuda_if_available());
    let net = SRCNN::new(&vs);
    // for variable in vs.variables() {
    //     println!("{:?}", variable);
    // }

    let out = net.resize_image2x(&DynamicImage::new_rgb8(64, 64)).unwrap();
    println!("{:?}", out);
}

