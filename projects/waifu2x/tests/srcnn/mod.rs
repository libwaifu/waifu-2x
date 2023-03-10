use image::DynamicImage;
use image::io::Reader;
use tch::Device;
use tch::nn::VarStore;

use waifu2x::SRCNN;

#[test]
pub fn run() {
    let image = Reader::open("tests/rename/miku_small_noisy_waifu2x.png").unwrap().with_guessed_format().unwrap().decode().unwrap();
    let vs = VarStore::new(Device::cuda_if_available());
    let net = SRCNN::new(&vs);
    // for variable in vs.variables() {
    //     println!("{:?}", variable);
    // }

    let out = net.resize_image2x(&image).unwrap();
    DynamicImage::ImageRgba32F(out).to_rgb8().save("tests/rename/miku_2x.png").unwrap();
}

