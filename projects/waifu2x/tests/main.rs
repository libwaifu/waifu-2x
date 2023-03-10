mod rename;
mod srcnn;

#[test]
fn ready() {
    println!("it works!")
}


#[test]
pub fn main(){
    tch::maybe_init_cuda();
    println!("{:?}", tch::Device::cuda_if_available());
    println!("{:?}", tch::Cuda::cudnn_is_available());
}