
impl Module for LeakyRelu {
    fn forward(&self, xs: &Tensor) -> Tensor {
        xs.max1(xs * self.slope)
    }
}

#[derive(Debug)]
pub struct LeakyRelu {
    slope: f32,
}

impl LeakyRelu {
    pub fn new(slope: f32) -> LeakyRelu {
        LeakyRelu { slope }
    }
}