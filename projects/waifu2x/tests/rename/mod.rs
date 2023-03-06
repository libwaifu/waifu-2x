use std::collections::BTreeMap;
use std::path::Path;

use safetensors::tensor::{SafeTensorError, SafeTensors, serialize_to_file, TensorView};

fn rename_srcnn(input: &str, output: &str) -> Result<(), SafeTensorError> {
    let mut map: BTreeMap<String, String> = BTreeMap::new();
    let mut tensor_map: BTreeMap<String, TensorView> = BTreeMap::new();
    let bytes = std::fs::read(input).unwrap();
    let safe_tensor = SafeTensors::deserialize(&bytes).unwrap();
    map.insert("convolution_W".to_string(), "noise.1.weight".to_string());
    map.insert("convolution_B".to_string(), "noise.1.bias".to_string());
    for i in 1..=6 {
        map.insert(format!("convolution{}_W", i), format!("noise.{}.weight", i + 1));
        map.insert(format!("convolution{}_B", i), format!("noise.{}.bias", i + 1));
    }
    for (name, tensor) in safe_tensor.tensors() {
        match map.get(&name) {
            None => {
                eprintln!("{} not found", name);
            }
            Some(s) => {
                tensor_map.insert(s.to_string(), tensor);
            }
        }
    }
    serialize_to_file(&tensor_map, &None, &Path::new(output))
}

#[test]
fn test() {
    rename_srcnn("tests/rename/out.safetensors", "tests/rename/srcnn_denoise0.safetensors").unwrap();
}