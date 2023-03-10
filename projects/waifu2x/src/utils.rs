use std::{path::Path, sync::Arc};

use ort::{session::Input, Environment, ExecutionProvider, OrtResult, Session, SessionBuilder};

pub fn make_session(runtime: &Arc<Environment>, model: &Path) -> OrtResult<Session> {
    let build = SessionBuilder::new(&runtime)?
        .with_execution_providers(&[ExecutionProvider::cuda(), ExecutionProvider::cpu()])?
        .with_model_from_file(model)?;
    Ok(build)
}

pub fn cancel_dimension(input: &mut Input, dimensions: &[usize]) {
    for dim in dimensions {
        match input.dimensions.get_mut(*dim) {
            Some(s) => *s = None,
            None => {
                log::info!("")
            }
        }
    }
}
