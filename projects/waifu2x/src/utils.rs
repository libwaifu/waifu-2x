use std::path::{Path, PathBuf};
use std::sync::Arc;

use ort::{Environment, ExecutionProvider, OrtResult, Session, SessionBuilder};

pub fn make_session(runtime: &Arc<Environment>, model: PathBuf) -> OrtResult<Session> {
    let build = SessionBuilder::new(&runtime)?
        .with_execution_providers(&[ExecutionProvider::cuda(), ExecutionProvider::cpu()])?
        .with_model_from_file(model)?;
    Ok(build)
}