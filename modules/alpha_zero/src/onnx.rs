use std::path::Path;

use tract_onnx::prelude::*;

pub struct OnnxModel {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, TypedModel>,
}

impl OnnxModel {
    pub fn load(path: impl AsRef<Path>, input_fact: InferenceFact) -> TractResult<Self> {
        let model = tract_onnx::onnx()
            .model_for_path(path)?
            .with_input_fact(0, input_fact)?
            .into_optimized()?
            .into_runnable()?;
        Ok(Self { model })
    }

    pub fn run(&self, input: Tensor) -> TractResult<TVec<Tensor>> {
        let outputs: TVec<TValue> = self.model.run(tvec!(input.into()))?;
        let mut out: TVec<Tensor> = TVec::new();
        for v in outputs {
            out.push(v.into_tensor());
        }
        Ok(out)
    }
}
