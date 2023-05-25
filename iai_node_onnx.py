# MIT
# https://github.com/microsoft/Olive/tree/main/examples/directml/stable_diffusion
# model https://huggingface.co/aluhrs13/stable-diffusion-v1-5-olive-optimized
# pip install -r requirements.txt

from typing import Literal
from pydantic import Field
from invokeai.app.invocations.baseinvocation import BaseInvocation, InvocationContext
from invokeai.app.invocations.image import ImageOutput, build_image_output
from invokeai.app.models.image import ImageType

import onnxruntime as ort
from diffusers import OnnxStableDiffusionPipeline


class OnnxPipelineInvocation(BaseInvocation):
    """Onnx Pipeline Invocation"""
    #fmt: off
    type: Literal["onnx_pipeline"] = "onnx_pipeline"
    prompt: str = Field(default=None, description="Prompt")
    num_inference_steps: int = Field(default=50, description="Number of steps in diffusion process")
    static_dims: bool = Field(default=True, description="Disable static shape optimization")
    model_dir: str = Field(default="D:\\Olive\\examples\\directml\\stable_diffusion\\models\\optimized\\runwayml\\stable-diffusion-v1-5", description="Model directory")
    #fmt: on

    def invoke(self, context: InvocationContext) -> ImageOutput:
        ort.set_default_logger_severity(3)
        batch_size = 1

        print("Loading models into ORT session...")
        sess_options = ort.SessionOptions()
        sess_options.enable_mem_pattern = False

        if self.static_dims:
            # Not necessary, but helps DML EP further optimize runtime performance.
            # batch_size is doubled for sample & hidden state because of classifier free guidance:
            # https://github.com/huggingface/diffusers/blob/46c52f9b9607e6ecb29c782c052aea313e6487b7/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L672
            sess_options.add_free_dimension_override_by_name("unet_sample_batch", batch_size * 2)
            sess_options.add_free_dimension_override_by_name("unet_sample_channels", 4)
            sess_options.add_free_dimension_override_by_name("unet_sample_height", 64)
            sess_options.add_free_dimension_override_by_name("unet_sample_width", 64)
            sess_options.add_free_dimension_override_by_name("unet_time_batch", 1)
            sess_options.add_free_dimension_override_by_name("unet_hidden_batch", batch_size * 2)
            sess_options.add_free_dimension_override_by_name("unet_hidden_sequence", 77)

        pipeline = OnnxStableDiffusionPipeline.from_pretrained(
            self.model_dir, provider="DmlExecutionProvider", sess_options=sess_options
        )

        result = pipeline(
            [self.prompt] * batch_size,
            num_inference_steps=self.num_inference_steps,
        )

        image_type = ImageType.RESULT
        image_name = context.services.images.create_name(
            context.graph_execution_state_id, self.id
        )
        metadata = context.services.metadata.build_metadata(
            session_id=context.graph_execution_state_id, node=self
        )
        context.services.images.save(image_type, image_name, result.images[0], metadata)

        return build_image_output(
            image_type=image_type, image_name=image_name, image=result.images[0]
        )
