# MIT
# https://github.com/microsoft/Olive/tree/main/examples/directml/stable_diffusion

from typing import Literal
from pydantic import Field
from invokeai.app.invocations.baseinvocation import BaseInvocation, InvocationContext
from invokeai.app.invocations.prompt import PromptOutput

import warnings
from pathlib import Path

import onnxruntime as ort
from diffusers import OnnxStableDiffusionPipeline
from packaging import version

"""
TODO:
- Conversions
"""

class OnnxPipelineInvocation(BaseInvocation):
    """Onnx Pipeline Invocation"""
    #fmt: off
    type: Literal["onnx_pipeline"] = "onnx_pipeline"
    prompt: str = Field(default=None, description="Prompt")
    num_images: int = Field(default=1, description="Number of images to generate")
    batch_size: int = Field(default=1, description="Number of images to generate per batch")
    num_inference_steps: int = Field(default=50, description="Number of steps in diffusion process")
    dynamic_dims: bool = Field(default=False, description="Disable static shape optimization")
    model_dir: str = Field(default="D:\\Olive\\examples\\directml\\stable_diffusion\\models\\optimized\\runwayml\\stable-diffusion-v1-5", description="Model directory")
    #fmt: on

    def invoke(self, context: InvocationContext) -> PromptOutput:
        if version.parse(ort.__version__) < version.parse("1.15.0"):
            print("This script requires onnxruntime-directml 1.15.0 or newer")
            exit(1)

        use_static_dims = not self.dynamic_dims

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.run_inference(
                self.model_dir,
                self.prompt,
                self.num_images,
                self.batch_size,
                self.num_inference_steps,
                use_static_dims,
            )

        return PromptOutput(prompt=self.prompt)


    def run_inference_loop(
        self, pipeline, prompt, num_images, batch_size, num_inference_steps, image_callback=None, step_callback=None
    ):
        images_saved = 0

        def update_steps(step, timestep, latents):
            if step_callback:
                step_callback((images_saved // batch_size) * num_inference_steps + step)

        while images_saved < num_images:
            print(f"\nInference Batch Start (batch size = {batch_size}).")
            result = pipeline(
                [prompt] * batch_size,
                num_inference_steps=num_inference_steps,
                callback=update_steps if step_callback else None,
            )
            passed_safety_checker = 0


            for image_index in range(batch_size):
                if not result.nsfw_content_detected[image_index]:
                    passed_safety_checker += 1
                    if images_saved < num_images:
                        output_path = f"result_{images_saved}.png"
                        result.images[image_index].save(output_path)
                        if image_callback:
                            image_callback(images_saved, output_path)
                        images_saved += 1
                        print(f"Generated {output_path}")

            print(f"Inference Batch End ({passed_safety_checker}/{batch_size} images passed the safety checker).")


    def run_inference(self, optimized_model_dir, prompt, num_images, batch_size, num_inference_steps, static_dims):
        ort.set_default_logger_severity(3)

        print("Loading models into ORT session...")
        sess_options = ort.SessionOptions()
        sess_options.enable_mem_pattern = False

        if static_dims:
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
            optimized_model_dir, provider="DmlExecutionProvider", sess_options=sess_options
        )

        self.run_inference_loop(pipeline, prompt, num_images, batch_size, num_inference_steps)
