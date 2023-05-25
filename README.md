# Onnx InvokeAI Node

An [InvokeAI](https://github.com/invoke-ai/InvokeAI) Node to run a basic Stable
Diffusion inference using an Onnx model, intended to be used to test an
[Olive optimized model](https://github.com/microsoft/Olive/tree/main/examples/directml/stable_diffusion).
I've uploaded a Stable Diffusion 1.5 olive optimized model on
[HuggingFace](https://huggingface.co/aluhrs13/stable-diffusion-v1-5-olive-optimized)
but using the directions linked to convert a model worked pretty easily for me.

Right now this is the bare minimum ported from Olive's sample. Depending on
difficult, I'll look at expanding it to work more cohesively with other Invoke
nodes.

## Requirements

I've only tested this with a developer install, probably won't work with a
normal one.

- In your InvokeAI venv run `pip install -r requirements.txt` (I'm not 100% sure
  these are right)
- Run InvokeAI
- Drag the Onnx node onto the canvas
