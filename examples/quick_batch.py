"""Example script showing how to run Difix with batches of input/reference images."""

from pathlib import Path

from diffusers.utils import load_image

from pipeline_difix import DifixPipeline


ASSETS = Path("assets")


if __name__ == "__main__":
    pipe = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
    pipe.to("cuda")

    input_path = ASSETS / "example_input.png"
    ref_path = ASSETS / "example_ref.png"

    input_images = [
        load_image(str(input_path)),
        load_image(str(input_path)),
    ]
    ref_images = [
        load_image(str(ref_path)),
        load_image(str(ref_path)),
    ]
    prompt = "remove degradation"

    result = pipe(
        prompt,
        image=input_images,
        ref_image=ref_images,
        num_inference_steps=1,
        timesteps=[199],
        guidance_scale=0.0,
    )
    for idx, image in enumerate(result.images):
        image.save(f"example_output_{idx}.png")
