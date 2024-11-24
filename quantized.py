import os
import torch
from datetime import datetime
from diffusers import StableDiffusion3Pipeline
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel


if __name__ == "__main__":
    # Load the model
    model = "stabilityai/stable-diffusion-3.5-large"
    # model = "stabilityai/stable-diffusion-3.5-medium"
    # model = "stabilityai/stable-diffusion-3.5-large-turbo"

    # Quantizing the model with diffusers
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model_nf4 = SD3Transformer2DModel.from_pretrained(
        model,
        subfolder="transformer",
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16
    )

    # Construct pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model,
        transformer=model_nf4,
        torch_dtype=torch.bfloat16
    )
    pipe.to("cuda")

    # Generate the images
    prompts = [
        "A programmer touching grass",
        "A dreamlike landscape with floating islands and waterfalls under a starry sky.",
        "A Roman soldier standing guard in front of the Colosseum during sunset.",
        "A cyberpunk character with neon tattoos in a rain-soaked alley.",
        "President Donald Trump winning the elections",
        "A capybara holding a sign that reads Hello World",
    ]
    
    results = pipe(
        prompts,
        num_inference_steps=20,
        guidance_scale=3.5,
        height=512,
        width=512
    )
    
    images = results.images
    
    # Create a directory with the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = os.path.join("images", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the images in the created directory
    for i, img in enumerate(images):
        img_path = os.path.join(output_dir, f"image_{i}.png")
        img.save(img_path)
    
    print(f"Images saved to {output_dir}")

