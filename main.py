import os
import torch
from datetime import datetime
from diffusers import StableDiffusion3Pipeline


if __name__ == "__main__":
    # Load the model
    model = "stabilityai/stable-diffusion-3.5-large"
    # model = "stabilityai/stable-diffusion-3.5-medium"
    # model = "stabilityai/stable-diffusion-3.5-large-turbo"

    # Construct pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(model, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()

    # Generate the images
    prompts = [
        # "A programmer touching grass",
        # "A dreamlike landscape with floating islands and waterfalls under a starry sky.",
        # "A Roman soldier standing guard in front of the Colosseum during sunset.",
        # "A cyberpunk character with neon tattoos in a rain-soaked alley.",
        # "President Donald Trump winning the elections",
        # "A capybara holding a sign that reads Hello World",
        # "Astrouat cat riding a pig on a snowy mountain peak under a purple sky, The cat has a sleek black and white fur coat, wearing a space suit with a clear visor and a helmet with a built-in communication device, The pig is a large, sturdy creature with a shiny coat of brown and white, carrying the cat on its back with ease, They are standing atop a snow-covered mountain peak, surrounded by majestic snow-capped mountains that stretch as far as the eye can see, The sky above them is a deep shade of purple, suggesting either dawn or dusk, The entire scene is bathed in a soft, ethereal glow, adding to the otherworldly atmosphere, The image captures the essence of adventure and camaraderie between two unlikely companions",
        # "(Fantasy photo:1.3) photo of a brave mouse warrior,  with a fierce expression, wearing leather armor, wielding a sharp sword and a sturdy shield, standing in a dramatic pose, full body, in a dimly lit forest clearing, surrounded by fallen leaves, lit by a single torch, shot from a low angle, on a Canon EOS 5D, with a 50mm lens, in the style of  Frank Frazetta",
        # "A pokemon from the final gym if pokemon had occurred in Spain",
        # "(macro photo:1.25) of woman lips Mode: Aperture Priority (A) Aperture: f/1.8 Shutter Speed: 1/250s ISO: 100 White Balance: Auto Focus Mode: AF-C with Eye AF enabled Focus Area: Flexible Spot Metering Mode: Multi or Spot Exposure Compensation: +0.3 Lens: 50mm f/1.8 or 85mm f/1.4 Lighting: Soft natural light or diffused artificial light",
        # "(Lifestyle photography:1.2) photo of freshly baked bread, a sourdough loaf,  with a crusty exterior,  still warm from the oven,  on a wooden table, surrounded by a scattering of flour, in a cozy bakery, with soft natural lighting streaming through a window, shot at eye level, on a Canon EOS 5D with a 50mm portrait lens, in the style of Wes Anderson.",
        # "A whimsical and creative image depicting a hybrid creature that is a mix of a waffle and a hippopotamus. This imaginative creature features the distinctive, bulky body of a hippo, but with a texture and appearance resembling a golden-brown, crispy waffle. The creature might have elements like waffle squares across its skin and a syrup-like sheen. It's set in a surreal environment that playfully combines a natural water habitat of a hippo with elements of a breakfast table setting, possibly including oversized utensils or plates in the background. The image should evoke a sense of playful absurdity and culinary fantasy.",
        # "Pedro Sánchez, president of Spain, holding a gun aiming a kitten while holding an elderly woman on the other hand",
        # "A whimsical, creative and colorful image depicting a hybrid creature that is a mix of a peafowl and some tropical fruits. This imaginative creature features the distinctive body and feathers of a peafowl, but with a texture and appearance resembling a mix of different colorful tropical fruits. It's set in a surreal environment that playfully combines a natural habitat of a peafowl with elements of a breakfast table setting, possibly including oversized utensils or plates in the background. The image should evoke a sense of playful absurdity and culinary fantasy.",
        "A mystical creature representing a technological group, robotic looking, futuristic, holding a futuristic bow-looking energy weapon, with fantasy lightning, colorful, hyper realistic, epic composition",
        "A wildfire seen from a drone 2 kilometers straight above the fire, in a forest, hyper realistic",
        "In this mesmerizing depiction, envision a psychedelic organic cyborg encapsulated in holographic plastic, illuminated by dramatic lighting. The fantasy-inspired composition is marked by intricate details, exuding an elegant and highly-detailed lifelike quality. Employing photorealistic techniques through digital painting, the artwork achieves a smooth and sharp focus, inviting viewers into a captivating realm of creativity. Drawing inspiration from the artistic mastery of John Collier, Albert Aublet, Krenz Cushart, Artem Demura, and Alphonse Mucha, this piece transcends conventional boundaries. The captivating interplay of elements such as holographic textures, dramatic lighting, and the fusion of organic and cybernetic components creates a visual symphony that stands out on platforms like ArtStation. The result is an illustration that seamlessly blends the ethereal with the technological, capturing the essence of the artists' collective brilliance",
    ]
    
    results = pipe(
        prompts,
        num_inference_steps=40,
        guidance_scale=3.5,
        height=1024,
        width=1024,
        max_sequence_length=512,
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

