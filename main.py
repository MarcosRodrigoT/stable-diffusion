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
        # # RANDOM
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
        # "A mystical creature representing a technological group, robotic looking, futuristic, holding a futuristic bow-looking energy weapon, with fantasy lightning, colorful, hyper realistic, epic composition",
        # "A wildfire seen from a drone 2 kilometers straight above the fire, in a forest, hyper realistic",
        # "In this mesmerizing depiction, envision a psychedelic organic cyborg encapsulated in holographic plastic, illuminated by dramatic lighting. The fantasy-inspired composition is marked by intricate details, exuding an elegant and highly-detailed lifelike quality. Employing photorealistic techniques through digital painting, the artwork achieves a smooth and sharp focus, inviting viewers into a captivating realm of creativity. Drawing inspiration from the artistic mastery of John Collier, Albert Aublet, Krenz Cushart, Artem Demura, and Alphonse Mucha, this piece transcends conventional boundaries. The captivating interplay of elements such as holographic textures, dramatic lighting, and the fusion of organic and cybernetic components creates a visual symphony that stands out on platforms like ArtStation. The result is an illustration that seamlessly blends the ethereal with the technological, capturing the essence of the artists' collective brilliance",
        # # WILDFIRE
        # """A breathtaking aerial view of a raging wildfire consuming a dense forest, captured from a UAV at high altitude. Smoke billows up into the sky, creating a dramatic contrast with the bright orange flames and the surrounding green canopy. The scene conveys urgency and raw natural power, photorealistic, vivid colors""",
        # """A close-up UAV perspective of a wildfire spreading across a dry savanna landscape. Flames are licking at the grass, with patches of charred earth visible in the foreground. Smoke is rising in thick plumes, partly obscuring the horizon, with golden sunlight filtering through""",
        # """An aerial shot of a wildfire from a UAV perspective at night, showing glowing embers scattered across a smoldering forest floor. Bright orange and red flames illuminate the surrounding darkness, with smoke swirling in the cold night air""",
        # """A high-resolution satellite image of a massive wildfire engulfing a forested area. Bright red and orange flames are clearly visible along the fire line, with a trail of dark smoke spreading across the image. Nearby rivers and roads are faintly visible, emphasizing the scale of the disaster""",
        # """A satellite view of a wildfire spreading across a hilly terrain, with vivid flames visible as glowing spots amidst thick gray smoke. The surrounding areas show patches of untouched vegetation, contrasting the charred landscape. The Earth's curvature is faintly visible in the background""",
        # """A composite satellite image of a wildfire during daylight, showing the fire's path cutting through dense forest regions. The smoke plume rises high into the atmosphere, blending with the clouds, while burned-out areas appear blackened against the lush green surroundings""",
        # """A dramatic ground-level photograph of a wildfire tearing through a forest, with towering flames consuming trees and thick smoke obscuring the sky. The ground is littered with ash and embers, and the intensity of the fire creates a glowing, apocalyptic scene""",
        # """A wildfire seen from the perspective of a firefighter on the ground. Flames rage through dry brush, with heat distortion visible in the air. The smoky atmosphere creates a hazy, surreal effect, with the sun dimly shining through the thick smoke""",
        # """A ground-level view of a wildfire burning through a rural area, showing a small wooden house in the foreground surrounded by dry grass, with flames encroaching in the background. The smoke-filled sky casts a reddish hue over the scene""",
        # """A panoramic ground-level view of a wildfire in a canyon. Flames leap up the canyon walls, while thick smoke creates an eerie, orange glow. A dirt trail leads into the scene, emphasizing the fire's scale and proximity to human access points""",
        # SOCCER
        # """A close-up shot of a soccer referee holding up a yellow card to a player in a professional soccer match, with the player's expression showing frustration, a stadium crowd blurred in the background, and the grass field visible.""",
        # """A medium shot of a referee giving a yellow card to a soccer player, with teammates standing nearby, the action occurring under stadium lights, and banners visible in the background.""",
        # """A wide-angle shot of a soccer match scene showing a referee displaying a yellow card to a player while other players look on from a distance, with the full field and stadium in view.""",
        # """A dynamic side-view shot of a soccer referee holding up a yellow card to a player mid-game, with the audience in the background and the team benches visible.""",
        # """A front-facing close-up shot of a soccer referee firmly showing a yellow card to a player wearing a red jersey, with the player gesturing in protest and the green grass field behind them.""",
        # """An aerial view of a soccer match showing a referee issuing a yellow card to a player near the center of the field, with other players scattered around and stadium seating surrounding the scene.""",
        # """A mid-range shot of a heated moment in a soccer game, with the referee holding a yellow card high in the air, players arguing nearby, and the goalposts visible in the background.""",
        # """A close-up shot from a player's perspective of a referee showing a yellow card directly at them, with the referee's stern face in focus and the player's hand partially visible in the foreground.""",
        # """A medium shot capturing a referee giving a yellow card to a soccer player, with the opposing team standing in the background and the corner flag visible on the side.""",
        # """A panoramic shot of a soccer match featuring a referee showing a yellow card to a player near the sideline, with the coach's technical area and cheering fans in the stadium clearly visible.""",
        # TOP-DOWN VIEW
        """A realistic top-down drone view of a wildfire burning through a dense forest. The perimeter of the fire is clearly visible, with bright orange flames forming irregular shapes amidst the charred black earth. Smoke rises in light gray plumes, partially obscuring some areas, with patches of unburnt green forest at the edges""",
        """A high-resolution aerial view of a wildfire consuming a dry grassland. The camera captures the entire fire perimeter, with glowing embers at the edge of the flames and smoke spreading outward in thin layers. The ground shows a stark contrast between blackened, burned areas and untouched golden grass""",
        """A top-down view of a wildfire spreading across hilly terrain, showing the fireline as a glowing orange ring encircling blackened land. The surrounding areas include patches of green vegetation and dry brush, with smoke visibly drifting away from the fire in a realistic manner""",
        """A cenital perspective of a large wildfire spreading across a canyon. The fire perimeter is clearly outlined, with intense flames concentrated along the canyon walls. Smoke rises and dissipates naturally, revealing scorched earth and rocky terrain within the fire zone""",
        """An aerial drone shot looking straight down at a wildfire burning through farmland. The fire has consumed several fields, with orange flames forming a jagged perimeter. Smoke trails are thin and wispy, and nearby irrigation channels and green patches remain untouched""",
        """A realistic top-down view of a wildfire consuming a section of a pine forest. Flames create a sharp boundary between the charred trees and the unburnt forest. Gray smoke rises from the perimeter, dispersing as it moves outward, with faint tracks of wildlife paths visible""",
        """A drone's cenital view of a wildfire in a savanna landscape. The fire spreads in a near-circular shape, with bright flames along the perimeter. Blackened soil and isolated tree clusters remain within the fire zone, while the surrounding grasslands are untouched and golden in color""",
        """A top-down drone image of a wildfire consuming a riverbank forest. The perimeter of the fire is sharply defined, with flames licking at the edge of the green canopy. A river winds through the scene, reflecting the orange glow of the flames, while smoke forms natural gradients across the frame""",
        """A drone's cenital view of a wildfire spreading through rocky terrain. The fire perimeter is irregular but well-defined, with fiery orange flames against the dark gray rocks. Sparse patches of dry vegetation ignite near the edges, with light smoke rising into the clear sky""",
        """A realistic aerial view of a wildfire burning in a marshy wetland area. The flames form distinct perimeters around dry grass patches, while small ponds and water bodies remain untouched, reflecting the chaotic glow of the fire. Smoke trails extend outward naturally, without distorting the scene""",
        # AIRBUS LOGO
        # """Airbus logo for a new project about wildfire detection and monitoring, must include a capital A in the logo, including path planning, simple logo, colorful, fire, containing aircrafts, futuristic, cybernetic, vector image""",
    ]

    results = pipe(
        prompts,
        num_inference_steps=40,
        guidance_scale=3.5,  # 3.5 - 5.5 -> The bigger this number the more the image will have to resemble the prompt
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
