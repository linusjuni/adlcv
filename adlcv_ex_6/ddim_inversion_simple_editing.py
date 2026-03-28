"""
Task: Image editing via DDIM inversion.

This script demonstrates the full DDIM-based editing pipeline:
  1. Encode the input image to latent space.
  2. Run DDIM inversion to obtain intermediate noisy latents.
  3. Start denoising from an intermediate step with a *new* prompt.

The key insight: by starting denoising from an intermediate inverted latent,
the model preserves the structure of the original image while applying the
new prompt.  The `start_step` parameter controls the trade-off:
  - Low start_step  → more faithful to original structure (less edit freedom)
  - High start_step → more aggressive editing (less structure preserved)

Try the script with your own group photo and edit prompt!
"""

import torch
from torchvision import transforms as tfms
from diffusers.utils import load_image
from PIL import Image

from pipeline_setup import pipe, device, vae_scale_factor
from ddim_sampling import sample
from ddim_inversion import invert


def edit(input_image, input_image_prompt, edit_prompt,
         num_steps=100, start_step=30, guidance_scale=3.5):
    """
    Edit an image by DDIM inversion + re-sampling with a new prompt.

    Parameters
    ----------
    input_image : PIL.Image
        The source image to edit (should be 512×512).
    input_image_prompt : str
        A text description of the *source* image (used during inversion).
    edit_prompt : str
        The target description for the edited image.
    num_steps : int
        Number of DDIM inversion/sampling steps. More steps → more accurate.
    start_step : int
        Which inverted latent to start sampling from.
        Larger values allow more structural change.
    guidance_scale : float
        CFG scale for the sampling pass.

    Returns
    -------
    PIL.Image
        The edited image.
    """
    # Encode the input image to latent space
    with torch.no_grad():
        latent = pipe.vae.encode(
            tfms.functional.to_tensor(input_image).unsqueeze(0).to(device) * 2 - 1
        )
    l = vae_scale_factor * latent.latent_dist.sample()

    # Run DDIM inversion to get the trajectory of noisy latents
    inverted_latents = invert(l, input_image_prompt, num_inference_steps=num_steps)

    # Sample (denoise) from the intermediate inverted latent with the new prompt
    final_im = sample(
        edit_prompt,
        start_latents=inverted_latents[-(start_step + 1)][None],
        start_step=start_step,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
    )[0]

    return final_im


if __name__ == "__main__":
    # ── Example 1: Puppy → cat ─────────────────────────────────────────────────
    input_image = load_image(
        "me2.png"
    ).resize((512, 512))

    print("Running edit: puppy → cat")
    result = edit(
        input_image,
        input_image_prompt="A grumpy student",
        edit_prompt="A happy studentt",
        num_steps=50,
        start_step=20,
        guidance_scale=3.3,
    )
    result.save("output/me/lower_guidanc.png")

   

    # ── Hyperparameter exploration ─────────────────────────────────────────────
    # Try varying these parameters and observe their effect:
    #
    # num_steps:       More steps → more accurate inversion but slower.
    #                  Range: 50–500
    #
    # start_step:      Higher → more of the image is re-generated → bigger edit.
    #                  Must be < num_steps.  Typical range: 5–50.
    #
    # guidance_scale:  Higher → output more strongly follows the edit prompt.
    #                  Too high can cause artifacts.  Typical range: 3–10.
