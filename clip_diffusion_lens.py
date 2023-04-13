"""Runs diffusion lens on a prompt and plots the similarity between a text and image over time."""

# %% Imports
from transformers import CLIPModel, CLIPProcessor
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import datetime

# %% 
def initialise_models(device):
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)
    pipe.safety_checker = lambda images, **kwargs: (images, False)
    _ = pipe("warmup", num_inference_steps=1)

    return model, processor, pipe
# %%
def get_ln_hidden_states(model, processor, prompt, device):
    """Get the layer norm hidden states for a given prompt. 
    Returns a list of hidden states, one for each layer except for the embedding layer."""
    ln_hidden_states = []

    # Run the prompt through the text model and get the hidden states
    tokens = processor.tokenizer(prompt, return_tensors="pt").to(device=device)
    output = model.text_model(**tokens, output_hidden_states=True)
    hidden_states = output.hidden_states

    # Remove the embedding layer
    hidden_states = hidden_states[1:]

    # For each layer, run the hidden state through final layer norm
    for hidden_state in hidden_states:
        ln_hidden_states.append(model.text_model.final_layer_norm(hidden_state))

    return ln_hidden_states

def generate_image_from_ln_hidden_states(pipe, ln_hidden_states, num_inference_steps=25):
    """Get an image from a list of layer normed hidden states."""
    images = []
    for hidden_state in tqdm(ln_hidden_states):
        image = pipe(prompt_embeds=hidden_state, num_inference_steps=num_inference_steps)
        images.append(image.images[0])

    return images

def display_list_of_images(images, prompt, save_image=False):
    """Display a list of images."""
    # Code to plot all 12 images in a grid
    fig, axs = plt.subplots(3, 4, figsize=(20, 15))
    for i, ax in enumerate(axs.flat):
        ax.imshow(images[i])
        ax.set_title(f"Layer {i}")
    # Set main title
    fig.suptitle(f"Prompt: {prompt}", fontsize=20)
    plt.show()

    if save_image:
        fig.savefig(f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")

# %% Initialisation
if __name__ == "__main__":
    USE_CPU = False

    # Initialise devices
    if torch.cuda.is_available() and not USE_CPU:
        device = torch.device("cuda")
    elif torch.has_mps and not USE_CPU:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Initialise models
    model, processor, pipe = initialise_models(device)

# %% Generate images, one for each layer
if __name__ == "__main__":
    prompt = "a photo of an astronaut riding a horse on Mars"
    ln_hidden_states = get_ln_hidden_states(model, processor, prompt, device)
    images = generate_image_from_ln_hidden_states(pipe, ln_hidden_states, num_inference_steps=25)

# %% Run the images through the image model to get the embeddings
if __name__ == "__main__":
    image_embeddings = []
    for image in tqdm(images):
        image = processor(images=image, return_tensors="pt").to(device=device)
        output = model.vision_model(image.pixel_values)
        image_embeddings.append(output)

# %% Get the similarity between a text embedding and the image embeddings
if __name__ == "__main__":
    # text = "A photo of an astronaut riding a horse on Mars"
    text = "Mars"
    text_tokens = processor(text=text, return_tensors="pt").to(device=device)
    text_embedding = model.text_model(**text_tokens)

    similarities = []
    for image_embedding in image_embeddings:
        image_proj = model.visual_projection(image_embedding.pooler_output)
        horse_proj = model.text_projection(text_embedding.pooler_output)
        similarity = F.cosine_similarity(image_proj, horse_proj)
        similarity = similarity.detach().cpu().numpy()
        similarities.append(similarity)

    # Plot the similarities
    plt.plot(similarities)
    plt.title(f"Similarity(Image, Embedding('{text}'))")
    plt.xlabel("Layer")
    plt.ylabel("Similarity")
    # Add grey vertical lines to show the layers
    for i in range(1, 12):
        plt.axvline(i, color="grey", linestyle="-", linewidth=0.2)
    plt.show()

# %%
# Display the images
if __name__ == "__main__":
    display_list_of_images(images, prompt, save_image=False)