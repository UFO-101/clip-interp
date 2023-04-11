# %% Imports
from transformers import CLIPModel, CLIPProcessor
import transformer_lens
import torch
import torch.nn.functional as F
USE_CPU = False

# %% Initialise devices
if torch.cuda.is_available() and not USE_CPU:
    device = torch.device("cuda")
elif torch.has_mps and not USE_CPU:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# %% 
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
# %%
prompt = "A photo of a cat"
# Run prompt through text model
text_input = processor(text=prompt, return_tensors="pt", padding=True).input_ids.to(device)
text_output = model.get_text_features(input_ids=text_input).to(device)

# %% 
# Define a randomly initialised image for the image model
# image_input = torch.randn(1, 3, 224, 224, device=device)

image_input = torch.zeros(1, 3, 224, 224, device=device) + 0.5
image_output = model.get_image_features(pixel_values=image_input)
# Define optimizer
optimizer = torch.optim.Adam([image_input], lr=0.05)
num_steps = 200
image_input.requires_grad = True
text_input.requires_grad = False

def similarity(text_output, image_input):
    # Get the cosine similarity between the text and image
    image_output = model.get_image_features(pixel_values=image_input)
    x = text_output
    y = image_output
    # return - F.cosine_similarity(text_output, image_output, dim=-1)
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), 'replicate')
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])

for step in range(num_steps):
    # Calculate the similarity between the text and image
    optimizer.zero_grad()
    image = image_input
    text = text_output.detach_()
    loss = similarity(text, image).mean()
    loss += tv_loss(image).mean() * 40
    loss.backward()
    # Run backpropagation on the similarity, and get the gradient of the image
    # Update the image
    optimizer.step()
    optimizer.zero_grad()
    with torch.no_grad():
        image_input.clamp_(0, 1)
    # Print the similarity
    print(f"Step {step}: {loss}")


# %%
# Show image_input using matplotlib
import matplotlib.pyplot as plt
image_numpy = image_input[0].detach().cpu().permute(1, 2, 0)
plt.imshow(image_input[0].detach().cpu().permute(1, 2, 0))

# %%
