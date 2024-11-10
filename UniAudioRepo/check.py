import torch

# Replace with the actual path to your checkpoint
checkpoint_path = "UniAudio/checkpoints/universal_model/ckpt_01455000.pth"

checkpoint = torch.load(checkpoint_path, map_location='cpu')
print("keys:", checkpoint.keys())  # Print out the keys to understand its structure