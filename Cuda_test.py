import torch
print("CUDA available:", torch.cuda.is_available())           # Should be True
print("Device count:", torch.cuda.device_count())             # Should be at least 1
print("Current device:", torch.cuda.current_device())         # Should be 0 or higher
print("Device name:", torch.cuda.get_device_name(0))          # Should display your GPU model
