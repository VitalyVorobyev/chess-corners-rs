import torch

print("built with mps:", torch.backends.mps.is_built())
print("mps available:", torch.backends.mps.is_available())

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
x = torch.randn(1024, 1024, device=device)
print(device, x.mean())
