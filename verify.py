import torch

print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print(torch.rand(2, 2))