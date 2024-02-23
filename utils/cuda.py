import torch


# Determine if your system supports CUDA
def cuda():
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print("CUDA is available. Utilize GPUs for computation")
        device = torch.device("cuda")
    else:
        print("CUDA is not available. Utilize CPUs for computation.")
        device = torch.device("cpu")
        return device, cuda_available
