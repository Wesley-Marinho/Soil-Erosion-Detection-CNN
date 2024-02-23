import subprocess


def gpuInfo():
    # Get the GPU information
    try:
        gpu_info = subprocess.check_output(["nvidia-smi"]).decode("utf-8")
        print(gpu_info)
    except subprocess.CalledProcessError:
        print("Not connected to a GPU")

    return gpu_info
