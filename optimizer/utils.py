import numpy as np

AWS_CPU_COST = 0.0665
AWS_K80_COST = 0.634
AWS_V100_COST = 2.528

GCP_CPU_COST = 0.039
GCP_K80_COST = 0.315
GCP_P100_COST = 1.022

BARE_METAL_CPU_COST = np.inf
BARE_METAL_TXP_COST = np.inf

# NOTE: This cost is the cost of physical cpus, not virtual cpus
def get_cpu_cost(cloud, num_cpus):
    if cloud == "aws":
        return num_cpus * AWS_CPU_COST
    elif cloud == "gcp":
        return num_cpus * GCP_CPU_COST
    elif cloud == "bare_metal":
        return num_cpus * BARE_METAL_CPU_COST
    else:
        raise Exception("Unsupported cloud: {}".format(cloud))


def get_gpu_cost(cloud, gpu_type, num_gpus=1):
    if cloud == "aws":
        if gpu_type == "k80":
            return num_gpus * AWS_K80_COST
        elif gpu_type == "v100":
            return num_gpus * AWS_V100_COST
        elif gpu_type == "none" or gpu_type is None:
            return 0
        else:
            raise Exception("Unsupported gpu type: {} for cloud {}".format(gpu_type, cloud))
    elif cloud == "gcp":
        if gpu_type == "k80":
            return num_gpus * GCP_K80_COST
        elif gpu_type == "p100":
            return num_gpus * GCP_P100_COST
        elif gpu_type == "none" or gpu_type is None:
            return 0
        else:
            raise Exception("Unsupported gpu type: {} for cloud {}".format(gpu_type, cloud))
    elif cloud == "bare_metal":
        if gpu_type == "TXP":
            return num_gpus * BARE_METAL_TXP_COST
        else:
            raise Exception("Unsupported gpu type: {} for cloud {}".format(gpu_type, cloud))
