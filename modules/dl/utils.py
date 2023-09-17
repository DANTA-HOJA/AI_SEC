import torch

from ..shared.clioutput import CLIOutput
# -----------------------------------------------------------------------------/


def set_gpu(cuda_idx:int, cli_out:CLIOutput=None):
    """
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Can't find any GPU")
    
    device = torch.device("cuda")
    
    torch.cuda.set_device(cuda_idx)
    device_num = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device_num)
    torch.cuda.empty_cache()
    
    if cli_out: cli_out.write(f"Using '{device}', device_name = '{device_name}'")
    
    return device
    # -------------------------------------------------------------------------/