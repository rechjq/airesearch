from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig

def scratch_model(ma:MambaConfig, device, dtype):
    return MambaLMHeadModel(ma, device=device, dtype=dtype)

def load_model(load_path,device, dtype):
    return MambaLMHeadModel.from_pretrained(load_path, device=device, dtype=dtype)
