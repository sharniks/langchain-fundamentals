from transformers import BitsAndBytesConfig
import torch
import bitsandbytes as bnb

print("bitsandbytes:", bnb.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0)
      if torch.cuda.is_available() else "None")

# Actual functional test - this is what matters
quant_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
print("Config created OK:", quant_config)
