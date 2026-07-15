"""
Standalone dtype/VRAM diagnostic — no Streamlit, no LangChain wrapper.
Loads the model directly with transformers so we can see exactly what's
happening, rather than going through HuggingFacePipeline's abstraction.

Run: python test_dtype.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def report_vram(label):
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"[{label}] allocated={allocated:.2f} GB | reserved={reserved:.2f} GB")


def load_and_check(kwarg_name, dtype_value):
    print(f"\n--- Loading with model_kwargs={{'{kwarg_name}': {dtype_value}}} ---")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    report_vram("before load")

    kwargs = {kwarg_name: dtype_value, "device_map": "auto"}
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)

    actual_dtype = next(model.parameters()).dtype
    print(f"Requested dtype : {dtype_value}")
    print(f"Actual dtype    : {actual_dtype}")
    print(f"Match?          : {actual_dtype == dtype_value}")

    report_vram("after load")
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak allocated  : {peak:.2f} GB")

    # cleanup before next test
    del model
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


if __name__ == "__main__":
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))

    # Test 1: the new-style key
    load_and_check("dtype", torch.float16)

    # Test 2: the legacy key (in case your transformers version only honors this one)
    load_and_check("torch_dtype", torch.float32)

    print("\nDone. Compare 'Requested dtype' vs 'Actual dtype' above.")
    print("If they don't match, that kwarg name isn't being honored by your installed transformers version.")