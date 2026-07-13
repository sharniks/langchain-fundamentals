import torch
from langchain_huggingface import HuggingFacePipeline
import streamlit as st

st.header('Research Tool')


@st.cache_resource
def load_llm():
    # FIX 2: Explicitly bypass standard bitsandbytes and load native torch.float16
    # A 1B model in fp16/bf16 takes up a tiny ~2.2 GB of VRAM—completely safe and stable.
    # This completely circumvents the broken Windows bitsandbytes binaries.

    return HuggingFacePipeline.from_model_id(
        model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        task='text-generation',
        device_map="auto",  # Safely targets your CUDA device
        # Cuts VRAM consumption exactly in half natively # bfloat16 can be used as well
        model_kwargs={"dtype": torch.float16},
        pipeline_kwargs={
            "max_new_tokens": 512,
            "temperature": 0.7,
            "do_sample": True,
        },
    )


llm = load_llm()
underlying_model = llm.pipeline.model
print("Actual dtype via LangChain wrapper:",
      next(underlying_model.parameters()).dtype)

user_input = st.text_input('Enter your prompt')

if st.button('Summarize'):
    if user_input:
        with st.spinner("Generating summary..."):
            # FIX 3: Wrap inference in a PyTorch memory-tracking guard
            with torch.inference_mode():
                result = llm.invoke(user_input)
                st.write(result)

            # Flush temporary tensors out of Windows display buffers
            torch.cuda.empty_cache()
    else:
        st.warning("Please enter a prompt first.")
