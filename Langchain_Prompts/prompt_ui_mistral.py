import torch
from transformers import BitsAndBytesConfig, GenerationConfig
from langchain_huggingface import HuggingFacePipeline
import streamlit as st

st.header('Research Tool')


@st.cache_resource
def load_llm():

    # 4-bit quantization config (use this OR the 8-bit one below, not both)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # "nf4" is better than "fp4" for LLM weights
        bnb_4bit_compute_dtype=torch.float16,
        # extra ~0.4 bits/param saved, negligible speed cost
        bnb_4bit_use_double_quant=True,
    )

    generation_config = GenerationConfig(
        max_new_tokens=512,
        temperature=0.7,
        do_sample=True,
    )
    # added just for reading not usable here
    # repetition_penalty=1.15,   # penalizes repeated tokens — critical for small models
    # no_repeat_ngram_size=3,    # hard-blocks repeating any 3-gram

    return HuggingFacePipeline.from_model_id(
        model_id='mistralai/Mistral-7B-Instruct-v0.3',
        task='text-generation',
        device_map="auto",  # Safely targets your CUDA device
        model_kwargs={"quantization_config": quant_config},
        pipeline_kwargs={"generation_config": generation_config},
    )


llm = load_llm()

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
