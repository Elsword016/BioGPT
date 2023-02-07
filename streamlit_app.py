##Create a streamlit app for BioGPT
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
#use cache to load the model and tokenizer
@st.cache(allow_output_mutation=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large")
    model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT-Large")
    return tokenizer, model

tokenizer, model = load_model()

def generate_text(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt")  # Batch size 1
    outputs = model.generate(**input_ids,max_length=200,length_penalty=1.0, num_beams=1)  #play around these param to make the output quality better.
    gen_txt = tokenizer.decode(outputs[0],skip_special_tokens=True)
    return gen_txt
st.title("BioGPT")
st.write("This is a demo of BioGPT, a large-scale pre-trained language model for biomedical text generation.")
st.write("You can use it to generate biomedical text, such as abstracts, titles, and sentences.")


#input text
prompt = st.text_input('Enter your text here:')

if st.button('Generate'):
    with st.spinner('Generating text...'):
        gen_txt = generate_text(prompt)
        st.write(gen_txt)
#also display the word count
    st.write("Word count: ", len(gen_txt.split()))
