import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community import llms
# Laden einfach zu Modellen 
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

## Funktion zum Abrufen der Antwort vom LLama3-Modell

def getResponse(input_text, no_words, topic_style):

    # einfache Überschriften schreiben 
    template = """
        Write a topic for {topic_style} for profile {input_text} within {no_words} words.
            """
    
    prompt = PromptTemplate(
        input_variables=["topic_style", "input_text","no_words"],
        template=template
    )

     
    formatted_prompt = prompt.format(
        topic_style=topic_style,
        input_text=input_text,
        no_words=no_words
    )

    # tokeinze 
    inputs = tokenizer(formatted_prompt, return_tensors="pt")

    # generate response 
    outputs = model.generate(
        inputs["input_ids"],
        max_length=int(no_words)+ len(inputs["input_ids"][0]),
        temperature =0.7, # for more randomness
        num_return_sequence =1,
        pad_token_id = tokenizer.eos_token_id 

    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


st.set_page_config(page_title="Generate Research Topics",
                   page_icon=':bar_chart:',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Generate Research Topic")

input_text = st.text_input("Enter your summary")

## creating columns for addintional feature

col1, col2 = st.columns([5,5])

with col1:
    no_words = st.text_input("number of words")

with col2:
    topic_style = st.selectbox('Searching the topic for', 
                               ('Researchers', 'Students', 'Professors'), index=0)
    
submit = st.button("Generate")

## Endlich Response
if submit:
    st.write(getResponse(input_text, no_words, topic_style))