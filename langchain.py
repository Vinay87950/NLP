# from transformers import pipeline
# from langchain_huggingface import HuggingFacePipeline

# # Define the model and task
# model_id = "qwen-7b-chat"  # Ensure this model ID exists on Hugging Face Hub
# pipe = pipeline("question-answering", model=model_id)

# # Create a LangChain HuggingFacePipeline instance
# llm = HuggingFacePipeline(pipeline=pipe)

# # Example question-answering input
# context = "LangChain is a framework for integrating language models."
# question = "What is LangChain?"

# # Use the model through the pipeline
# response = llm({"context": context, "question": question})

# print(response)

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "model-00012-of-00017.safetensors"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "How many r in strawberry."
messages = [
    {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
