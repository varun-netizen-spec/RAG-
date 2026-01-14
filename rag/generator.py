from transformers import pipeline

llm = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    max_new_tokens=200,
    temperature=0.3,
    do_sample=True
)

def generate_answer(context, query):
    prompt = f"""
You are an agricultural market advisor for farmers.

Context:
{chr(10).join(context)}

Question:
{query}

Give clear selling advice with reasoning.
"""
    result = llm(prompt)
    return result[0]["generated_text"]
