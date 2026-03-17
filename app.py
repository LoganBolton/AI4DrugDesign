import gradio as gr
import os
from openai import OpenAI
from dotenv import load_dotenv

from tabs import (
    integrated_pipeline,
)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chat_fn(message, history):
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})

    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model="gpt-5-nano-2025-08-07",
        messages=messages
    )

    return response.choices[0].message.content


with gr.Blocks() as demo:
    integrated_pipeline.create_tab()

if __name__ == "__main__":
    demo.launch()