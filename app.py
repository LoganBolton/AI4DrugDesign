import gradio as gr
import os
from openai import OpenAI
from dotenv import load_dotenv

from tabs import (
    visualization_3d,
    chat,
    compound_optimization,
    compound_visualization,
    #molecular_docking,
    protein_analysis,
    protein_image_tab,
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
    chat.create_tab()
    protein_analysis.create_tab()
    compound_optimization.create_tab()
    compound_visualization.create_tab()
    #molecular_docking.create_tab()
    protein_image_tab.create_tab()
    visualization_3d.create_tab()

demo.launch()