import gradio as gr
import time
from openai import OpenAI
from taskgen import *
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the LLM function with streaming support
def llm(system_prompt: str, user_prompt: str) -> str:
    ''' Function to interact with LLM API and stream response '''
    client = OpenAI(
        api_key=os.getenv("TOGETHER_API_KEY"),
        base_url="https://api.together.xyz/v1",
    )
    response = client.chat.completions.create(
        model='meta-llama/Llama-3.3-70B-Instruct-Turbo-Free',
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        stream=True
    )

    # Process the streaming response and yield content progressively
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content

# Define the Agent
agent = Agent(
    'Psychology counsellor',
    "Helps to understand and respond to User's emotion and situation. Reply User based on User Requests for the Conversation",
    llm=llm
)

# Define the ConversationWrapper
my_agent = ConversationWrapper(
    agent,
    persistent_memory={
        'User Requests for the Conversation': '',
        'User Emotion': '',
        'Summary of Key Incidents': "Key incidents relevant to understanding User's situation in one line"
    }
)

# Function to be used with Gradio with streaming response
def counsellor_chat(message,history):
    # Get the full response progressively using streaming
    system_prompt = "You are a psychology counsellor."
    user_prompt = message
    # Yielding progressively as the LLM responds
    response = llm(system_prompt, user_prompt)
    partial_response = ""

    for chunk in response:
        partial_response += chunk
        yield partial_response
        # time.sleep(0.05)  # Adjust the speed of streaming response

# Create Gradio interface
demo = gr.ChatInterface(
    counsellor_chat,
    type="messages",
    chatbot=gr.Chatbot(height=600, label="Chat with Psychology Counsellor"),
    title="Psychology Counsellor",
    description="Share how you're feeling, and the counsellor will help you understand your emotions and situation.",
    theme="soft",
    examples=["I've been feeling really down lately.",
              "I'm having trouble at work with my colleagues.",
              "I'm feeling anxious about my upcoming exam."],
    cache_examples=False,
    # flagging_mode="manual",
    # flagging_options=["Helpful", "Not Helpful", "Inaccurate", "Other"],
    # save_history=True,
)

if __name__ == "__main__":
    demo.launch(debug=True)
