from mlx_lm import load, generate
from mlx_lm.models.cache import make_prompt_cache

# Load the model
model, tokenizer = load("mlx-community/gemma-3-1b-it-4bit-DWQ")

# Create a cache for efficient multi-turn conversations
prompt_cache = make_prompt_cache(model)

# Set up our chat history
messages = []

def chat(user_input):
    # Add user message to chat history
    messages.append({"role": "user", "content": user_input})

    # Format the conversation using the chat template
    prompt = tokenizer.apply_chat_template(messages,add_generation_prompt=True)

    # Generate response using cached key-value pairs for efficiency
    response = generate(
        model,
        tokenizer,
        prompt,
        max_tokens=1024,
        prompt_cache=prompt_cache
    )

    # Add assistant response to chat history
    messages.append({"role": "assistant", "content": response})
    return response

# Example conversation
response1 = chat("What is LLM so powerful?")
print(response1)
response2 = chat("Will it replace humans?")
print(response2)
