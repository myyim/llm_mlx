from mlx_lm import stream_generate

messages = [{"role": "user", "content": "Write a poem about LLMs."}]

prompt = tokenizer.apply_chat_template(messages,add_generation_prompt=True)

# Stream the response token by token
for response in stream_generate(model, tokenizer, prompt, max_tokens=512):
    print(response.text, end="", flush=True)
