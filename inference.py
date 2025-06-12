from mlx_lm import load, generate

model, tokenizer = load("mlx-community/gemma-3-1b-it-4bit-DWQ")

prompt = "Write a poem about LLMs."

response = generate(
  model,
  tokenizer,
  prompt,
  max_tokens=256,
# verbose=True
)

print(response)
