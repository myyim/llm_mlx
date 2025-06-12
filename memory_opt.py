from mlx_lm import generate
from mlx_lm.models.cache import make_prompt_cache
import mlx.core as mx

# Create a memory-efficient prompt cache
prompt_cache = make_prompt_cache(
  model,
  max_kv_size=4096 # Limit cache size to prevent memory bloat
)

# Generate with quantized KV cache for better memory efficiency
response = generate(
  model,
  tokenizer,
  prompt,
  prompt_cache=prompt_cache,
  kv_bits=4, # Quantize cache to 4 bits
  kv_group_size=64 # Quantization group size
)

# Clear GPU memory cache after large operations
mx.clear_cache()
