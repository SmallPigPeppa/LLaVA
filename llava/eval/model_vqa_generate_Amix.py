from openai import OpenAI

client = OpenAI(
  base_url='https://api.ppinfra.com/llm/v1',
  api_key='dcf59e46-a998-4a69-adfd-31aeabac1760',
)

completion_res = client.completions.create(
  model='Qwen2-7B-Instruct',
  prompt='派欧算力云提供 GPU 云产品能用于哪些场景？',
  stream=True,
  max_tokens=512,
)
