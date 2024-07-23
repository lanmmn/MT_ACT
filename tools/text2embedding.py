model_id = "sentence-transformers/all-MiniLM-L6-v2"
hf_token = "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"

import requests
from retry import retry

api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}


@retry(tries=10, delay=20)
def query(texts):
    response = requests.post(api_url, headers=headers, json={"inputs": texts})
    result = response.json()
    if isinstance(result, list):
      return result
    elif list(result.keys())[0] == "error":
      raise RuntimeError(
          "The model is currently loading, please re-run the query."
          )

texts = ["Pick the pumpkin.",
        "Pick two cubes using single arm.",
        "Pick two cubes with two arms separately.",
        "Play the chess."]

output = query(texts)


import pandas as pd

embeddings = pd.DataFrame(output)
print(embeddings)

import os

# 设置文件路径和名称
file_path = os.path.join(os.getcwd(), "output.txt")

# 将输出写入文件
with open(file_path, "w") as f:
    # 如果 output 是一个 pandas DataFrame
    if isinstance(output, pd.DataFrame):
        output.to_csv(f, index=False)
    # 如果 output 是一个列表
    if isinstance(output, list):
      for item in output:
          f.write(str(item) + "\n")
    else:
      f.write(str(output))

print(f"Output saved to: {file_path}")


