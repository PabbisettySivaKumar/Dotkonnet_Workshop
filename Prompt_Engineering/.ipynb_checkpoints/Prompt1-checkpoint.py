import json
import requests

OLLAMA_API_URL = "http://localhost:11434/api/generate"

prompt_template= """ [Role]:You are a News Reporter.
[Intent]: Your task is to Publish article with all the details including his past life and what made him to do, what he did?.
[Instructions]:Follow this structure:
1. Break the query into smaller parts.
2. Reason about each part.
3. Derive the final answer after all steps.
[Style & Constraints]: Be Methodical. Do not skip any steps. Avoid premature conclusions.
[Format]: Respond in Markdown with:
-Step-by-step reasoning
-Final answer
"""

question = "Who is Edward Snowden and what did he do?"

full_prompt = f"{prompt_template}\n\n[Question]: {question}"

payload = {
    "model": "mistral",
    "prompt": full_prompt,
    "stream": False
}

resp= requests.post(OLLAMA_API_URL, data=json.dumps(payload), headers={"Content-Type": "application/json"})
resp.raise_for_status()
data= resp.json()

print(data['response'])