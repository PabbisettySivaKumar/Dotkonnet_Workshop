import requests
import json

class PromptOptimizationFramework:
    def __init__(self, model="mistral"):
        self.model = model
        # Predefined prompt templates
        self.prompt_templates = {
            "summarization": "Summarize the following text in a clear and concise way:\n\n{text}\n\nSummary:",
            "classification": "Classify the following text into one of the categories: {labels}\n\nText: {text}\n\nCategory:",
            "creative": "Write a creative and engaging response to the following:\n\n{text}\n\nCreative Output:"
        }

    def detect_task(self, text: str, labels: list = None):
        """
        Detects the task type based on input.
        If labels are provided, we assume it's classification.
        If text is long -> summarization.
        Otherwise -> creative generation.
        """
        if labels:
            return "classification"
        elif len(text.split()) > 80:  # Threshold for summarization
            return "summarization"
        else:
            return "creative"

    def optimize_prompt(self, text: str, labels: list = None):
        """
        Selects and optimizes prompt based on detected task.
        """
        task = self.detect_task(text, labels)
        if task == "classification":
            return self.prompt_templates[task].format(text=text, labels=", ".join(labels))
        else:
            return self.prompt_templates[task].format(text=text)

    def run_ollama(self, prompt: str):
        """
        Sends the optimized prompt to Ollama and returns the model's response.
        """
        url = "http://localhost:11434/api/generate"
        payload = {"model": self.model, "prompt": prompt}
        
        response = requests.post(url, json=payload, stream=True)
        output = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                output += data.get("response", "")
        return output

    def process(self, text: str, labels: list = None):
        """
        End-to-end pipeline:
        1. Detects task
        2. Optimizes prompt
        3. Runs on Mistral via Ollama
        4. Returns response
        """
        optimized_prompt = self.optimize_prompt(text, labels)
        result = self.run_ollama(optimized_prompt)
        return result


# -------------------- Example Usage -------------------- #
framework = PromptOptimizationFramework(model="mistral")

# 1. Summarization
long_text = "Artificial Intelligence is a field of computer science that enables machines to learn from experience and perform tasks that typically require human intelligence such as perception, reasoning, and decision-making..."
print("\n🔹 SUMMARIZATION RESULT:\n", framework.process(long_text))

# 2. Classification
classification_text = "This product is amazing, I love it!"
print("\n🔹 CLASSIFICATION RESULT:\n", framework.process(classification_text, labels=["Positive", "Negative", "Neutral"]))

# 3. Creative Generation
short_text = "Write a story about a robot who learns emotions."
print("\n🔹 CREATIVE RESULT:\n", framework.process(short_text))