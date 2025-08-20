import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from typing import List

# ----------------------------
# 1. Logging Configuration
# ----------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ----------------------------
# 2. Model Configuration
# ----------------------------
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"  # Sentiment analysis

# Detect device (Apple Silicon MPS, CUDA, or CPU)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    logger.info("Using Apple MPS backend (GPU acceleration)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    logger.info("Using NVIDIA CUDA backend")
else:
    DEVICE = torch.device("cpu")
    logger.info("Using CPU backend")

def load_model(model_name: str, device: torch.device):
    """Load tokenizer and model from Hugging Face."""
    try:
        logger.info(f"Loading model '{model_name}' on {device}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model = model.to(device)  # Move to detected device
        model.eval()
        return tokenizer, model
    except Exception as e:
        logger.exception("Error loading the model.")
        raise RuntimeError(f"Failed to load model: {e}")

# ----------------------------
# 3. Inference Function
# ----------------------------
def run_inference(texts: List[str], tokenizer, model, device: torch.device):
    """Run inference with batching."""
    try:
        logger.info("Starting inference...")

        # Create pipeline
        nlp_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=0 if device.type in ["cuda", "mps"] else -1
        )

        batch_size = 4
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} with {len(batch)} items")
            batch_results = nlp_pipeline(batch)
            results.extend(batch_results)

        return results
    except Exception as e:
        logger.exception("Error during inference.")
        raise RuntimeError(f"Inference failed: {e}")

# ----------------------------
# 4. Main Execution
# ----------------------------
if __name__ == "__main__":
    try:
        tokenizer, model = load_model(MODEL_NAME, DEVICE)

        sample_inputs = [
            "I love Hugging Face models!",
            "This movie was the worst experience of my life.",
            "The food was okay, nothing special."
        ]

        predictions = run_inference(sample_inputs, tokenizer, model, DEVICE)

        logger.info("Inference Results:")
        for text, pred in zip(sample_inputs, predictions):
            logger.info(f"Text: '{text}' → {pred}")

    except Exception as e:
        logger.error(f"Script failed: {e}")
