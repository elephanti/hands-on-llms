# Entropix Method

**Version:** 1.0.0  
**Last Updated:** 2025-02-01

---

## Overview

The **Entropix Method** is an advanced entropy-based sampling strategy designed to improve response accuracy and faithfulness in financial chatbot systems. By dynamically adjusting sampling parameters based on real-time uncertainty metrics, Entropix enhances model performance without requiring additional fine-tuning.

This method is integrated into an inference pipeline and is particularly effective in scenarios requiring high precision and factual consistency, such as financial advisory applications.

---

## Key Features

- **Dynamic Sampling Adjustment:** Adapts token selection strategy based on uncertainty metrics.
- **Improved Accuracy:** Enhances response similarity to ground truth answers.
- **Increased Faithfulness:** Ensures factual consistency with provided context.
- **Seamless Integration:** Works with existing LLM-based chatbot architectures.
- **Scalability:** Efficiently processes large-scale financial datasets.

---


### Usage

```python
import FalconForCausalLMWithEntropix
from transformers import AutoTokenizer

# Load the model and tokenizer
model = FalconForCausalLMWithEntropix.from_pretrained("tiiuae/falcon-7b-instruct")
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")

# Example usage
input_text = "What is the latest trend in the financial market?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Generate response
output = model.generate(input_ids, max_length=100)
response = tokenizer.decode(output[0], skip_special_tokens=True)
print(response)
```

---

## Configuration

- **Sampling Strategies:** Greedy, High Uncertainty, Exploration, Clarification, Adaptive Sampling
- **Uncertainty Metrics:** Logits Entropy, Logits Varentropy, Attention Entropy, Attention Varentropy, Attention Agreement, Atthention Strength
- **Base Hyperparameters:**
  ```yaml
  temperature: 0.666
  top-k: 27
  top-p: 0.9
  min_p: 0.0
  ```
---

## Results

Performance evaluation comparing a baseline model and Entropix-enhanced model:

| Metric                | Baseline | Entropix | Improvement |
|-----------------------|---------|----------|------------|
| Avg Answer Similarity | 0.28    | 0.352    | +25.7%     |
| Avg Faithfulness      | 0.158   | 0.214    | +35.4%     |

---
