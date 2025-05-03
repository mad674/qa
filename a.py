import torch
import json
from generator import PointerProgramGenerator  # or your import path

# Load Generator model
with open("data/finqa_vocab.json", "r") as f:
    vocab_dict = json.load(f)
    vocab = list(vocab_dict.keys())

# 1. Instantiate and load your trained generator
generator = PointerProgramGenerator(vocab_dict)
generator.load_state_dict(torch.load("models/100egen.pt", map_location="cpu"))
generator.eval()

# 2. Apply dynamic quantization
quantized_generator = torch.quantization.quantize_dynamic(
    generator,
    {torch.nn.Linear, torch.nn.LSTM},
    dtype=torch.qint8
)

# 3. Save the entire quantized module
torch.save(quantized_generator, "quantized_generator_model.pt")
print("✅ Quantized model saved to quantized_generator_model.pt")
