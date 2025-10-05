# Qwen1.5-0.5B Fine-tuning for Person Information Extraction

This project demonstrates fine-tuning the Qwen1.5-0.5B-Chat language model to extract structured person information from descriptive text using LoRA (Low-Rank Adaptation).

## 📋 Project Overview

The model is trained to parse natural language descriptions of people and extract key attributes in a structured format:
- **Name**
- **Age** 
- **Job/Profession**
- **Gender**

### Example:
**Input (Prompt):**

Within an echoing cathedral, Zoey, currently 39 years old builds a career as a lawyer. She finds peace in practicing Japanese calligraphy in quiet solitude.

```

**Output (Completion):**
```

name: Zoey, age: 39, job: lawyer, gender: female

```
```

## 🛠️ Technical Implementation

### Model Architecture
- **Base Model**: Qwen1.5-0.5B-Chat (500M parameters)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **LoRA Configuration**:
  - Rank (r): 8
  - Alpha: 16
  - Dropout: 0.01
  - Target Modules: q_proj, k_proj, v_proj

### Training Details
- **Dataset**: 300 custom examples (225 train, 75 test)
- **Epochs**: 7
- **Learning Rate**: 0.001
- **Batch Size**: Default (auto-configured)
- **Sequence Length**: 128 tokens
- **Trainable Parameters**: 1.18M (0.25% of total)

## 📊 Results

### Performance Metrics
- **Training Loss**: 0.196 (final)
- **Evaluation Loss**: 0.269
- **Accuracy**: 61% (token-level)
- **Sample Accuracy**: 70% (exact match on 10 test samples)

### Example Predictions
The model successfully learned to:
- Extract names from context
- Identify professions from descriptive text
- Infer gender from pronouns and context
- Handle missing information gracefully

## 🚀 Usage

### Installation
```bash
pip install transformers datasets peft accelerate torch
```
Loading the Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

model_path = "path/to/Qwen1.5-0.5B-Chat"
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_path)
```

Inference

```python
def extract_person_info(prompt, model, tokenizer, max_new_tokens=64):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example usage
prompt = "Under the scorching summer sun, Maya works as a biologist. She is known among friends for learning sign language."
result = extract_person_info(prompt, model, tokenizer)
print(result)
```

📁 Project Structure

```
├── Fine_tuning_test.ipynb          # Main training notebook
├── /content/drive/MyDrive/
│   ├── HuggingFace_Model/
│   │   └── Qwen1.5-0.5B-Chat/      # Base model files
│   └── Fine_Tuning_Data/
│       └── people_data.json        # Training dataset
├── results/                        # Training outputs (optional)
└── README.md
```

🎯 Key Features

· Efficient Fine-tuning: Uses LoRA for parameter-efficient adaptation
· Structured Output: Consistent formatting for easy parsing
· Context Understanding: Handles varied sentence structures and contexts
· Error Handling: Gracefully manages missing or ambiguous information

🔧 Customization

The model can be further fine-tuned for:

· Different attribute extraction tasks
· Various output formats
· Domain-specific person descriptions
· Multi-language support

📈 Future Improvements

· Expand training dataset size and diversity
· Experiment with different LoRA configurations
· Add validation for extracted information
· Implement confidence scoring for predictions
· Support for additional attributes (location, interests, etc.)

🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for suggestions and improvements.

📄 License

This project uses the Qwen1.5 model which is subject to its original license terms. Please refer to the official Qwen repository for licensing details.

---

Note: This project was developed in Google Colab with T4 GPU acceleration.

```
