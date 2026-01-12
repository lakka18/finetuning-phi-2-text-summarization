# finetuning-phi-2-text-summarization

---
Name      : Arthur Trageser
NIM        : 1103223090
Class      : DL TK-46-Gab
---
# Task 3: Abstractive Text Summarization with Phi-2

## Overview

This task implements an end-to-end abstractive text summarization system using **Phi-2**, a decoder-only Large Language Model (LLM) from Microsoft. The model is fine-tuned on the **XSum dataset** to generate concise, single-sentence summaries that capture the essence of news articles. Unlike extractive summarization, this approach learns to rewrite and compress information into highly abstractive summaries.

## Model Architecture

- **Model**: `microsoft/phi-2` (2.7B parameters)
- **Type**: Decoder-only Transformer (Causal Language Model)
- **Training Method**: Parameter-Efficient Fine-Tuning (PEFT) using LoRA (Low-Rank Adaptation)
- **Tokenizer**: CodeGen tokenizer (used by Phi-2)

### LoRA Configuration

```python
LoRA Parameters:
- r (rank): 16
- alpha: 32
- Target modules: ['q_proj', 'k_proj', 'v_proj', 'dense']
- dropout: 0.05
- bias: none
- task_type: CAUSAL_LM
```

## Dataset

### XSum (Extreme Summarization)

- **Source**: BBC news articles
- **Task**: Generate single-sentence summaries
- **Characteristics**: Highly abstractive (summaries often paraphrase rather than extract)
- **Training set**: 5,000 samples (subset used)
- **Validation set**: 500 samples (subset used)

### Data Format

Each example contains:
- **document**: Full news article text
- **summary**: Single-sentence abstractive summary (target)

### Preprocessing

```python
Prompt Template:
"Summarize this article in one sentence:\n{document}\n\nSummary:"

Tokenization Settings:
- Max input length: 512 tokens
- Max target length: 64 tokens
- Padding: right
- Truncation: enabled
```

## Training Configuration

### Hyperparameters

```python
Training Arguments:
- Batch size: 4 (per device)
- Gradient accumulation: 4 steps (effective batch = 16)
- Learning rate: 2e-4
- Epochs: 3
- Warmup steps: 100
- Weight decay: 0.01
- Optimizer: AdamW (8-bit)
- Learning rate scheduler: Linear
- fp16: True (mixed precision)
- Gradient checkpointing: True
```

### Hardware Requirements

- **Recommended**: NVIDIA A100 (80GB) or similar high-memory GPU
- **Minimum**: GPU with at least 16GB VRAM (with reduced batch size)
- **Training time**: ~45-60 minutes per epoch on A100

### Memory Optimization Techniques

1. **LoRA**: Fine-tunes only 0.3% of model parameters (~9M out of 2.7B)
2. **8-bit AdamW**: Reduces optimizer memory footprint
3. **Gradient Checkpointing**: Trades computation for memory
4. **Mixed Precision (fp16)**: Halves activation memory usage

## Implementation Details

### 1. Data Loading and Preprocessing

```python
# Load dataset
dataset = load_dataset("EdinburghNLP/xsum")

# Filter to subset
train_dataset = dataset['train'].select(range(5000))
val_dataset = dataset['validation'].select(range(500))

# Tokenize with prompt template
def preprocess_function(examples):
    prompts = [
        f"Summarize this article in one sentence:\n{doc}\n\nSummary:"
        for doc in examples['document']
    ]
    model_inputs = tokenizer(prompts, max_length=512, truncation=True)
    labels = tokenizer(examples['summary'], max_length=64, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs
```

### 2. Model Setup with LoRA

```python
# Load base model in 8-bit
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    load_in_8bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Apply LoRA
from peft import get_peft_model, LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'dense'],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
```

### 3. Training

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()
```

### 4. Inference

```python
def generate_summary(text, max_length=64):
    prompt = f"Summarize this article in one sentence:\n{text}\n\nSummary:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_length,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
        temperature=0.7
    )
    
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary = summary.split("Summary:")[-1].strip()
    return summary
```

## Evaluation Metrics

### ROUGE Scores

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures n-gram overlap between generated and reference summaries.

#### Results (100 validation samples):

```
ROUGE-1: 0.1990  (unigram overlap)
ROUGE-2: 0.0870  (bigram overlap)
ROUGE-L: 0.1524  (longest common subsequence)
```

### Interpretation

- **ROUGE-1 (19.90%)**: Indicates moderate word-level similarity
- **ROUGE-2 (8.70%)**: Shows lower phrase-level matching (expected for abstractive summaries)
- **ROUGE-L (15.24%)**: Captures sentence structure similarity

**Note**: Lower ROUGE scores for abstractive summarization are normal since the model generates novel phrasings rather than extracting sentences. Qualitative evaluation is equally important.

## Results and Analysis

### Training Metrics

```
Epoch 1:
- Training Loss: 1.8523
- Validation Loss: 1.7291

Epoch 2:
- Training Loss: 1.6847
- Validation Loss: 1.6982

Epoch 3:
- Training Loss: 1.5432
- Validation Loss: 1.6845
```

### Sample Predictions

| True Summary | Generated Summary |
|--------------|-------------------|
| *"A man has been arrested after a police officer was injured..."* | *"Police officer injured in incident, man arrested"* |
| *"Scientists discover new species of deep-sea fish..."* | *"New deep-sea fish species discovered by scientists"* |

### Key Observations

1. **Abstractive Quality**: Model successfully paraphrases rather than copies
2. **Conciseness**: Summaries are single-sentence as required
3. **Information Retention**: Key facts preserved despite compression
4. **Challenges**: 
   - Occasional repetition in longer articles
   - Some loss of specific details
   - Generic phrasing for complex topics

## Project Structure

```
task3_xsum/
├── notebooks/
│   └── task3_xsum.ipynb          # Main training notebook
├── checkpoints/
│   ├── adapter_model.safetensors # LoRA weights (33MB)
│   ├── adapter_config.json       # LoRA configuration
│   ├── tokenizer files           # Tokenizer artifacts
│   └── training_args.bin         # Training configuration
├── reports/
│   ├── training_metrics.csv      # Loss curves
│   └── sample_predictions.csv    # Validation examples
├── logs/
│   └── training.log              # Detailed training logs
└── output/
    └── phi2_summarization.zip    # Packaged model (~33MB)
```

## How to Run

### 1. Environment Setup

```bash
# Install dependencies
pip install transformers datasets accelerate peft bitsandbytes evaluate rouge-score

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### 2. Training

Open `task3_xsum.ipynb` in Google Colab or Jupyter and run all cells sequentially:

1. **Setup Cell**: Installs packages, checks GPU, mounts Drive
2. **Import Libraries**: Loads required packages
3. **Load Dataset**: Downloads and preprocesses XSum
4. **Tokenization**: Prepares data with prompt template
5. **Model Setup**: Loads Phi-2 with LoRA
6. **Training**: Fine-tunes for 3 epochs
7. **Evaluation**: Computes ROUGE scores
8. **Inference**: Generates sample summaries

### 3. Inference with Saved Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    load_in_8bit=True,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "path/to/checkpoints")
tokenizer = AutoTokenizer.from_pretrained("path/to/checkpoints")

# Generate summary
def summarize(text):
    prompt = f"Summarize this article in one sentence:\n{text}\n\nSummary:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=64, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("Summary:")[-1].strip()
```

## Challenges and Solutions

### 1. Memory Constraints
**Problem**: Phi-2 (2.7B parameters) requires ~6GB just to load in fp16  
**Solution**: 
- 8-bit quantization (reduces to ~3GB)
- LoRA (fine-tune only 9M parameters)
- Gradient checkpointing
- Effective batch size via gradient accumulation

### 2. Long Training Times
**Problem**: Large model + 5,000 samples = slow training  
**Solution**:
- A100 GPU (much faster than T4)
- Mixed precision training (fp16)
- Efficient data collation
- Early stopping considerations

### 3. Generic Summaries
**Problem**: Model sometimes produces vague summaries  
**Solution**:
- Temperature tuning (0.7 for creativity)
- Beam search (4 beams for quality)
- No-repeat n-gram penalty (prevents repetition)
- Consider increasing training data in production

### 4. Evaluation Discrepancy
**Problem**: Good qualitative summaries but modest ROUGE scores  
**Solution**:
- Human evaluation supplement
- Multiple reference summaries
- Focus on coherence and factuality
- Task-specific metrics (BLEU, BERTScore)

## Future Improvements

1. **Dataset Expansion**: Train on full 200K+ XSum samples
2. **Prompt Engineering**: Experiment with different instruction formats
3. **Hyperparameter Tuning**: Grid search for optimal LoRA rank/alpha
4. **Multi-Reference Evaluation**: Use multiple gold summaries
5. **Post-Processing**: Add grammar correction, length constraints
6. **Alternative Models**: Try Llama-2-7B, Mistral-7B for comparison
7. **Reinforcement Learning**: RLHF for human preference alignment

## License

This project is for educational purposes as part of a Deep Learning for NLP assignment.

## Author

Task completed as part of NLP course assignment on Transformer-based architectures.
