# RocLM: Building a Modern Language Model from Scratch

## 🎯 Assignment Overview

This assignment will teach you how to design, implement, and deploy a modern transformer-based language model called **RocLM**. By completing this project, you will gain hands-on experience with:

- **Attention Mechanisms**: Implement multi-head attention with Grouped Query Attention (GQA)
- **Feed-Forward Networks**: Build SwiGLU-activated MLPs for efficient computation
- **Modern Normalization**: Use RMSNorm instead of traditional LayerNorm
- **Positional Encoding**: Implement Rotary Position Embedding (RoPE)
- **Model Deployment**: Load and run interactive demos with your trained model

## 🏗️ Architecture Overview

RocLM is a decoder-only transformer model inspired by modern architectures like LLaMA and Qwen. It features:

### Core Components
- **RocLMRMSNorm**: Root Mean Square Normalization (more efficient than LayerNorm)
- **RocLMMLP**: Multi-Layer Perceptron with SwiGLU gated activation
- **RocLMAttention**: Multi-head attention with Grouped Query Attention (GQA)
- **RocLMRotaryEmbedding**: Rotary Position Embedding for positional encoding

### Key Features
- **Grouped Query Attention (GQA)**: Reduces memory usage while maintaining performance
- **SwiGLU Activation**: Gated activation function for better expressiveness
- **RMSNorm**: Efficient normalization technique
- **RoPE**: Relative positional encoding that handles variable sequence lengths

## 📚 Learning Objectives

After completing this assignment, you will understand:

1. **Transformer Architecture**: How attention mechanisms work and why they're effective
2. **Modern Attention Variants**: GQA for efficiency, causal masking for autoregressive generation
3. **Activation Functions**: SwiGLU gating mechanism and its benefits
4. **Normalization Techniques**: RMSNorm vs LayerNorm trade-offs
5. **Positional Encoding**: How RoPE encodes relative positions
6. **Model Engineering**: Loading checkpoints, tokenization, and text generation
7. **Deployment Skills**: Interactive demos and chat interfaces

## 🛠️ Implementation Tasks

### Task 1: RMSNorm Implementation (`RocLMRMSNorm`)
**Location**: Lines 98-142 in `roclm_model.py`

**What to implement**:
- Initialize learnable weight parameter
- Calculate root mean square of input
- Normalize by dividing by RMS + epsilon
- Apply learnable scaling

**Key concepts**:
- RMSNorm is simpler than LayerNorm (no mean centering)
- More efficient for large models
- Formula: `RMSNorm(x) = weight * x / sqrt(mean(x^2) + epsilon)`

### Task 2: SwiGLU MLP Implementation (`RocLMMLP`)
**Location**: Lines 167-216 in `roclm_model.py`

**What to implement**:
- Initialize three linear projections (gate_proj, up_proj, down_proj)
- Choose activation function based on config
- Implement SwiGLU forward pass: `SwiGLU(x) = down_proj(Swish(gate_proj(x)) * up_proj(x))`

**Key concepts**:
- Gated architecture with two parallel transformations
- SwiGLU = Swish(gate) * up_projection
- More expressive than standard ReLU/GELU

### Task 3: Multi-Head Attention with GQA (`RocLMAttention`)
**Location**: Lines 398-543 in `roclm_model.py`

**What to implement**:
- Calculate GQA parameters (num_key_value_groups, scaling factor)
- Initialize Q, K, V, and output projections
- Initialize query and key normalization layers
- Implement forward pass with:
  - Project and reshape Q, K, V
  - Apply RoPE to Q and K
  - Repeat K,V for GQA
  - Compute attention scores
  - Apply causal mask
  - Softmax and dropout
  - Apply attention to values

**Key concepts**:
- GQA shares key-value heads across multiple query heads
- Causal masking for autoregressive generation
- Query/Key normalization for training stability

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Model Checkpoint Download

You need to download two files from Google Drive and place them in the `ckpt/` directory:

1. **Model weights**: Download `custom_model_state_dict.pth` from:
   - [https://drive.google.com/file/d/1MJaGhBYN7lL1b0QrFgasHLC_706p5f8v/view?usp=sharing](https://drive.google.com/file/d/1MJaGhBYN7lL1b0QrFgasHLC_706p5f8v/view?usp=sharing)

2. **Model configuration**: Download `model_config.json` from:
   - [https://drive.google.com/file/d/1qZNM4DILFLE2JPGgDXr0CFMUUkO7LmsM/view?usp=sharing](https://drive.google.com/file/d/1qZNM4DILFLE2JPGgDXr0CFMUUkO7LmsM/view?usp=sharing)

**Alternative download method using gdown**:
```bash
# Download model weights
gdown 1MJaGhBYN7lL1b0QrFgasHLC_706p5f8v -O ckpt/custom_model_state_dict.pth

# Download model config
gdown 1qZNM4DILFLE2JPGgDXr0CFMUUkO7LmsM -O ckpt/model_config.json
```

Make sure both files are placed in the `ckpt/` directory before running the demo.

### Tokenizer Lab (Required)
- Open `Tokenizer_Lab.ipynb` in Jupyter and follow the guided exercises to:
  - Load the local tokenizer in `tokenizer/`
  - Explore how text maps to token IDs and back
  - Complete the assignment to tokenize and analyze your student ID

### Project Structure
```
CSC445_HW2/
├── roclm_model.py              # Main model implementation (YOUR TASKS)
├── tokenizer/                  # Tokenizer files
├── ckpt/                       # Model checkpoints (download from google drive)
├── roc_demo.py                 # Interactive demo script
├── Tokenizer_Lab.ipynb         # Tokenizer exploration & assignment (REQUIRED)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

### Testing Your Implementation

1. **Run the model test**:
```bash
python roclm_model.py
```
This will test your implementation and show parameter counts.

2. **Load and test with demo**:
```bash
python roc_demo.py
```
This will load a pre-trained checkpoint and run interactive demos.

## 🧪 Implementation Guide

### Step 1: RMSNorm (Start Here)
```python
# In RocLMRMSNorm.__init__:
self.weight = nn.Parameter(torch.ones(hidden_size))

# In RocLMRMSNorm.forward:
variance = hidden_states.pow(2).mean(-1, keepdim=True)
hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
return self.weight * hidden_states.to(input_dtype)
```

### Step 2: SwiGLU MLP
```python
# In RocLMMLP.__init__:
self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

# In RocLMMLP.forward:
return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

### Step 3: Attention Mechanism
```python
# In RocLMAttention.__init__:
self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
self.scaling = self.head_dim**-0.5

# Initialize projections and normalization layers...

# In RocLMAttention.forward:
# Project Q, K, V and apply normalization
# Apply RoPE
# Repeat K,V for GQA
# Compute attention scores
# Apply causal mask
# Softmax + dropout
# Apply to values and project output
```

## 🎮 Interactive Demo

Once your implementation is complete, you can run the interactive demo:

```bash
python roc_demo.py
```

**Available demos**:
1. **Text Generation**: Generate text from prompts
2. **Chat Completion**: Conversational AI with thinking mode
3. **Interactive Chat**: Real-time chat interface

**Example usage**:
```python
# Text generation
prompt = "The future of AI is"
generated = engine.generate_text(prompt, max_new_tokens=100, temperature=0.7)

# Chat completion
messages = [{"role": "user", "content": "Explain machine learning"}]
response = engine.chat_completion(messages, enable_thinking=True)
```

## 🔍 Key Concepts Explained

### Grouped Query Attention (GQA)
- **Problem**: Standard multi-head attention is memory-intensive
- **Solution**: Share key-value heads across multiple query heads
- **Benefit**: Reduces memory usage while maintaining performance
- **Example**: 16 query heads, 8 key-value heads → each K,V pair serves 2 Q heads

### SwiGLU Activation
- **Standard**: `MLP(x) = down_proj(activation(up_proj(x)))`
- **SwiGLU**: `SwiGLU(x) = down_proj(Swish(gate_proj(x)) * up_proj(x))`
- **Benefit**: Gating mechanism allows selective information flow

### RMSNorm vs LayerNorm
- **LayerNorm**: `(x - mean) / sqrt(variance + epsilon)`
- **RMSNorm**: `x / sqrt(mean(x^2) + epsilon)`
- **Benefit**: No mean computation, more efficient

### Rotary Position Embedding (RoPE)
- **Problem**: Absolute positional embeddings don't generalize to longer sequences
- **Solution**: Rotate query and key vectors based on position
- **Benefit**: Naturally handles variable sequence lengths

## 🐛 Debugging Tips

1. **Check tensor shapes**: Print shapes at each step of attention computation
2. **Verify GQA**: Ensure K,V repetition matches expected head counts
3. **Test normalization**: Check that RMSNorm outputs have reasonable scales
4. **Validate RoPE**: Ensure position embeddings are applied correctly

## 📊 Expected Results

After successful implementation:
- Model should have ~1.3B parameters
- Forward pass should work without errors
- Generated text should be coherent (though may be basic without training)
- Interactive demo should load and respond

## 🎓 What You'll Learn

This assignment provides hands-on experience with:

1. **Modern Transformer Architecture**: Understanding the building blocks of LLMs
2. **Attention Mechanisms**: How self-attention works and why it's powerful
3. **Efficiency Techniques**: GQA, RMSNorm, and other optimizations
4. **Engineering Skills**: Model loading, tokenization, and deployment
5. **Research Insights**: Why these architectural choices matter

## 🚀 Next Steps

After completing the implementation:
1. Experiment with different generation parameters (temperature, top-k, top-p)
2. Try the interactive chat to see your model in action
3. Explore the model architecture and understand each component
4. Consider how you might extend or modify the model

## 📝 Submission

Submit your completed `roclm_model.py` file with all TODO sections implemented. The model should pass the test in the `__main__` section and work with the interactive demo.

---

**Happy coding! 🚀** This assignment will give you a solid foundation in modern language model architecture and implementation.
