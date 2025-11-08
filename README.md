# CSC445 Project 2: Deep Learning Fundamentals

## 🎯 Project Overview

This project consists of two comprehensive assignments that will give you hands-on experience with modern deep learning techniques. You'll work with both computer vision and natural language processing, implementing state-of-the-art models from scratch and applying transfer learning techniques.

## 📋 Project Structure

```
CSC445_HW2/
├── part1/                          # Image Classification to compare CNN and Transformer-based models
│   ├── main.py                     # Main training script
│   ├── data_utils.py               # Data loading utilities
│   ├── model_utils.py              # Model setup utilities
│   ├── training_utils.py           # Training loop utilities
│   ├── experiment_tracker.py       # Experiment tracking with wandb
│   ├── IMPLEMENTATION_GUIDE.md     # Detailed part 1 instructions
│   ├── README.md                   # Part 1 specific documentation
│   ├── requirements.txt            # Part 1 dependencies
│   ├── data/                       # Dataset storage
│   └── results/                    # Generated plots and analysis
├── part2/                          # Language Model Implementation
│   ├── roclm_model.py              # Your model implementation
│   ├── roc_demo.py                 # Interactive demo script
│   ├── Tokenizer_Lab.ipynb         # Tokenizer exploration (REQUIRED)
│   ├── README.md                   # Part 2 specific documentation
│   ├── tokenizer/                  # Tokenizer files
│   └── ckpt/                       # Model checkpoints
├── requirements.txt                # Global dependencies
└── README.md                       # This file
```

## 🚀 Getting Started

### Prerequisites

1. **Python Environment**: Python 3.8+ recommended
2. **CUDA**: GPU support recommended for faster training (optional but highly recommended)
3. **Git**: For version control

### Git and GitHub Setup (macOS and Windows, Optional, if you want to use git to manage your code and do version control for debugging)

Follow these steps to get set up with Git and GitHub and learn the basic workflow used in this class.

#### 1 Create a GitHub account
- Go to `https://github.com` and sign up (use your school email if possible).
- Verify your email.

#### 2 Install Git
- macOS:
  - Option A: Install via Homebrew (recommended):
  ```bash
  brew install git
  ```
  - Option B: Install Xcode Command Line Tools (includes Git):
  ```bash
  xcode-select --install
  ```
- Windows:
  - Install "Git for Windows" from `https://git-scm.com/download/win`.
  - During setup, accept defaults (includes Git Bash and the Credential Manager).

Verify install:
```bash
git --version
```

#### 3 Configure your Git identity (once per machine)
```bash
git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"
```

Optional but recommended:
```bash
git config --global pull.rebase false   # use merge on pull
git config --global init.defaultBranch main
git config --global core.autocrlf input # macOS/Linux (Windows default is fine)
```

#### 4 Set up SSH keys (recommended) or use HTTPS
SSH avoids typing your password each time.

- Generate an SSH key (use your GitHub email):
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```
Press Enter to accept defaults (path `~/.ssh/id_ed25519`). Optionally add a passphrase.

- Start ssh-agent and add your key:
  - macOS (zsh/bash):
  ```bash
  eval "$(ssh-agent -s)"
  ssh-add ~/.ssh/id_ed25519
  ```
  - Windows (Git Bash):
  ```bash
  eval "$(ssh-agent -s)"
  ssh-add ~/.ssh/id_ed25519
  ```

- Copy your public key and add it to GitHub:
```bash
cat ~/.ssh/id_ed25519.pub
```
Copy the output, then go to GitHub → Settings → SSH and GPG keys → New SSH key → paste → Save.

- Test connection:
```bash
ssh -T git@github.com
```
Type "yes" if prompted. You should see a success message.

If you prefer HTTPS, you can skip SSH and use your GitHub credentials or a Personal Access Token when prompted.

#### 5 Add this repository to your github (Please make it private)
First, create a new repository on your github account. Then, add this repository to your github.
Make sure it is private. Please do not use public repository to prevent cheating.
```bash
git remote add origin git@github.com:<your-org-or-user>/<your-repository-name>.git
git push -u origin main
```

#### 6 Basic Git workflow you will use
- Create a new branch for your work:
```bash
git checkout -b your-feature-branch
```
- Check status and what changed:
```bash
git status
git diff        # view changes
```
- Stage and commit your changes:
```bash
git add -A      # or: git add path/to/file.py
git commit -m "Implement part1 training loop"
```
- Sync with remote:
```bash
git pull origin main   # bring latest main into your branch (merge)
git push -u origin your-feature-branch
```
- Open a Pull Request (PR) on GitHub from your branch into `main`. Add a concise description of what you changed.

#### 7 Keeping your branch up to date
Before you push or open a PR, update your branch with the latest `main`:
```bash
git checkout your-feature-branch
git pull origin main
```

If there are merge conflicts, Git will mark the files. Open the files, look for conflict markers `<<<<<<<`, `=======`, `>>>>>>>`, resolve them, then:
```bash
git add resolved_file.py
git commit
git push
```

#### 8 Optional: GitHub Desktop
If you prefer a GUI, install GitHub Desktop from `https://desktop.github.com`. You can clone, commit, and push visually while still using Git under the hood.

#### 9 Helpful tips
- Use small, frequent commits with clear messages.
- Never commit large data files or secrets. Use `.gitignore`.
- If you get stuck, run `git status` and read its guidance.

### Installation

1. **Clone the repository** (if not already done):
```bash
git clone <repository-url>
cd CSC445_HW2
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📚 Part 1: Image Classification with Transfer Learning

### 🎯 Learning Objectives

- **Transfer Learning**: Understand the difference between feature extraction and fine-tuning
- **Model Architecture**: Work with both CNN (ResNet-50) and Transformer (Swin-T) architectures
- **Data Augmentation**: Implement appropriate transforms for image classification
- **Evaluation**: Analyze model performance and error patterns

### 📖 What You'll Implement

1. **Data Preprocessing**: Image transforms and normalization
2. **Model Setup**: ResNet-50 and Swin-T fine-tuning
3. **Training Functions**: Complete training and validation loops
4. **Analysis Functions**: Loss curves, accuracy tables, and error analysis

### 🏃‍♂️ Quick Start

1. **Navigate to Part 1**:
```bash
cd part1
```

2. **Read the detailed implementation guide**:
```bash
cat IMPLEMENTATION_GUIDE.md
```

3. **Start implementing**:
   - Follow the step-by-step instructions in `IMPLEMENTATION_GUIDE.md`
   - Implement functions in `data_utils.py`, `model_utils.py`, and `training_utils.py`
   - Test with a small number of epochs first

4. **Run the experiment**:
```bash
python main.py
```

### 📊 Expected Deliverables

- **Loss Curves**: Training/validation loss comparison plots
- **Results Table**: Accuracy comparison across models and strategies
- **Error Analysis**: Discussion of challenging categories and model differences
- **Code Implementation**: Complete `student_template.py` with all TODOs filled

### 🔍 Key Concepts

- **Feature Extraction**: Freeze backbone, train only final layer
- **Full Fine-tuning**: Train entire network with lower learning rates
- **Data Augmentation**: Random crops, flips, and normalization
- **Model Comparison**: ResNet-50 vs Swin-T performance analysis

---

## 🤖 Part 2: Building RocLM - A Modern Language Model

### 🎯 Learning Objectives

- **Attention Mechanisms**: Implement multi-head attention with Grouped Query Attention (GQA)
- **Feed-Forward Networks**: Build SwiGLU-activated MLPs for efficient computation
- **Modern Normalization**: Use RMSNorm instead of traditional LayerNorm
- **Positional Encoding**: Implement Rotary Position Embedding (RoPE)
- **Model Deployment**: Load and run interactive demos with your trained model

### 📖 What You'll Implement

1. **RMSNorm**: Root Mean Square Normalization
2. **SwiGLU MLP**: Gated activation function for better expressiveness
3. **Multi-Head Attention**: With Grouped Query Attention (GQA) for efficiency
4. **Model Integration**: Complete transformer architecture

### 🏃‍♂️ Quick Start

1. **Navigate to Part 2**:
```bash
cd part2
```

2. **Complete the Tokenizer Lab** (REQUIRED):
```bash
jupyter notebook Tokenizer_Lab.ipynb
```

3. **Implement the model**:
   - Open `roclm_model.py`
   - Complete the three main tasks:
     - `RocLMRMSNorm` (lines 98-142)
     - `RocLMMLP` (lines 167-216)
     - `RocLMAttention` (lines 398-543)

4. **Test your implementation**:
```bash
python roclm_model.py
```

5. **Run the interactive demo**:
```bash
python roc_demo.py
```

### 🧪 Implementation Tasks

#### Task 1: RMSNorm Implementation
- Initialize learnable weight parameter
- Calculate root mean square of input
- Normalize by dividing by RMS + epsilon
- Apply learnable scaling

#### Task 2: SwiGLU MLP Implementation
- Initialize three linear projections (gate_proj, up_proj, down_proj)
- Implement SwiGLU forward pass with gating mechanism
- More expressive than standard ReLU/GELU

#### Task 3: Multi-Head Attention with GQA
- Calculate GQA parameters for memory efficiency
- Initialize Q, K, V, and output projections
- Implement forward pass with causal masking
- Apply RoPE for positional encoding

### 🎮 Interactive Demo

Once implemented, you can run:
```bash
python roc_demo.py
```

**Available demos**:
1. **Text Generation**: Generate text from prompts
2. **Chat Completion**: Conversational AI with thinking mode
3. **Interactive Chat**: Real-time chat interface

### 🔍 Key Concepts

- **Grouped Query Attention (GQA)**: Reduces memory usage while maintaining performance
- **SwiGLU Activation**: Gated architecture for selective information flow
- **RMSNorm**: More efficient than LayerNorm (no mean centering)
- **RoPE**: Relative positional encoding for variable sequence lengths

---

## 📝 Submission Guidelines

### Part 1 Submission
- Complete `part1/student_template.py` with all TODO sections implemented
- Include generated plots and analysis in `part1/results/`
- Provide written analysis of your findings

### Part 2 Submission
- Complete `part2/roclm_model.py` with all three tasks implemented
- Complete `part2/Tokenizer_Lab.ipynb` (required)
- Ensure the model passes tests and works with the demo

### General Requirements
- Code should run without errors
- Include comments explaining your implementation choices
- Follow the existing code style and structure
- Test your implementations thoroughly

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in training scripts
   - Use gradient accumulation for effective larger batch sizes

2. **Import Errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility

3. **Model Loading Issues**:
   - Verify checkpoint files are in the correct directories
   - Check file permissions and paths

4. **Training Convergence**:
   - Start with feature extraction (faster convergence)
   - Use appropriate learning rates (1e-2 for feature extraction, 1e-3 for fine-tuning)

### Debugging across multiple files (CLI and IDE)

When working in `part1/` and `part2/`, you will touch several modules. Here are reliable ways to debug.

#### A Run modules from the project root
- Always run from the repository root so imports resolve correctly:
```bash
pwd  # should end with CSC445_HW2
python -m part1.main
python -m part2.roclm_model
```
- To pass args: `python -m part1.main --epochs 1`.

#### B Use breakpoints with the Python debugger (pdb)
- Add this line where you want to pause:
```python
import pdb; pdb.set_trace()
```
- Then step through:
  - `n` next line, `s` step into, `c` continue, `p var` print variable.
- Use `where` to view the stack across files.

If pdb feels too hard, use the built-in breakpoint() (Python 3.7+):
```python
def train_one_epoch(model, optimizer, dataloader):
    for step, batch in enumerate(dataloader):
        # Inspect variables here
        breakpoint()  # pauses execution; use n/s/c and p var to explore
        outputs = model(batch[0])
        loss = outputs.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```
Run your script normally (from repo root):
```bash
python -m part1.main
```

Quick one-liner to start at main entry:
```bash
python -m pdb -m part1.main
```

#### C Use `ipdb` for a nicer debugger (optional)
```bash
pip install ipdb
```
Then:
```python
import ipdb; ipdb.set_trace()
```

#### D Structured logging to trace flow
- At the top of your entry file (e.g., `part1/main.py`):
```python
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
```
- Replace prints with:
```python
logger.info("Loaded %d samples", len(dataset))
logger.debug("Batch shape: %s", batch.shape)
```
- To see debug logs, set `level=logging.DEBUG` or export env var:
```bash
PYTHONLOGLEVEL=DEBUG python -m part1.main
```

#### E Debug configuration in VS Code
1. Open the repo in VS Code.
2. Create `.vscode/launch.json` with a module configuration:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug Part 1",
      "type": "python",
      "request": "launch",
      "module": "part1.main",
      "cwd": "${workspaceFolder}",
      "args": ["--epochs", "1"]
    },
    {
      "name": "Debug Part 2 Model",
      "type": "python",
      "request": "launch",
      "module": "part2.roclm_model",
      "cwd": "${workspaceFolder}"
    }
  ]
}
```
- Set breakpoints in any file (`data_utils.py`, `training_utils.py`, `roclm_model.py`) and press F5.
- Works on macOS and Windows.

#### F Debug configuration in PyCharm
- Open the project, then Run → Edit Configurations → + → Python.
  - Name: "Part 1"
  - Module name: `part1.main` (select "Module name" instead of script path)
  - Working directory: project root (`CSC445_HW2`)
  - Parameters: `--epochs 1`
  - Python interpreter: your env for this repo
- Repeat for `part2.roclm_model`.

#### G Tests and quick checks
- Sanity-run minimal epochs to catch shape errors early:
```bash
python -m part1.main --epochs 1 --batch-size 4
python -m part2.roclm_model
```
- Use `assert` statements in critical functions to validate shapes and ranges.

### Getting Help

- Check the detailed README files in each part directory
- Review the student instruction files for specific guidance
- Test with small datasets/epochs first to verify implementation
- Use the provided test scripts to validate your code

## 🎓 Learning Outcomes

After completing this project, you will have:

1. **Deep Understanding** of modern deep learning architectures
2. **Hands-on Experience** with transfer learning and fine-tuning
3. **Implementation Skills** for transformer-based language models
4. **Analysis Abilities** for model performance and error patterns
5. **Deployment Knowledge** for interactive AI applications

## 📚 Additional Resources

### Part 1 Resources
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Oxford Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/)

### Part 2 Resources
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [LLaMA Paper](https://arxiv.org/abs/2302.13971) - Modern transformer architecture
- [RoPE Paper](https://arxiv.org/abs/2104.09864) - Rotary Position Embedding

---

**Good luck with your implementation! 🚀**

This project will give you a solid foundation in modern deep learning techniques and prepare you for advanced AI/ML coursework and research/engineering roles.
