#!/usr/bin/env python3
"""
RocLM Simple Demo - Load from Checkpoint

This script loads the custom RocLM model from checkpoint
and tokenizer from local files for quick inference.
"""

import os
import sys
import json
import torch
from typing import Dict, List

try:
    from roclm_model import create_roclm_model
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("❌ transformers library required")
    sys.exit(1)


class RocLMSimpleEngine:
    """
    Simple RocLM engine that loads from checkpoint
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_path = os.path.join(os.path.dirname(__file__), "ckpt")
        self.tokenizer_path = os.path.join(os.path.dirname(__file__), "tokenizer")
        
        self.load_model_and_tokenizer()

    def load_model_and_tokenizer(self):
        """Load model from checkpoint and tokenizer from local files"""
        print("🚀 RocLM Simple Demo - Loading from Checkpoint")
        print("=" * 50)
        
        # Load tokenizer
        print(f"📚 Loading tokenizer from: {self.tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_path, trust_remote_code=True
        )
        
        # Load model from checkpoint
        checkpoint_file = os.path.join(self.ckpt_path, "custom_model_state_dict.pth")
        config_file = os.path.join(self.ckpt_path, "model_config.json")
        
        if not os.path.exists(checkpoint_file) or not os.path.exists(config_file):
            print("❌ Checkpoint not found!")
            print(f"   Expected files:")
            print(f"   - {checkpoint_file}")
            print(f"   - {config_file}")
            sys.exit(1)
        
        print(f"📂 Loading model from: {checkpoint_file}")
        
        # Load config
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        
        # Create model
        print("🔧 Creating model from saved config...")
        self.model = create_roclm_model(config_dict)
        
        # Load state dict
        state_dict = torch.load(checkpoint_file, map_location='cpu')
        self.model.load_state_dict(state_dict, strict=True)
        
        # Move to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Vocab size: {len(self.tokenizer):,}")
        print(f"   Device: {self.device}")
        print("=" * 50)

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> str:
        """Generate text from a prompt"""
        
        print(f"📝 Generating text for: '{prompt}'")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=inputs.input_ids.shape[1] + max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
            )
        
        # Decode result
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        new_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        
        print(f"✨ Generated: {new_text}")
        return generated_text

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        enable_thinking: bool = False,
    ) -> Dict[str, str]:
        """Perform chat completion"""
        
        print(f"💬 Chat completion (thinking: {'on' if enable_thinking else 'off'})")
        
        # Apply chat template
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except Exception as e:
            print(f"⚠️  Chat template error: {e}, using simple format")
            text = ""
            for message in messages:
                role = message.get("role", "user").capitalize()
                content = message.get("content", "")
                text += f"{role}: {content}\n"
            text += "Assistant: "
        
        # Tokenize
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=inputs.input_ids.shape[1] + max_new_tokens,
                temperature=temperature,
                top_k=50,
                top_p=0.9,
                do_sample=True,
            )
        
        # Extract new tokens
        output_ids = outputs[0][inputs.input_ids.shape[1]:].tolist()
        
        # Parse thinking content if enabled
        thinking_content = ""
        content = ""
        
        if enable_thinking:
            try:
                # Look for </think> token (151668)
                think_end_indices = [
                    i for i, token_id in enumerate(output_ids) if token_id == 151668
                ]
                if think_end_indices:
                    index = think_end_indices[-1] + 1
                    thinking_content = self.tokenizer.decode(
                        output_ids[:index], skip_special_tokens=True
                    ).strip()
                    content = self.tokenizer.decode(
                        output_ids[index:], skip_special_tokens=True
                    ).strip()
                else:
                    content = self.tokenizer.decode(
                        output_ids, skip_special_tokens=True
                    ).strip()
            except Exception as e:
                print(f"⚠️  Error parsing thinking: {e}")
                content = self.tokenizer.decode(
                    output_ids, skip_special_tokens=True
                ).strip()
        else:
            content = self.tokenizer.decode(
                output_ids, skip_special_tokens=True
            ).strip()
        
        return {
            "thinking_content": thinking_content,
            "content": content,
            "full_response": self.tokenizer.decode(
                output_ids, skip_special_tokens=False
            ),
        }


def demo_text_generation():
    """Demo basic text generation"""
    print("\n📝 Text Generation Demo")
    print("=" * 40)
    
    engine = RocLMSimpleEngine()
    
    test_prompts = [
        "The future of artificial intelligence is",
        "Python programming is great for",
        "Machine learning can help us",
    ]
    
    for prompt in test_prompts:
        print(f"\n🎯 Prompt: {prompt}")
        generated = engine.generate_text(prompt, max_new_tokens=50, temperature=0.7)
        print(f"📄 Complete: {generated}")
        print("-" * 30)


def demo_chat():
    """Demo chat completion"""
    print("\n💬 Chat Completion Demo")
    print("=" * 40)
    
    engine = RocLMSimpleEngine()
    
    conversations = [
        [{"role": "user", "content": "Explain machine learning in simple terms."}],
        [{"role": "user", "content": "What are the benefits of renewable energy?"}],
    ]
    
    for i, messages in enumerate(conversations, 1):
        print(f"\n🔄 Conversation {i}:")
        print(f"User: {messages[0]['content']}")
        
        result = engine.chat_completion(
            messages, max_new_tokens=150, temperature=0.7, enable_thinking=True
        )
        
        if result["thinking_content"]:
            print(f"🧠 Thinking: {result['thinking_content']}")
        print(f"🤖 Assistant: {result['content']}")
        print("-" * 40)


def interactive_chat():
    """Interactive chat demo"""
    print("\n🎮 Interactive Chat")
    print("=" * 40)
    print("Type 'quit' to exit, 'thinking on/off' to toggle thinking mode")
    
    engine = RocLMSimpleEngine()
    
    messages = []
    enable_thinking = True
    
    print("🤖 Ready to chat! Ask me anything.")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == "thinking on":
                enable_thinking = True
                print("🧠 Thinking mode enabled")
                continue
            elif user_input.lower() == "thinking off":
                enable_thinking = False
                print("🗣️  Thinking mode disabled")
                continue
            
            if not user_input:
                continue
            
            # Add user message
            messages.append({"role": "user", "content": user_input})
            
            # Keep conversation manageable
            if len(messages) > 6:
                messages = messages[-6:]
            
            # Generate response
            result = engine.chat_completion(
                messages,
                max_new_tokens=200,
                temperature=0.8,
                enable_thinking=enable_thinking,
            )
            
            if enable_thinking and result["thinking_content"]:
                print(f"\n🧠 Thinking: {result['thinking_content']}")
            
            print(f"🤖 Assistant: {result['content']}")
            
            # Add assistant response
            messages.append({"role": "assistant", "content": result["content"]})
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main demo function"""
    print("🚀 RocLM Simple Demo - Load from Checkpoint")
    print("=" * 50)
    
    if not TRANSFORMERS_AVAILABLE:
        print("❌ This demo requires the transformers library")
        return 1
    
    try:
        print("Available demos:")
        print("1. Text Generation Demo")
        print("2. Chat Completion Demo") 
        print("3. Interactive Chat")
        print("4. Run All Demos")
        
        choice = input("\nSelect demo (1-4): ").strip()
        
        if choice == "1":
            demo_text_generation()
        elif choice == "2":
            demo_chat()
        elif choice == "3":
            interactive_chat()
        elif choice == "4":
            demo_text_generation()
            demo_chat()
            interactive_chat()
        else:
            print("Invalid choice. Running text generation demo...")
            demo_text_generation()
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
