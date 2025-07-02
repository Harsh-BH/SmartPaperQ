"""
Simple text generation model using HuggingFace transformers.
This provides a fallback when no other LLM is available.
"""

import os
import sys
import argparse

def setup_simple_model():
    """Set up a simple text generation model and return a function to use it"""
    try:
        from transformers import pipeline
        
        print("Setting up simple text generation model...")
        # Use a small model that should work on most systems
        model_name = "distilgpt2"  # Very small model (~500MB)
        generator = pipeline('text-generation', model=model_name)
        
        def generate_text(prompt, max_length=100):
            """Generate text from a prompt"""
            result = generator(prompt, max_length=max_length, do_sample=True, temperature=0.7)
            return result[0]['generated_text']
            
        print(f"Successfully loaded {model_name}")
        return generate_text
        
    except ImportError:
        print("Error: transformers library not installed.")
        print("Please install it with: pip install transformers torch")
        return None
    except Exception as e:
        print(f"Error setting up model: {e}")
        return None

def test_generation():
    """Test the text generation"""
    generator = setup_simple_model()
    if generator:
        prompt = "The key findings of this research paper include"
        result = generator(prompt)
        print("\nTest generation:")
        print(f"Prompt: {prompt}")
        print(f"Result: {result}")
    else:
        print("Failed to set up the model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple text generation using HuggingFace transformers")
    parser.add_argument("--test", action="store_true", help="Run a test generation")
    parser.add_argument("--prompt", type=str, help="Text prompt to generate from")
    
    args = parser.parse_args()
    
    if args.test:
        test_generation()
    elif args.prompt:
        generator = setup_simple_model()
        if generator:
            result = generator(args.prompt)
            print(result)
    else:
        print("Please specify --test or --prompt")
