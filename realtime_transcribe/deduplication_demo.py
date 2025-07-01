import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_trained_model(model_path):
    """Load the trained deduplication model"""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.eos_token_id = 0
    
    return model, tokenizer

def predict_full_text(model, tokenizer, chunks, max_length=512):
    """Predict full text from chunks using the trained model"""
    # Format input as: chunk1\n->chunk2\n->chunk3\n->chunk4\n=
    chunk_text = "\n->".join(chunks)
    prompt = f"{chunk_text}\n="
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate prediction
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode the full output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract only the predicted part (after the "=")
    if "\n=" in generated_text:
        predicted_text = generated_text.split("\n=", 1)[1].strip()
    else:
        predicted_text = generated_text
    
    return predicted_text

def demo():
    """Demo the deduplication model with sample chunks"""
    model_path = "deduplicator-135M"
    
    print("Loading trained deduplication model...")
    model, tokenizer = load_trained_model(model_path)
    print("Model loaded successfully!")
    
    # Sample chunks for testing
    test_cases = [
        {
            "chunks": [
                "The quick brown fox jumps over",
                "brown fox jumps over the lazy dog",
                "jumps over the lazy dog and runs",
                "the lazy dog and runs away quickly"
            ],
            "Ground Truth": "The quick brown fox jumps over the lazy dog and runs away quickly"
        },
        {
            "chunks": [
                "the attacks were particularly severe in Vienna, where most of the city",
                "attacks were particularly severe in Vienna, where most of the city synagogues were burnt, as the capital's",
                "where most of the city's synagogues were burnt as the capital's people and fire departments looked on and",
                "as the capital's people and fire departments looked on and watched."
            ],
            "Ground Truth": "The attacks were particularly severe in Vienna, where most of the city's synagogues were burnt as the capital's people and fire departments looked on and watched."
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_case['chunks'][0]} ---")
        print("Input chunks:")
        for j, chunk in enumerate(test_case['chunks'], 1):
            print(f"  {j}. '{chunk}'")
        print(f"Ground Truth: {test_case.get('Ground Truth', 'N/A')}")
        
        predicted_text = predict_full_text(model, tokenizer, test_case['chunks'])
        print(f"\nPredicted full text: '{predicted_text}'")


if __name__ == "__main__":
    demo()
