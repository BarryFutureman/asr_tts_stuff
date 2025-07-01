from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

app = Flask(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None


def load_model():
    """Load the HuggingFace model and tokenizer"""
    global model, tokenizer
    model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure bitsandbytes for quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
        cache_dir="cache",
    )
    print("Model loaded successfully!")


def create_deduplication_prompt(past_context, new_transcription):
    """Create a prompt for deduplication task"""
    messages = [
        {
            "role": "system",
            "content": "You are a text processing assistant. Your job is to intelligently merge new transcribed text with existing context, removing duplicates and creating a coherent, flowing text. Only output the final merged text without explanations."
        },
        {
            "role": "user", 
            "content": f"Merge this new transcription with the existing context. Remove any repeated or duplicate phrases while maintaining meaning and flow.\n\nExisting context:\n{past_context}\n\nNew transcription:\n{new_transcription}"
        }
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    half_tokens = " ".join(past_context.split(" ")[:len(past_context.split(" "))//2])
    partial_context = half_tokens
    prompt += f"Merged text: [{partial_context}"
    
    return prompt


@app.route('/deduplication', methods=['POST'])
def deduplication():
    """Endpoint to deduplicate and merge transcribed text"""
    try:
        data = request.get_json()
        
        past_context = data.get('past_context', '')
        new_transcription = data.get('new_transcription', '')
        max_tokens = data.get('max_tokens', 200)
        temperature = data.get('temperature', 0.3)
        
        if not new_transcription:
            return jsonify({"error": "new_transcription is required"}), 400
        
        # Create the deduplication prompt
        prompt = create_deduplication_prompt(past_context, new_transcription)
        
        # Tokenize input
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=torch.ones_like(inputs),
                repetition_penalty=1.1
            )
        
        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Full response: {full_response}")
        
        # Extract only the generated part (after the prompt)
        prompt_text = tokenizer.decode(inputs[0], skip_special_tokens=True)
        generated_text = full_response[len(prompt_text):].strip()

        generated_text = generated_text.split("]", 1)[0].strip()
        
        response = {
            "updated_context": generated_text,
            "original_past_context": past_context,
            "original_new_transcription": new_transcription
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": model is not None})

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Start Flask server
    app.run(host='0.0.0.0', port=8069, debug=True)
