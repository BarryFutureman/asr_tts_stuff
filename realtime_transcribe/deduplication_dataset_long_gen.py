import os
import random
import re
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# Global variable to control number of items to process
NUM_ITEMS = 1000
# Global variable to control number of splits
NUM_SPLITS = 2

# Global list to collect dataset entries
dataset_entries = []

def split_text_into_sentences(text):
    """Split text into sentences using regex"""
    # Simple sentence splitting on periods, exclamation marks, and question marks
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def create_overlapping_text_chunks(text, num_splits=NUM_SPLITS):
    """Split text into overlapping chunks at word level"""
    words = text.split()
    total_words = len(words)
    
    if total_words < 10:  # Skip very short texts
        return None, None
    
    # Ensure first chunk is at least half the text length
    min_first_chunk_size = total_words * 3 // 4
    
    # Create split points at word boundaries
    split_points = []
    for i in range(1, num_splits):
        if i == 1:
            # First split point must be at least at half the text
            split_point = max(min_first_chunk_size, int((i / num_splits) * total_words))
        else:
            split_point = int((i / num_splits) * total_words)
        split_points.append(split_point)
    
    # Random overlap: 10-30% of average chunk size
    avg_chunk_size = total_words // num_splits
    
    chunks = []
    chunk_word_ranges = []
    
    # Create overlapping chunks
    for i in range(num_splits):
        if i == 0:
            # First chunk: from start to first split point + overlap (ensuring at least half text length)
            start = 0
            overlap_words = random.randint(max(1, int(0.1 * avg_chunk_size)), int(0.4 * avg_chunk_size))
            end = min(split_points[0] + overlap_words, total_words)
            # Ensure first chunk is at least half the text
            end = max(end, min_first_chunk_size)
        elif i == num_splits - 1:
            # Last chunk: from last split point - overlap to end
            overlap_words = random.randint(max(1, int(0.1 * avg_chunk_size)), int(0.4 * avg_chunk_size))
            start = max(split_points[-1] - overlap_words, 0)
            end = total_words
        else:
            # Middle chunks: from previous split point - overlap to next split point + overlap
            overlap_words = random.randint(max(1, int(0.1 * avg_chunk_size)), int(0.4 * avg_chunk_size))
            start = max(split_points[i-1] - overlap_words, 0)
            end = min(split_points[i] + overlap_words, total_words)
        
        chunk_words = words[start:end]
        chunk_text = ' '.join(chunk_words)
        chunks.append(chunk_text)
        chunk_word_ranges.append((start, end))
        
        print(f"  Chunk {i+1}: words {start}-{end} ({end-start} words)")
        print(f"    Preview: <<{chunk_text}>>")
    
    return chunks, chunk_word_ranges

def create_processed_full_text(text, chunk_ranges):
    """Create full_text starting from first overlapping sentence"""
    if len(chunk_ranges) < 2:
        return text
    
    words = text.split()
    
    # Find overlap between first and second chunk
    first_chunk_start, first_chunk_end = chunk_ranges[0]
    second_chunk_start, second_chunk_end = chunk_ranges[1]
    
    # Find the overlapping region
    overlap_start = max(first_chunk_start, second_chunk_start)
    overlap_end = min(first_chunk_end, second_chunk_end)
    
    if overlap_start >= overlap_end:
        # No overlap, return original text
        return text
    
    # Get the overlapping text
    overlap_words = words[overlap_start:overlap_end]
    overlap_text = ' '.join(overlap_words)
    
    print(f"    Overlap region: words {overlap_start}-{overlap_end}: '{overlap_text}'")
    
    # Find the sentence that contains the overlap start
    # Convert word position back to character position in original text
    words_before_overlap = words[:overlap_start]
    char_pos_estimate = len(' '.join(words_before_overlap))
    if words_before_overlap:
        char_pos_estimate += 1  # Add space before overlap
    
    # Find the sentence containing this position
    full_text_sentences = split_text_into_sentences(text)
    
    # Reconstruct text to find sentence boundaries
    current_pos = 0
    target_sentence_idx = 0
    
    for i, sentence in enumerate(full_text_sentences):
        sentence_in_text = sentence.strip()
        sentence_start = text.find(sentence_in_text, current_pos)
        sentence_end = sentence_start + len(sentence_in_text)
        
        if sentence_start <= char_pos_estimate <= sentence_end:
            target_sentence_idx = i
            break
        
        current_pos = sentence_end
    
    print(f"    Starting from sentence {target_sentence_idx}: '{full_text_sentences[target_sentence_idx] if target_sentence_idx < len(full_text_sentences) else 'NOT FOUND'}'")
    
    # Start from the target sentence, preserving original punctuation
    if target_sentence_idx < len(full_text_sentences):
        # Find where this sentence starts in the original text
        target_sentence = full_text_sentences[target_sentence_idx].strip()
        sentence_start_pos = text.find(target_sentence)
        
        if sentence_start_pos >= 0:
            processed_text = text[sentence_start_pos:]
        else:
            # Fallback: start from overlap point
            raise NotImplementedError()
            processed_text = ' '.join(words[overlap_start:])
    
    return processed_text

def process_text_sample(text, file_index):
    """Process a single text sample into chunks and processed full text"""
    # Filter out very short texts
    word_count = len(text.split())
    if word_count < 50:  # Minimum 50 words
        print(f"Filtering out text {file_index}: only {word_count} words")
        return
    
    print(f"\nProcessing text {file_index} ({word_count} words)")

    # Arbitrarily chop off text from the end
    words = text.split()
    total_words = len(words)
    chop_percentage = random.uniform(0.14, 0.4)
    words_to_remove = int(total_words * chop_percentage)
    print(words[-words_to_remove:])
    words = words[:-words_to_remove]
    chopped_text = ' '.join(words)
    print(f"  Arbitrarily chopped off {words_to_remove} words from end")
    
    # Create overlapping chunks
    chunks, chunk_ranges = create_overlapping_text_chunks(chopped_text)
    
    if chunks is None:
        return
    
    # Create processed full text
    processed_full_text = create_processed_full_text(chopped_text, chunk_ranges)
    
    print(f"Original text preview: '{text}'")
    print(f"Processed full text preview: '{processed_full_text}'")
    print(f"Number of chunks: {len(chunks)}")
    
    # Add to dataset
    dataset_entries.append({
        "chunks": chunks,
        "full_text": processed_full_text,
        "original_text": text
    })

def should_filter_sample(batch, dataset_type="human_llms"):
    """Check if sample should be filtered out"""
    if dataset_type == "human_llms":
        text = batch["chosen"][0]
    else:  # cosmopedia
        text = batch["text"][0]
    
    # Filter out very short texts
    word_count = len(text.split())
    if word_count < 50:
        return True
    
    # Filter out very long texts to avoid memory issues
    if word_count > 2000:
        return True
    
    return False

def create_and_push_dataset():
    """Create dataset from collected entries and push to Hub"""
    if not dataset_entries:
        print("No dataset entries to save!")
        return
    
    # Create dataset
    dataset = Dataset.from_list(dataset_entries)
    
    # Print dataset info
    print(f"\nCreated dataset with {len(dataset)} entries")
    print("Sample entry keys:", list(dataset[0].keys()))
    print("Sample chunks count:", len(dataset[0]["chunks"]))
    
    # Save locally first
    dataset.save_to_disk("text_deduplication_long_dataset")
    print("Dataset saved locally to 'text_deduplication_long_dataset'")
    
    # Push to Hub
    try:
        dataset.push_to_hub("BarryFutureman/text-based-deduplication-long", private=False)
        print("Dataset successfully pushed to Hub as 'text-based-deduplication-long'")
    except Exception as e:
        print(f"Error pushing to Hub: {e}")
        print("Make sure you're logged in with `huggingface-cli login`")

def main():
    # Process both datasets
    datasets_to_process = [
        {
            "name": "HumanLLMs/Human-Like-DPO-Dataset",
            "field": "chosen",
            "type": "human_llms"
        },
        {
            "name": "HuggingFaceTB/cosmopedia-20k", 
            "field": "text",
            "type": "cosmopedia"
        }
    ]
    
    processed_count = 0
    
    # Create progress bar
    pbar = tqdm(total=NUM_ITEMS, desc="Processing text samples")
    
    for dataset_info in datasets_to_process:
        processed_count = 0
            
        print(f"\nProcessing dataset: {dataset_info['name']}")
        
        # Load dataset
        dataset = load_dataset(dataset_info["name"], split="train", streaming=True)
        
        # Create dataloader with batch size 1
        dataloader = DataLoader(dataset, batch_size=1)
        
        for i, batch in enumerate(dataloader):
            if processed_count >= NUM_ITEMS:
                break
                
            # Filter out unwanted samples
            if should_filter_sample(batch, dataset_info["type"]):
                continue

            # Extract text from batch based on dataset type
            if dataset_info["type"] == "human_llms":
                text = batch["chosen"][0]
            else:  # cosmopedia
                text = batch["text"][0]
            
            # Process the text
            process_text_sample(text, processed_count + 1)
            processed_count += 1
            
            # Update progress bar
            pbar.update(1)
    
    pbar.close()
    
    print(f"Finished processing {processed_count} text samples.")
    
    # Create and push dataset
    create_and_push_dataset()

if __name__ == "__main__":
    main()
