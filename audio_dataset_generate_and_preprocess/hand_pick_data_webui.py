import gradio as gr
from datasets import load_dataset, Dataset, Features, Audio, Value
import numpy as np

# --- Config ---
DATASET_PATH = "BarryFutureman/kara2"
AUDIO_KEY = "audio"
TEXT_KEY = "text"
PUSH_TO_HUB_NAME = "BarryFutureman/kara2_handpicked"

# --- Load dataset ---
dataset = load_dataset(DATASET_PATH, split="train", cache_dir="cache")
total = len(dataset)

# --- State ---
def get_item(idx):
    item = dataset[idx]
    audio = item[AUDIO_KEY]
    text = item[TEXT_KEY]
    return audio["array"], audio["sampling_rate"], text

def update(idx, handpicked):
    idx = max(0, min(idx, total - 1))
    audio, sr, text = get_item(idx)
    # Remove tag at the start, e.g., <tag> text
    if text.startswith("<") and ">" in text:
        text = text[text.find(">")+1:].lstrip()
    # Trim last 0.5 seconds from audio
    trim_samples = int(sr * 1)
    trim_samples_start = int(sr * 0.2)
    if len(audio) > trim_samples:
        audio = audio[trim_samples_start:-trim_samples]
    else:
        audio = audio * 0  # If too short, just silence

    # handpicked now stores tuples (idx, audio, sr, text)
    handpicked_indices = [t[0] for t in handpicked]
    if idx in handpicked_indices:
        btn_label = "Delete from hand picked"
    else:
        btn_label = "Add to hand picked"

    return (
        (sr, np.array(audio, dtype=np.float32)),
        text,
        idx,
        f"{idx+1}/{total}",
        handpicked,
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(value=btn_label),
    )

def add_or_delete_handpicked(idx, edited_text, handpicked):
    audio, sr, orig_text = get_item(idx)
    # Remove tag for comparison
    if orig_text.startswith("<") and ">" in orig_text:
        orig_text = orig_text[orig_text.find(">")+1:].lstrip()
    # Trim audio as in update
    trim_samples = int(sr * 1)
    trim_samples_start = int(sr * 0.2)
    if len(audio) > trim_samples:
        audio = audio[trim_samples_start:-trim_samples]
    else:
        audio = audio * 0
    # Remove if present, else add, using idx
    indices = [t[0] for t in handpicked]
    if idx in indices:
        handpicked = [t for t in handpicked if t[0] != idx]
        print(f"Deleted {idx} from handpicked")
    else:
        handpicked.append((idx, audio, sr, edited_text))
    return handpicked

def push_to_hub(handpicked):
    # handpicked is a list of (idx, audio, sr, text)
    new_data = []
    for _, audio, sr, text in handpicked:
        audio_dict = {"array": np.array(audio, dtype=np.float32), "sampling_rate": sr}
        new_data.append({"audio": audio_dict, "text": text})
    features = Features({"audio": Audio(sampling_rate=None), "text": Value("string")})
    new_ds = Dataset.from_list(new_data, features=features)
    save_path = "./handpicked_dataset"
    new_ds.save_to_disk(save_path)
    new_ds.push_to_hub(PUSH_TO_HUB_NAME)
    return gr.update(value=f"Saved to {save_path} and pushed to hub!")

with gr.Blocks(title="Hand Pick Dataset") as demo:
    gr.Markdown("# Hand Pick Dataset Entries")

    idx_state = gr.State(0)
    handpicked_state = gr.State([])

    # --- Tag buttons ---
    TAGS = [
        "happy", "normal", "digust", "disgust", "longer", "sad", "frustrated", "slow", "excited",
        "whisper", "panicky", "curious", "surprise", "fast", "crying", "deep", "sleepy", "angry", "high", "shout", "worried", "struggling"
    ]

    def add_tag_to_text(tag, text):
        import re
        # Remove existing tag at front
        text = re.sub(r"^<[^>]+>\s*", "", text)
        return f"<{tag}> {text}"

    with gr.Row():
        tag_buttons = []
        for tag in TAGS:
            btn = gr.Button(tag)
            tag_buttons.append(btn)

    with gr.Row():
        audio = gr.Audio(label="Audio", autoplay=True, interactive=False)
        text = gr.Textbox(label="Edit Text", interactive=True)

    # Tag button click events
    for btn, tag in zip(tag_buttons, TAGS):
        btn.click(add_tag_to_text, inputs=[gr.State(tag), text], outputs=text)

    idx_display = gr.Textbox(label="Index", interactive=False)
    total_display = gr.Textbox(label="Total", value=f"1/{total}", interactive=False)

    with gr.Row():
        btn_last = gr.Button("Last")
        btn_next = gr.Button("Next")
        btn_add = gr.Button("Add to hand picked")  # Initial label

    gr.Markdown("## Hand Picked List")
    handpicked_list = gr.Dataframe(headers=["Text"], datatype=["str"], interactive=False)
    push_btn = gr.Button("Push to hub")
    push_status = gr.Textbox(label="", interactive=False)

    # --- Button logic ---
    def last_fn(idx, handpicked):
        return update(idx - 1, handpicked)

    def next_fn(idx, handpicked):
        return update(idx + 1, handpicked)

    def add_or_delete_fn(idx, text, handpicked):
        handpicked = add_or_delete_handpicked(idx, text, handpicked)
        # Only show text in the dataframe
        return [[t[3]] for t in handpicked], handpicked

    btn_last.click(
        last_fn,
        [idx_state, handpicked_state],
        [audio, text, idx_state, total_display, handpicked_list, btn_last, btn_next, btn_add, btn_add]
    )
    btn_next.click(
        next_fn,
        [idx_state, handpicked_state],
        [audio, text, idx_state, total_display, handpicked_list, btn_last, btn_next, btn_add, btn_add]
    )
    btn_add.click(
        add_or_delete_fn,
        [idx_state, text, handpicked_state],
        [handpicked_list, handpicked_state]
    )

    push_btn.click(push_to_hub, [handpicked_state], [push_status])

    demo.load(
        update,
        [idx_state, handpicked_state],
        [audio, text, idx_state, total_display, handpicked_list, btn_last, btn_next, btn_add, btn_add]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7862)
