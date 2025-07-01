import os
import queue
import threading
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import click
import numpy as np
import torch
import torch._inductor.config
import torchaudio

# for dac
import soundfile as sf
from modded_dac import DAC, ModelArgs

from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

from content_sequence import (
    ContentSequence,
    TextPart,
    VQPart,
)
from tokenizer import IM_END_TOKEN

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True

if hasattr(torch._inductor.config, "fx_graph_cache"):
    # Experimental feature to reduce compilation times, will be on by default in future
    torch._inductor.config.fx_graph_cache = True


from torch.nn.attention import SDPBackend, sdpa_kernel

from llama import (
    BaseTransformer,
    DualARTransformer,
    NaiveTransformer,
)


def multinomial_sample_one_no_sync(
    probs_sort,
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)


def logits_to_probs(
    logits,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: torch.Tensor,
    previous_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Apply repetition penalty
    if previous_tokens is not None:
        previous_tokens = previous_tokens.long()
        score = torch.gather(logits, dim=-1, index=previous_tokens)
        score = torch.where(
            score < 0, score * repetition_penalty, score / repetition_penalty
        )
        logits.scatter_(dim=-1, index=previous_tokens, src=score)

    # Apply top-p sampling
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[0] = False  # keep at least one option
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, -float("Inf"))
    logits = logits / torch.clip(temperature, min=1e-5)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def sample(
    logits,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: torch.Tensor,
    previous_tokens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    probs = logits_to_probs(
        logits=logits[0, -1],
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        previous_tokens=previous_tokens,
    )
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


def decode_one_token_ar(
    model: DualARTransformer,
    x: torch.Tensor,
    input_pos: torch.Tensor,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: torch.Tensor,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    previous_tokens: torch.Tensor = None,
) -> torch.Tensor:
    # print(x, torch.count_nonzero(vq_masks))
    x = model.forward_generate(
        x,
        input_pos,
        audio_masks=audio_masks,
        audio_parts=audio_parts,
    )
    logits = x.logits  # [:, -1:]
    hidden_states = x.hidden_states  # [:, -1:]

    codebooks = [
        sample(
            logits,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            previous_tokens=(
                previous_tokens[:, 0] if previous_tokens is not None else None
            ),
        )[0]
    ]

    # Cleanup the cache
    for layer in model.fast_layers:
        layer.attention.kv_cache.k_cache.fill_(0)
        layer.attention.kv_cache.v_cache.fill_(0)

    input_pos = torch.tensor([0], device=hidden_states.device, dtype=torch.long)
    model.forward_generate_fast(hidden_states, input_pos)
    a = codebooks[0] - model.tokenizer.semantic_begin_id
    a[a < 0] = 0
    hidden_states = model.fast_embeddings(a)
    codebooks.append(a)

    for codebook_idx in range(1, model.config.num_codebooks):
        input_pos = torch.tensor(
            [codebook_idx], device=hidden_states.device, dtype=torch.long
        )
        logits = model.forward_generate_fast(hidden_states, input_pos)

        short_logits = logits[:, :, :1024]

        # Convert logits to probs
        a = sample(
            short_logits,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            previous_tokens=(
                previous_tokens[codebook_idx + 1]
                if previous_tokens is not None
                else None
            ),
        )[0]

        hidden_states = model.fast_embeddings(a)
        codebooks.append(a)

    codebooks = torch.stack(codebooks, dim=1)
    return codebooks.T


def decode_n_tokens(
    model: NaiveTransformer,
    cur_token: torch.Tensor,
    input_pos: torch.Tensor,
    num_new_tokens: int,
    temperature: torch.Tensor,
    top_p: torch.Tensor,
    repetition_penalty: torch.Tensor,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    decode_one_token=decode_one_token_ar,
):
    previous_tokens = torch.zeros(
        (model.config.num_codebooks + 1, model.config.max_seq_len),
        dtype=torch.int,
        device=cur_token.device,
    )

    for i in tqdm(range(num_new_tokens)):
        # We need to get windowed repeat penalty
        win_size = 16
        if i < win_size:
            window = previous_tokens[:, :win_size]
        else:
            window = previous_tokens[:, i - win_size : i]

        with sdpa_kernel(
            SDPBackend.MATH
        ):  # Actually better for Inductor to codegen attention here
            next_token = decode_one_token(
                model=model,
                x=cur_token,
                input_pos=input_pos,
                previous_tokens=window,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                audio_masks=audio_masks,
                audio_parts=audio_parts,
            ).clone()

        input_pos += 1
        cur_token = next_token.view(1, model.config.num_codebooks + 1, -1)
        previous_tokens[:, i : i + 1] = next_token.view(
            model.config.num_codebooks + 1, -1
        )

        if cur_token[0, 0, -1] == model.tokenizer.get_token_id(IM_END_TOKEN):
            break

    return previous_tokens[:, : i + 1]


@torch.no_grad()
@torch.inference_mode()
def generate(
    *,
    model: BaseTransformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    audio_masks: torch.Tensor,
    audio_parts: torch.Tensor,
    decode_one_token=decode_one_token_ar,
    num_samples: int = 1,
    **sampling_kwargs,
):
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(1)
    prompt = prompt[None].repeat(num_samples, 1, 1)

    if T >= model.config.max_seq_len:
        raise ValueError(
            f"Input sequence length {T} exceeds max_seq_len {model.config.max_seq_len}"
        )

    if max_new_tokens:
        if T + max_new_tokens > model.config.max_seq_len:
            max_new_tokens = model.config.max_seq_len - T

        T_new = T + max_new_tokens
    else:
        T_new = model.config.max_seq_len
        max_new_tokens = T_new - T

    device, dtype = prompt.device, prompt.dtype
    with torch.device(device):
        model.setup_caches(
            max_batch_size=num_samples,
            max_seq_len=model.config.max_seq_len,
            dtype=next(model.parameters()).dtype,
        )

    codebook_dim = 1 + model.config.num_codebooks
    input_pos = torch.arange(0, T, device=device)
    empty = torch.empty(
        (codebook_dim, model.config.max_seq_len), dtype=dtype, device=device
    )
    empty[:, :T] = prompt
    seq = empty

    temperature = torch.tensor(
        sampling_kwargs["temperature"], device=device, dtype=torch.bfloat16
    )
    top_p = torch.tensor(sampling_kwargs["top_p"], device=device, dtype=torch.bfloat16)
    repetition_penalty = torch.tensor(
        sampling_kwargs["repetition_penalty"], device=device, dtype=torch.bfloat16
    )

    prefill_decode = decode_one_token_ar

    first_token = prefill_decode(
        model,
        prompt.view(1, codebook_dim, -1),
        input_pos,
        temperature,
        top_p,
        repetition_penalty,
        audio_masks,
        audio_parts,
    )
    seq[:, T : T + 1] = first_token

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    x = decode_n_tokens(
        model,
        first_token.view(1, codebook_dim, -1),
        input_pos,
        max_new_tokens - 1,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        audio_masks=audio_masks,
        audio_parts=audio_parts,
        decode_one_token=decode_one_token,
    )
    seq = seq[:, : T + 1 + x.size(1)]
    seq[:, T + 1 :] = x
    return seq


def load_dac_model(config_name="modded_dac_vq", checkpoint_path=None, device="cuda"):
    """Load DAC model for audio generation"""
    if checkpoint_path is None:
        return None
    
    # Import quantizer components
    from rvq import DownsampleResidualVectorQuantize
    from modded_dac import WindowLimitedTransformer
    
    # Create transformer config for quantizer
    quantizer_transformer_config = ModelArgs(
        block_size=4096,
        n_layer=8,
        n_head=16,
        dim=1024,
        intermediate_size=3072,
        n_local_heads=-1,
        head_dim=64,
        rope_base=10000,
        norm_eps=1e-5,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        channels_first=True,
    )
    
    # Create transformer module for quantizer
    transformer_module = WindowLimitedTransformer(
        causal=True,
        window_size=128,
        input_dim=1024,
        config=quantizer_transformer_config,
    )
    
    # Create quantizer
    quantizer = DownsampleResidualVectorQuantize(
        input_dim=1024,
        n_codebooks=9,
        codebook_size=1024,
        codebook_dim=8,
        quantizer_dropout=0.5,
        downsample_factor=[2, 2],
        post_module=transformer_module,
        pre_module=transformer_module,
        semantic_codebook_size=4096,
    )
    
    # Create transformer config for main model
    transformer_config = ModelArgs(
        block_size=16384,
        n_layer=8,
        n_head=16,
        dim=1024,
        intermediate_size=3072,
        n_local_heads=-1,
        head_dim=64,
        rope_base=10000,
        norm_eps=1e-5,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        channels_first=True,
    )
    
    # Create DAC model with quantizer
    model = DAC(
        encoder_dim=64,
        encoder_rates=[2, 4, 8, 8],
        latent_dim=1024,
        decoder_dim=1536,
        decoder_rates=[8, 8, 4, 2],
        quantizer=quantizer,
        sample_rate=44100,
        causal=True,
        encoder_transformer_layers=[0, 0, 0, 4],
        decoder_transformer_layers=[4, 0, 0, 0],
        transformer_general_config=lambda **kwargs: ModelArgs(**{**transformer_config.__dict__, **kwargs}),
    )

    state_dict = torch.load(
        checkpoint_path, map_location=device, mmap=True, weights_only=True
    )
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    if any("generator" in k for k in state_dict):
        state_dict = {
            k.replace("generator.", ""): v
            for k, v in state_dict.items()
            if "generator." in k
        }

    result = model.load_state_dict(state_dict, strict=False, assign=True)
    model.eval()
    model.to(device)

    logger.info(f"Loaded DAC model: {result}")
    return model


def init_model(checkpoint_path, device, precision, compile=False):
    # Check if checkpoint_path is a HuggingFace repo ID
    if isinstance(checkpoint_path, str) and "/" in checkpoint_path and not Path(checkpoint_path).exists():
        logger.info(f"Downloading model from HuggingFace: {checkpoint_path}")
        local_dir = snapshot_download(repo_id=checkpoint_path, cache_dir="cache")
        checkpoint_path = Path(local_dir)
        logger.info(f"Model downloaded to: {checkpoint_path}")
    
    # Try to find codec.pth in the model directory
    dac_checkpoint_path = checkpoint_path / "codec.pth"
    
    model = DualARTransformer.from_pretrained(checkpoint_path, load_weights=True)

    model = model.to(device=device, dtype=precision)
    logger.info(f"Restored model from checkpoint")

    # Load DAC model if codec.pth found
    dac_model = load_dac_model(checkpoint_path=dac_checkpoint_path, device=device)

    if isinstance(model, DualARTransformer):
        decode_one_token = decode_one_token_ar
        prefill_n_tokens = decode_one_token_ar
        logger.info("Using DualARTransformer")
    else:
        raise ValueError("Unsupported model type")

    # Initialize cache
    with torch.device(device):
        model.setup_caches(
            max_batch_size=1,
            max_seq_len=model.config.max_seq_len,
            dtype=next(model.parameters()).dtype,
        )

    if compile:
        logger.info("Compiling function...")
        decode_one_token = torch.compile(
            decode_one_token,
            # mode="max-autotune-no-cudagraphs",
            backend="inductor" if torch.cuda.is_available() else "aot_eager",
            mode="reduce-overhead" if torch.cuda.is_available() else None,
            fullgraph=True,
        )

    return model.eval(), decode_one_token, dac_model


@dataclass
class GenerateResponse:
    action: Literal["sample", "next"]
    codes: Optional[torch.Tensor] = None
    text: Optional[str] = None
    audio_path: Optional[str] = None


def generate_long(
    *,
    model,
    device: str | torch.device,
    decode_one_token: callable,
    text: str,
    num_samples: int = 1,
    max_new_tokens: int = 0,
    top_p: int = 0.8,
    repetition_penalty: float = 1.1,
    temperature: float = 0.8,
    compile: bool = False,
    iterative_prompt: bool = True,
    chunk_length: int = 512,
    prompt_text: Optional[str | list[str]] = None,
    prompt_tokens: Optional[torch.Tensor | list[torch.Tensor]] = None,
):
    assert 0 < top_p <= 1, "top_p must be in (0, 1]"
    assert 0 < repetition_penalty < 2, "repetition_penalty must be in (0, 2)"
    assert 0 < temperature < 2, "temperature must be in (0, 2)"

    use_prompt = prompt_text is not None and prompt_tokens is not None
    if use_prompt and isinstance(prompt_text, str):
        prompt_text = [prompt_text]
        prompt_tokens = [prompt_tokens]

    assert use_prompt is False or len(prompt_text) == len(
        prompt_tokens
    ), "Prompt text and tokens must have the same length"

    prompt_tokens = [i.cpu() for i in prompt_tokens]

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tokenizer = model.tokenizer
    base_content_sequence = ContentSequence(modality="interleave")

    max_length = model.config.max_seq_len
    if use_prompt:
        for t, c in zip(prompt_text, prompt_tokens):
            base_content_sequence.append(
                [
                    TextPart(text=t),
                    VQPart(codes=c),
                ],
                add_end=True,
                speaker=0,
            )
    base_content_sequence.append(
        [
            TextPart(text=text),
        ],
        add_end=False,
        speaker=0,
    )

    encoded, audio_masks, audio_parts = base_content_sequence.encode_for_inference(
        tokenizer, num_codebooks=model.config.num_codebooks
    )
    if encoded.size(1) > max_length - 2048:
        raise ValueError(f"Prompt is too long: {encoded.size(1)} > {max_length - 2048}")

    encoded = encoded.to(device=device)
    logger.info(f"Encoded text: {text}")

    # Move temperature, top_p, repetition_penalty to device
    # This is important so that changing params doesn't trigger recompile
    temperature = torch.tensor(temperature, device=device, dtype=torch.float)
    top_p = torch.tensor(top_p, device=device, dtype=torch.float)
    repetition_penalty = torch.tensor(
        repetition_penalty, device=device, dtype=torch.float
    )

    for sample_idx in range(num_samples):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        global_encoded = []
        seg_idx = 0
        prompt_length = encoded.size(1)

        t0 = time.perf_counter()
        y = generate(
            model=model,
            prompt=encoded,
            max_new_tokens=max_new_tokens,
            audio_masks=audio_masks,
            audio_parts=audio_parts,
            decode_one_token=decode_one_token,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        if sample_idx == 0 and seg_idx == 0 and compile:
            logger.info(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t = time.perf_counter() - t0

        tokens_generated = y.size(1) - prompt_length
        tokens_sec = tokens_generated / t
        logger.info(
            f"Generated {tokens_generated} tokens in {t:.02f} seconds, {tokens_sec:.02f} tokens/sec"
        )
        logger.info(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")

        if torch.cuda.is_available():
            logger.info(
                f"GPU Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB"
            )

        # Put the generated tokens
        # since there is <im_end>, we remove last token
        codes = y[1:, prompt_length:-1].clone()
        assert (codes >= 0).all(), f"Negative code found"

        decoded = y[:, prompt_length:].clone()
        # But for global encoding, we should keep the <im_end> token

        global_encoded.append(decoded.cpu())
        assert (codes >= 0).all(), f"Negative code found: {codes}"
        yield GenerateResponse(action="sample", codes=codes, text=text)
        seg_idx += 1

        # This indicates the end of the current sample
        yield GenerateResponse(action="next")


@dataclass
class WrappedGenerateResponse:
    status: Literal["success", "error"]
    response: Optional[GenerateResponse | Exception] = None


@dataclass
class GenerateRequest:
    request: dict
    response_queue: queue.Queue


def launch_thread_safe_queue(
    checkpoint_path,
    device,
    precision,
    compile: bool = False,
):
    input_queue = queue.Queue()
    init_event = threading.Event()

    def worker():
        model, decode_one_token, dac_model = init_model(
            checkpoint_path, device, precision, compile=compile
        )
        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )
        init_event.set()

        while True:
            item: GenerateRequest | None = input_queue.get()
            if item is None:
                break

            kwargs = item.request
            response_queue = item.response_queue

            try:
                for chunk in generate_long(
                    model=model, decode_one_token=decode_one_token, **kwargs
                ):
                    response_queue.put(
                        WrappedGenerateResponse(status="success", response=chunk)
                    )
            except Exception as e:
                logger.error(traceback.format_exc())
                response_queue.put(WrappedGenerateResponse(status="error", response=e))

    threading.Thread(target=worker, daemon=True).start()
    init_event.wait()

    return input_queue


def process_prompt_audio(audio_path: Path, dac_model, device: str):
    """Convert audio file to tokens using DAC model"""
    if dac_model is None:
        raise ValueError("DAC model is required to process audio prompts")
    
    # Load audio
    audio, sr = torchaudio.load(str(audio_path))
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)
    audio = torchaudio.functional.resample(audio, sr, dac_model.sample_rate)

    audios = audio[None].to(device)
    logger.info(f"Loaded prompt audio with {audios.shape[2] / dac_model.sample_rate:.2f} seconds")

    # Encode to tokens
    audio_lengths = torch.tensor([audios.shape[2]], device=device, dtype=torch.long)
    indices, indices_lens = dac_model.encode(audios, audio_lengths)

    if indices.ndim == 3:
        indices = indices[0]

    logger.info(f"Generated prompt tokens of shape {indices.shape}")
    return indices.cpu()


@click.command()
@click.option(
    "--text",
    type=str,
    default="Always stay healthy and happy. I love you guys.",
)
@click.option("--prompt-text", type=str, default=("I'm Chloe. And you, what's your name? Delighted to meet you, John. Of course. I'm the first personal assistant built by CyberLife. I take care of most everyday tasks like cooking, housework, or managing your appointments, for example. I really didn't do much, you know. I just spoke with a few humans to see if they could tell the difference between me and a real person. It was a really interesting experience. Absolutely. But I only exist thanks to the intelligence of the humans who designed me. And, you know, they have something I could never have. A soul.",), multiple=True)
@click.option(
    "--prompt-audio",
    type=click.Path(path_type=Path, exists=True),
    default=("chloe35s.mp3",),
    multiple=True,
    help="Audio files to use as prompts (will be converted to tokens)"
)
@click.option("--num-samples", type=int, default=1)
@click.option("--max-new-tokens", type=int, default=0)
@click.option("--top-p", type=float, default=0.8)
@click.option("--repetition-penalty", type=float, default=1.1)
@click.option("--temperature", type=float, default=0.8)
@click.option(
    "--checkpoint-path",
    type=str,
    default="fishaudio/openaudio-s1-mini",
)
@click.option("--device", type=str, default="cuda")
@click.option("--compile/--no-compile", default=False)
@click.option("--seed", type=int, default=42)
@click.option("--half/--no-half", default=False)
@click.option("--iterative-prompt/--no-iterative-prompt", default=True)
@click.option("--chunk-length", type=int, default=300)
@click.option("--output-dir", type=Path, default="temp")
@click.option("--save-codes/--no-save-codes", default=False, help="Save intermediate codes")
def main(
    text: str,
    prompt_text: Optional[list[str]],
    prompt_audio: Optional[list[Path]],
    num_samples: int,
    max_new_tokens: int,
    top_p: int,
    repetition_penalty: float,
    temperature: float,
    checkpoint_path: str,
    device: str,
    compile: bool,
    seed: int,
    half: bool,
    iterative_prompt: bool,
    chunk_length: int,
    output_dir: Path,
    save_codes: bool,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    precision = torch.half if half else torch.bfloat16

    # Validate prompt inputs
    if prompt_text is not None and prompt_audio is not None:
        if len(prompt_text) != len(prompt_audio):
            raise ValueError(
                f"Number of prompt texts ({len(prompt_text)}) must match number of prompt audio files ({len(prompt_audio)})"
            )

    logger.info("Loading model ...")
    t0 = time.time()
    model, decode_one_token, dac_model = init_model(
        checkpoint_path, device, precision, compile=compile
    )
    with torch.device(device):
        model.setup_caches(
            max_batch_size=1,
            max_seq_len=model.config.max_seq_len,
            dtype=next(model.parameters()).dtype,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    logger.info(f"Time to load model: {time.time() - t0:.02f} seconds")

    # Process prompt tokens from audio files
    prompt_tokens = None
    if prompt_audio is not None:
        prompt_tokens = []
        for audio_path in prompt_audio:
            tokens = process_prompt_audio(audio_path, dac_model, device)
            prompt_tokens.append(tokens)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    generator = generate_long(
        model=model,
        device=device,
        decode_one_token=decode_one_token,
        text=text,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        compile=compile,
        iterative_prompt=iterative_prompt,
        chunk_length=chunk_length,
        prompt_text=prompt_text,
        prompt_tokens=prompt_tokens,
    )

    idx = 0
    codes = []

    for response in generator:
        if response.action == "sample":
            codes.append(response.codes)
            logger.info(f"Sampled text: {response.text}")
        elif response.action == "next":
            if codes:
                combined_codes = torch.cat(codes, dim=1)
                
                # Save codes if requested
                if save_codes:
                    codes_npy_path = os.path.join(output_dir, f"codes_{idx}.npy")
                    np.save(codes_npy_path, combined_codes.cpu().numpy())
                    logger.info(f"Saved codes to {codes_npy_path}")
                
                # Generate audio if DAC model is available
                if dac_model is not None:
                    try:
                        audio_path = os.path.join(output_dir, f"audio_{idx}.wav")
                        
                        # Convert codes to audio - detach to remove gradients
                        indices = combined_codes.detach().to(device).long()
                        indices_lens = torch.tensor([indices.shape[1]], device=device, dtype=torch.long)
                        
                        fake_audios, audio_lengths = dac_model.decode(indices, indices_lens)
                        fake_audio = fake_audios[0, 0].detach().float().cpu().numpy()
                        
                        # Save audio
                        sf.write(audio_path, fake_audio, dac_model.sample_rate)
                        logger.info(f"Saved audio to {audio_path}")
                        
                    except Exception as e:
                        logger.error(f"Failed to generate audio: {e}")
                else:
                    logger.warning("DAC model not available, skipping audio generation")
                    
            logger.info(f"Next sample")
            codes = []
            idx += 1
        else:
            logger.error(f"Error: {response}")


if __name__ == "__main__":
    main()
