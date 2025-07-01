# coding=utf-8
# Copyright 2024 and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Parler-TTS model configuration"""

from transformers import AutoConfig, logging
from transformers.configuration_utils import PretrainedConfig


logger = logging.get_logger(__name__)

MUSICGEN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/parler_tts-small": "https://huggingface.co/facebook/parler_tts-small/resolve/main/config.json",
    # See all ParlerTTS models at https://huggingface.co/models?filter=parler_tts
}


class ParlerTTSDecoderConfig(PretrainedConfig):
    model_type = "parler_tts_decoder"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=2049,  # vocab size = 2048 (encodec vocab size) + 1 (eos)
        max_position_embeddings=2048,
        num_hidden_layers=24,
        ffn_dim=4096,
        num_attention_heads=16,
        layerdrop=0.0,
        use_cache=True,
        activation_function="gelu",
        hidden_size=1024,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        initializer_factor=0.02,
        scale_embedding=False,
        num_codebooks=4,
        pad_token_id=2048,
        bos_token_id=2049,
        eos_token_id=2048,
        tie_word_embeddings=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.ffn_dim = ffn_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.initializer_factor = initializer_factor
        self.layerdrop = layerdrop
        self.use_cache = use_cache
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.num_codebooks = num_codebooks

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class ParlerTTSConfig(PretrainedConfig):

    model_type = "parler_tts"
    is_composition = True

    def __init__(self, vocab_size=1024, **kwargs):
        super().__init__(**kwargs)
        if "audio_encoder" not in kwargs or "decoder" not in kwargs:
            raise ValueError("Config has to be initialized with text_encoder, audio_encoder and decoder config")

        audio_encoder_config = kwargs.pop("audio_encoder")
        audio_encoder_model_type = audio_encoder_config.pop("model_type")

        decoder_config = kwargs.pop("decoder")

        self.vocab_size = vocab_size
        self.audio_encoder = AutoConfig.for_model(audio_encoder_model_type, **audio_encoder_config)
        self.decoder = ParlerTTSDecoderConfig(**decoder_config)
        self.is_encoder_decoder = False

    @classmethod
    def from_sub_models_config(
        cls,
        audio_encoder_config: PretrainedConfig,
        decoder_config: ParlerTTSDecoderConfig,
        **kwargs,
    ):
        r"""
        Instantiate a [`ParlerTTSConfig`] (or a derived class) from text encoder, audio encoder and decoder
        configurations.

        Returns:
            [`ParlerTTSConfig`]: An instance of a configuration object
        """

        return cls(
            audio_encoder=audio_encoder_config.to_dict(),
            decoder=decoder_config.to_dict(),
            **kwargs,
        )

    @property
    # This is a property because you might want to change the codec model on the fly
    def sampling_rate(self):
        return self.audio_encoder.sampling_rate
