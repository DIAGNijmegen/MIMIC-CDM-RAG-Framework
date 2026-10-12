from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
    ExLlamaV2Lora,
)
from exllamav2.generator import ExLlamaV2Sampler
import torch
import random


class ExLlamaV2BaseGeneratorRAG:
    # Internal state

    model: ExLlamaV2
    cache: ExLlamaV2Cache
    tokenizer: ExLlamaV2Tokenizer

    sequence_ids: torch.tensor = None

    def __init__(self, model, cache, tokenizer):
        self.model = model
        self.cache = cache
        self.tokenizer = tokenizer

        # Check if the model is on GPU
        gpu_count = 0
        for context in self.model.device_context:
            if context is not None and context.device_idx >= 0:
                gpu_count += 1

        if gpu_count > 0:
            print(f"Model loaded on {gpu_count} GPU(s).")
        else:
            print("Model is loaded on CPU.")

        #Print on which GPU the model is loaded
        for context in self.model.device_context:
            if context is not None and context.device_idx >= 0:
                print(f"Model loaded on GPU {context.device_idx}.")

        from exllamav2.attn import has_flash_attn
        if has_flash_attn:
            print("FlashAttention is enabled.")
        else:
            print("FlashAttention is disabled.")

    # For testing purposes, run a forward pass to make sure CUDA is fully initialized

    def warmup(self):
        input_ids = torch.zeros((1, 2), dtype=torch.long)
        self.model.forward(input_ids, cache=None, input_mask=None, preprocess_only=True)

    def full(self):
        return self.sequence_ids.shape[-1] >= self.model.config.max_seq_len

    # TODO: Argument to allow different random samples over batch dimension

    def generate_simple(
        self,
        prompt: str or list,
        gen_settings: ExLlamaV2Sampler.Settings,
        num_tokens: int,
        seed=None,
        token_healing=False,
        encode_special_tokens=True,
        decode_special_tokens=False,
        loras=None,
        stop_criteria=None,
    ):
        # Accept LoRA or list of LoRAs
        if loras is not None and isinstance(loras, ExLlamaV2Lora):
            loras = [loras]

        # Apply seed
        if seed is not None:
            random.seed(seed)

        # Tokenize input and produce padding mask if needed
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        ids = self.tokenizer.encode(prompt, encode_special_tokens=encode_special_tokens)

        overflow = ids.shape[-1] + num_tokens - self.model.config.max_seq_len
        if overflow > 0:
            ids = ids[:, overflow:]

        mask = self.tokenizer.padding_mask(ids) if batch_size > 1 else None

        # Prepare for healing
        unhealed_token = None
        if ids.shape[-1] < 2:
            token_healing = False
        if token_healing:
            unhealed_token = ids[:, -1:]
            ids = ids[:, :-1]

        # Process prompt and begin generation
        self._gen_begin_base(ids, mask, loras)

        #FIX: Removed Begin filters as it is now called in the sample function
        # # Begin filters

        # gen_settings.begin_filters(
        #     self.tokenizer.get_id_to_piece_list()[unhealed_token]
        #     if unhealed_token is not None
        #     else None
        # )

        # Generate tokens
        probabilities_sequence = torch.tensor([[]])
        filters = []  # FIX: Initialize filters if required
        for _ in range(num_tokens):
            logits = (
                self.model.forward(
                    self.sequence_ids[:, -1:], self.cache, input_mask=mask, loras=loras
                )
                .float()
                .cpu()
            )

            # Updated sample function to match new signature:
            # FIX: Now returns 5 values instead of the previous 3. The placeholders `_` are used for values we don't need (top_tokens and top_probs).
            token, _, _, probabilities, eos = ExLlamaV2Sampler.sample(
                logits,
                gen_settings,
                self.sequence_ids,
                random.random(),
                self.tokenizer,
                prefix_token=unhealed_token,
                filters=filters, #FIX: Added filters to the sample function call
            )
            
            self.sequence_ids = torch.cat([self.sequence_ids, token], dim=1)
            probabilities_sequence = torch.cat(
                [probabilities_sequence, probabilities], dim=1
            )
            #FIX: Removed the call to gen_settings.feed_filters as it does not exist anymore
            # gen_settings.feed_filters(token)

            # Check for stop tokens
            # FIX: Added a conditional check to ensure stop_criteria is not None before calling it.
            if stop_criteria and stop_criteria(self.sequence_ids, None):
                break

            unhealed_token = None
            if eos:
                break

        return self.sequence_ids, probabilities_sequence
    
        # Decode
        text = self.tokenizer.decode(
            self.sequence_ids, decode_special_tokens=decode_special_tokens
        )

        if isinstance(prompt, str):
            return text[0]
        return text

    def _gen_begin_base(self, input_ids, mask=None, loras=None):
        self.cache.current_seq_len = 0
        self.model.forward(
            input_ids[:, :-1],
            self.cache,
            input_mask=mask,
            preprocess_only=True,
            loras=loras,
        )

        self.sequence_ids = input_ids.clone()
        self.sequence_ids = input_ids
