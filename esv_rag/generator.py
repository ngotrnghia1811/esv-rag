"""
LLM generation backend for ESV-RAG.

Supports two backends:
  - OpenAI-compatible API (online, including vLLM-served models)
  - HuggingFace / vLLM local inference (offline)

The Generator class is the single interface used by all ESV actions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import GeneratorConfig

logger = logging.getLogger(__name__)


class _Cache:
    """Disk-backed prompt→response cache."""

    def __init__(self, cache_dir: str):
        self.path = Path(cache_dir)
        self.path.mkdir(parents=True, exist_ok=True)

    def _key(self, prompt: str, model: str, temperature: float) -> str:
        raw = json.dumps({"p": prompt, "m": model, "t": temperature}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, prompt: str, model: str, temperature: float) -> Optional[str]:
        fp = self.path / (self._key(prompt, model, temperature) + ".json")
        if fp.exists():
            return json.loads(fp.read_text())["response"]
        return None

    def set(self, prompt: str, model: str, temperature: float, response: str) -> None:
        fp = self.path / (self._key(prompt, model, temperature) + ".json")
        fp.write_text(json.dumps({"response": response}))


class Generator:
    """
    Unified LLM interface for ESV actions.

    Parameters
    ----------
    config : GeneratorConfig
        Backend and generation hyperparameters.
    """

    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        self._client = None
        self._cache = _Cache(self.config.cache_dir) if self.config.use_cache else None
        self._build_client()

    # ------------------------------------------------------------------
    # Client construction
    # ------------------------------------------------------------------

    def _build_client(self) -> None:
        if self.config.api_url is not None:
            self._client = self._build_openai_client()
            logger.info("Generator → OpenAI-compatible API at %s", self.config.api_url)
        elif self.config.use_vllm:
            self._client = self._build_vllm_client()
            logger.info("Generator → vLLM local (%s)", self.config.model_name)
        else:
            self._client = self._build_hf_client()
            logger.info("Generator → HuggingFace transformers (%s)", self.config.model_name)

    def _build_openai_client(self):
        import openai
        api_key = os.environ.get("OPENAI_API_KEY", self.config.api_key)
        return openai.OpenAI(base_url=self.config.api_url, api_key=api_key)

    def _build_vllm_client(self):
        from vllm import LLM, SamplingParams
        llm = LLM(
            model=self.config.model_name,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            trust_remote_code=True,
        )
        return {"llm": llm, "SamplingParams": SamplingParams}

    def _build_hf_client(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        return {"tokenizer": tokenizer, "model": model}

    # ------------------------------------------------------------------
    # Public generation API
    # ------------------------------------------------------------------

    def generate(self, prompt: str,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 system_message: str = "You are a helpful assistant.") -> str:
        """
        Generate a single text response.

        Returns the generated string (stripped).
        """
        temp = temperature if temperature is not None else self.config.temperature
        maxt = max_tokens if max_tokens is not None else self.config.max_tokens

        if self._cache:
            cached = self._cache.get(prompt, self.config.model_name, temp)
            if cached is not None:
                logger.debug("Cache hit for prompt hash")
                return cached

        response = self._generate_impl(prompt, temp, maxt, system_message)

        if self._cache:
            self._cache.set(prompt, self.config.model_name, temp, response)

        return response

    def generate_batch(self, prompts: List[str],
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None) -> List[str]:
        """Generate responses for a batch of prompts."""
        return [self.generate(p, temperature, max_tokens) for p in prompts]

    # ------------------------------------------------------------------
    # Backend-specific generation
    # ------------------------------------------------------------------

    def _generate_impl(self, prompt: str, temperature: float,
                       max_tokens: int, system_message: str) -> str:
        if self.config.api_url is not None:
            return self._generate_openai(prompt, temperature, max_tokens, system_message)
        elif self.config.use_vllm:
            return self._generate_vllm(prompt, temperature, max_tokens)
        else:
            return self._generate_hf(prompt, temperature, max_tokens)

    def _generate_openai(self, prompt: str, temperature: float,
                          max_tokens: int, system_message: str) -> str:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user",   "content": prompt},
        ]
        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=self.config.top_p,
                )
                return resp.choices[0].message.content.strip()
            except Exception as exc:
                logger.warning("OpenAI attempt %d failed: %s", attempt + 1, exc)
                time.sleep(2 ** attempt)
        raise RuntimeError("OpenAI generation failed after 3 attempts")

    def _generate_vllm(self, prompt: str, temperature: float, max_tokens: int) -> str:
        SamplingParams = self._client["SamplingParams"]
        params = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                                top_p=self.config.top_p)
        outputs = self._client["llm"].generate([prompt], params)
        return outputs[0].outputs[0].text.strip()

    def _generate_hf(self, prompt: str, temperature: float, max_tokens: int) -> str:
        import torch
        tokenizer = self._client["tokenizer"]
        model     = self._client["model"]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=self.config.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()
