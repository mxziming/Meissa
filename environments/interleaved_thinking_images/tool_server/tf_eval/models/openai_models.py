"""
OpenAI-compatible API backend.

Works with any OpenAI-compatible endpoint, including:
  - DashScope (Alibaba Cloud Qwen): https://dashscope.aliyuncs.com/compatible-mode/v1
  - Together AI, OpenRouter, local vLLM server, etc.

Example launch commands:

  # Qwen2.5-VL-72B via DashScope
  python -m tool_server.tf_eval \
    --model openai_models \
    --model_args "model_name=qwen-vl-max,api_key=sk-xxx,base_url=https://dashscope.aliyuncs.com/compatible-mode/v1" \
    --task_name pathvqa --max_rounds 5 ...

  # Qwen3-VL-32B served locally via vLLM
  python -m tool_server.tf_eval \
    --model openai_models \
    --model_args "model_name=Qwen3-VL-32B-Instruct,base_url=http://localhost:8000/v1,api_key=EMPTY" \
    --task_name pathvqa --max_rounds 5 ...
"""

from .abstract_model import tp_model
from .gemini import GEMINI_SYSTEM_PROMPT_BACKWARD

import uuid
import time
import base64
import os
from io import BytesIO
from PIL import Image
from typing import List

from ..utils.utils import *
from ..tool_inferencer.dynamic_batch_manager import DynamicBatchItem
from ..utils.log_utils import get_logger

logger = get_logger(__name__)


class OpenaiModels(tp_model):
    def __init__(
        self,
        model_name: str = "qwen-vl-max",
        api_key: str = None,
        base_url: str = None,
        max_retry: int = 5,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        from openai import OpenAI

        self.model_name = model_name
        self.max_retry = int(max_retry)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        if not resolved_key:
            raise ValueError("No API key provided. Pass api_key= or set OPENAI_API_KEY / DASHSCOPE_API_KEY.")

        client_kwargs = {"api_key": resolved_key}
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    def set_generation_config(self, config):
        pass

    def getitem_fn(self, meta_data, idx):
        item = meta_data[idx]
        image_path = item.get("image_path") or item.get("image_file")
        image = Image.open(image_path).convert("RGB") if image_path else None
        return dict(image=image, text=item["text"], idx=item["idx"])

    def _pil_to_data_url(self, image: Image.Image) -> str:
        buf = BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"

    def generate_conversation_fn(self, text, image, role="user"):
        messages = [
            {"role": "system", "content": GEMINI_SYSTEM_PROMPT_BACKWARD}
        ]
        content = [{"type": "text", "text": text}]
        if image is not None:
            if isinstance(image, str):
                # already a data URL or base64
                url = image if image.startswith("data:") else f"data:image/jpeg;base64,{image}"
            else:
                url = self._pil_to_data_url(image)
            content.append({"type": "image_url", "image_url": {"url": url}})
        messages.append({"role": role, "content": content})
        return messages

    def append_conversation_fn(self, conversation, text, image, role):
        content = [{"type": "text", "text": text}]
        if image is not None:
            if isinstance(image, str):
                if "," in image:
                    image = image.split(",")[1]
                url = f"data:image/jpeg;base64,{image}"
            else:
                url = self._pil_to_data_url(image)
            content.append({"type": "image_url", "image_url": {"url": url}})
        conversation.append({"role": role, "content": content})
        return conversation

    def generate(self, batch: List[DynamicBatchItem]):
        if not batch:
            return

        # API models run one item at a time (rate-limit friendly)
        for item in batch:
            # Strip system role — convert to first user message prefix if needed
            messages = []
            for msg in item.conversation:
                if msg["role"] == "system":
                    # Prepend system content as text into the first user message below
                    messages.append(msg)
                else:
                    messages.append(msg)

            fail_times = 0
            output_text = ""
            while fail_times < self.max_retry:
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )
                    output_text = response.choices[0].message.content or ""
                    # strip markdown fences if present
                    t = output_text.strip()
                    if t.startswith("```json"):
                        t = t[7:].strip()
                    if t.startswith("```"):
                        t = t[3:].strip()
                    if t.endswith("```"):
                        t = t[:-3].strip()
                    output_text = t
                    break
                except Exception as e:
                    fail_times += 1
                    wait = fail_times * 5
                    logger.error(f"API error for {item.meta_data.get('idx')}: {e}. Retry {fail_times}/{self.max_retry} in {wait}s")
                    time.sleep(wait)

            if fail_times >= self.max_retry:
                logger.error(f"API failed after {self.max_retry} retries for {item.meta_data.get('idx')}")
                output_text = ""

            item.model_response.append(output_text)
            self.append_conversation_fn(item.conversation, output_text, None, "assistant")
