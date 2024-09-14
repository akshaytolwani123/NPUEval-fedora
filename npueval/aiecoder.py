# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import os
import re
import json
from typing import Dict, List, Optional

import openai
from anthropic import Anthropic

from .tools import aie_compiler

SYSTEM_PROMPT = """You are a part of a code generation system for AIE (AI Engines).

* Your job is to write C++ code for a single kernel that will run on an AIE tile.
* Produce only the C++ code for the requested kernel including any required headers and imports.
* Make sure the C++ code is complete and self contained in a single code block.
* Name the function exactly as specified in the request, and output only the kernel (no main(), examples, explanations or extra code).
"""

class AIECoder:
    """A basic agent that uses either OpenAI or Anthropic APIs to generate AIE kernels."""
    
    def __init__(self, model: str = "gpt-4",
                       temperature: float = 0.0,
                       top_p: float = 1.0,
                       attempts: int = 1,
                       base_url: Optional[str] = None,
                       api_key: Optional[str] = None):
        self.system_prompt = SYSTEM_PROMPT
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.attempts = attempts
        self.api_key = api_key
        self.token_usage = {'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0}
        
        # Determine if it's a reasoning model
        self.reasoning = model.startswith('o1')

        # Determine if we should use Anthropic
        self.use_anthropic = model.startswith('claude')
        
        if self.use_anthropic:
            self.client = Anthropic()
        else:
            if self.api_key:
                self.client = openai.OpenAI(base_url=base_url, api_key=self.api_key)
            else:
                self.client = openai.OpenAI(base_url=base_url)

    def __call__(self, prompt: str) -> Dict:
        """Generate code based on the prompt, with optional compilation verification."""
        self.messages.append({"role": "user", "content": prompt})

        if self.attempts == 1:
            response = self.generate_code()
            return {"response": response, "attempt": 0, "token_usage": self.token_usage, "history": self.messages}
        
        for attempt in range(self.attempts):
            response = self.generate_code()
            code = self.extract_codeblock(response)

            if not code:
                self.messages.append({"role": "user", "content": "Expected a single codeblock but no code provided."})
                continue

            compilation_result = aie_compiler(code)
            compile_pass = compilation_result.split('\n')[0] == 'Compilation successful.'
            
            if compile_pass:
                return {
                    "response": response,
                    "attempt": attempt,
                    "token_usage": self.token_usage,
                    "history": self.messages
                }
            else:
                self.messages.append({"role": "user", "content": f"Compilation failed with:\n{compilation_result}"})

        return {
            "response": response,
            "attempt": self.attempts,
            "token_usage": self.token_usage,
            "history": self.messages
        }

    def generate_code(self) -> str:
        """Lightweight wrapper for OpenAI/Anthropic API."""
        if self.use_anthropic:
            system_message = next((msg['content'] for msg in self.messages if msg['role'] == 'system'), None)
            user_messages = [msg for msg in self.messages if msg['role'] in ['user', 'assistant']]
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8192,
                temperature=self.temperature,
                system=system_message,
                messages=[{"role": m["role"], "content": m["content"]} for m in user_messages]
            )
            
            self.update_tokens({
                'completion_tokens': response.usage.output_tokens,
                'prompt_tokens': response.usage.input_tokens,
                'total_tokens': response.usage.input_tokens + response.usage.output_tokens
            })
            response_text = response.content[0].text
            self.messages.append({"role": "assistant", "content": response_text})
            
        else:
            
            if self.reasoning:
                if self.messages[0]['role'] == "system":
                    self.messages[0]['role'] = "user"
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    top_p=self.top_p,
                    seed=42
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    seed=42
                )
            
            usage = response.usage.model_dump()
            self.update_tokens(usage)
            response_text = response.choices[0].message.content
            
            self.messages.append({"role": "assistant", "content": response_text})
        return response_text

    def update_tokens(self, usage: Dict) -> None:
        """Update token usage statistics."""
        self.token_usage['completion_tokens'] += usage['completion_tokens']
        self.token_usage['prompt_tokens'] += usage['prompt_tokens']
        self.token_usage['total_tokens'] += usage['total_tokens']

    @staticmethod
    def extract_codeblock(text: str) -> Optional[str]:
        """Extract code from markdown codeblocks."""
        code_blocks = re.findall(r'```(?:[a-zA-Z0-9]+)?\n(.*?)```|```(.*?)```', text, re.DOTALL)
        code_blocks = [block for match in code_blocks for block in match if block]
        return code_blocks[0].strip() if code_blocks else None

    def reset_history(self) -> None:
        """Reset the conversation history and token usage statistics."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.token_usage = {'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0}
