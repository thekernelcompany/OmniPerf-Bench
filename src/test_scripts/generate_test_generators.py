#!/usr/bin/env python3
"""
Generate per-commit "test-case-generator" scripts from commit extraction JSONs.

For each JSON in commit_extractions/, we send the FULL JSON content to an LLM
and ask for a standalone Python script that generates realistic, domain-faithful
test cases. Returned code is syntax-checked and saved to:
  generated_test_generators/<hash8>_test_case_generator.py

Environment:
- Set OPENAI_API_KEY for OpenAI, or ANTHROPIC_API_KEY for Anthropic.
- Optionally set LLM_PROVIDER=openai|anthropic and model via OPENAI_MODEL/ANTHROPIC_MODEL.
"""

import argparse
import ast
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

# Optional providers
try:
    import anthropic  # type: ignore
except Exception:
    anthropic = None  # type: ignore

try:
    from openai import OpenAI  # type: ignore  # OpenAI Python SDK >= 1.0.0
except Exception:
    OpenAI = None  # type: ignore

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


def list_extraction_files(extractions_dir: str) -> List[str]:
    files = glob.glob(os.path.join(extractions_dir, "*.json"))
    files = [f for f in files if not f.endswith("extraction_summary.json")]
    files.sort()
    return files


def load_extraction(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


PROMPT_TEMPLATE_PATH = os.getenv("TEST_CASE_GENERATOR_PROMPT", "/root/OmniPerf-Bench/third-party/effibench/prompts/focused_test_case_generator_prompt.md")


def read_prompt_template(path: str = PROMPT_TEMPLATE_PATH) -> str:
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        raise RuntimeError(f"Failed to read prompt template at {path}: {e}")


def build_prompt_from_template(prompt_template: str, full_json_text: str, commit_hash: str) -> str:
    # Compose final prompt per the provided comprehensive template: include the JSON input explicitly.
    return (
        f"{prompt_template}\n\n"
        f"<!-- Commit: {commit_hash} -->\n"
        f"Here is the commit extraction JSON you must base the tests on:\n\n"
        f"```json\n{full_json_text}\n```\n"
    )


def clean_llm_code_response(text: str) -> str:
    if not text:
        return ""
    s = text.strip()
    if "```" in s:
        parts = s.split("```")
        candidates = [p.strip() for p in parts if p.strip()]
        if candidates:
            for c in candidates:
                if c.startswith("python"):
                    return c[len("python"):].lstrip("\n")
            return candidates[0]
    if s and not (s.startswith("import") or s.startswith("from") or s.startswith("class") or s.startswith("def")):
        lines = s.split("\n")
        for i, line in enumerate(lines):
            if line.startswith(("import", "from", "class", "def")):
                return "\n".join(lines[i:])
    return s


def validate_python_syntax(code: str) -> Tuple[bool, Optional[str]]:
    try:
        ast.parse(code)
        return True, None
    except Exception as e:
        return False, str(e)


class LLMClient:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None,
                 temperature: float = 0.1, max_tokens: int = 8000) -> None:
        self.provider = provider or os.getenv("LLM_PROVIDER") or ("openai" if os.getenv("OPENAI_API_KEY") else "anthropic")
        self.model = model or os.getenv("OPENAI_MODEL") or os.getenv("ANTHROPIC_MODEL") or "gpt-5-mini-2025-08-07"
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._openai_client = None
        self._anthropic_client = None
        if self.provider == "openai" and OpenAI is not None and os.getenv("OPENAI_API_KEY"):
            self._openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if self.provider == "anthropic" and anthropic is not None and os.getenv("ANTHROPIC_API_KEY"):
            self._anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def generate(self, prompt: str) -> str:
        if self.provider == "openai":
            return self._call_openai(prompt)
        if self.provider == "anthropic":
            return self._call_anthropic(prompt)
        # Fallbacks
        if self._openai_client is not None:
            return self._call_openai(prompt)
        if self._anthropic_client is not None:
            return self._call_anthropic(prompt)
        raise RuntimeError("No usable LLM provider configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.")

    def _call_openai(self, prompt: str) -> str:
        if self._openai_client is None:
            return ""

        def responses_call(include_temperature: bool, tokens_param: Optional[str]) -> str:
            kwargs: Dict[str, Any] = {"model": self.model, "input": prompt}
            if tokens_param:
                kwargs[tokens_param] = self.max_tokens
            if include_temperature:
                kwargs["temperature"] = self.temperature
            resp = self._openai_client.responses.create(**kwargs)
            content = getattr(resp, "output_text", None)
            if content:
                return content
            chunks: List[str] = []
            for item in getattr(resp, "output", []) or []:
                for c in getattr(item, "content", []) or []:
                    t = getattr(getattr(c, "text", None), "value", None)
                    if t:
                        chunks.append(t)
            return "".join(chunks)

        def chat_call(include_temperature: bool) -> str:
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
            }
            if include_temperature:
                kwargs["temperature"] = self.temperature
            # Ensure older chat API also respects output length
            kwargs["max_tokens"] = self.max_tokens
            chat = self._openai_client.chat.completions.create(**kwargs)
            return chat.choices[0].message.content or ""

        # Try Responses API with different parameter variants and gracefully
        # retry when temperature is unsupported by the model.
        last_err: Optional[Exception] = None
        for tokens_param in ("max_output_tokens", "max_completion_tokens", "max_tokens", None):
            for include_temperature in (True, False):
                try:
                    return responses_call(include_temperature, tokens_param)
                except Exception as e:  # noqa: BLE001
                    msg = str(e)
                    # Retry without temperature if model disallows it
                    if include_temperature and (
                        "unsupported_value" in msg
                        or "does not support" in msg
                        or "temperature" in msg
                    ):
                        continue
                    # If invalid kw, try next tokens param variant
                    if (
                        "unexpected keyword" in msg
                        or "invalid_request_error" in msg and "max_" in msg
                    ):
                        last_err = e
                        break
                    last_err = e
            # proceed to next tokens_param

        # Fallback to Chat Completions (older API) if Responses consistently fails
        for include_temperature in (True, False):
            try:
                return chat_call(include_temperature)
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if include_temperature and (
                    "unsupported_value" in msg or "does not support" in msg
                ):
                    continue
                last_err = e
                break

        if last_err is not None:
            print(f"Error calling OpenAI: {last_err}")
        return ""

    def _call_anthropic(self, prompt: str) -> str:
        if self._anthropic_client is None:
            return ""
        try:
            message = self._anthropic_client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            if hasattr(message, "content") and message.content:
                blk = message.content[0]
                text = getattr(blk, "text", None)
                if text:
                    return text
            return ""
        except Exception as e:
            print(f"Error calling Anthropic: {e}")
            return ""


def save_script(out_dir: str, commit_hash: str, code: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{commit_hash[:8]}_test_case_generator.py"
    path = os.path.join(out_dir, fname)
    with open(path, "w") as f:
        f.write(code)
    return path


def process_extraction_file(path: str, out_dir: str, client: LLMClient) -> Optional[Dict[str, Any]]:
    try:
        raw = load_extraction(path)
    except Exception as e:
        print(f"Failed to load {os.path.basename(path)}: {e}")
        return None

    commit_hash = raw.get("commit_hash", os.path.splitext(os.path.basename(path))[0])
    try:
        with open(path, "r") as f:
            full_json_text = f.read()
    except Exception as e:
        print(f"Failed to read JSON text from {path}: {e}")
        return None

    # Build prompt using the comprehensive template and write it to eg_test_generator.txt for transparency.
    template_text = read_prompt_template()
    prompt = build_prompt_from_template(template_text, full_json_text, commit_hash)
    try:
        with open("./eg_test_generator.txt", "w") as f:
            f.write(prompt)
    except Exception as e:
        print(f"Warning: failed to write prompt to eg_test_generator.txt: {e}")
    llm_text = client.generate(prompt)
    code = clean_llm_code_response(llm_text)

    ok, err = validate_python_syntax(code)
    if not ok:
        print(f"Syntax error for {commit_hash[:8]}: {err}")
        return None

    script_path = save_script(out_dir, commit_hash, code)
    return {
        "commit_hash": commit_hash,
        "script_path": script_path,
        "category": raw.get("csv_metadata", {}).get("category"),
        "message": raw.get("message"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-commit test-case-generator scripts via LLM")
    parser.add_argument("--extractions-dir", default="./commit_extractions", help="Directory containing commit JSONs")
    parser.add_argument("--out-dir", default="./generated_test_generators", help="Output directory for scripts")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of commits (0 = no limit)")
    parser.add_argument("--only-hash", default="", help="If set, only process commits whose filename contains this substring")
    parser.add_argument("--provider", choices=["openai", "anthropic"], default=None, help="LLM provider")
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=8000)
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        return

    files = list_extraction_files(args.extractions_dir)
    if args.only_hash:
        files = [f for f in files if args.only_hash in os.path.basename(f)]
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    if not files:
        print("No commit extraction JSON files found.")
        return

    client = LLMClient(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    index: List[Dict[str, Any]] = []
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {os.path.basename(path)}")
        result = process_extraction_file(path, args.out_dir, client)
        if result:
            print(f"  -> Saved: {result['script_path']}")
            index.append(result)
        else:
            print("  -> Skipped")

    if index:
        os.makedirs(args.out_dir, exist_ok=True)
        idx_path = os.path.join(args.out_dir, "index.json")
        with open(idx_path, "w") as f:
            json.dump(index, f, indent=2)
        print(f"\nIndex written: {idx_path}")
    print(f"Done. Successful: {len(index)}/{len(files)}")


if __name__ == "__main__":
    main()



