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


def is_nontrivial_code(code: str) -> bool:
    if not code or not code.strip():
        return False
    s = code.strip()
    if len(s) < 40:
        return False
    tokens = ("def ", "class ", "import ", "from ", "if __name__ == \"__main__\":")
    return any(t in s for t in tokens)


def build_repair_prompt(original_code: str, error_message: str) -> str:
    return (
        "You produced a Python script that fails to parse.\n"
        "Task: Fix ONLY the syntax/structural issues to make it valid Python.\n"
        "- Keep the intent and behavior; do not add placeholders.\n"
        "- Output a single, complete Python file.\n"
        "- Do NOT include markdown fences.\n"
        "- Ensure all try blocks have matching except/finally.\n"
        f"Compiler error: {error_message}\n\n"
        "Here is the code to repair:\n\n"
        f"{original_code}\n"
    )


class LLMClient:
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None,
                 temperature: float = 0.1, max_tokens: int = 4096,
                 reasoning_effort: Optional[str] = None) -> None:
        self.provider = provider or os.getenv("LLM_PROVIDER") or ("openai" if os.getenv("OPENAI_API_KEY") else "anthropic")
        # Use GPT-5 for simple tasks, GPT-4-turbo for complex test generation
        default_model = "gpt-5-2025-08-07"  # GPT-5 may not handle complex prompts well
        if os.getenv("FORCE_GPT5"):
            default_model = "gpt-5-2025-08-07"
        self.model = model or os.getenv("OPENAI_MODEL") or os.getenv("ANTHROPIC_MODEL") or default_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Note: reasoning_effort is not used in current OpenAI API, keeping for compatibility
        self.reasoning_effort = (
            reasoning_effort
            or os.getenv("OPENAI_REASONING_EFFORT")
            or os.getenv("REASONING_EFFORT")
            or "high"
        )

        self._openai_client = None
        self._anthropic_client = None
        if self.provider == "openai" and OpenAI is not None and os.getenv("OPENAI_API_KEY"):
            self._openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if self.provider == "anthropic" and anthropic is not None and os.getenv("ANTHROPIC_API_KEY"):
            self._anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def generate(self, prompt: str) -> str:
        print(f"LLMClient.generate called with provider: {self.provider}, model: {self.model}")
        print(f"Prompt length: {len(prompt)} characters")

        if self.provider == "openai":
            result = self._call_openai(prompt)
            print(f"OpenAI response length: {len(result)} characters")
            return result
        if self.provider == "anthropic":
            result = self._call_anthropic(prompt)
            print(f"Anthropic response length: {len(result)} characters")
            return result
        # Fallbacks
        if self._openai_client is not None:
            print("Falling back to OpenAI")
            result = self._call_openai(prompt)
            print(f"OpenAI fallback response length: {len(result)} characters")
            return result
        if self._anthropic_client is not None:
            print("Falling back to Anthropic")
            result = self._call_anthropic(prompt)
            print(f"Anthropic fallback response length: {len(result)} characters")
            return result
        raise RuntimeError("No usable LLM provider configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY.")

    def _call_openai(self, prompt: str) -> str:
        if self._openai_client is None:
            print("OpenAI client not initialized")
            return ""

        try:
            print(f"Attempting to use model: {self.model}")

            # Check if this is GPT-5 which uses different parameter names
            if "gpt-5" in self.model:
                print("Using GPT-5 specific parameters")
                response = self._openai_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    # GPT-5 doesn't support temperature parameter, only default (1)
                    max_completion_tokens=self.max_tokens,  # GPT-5 uses max_completion_tokens
                )
            else:
                # Standard GPT models
                response = self._openai_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

            content = response.choices[0].message.content
            print(f"Successfully got response from {self.model}")
            print(f"Raw response content: '{content}'")
            print(f"Response object details: {response.choices[0].message}")
            return content or ""
        except Exception as e:
            print(f"Error calling OpenAI with model {self.model}: {e}")
            print(f"Available models may include: gpt-4, gpt-4-turbo, gpt-3.5-turbo, gpt-5-2025-08-07, etc.")
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
    print(f"Generating LLM response for commit {commit_hash[:8]}")
    llm_text = client.generate(prompt)
    print(f"Raw LLM response length: {len(llm_text)}")
    print(f"Raw LLM response preview: {llm_text[:200]}...")

    code = clean_llm_code_response(llm_text)
    print(f"Cleaned code length: {len(code)}")
    print(f"Cleaned code preview: {code[:200]}...")

    # If the code is empty or clearly nontrivial, attempt regeneration before syntax validation
    if not is_nontrivial_code(code):
        print(f"Empty or trivial code produced for {commit_hash[:8]}; attempting regeneration")
        # Try up to two regenerations with the same prompt
        for attempt in range(1, 3):
            print(f"Regeneration attempt {attempt} for {commit_hash[:8]}")
            regen_text = client.generate(prompt)
            regen_code = clean_llm_code_response(regen_text)
            print(f"Regeneration attempt {attempt} code length: {len(regen_code)}")
            if is_nontrivial_code(regen_code):
                print(f"Regenerated nontrivial code on attempt {attempt} for {commit_hash[:8]}")
                code = regen_code
                break
            else:
                print(f"Regeneration attempt {attempt} still trivial for {commit_hash[:8]}")
        # If still trivial, force an explicit instruction to output full python file
        if not is_nontrivial_code(code):
            print(f"Code still trivial, trying forced regeneration for {commit_hash[:8]}")
            force_prompt = (
                prompt
                + "\n\nYour previous response was empty or incomplete. Output ONLY a complete, executable Python file implementing the requested test-case generator."
            )
            forced_text = client.generate(force_prompt)
            forced_code = clean_llm_code_response(forced_text)
            print(f"Forced regeneration code length: {len(forced_code)}")
            if is_nontrivial_code(forced_code):
                print(f"Forced regeneration produced nontrivial code for {commit_hash[:8]}")
                code = forced_code
            else:
                print(f"Forced regeneration still trivial for {commit_hash[:8]}")

    ok, err = validate_python_syntax(code)
    if not ok:
        print(f"Syntax error for {commit_hash[:8]}: {err}")
        # Attempt up to two repair passes by providing the exact error and code back to the model
        for attempt in range(1, 3):
            repair_prompt = build_repair_prompt(code, err or "")
            repaired_text = client.generate(repair_prompt)
            repaired_code = clean_llm_code_response(repaired_text)
            ok2, err2 = validate_python_syntax(repaired_code)
            if ok2:
                print(f"Repaired syntax on attempt {attempt} for {commit_hash[:8]}")
                code = repaired_code
                err = None
                break
            else:
                print(f"Repair attempt {attempt} failed for {commit_hash[:8]}: {err2}")
                err = err2
        if err is not None:
            return None

    # Final sanity: ensure code is still nontrivial before saving
    if not is_nontrivial_code(code):
        print(f"Generated code is still trivial/empty for {commit_hash[:8]} after attempts; skipping")
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



