#!/usr/bin/env python3
"""
Generate per-commit "test-case-generator" scripts from commit extraction JSONs.

For each JSON in commit_extractions/, we send the FULL JSON content to an LLM
and ask for a standalone Python script that generates realistic, domain-faithful
test cases. Returned code is syntax-checked and saved to:
  generated_test_generators/<hash8>_test_case_generator.py

Environment:
- Set OPENAI_API_KEY for OpenAI, ANTHROPIC_API_KEY for Anthropic, or OPENROUTER_API_KEY for OpenRouter.
- For AWS Bedrock: Configure AWS credentials via AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, 
  AWS credential files (~/.aws/credentials), or IAM roles. Set AWS_REGION (defaults to us-west-2).
- Optionally set LLM_PROVIDER=openai|anthropic|openrouter|bedrock and model via OPENAI_MODEL/ANTHROPIC_MODEL.
- For Bedrock, models are automatically mapped to Bedrock model IDs (e.g., claude-4 -> anthropic.claude-sonnet-4-20250514-v1:0).
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
    from anthropic import AnthropicBedrock  # type: ignore
except Exception:
    AnthropicBedrock = None  # type: ignore

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


PROMPT_TEMPLATE_PATH = os.getenv("TEST_CASE_GENERATOR_PROMPT", "/root/OmniPerf-Bench/third-party/effibench/prompts/claude_4_prompt_v2.md")


def read_prompt_template(path: str = PROMPT_TEMPLATE_PATH) -> str:
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        raise RuntimeError(f"Failed to read prompt template at {path}: {e}")




def hierarchical_api_search(affected_apis: List[str], manifest_symbols: List[Dict[str, Any]]) -> List[str]:
    """Perform hierarchical search for API matches in manifest.

    For each affected API, search progressively from most specific to least specific:
    vllm.v1.engine.core_client.AsyncMPClient.get_output_async
    1. get_output_async
    2. AsyncMPClient
    3. core_client
    4. engine
    5. v1
    6. vllm
    """
    relevant_symbols = []

    for api in affected_apis:
        if not api or "." not in api:
            continue

        parts = api.split(".")
        found_matches = False

        # Search from most specific to least specific
        for i in range(len(parts) - 1, -1, -1):  # Start from end, go backwards
            search_term = parts[i]

            # Find symbols that contain this term
            for symbol in manifest_symbols:
                qualname = symbol.get('qualname', '')
                name = symbol.get('name', '')

                # Check if this symbol matches our search term
                if (search_term in qualname or
                    search_term in name or
                    qualname.endswith('.' + search_term) or
                    name == search_term):

                    relevant_symbols.append(symbol)
                    found_matches = True

            # If we found matches at this level, we can stop for this API
            # (we want the most specific matches possible)
            if found_matches:
                break

    return relevant_symbols



def build_prompt_from_template(prompt_template: str, full_json_text: str, commit_hash: str, api_manifest_path: Optional[str] = None) -> str:
    # Parse JSON to extract template variables
    try:
        import json
        commit_data = json.loads(full_json_text)
    except:
        commit_data = {}

    # Extract template variables
    commit_message = commit_data.get("message", "")
    git_diff = commit_data.get("files", [])
    changed_symbols_json = json.dumps(commit_data.get("affected_apis", []), indent=2)
    changed_files_json = json.dumps([f.get("file_path", "") for f in git_diff], indent=2)

    # Load and format API manifest symbols
    api_manifest_symbols = ""
    if api_manifest_path and os.path.exists(api_manifest_path):
        try:
            with open(api_manifest_path, 'r') as f:
                manifest_data = json.load(f)
            symbols = manifest_data.get("manifest", {}).get("symbols", [])

            # Hierarchical search for relevant symbols based on affected APIs
            relevant_symbols = symbols

            affected_apis = commit_data.get("affected_apis", [])
            if affected_apis:
                # Use hierarchical search to find relevant symbols
                relevant_symbols = hierarchical_api_search(affected_apis, symbols)

                if relevant_symbols:
                    print(f"Hierarchical search found {len(relevant_symbols)} relevant symbols from {len(symbols)} total")
                else:
                    print(f"No hierarchical matches found, using all {len(symbols)} symbols")
                    relevant_symbols = symbols
            else:
                print(f"No affected APIs in commit, using all {len(symbols)} symbols")

            # Limit to reasonable number to avoid token limits
            display_limit = min(50, len(relevant_symbols))
            symbol_list = []
            for symbol in relevant_symbols[:display_limit]:
                qualname = symbol.get('qualname', '')
                kind = symbol.get('kind', 'unknown')
                symbol_list.append(f"- {qualname} ({kind})")

            api_manifest_symbols = "\n".join(symbol_list)
            if len(relevant_symbols) > display_limit:
                api_manifest_symbols += f"\n... and {len(relevant_symbols) - display_limit} more relevant symbols"

        except Exception as e:
            api_manifest_symbols = f"Error loading API manifest: {e}"

    # Fill template placeholders
    prompt = prompt_template.format(
        commit_hash=commit_hash,
        commit_message=commit_message,
        git_diff=json.dumps(git_diff, indent=2),
        changed_symbols_json=changed_symbols_json,
        changed_files_json=changed_files_json,
        api_manifest_symbols=api_manifest_symbols,
        module_hint="",  # Could be extracted from commit data
        symbol_hint="",  # Could be extracted from commit data
        impl_tag="child",
        commit_role="optimization",
        default_device="cuda",
        default_dtype="torch.float16",
        opt_gates_json="{}"
    )

    # Backward-compatible: append the full JSON for additional context
    return (
        f"{prompt}\n\n"
        f"<!-- Commit: {commit_hash} -->\n"
        f"Here is the commit extraction JSON you must base the tests on:\n\n"
        f"```json\n{full_json_text}\n```\n"
    )


def build_system_and_user_messages(prompt_template: str, full_json_text: str, commit_hash: str) -> List[Dict[str, str]]:
    """Construct a two-message chat: system carries policy/instructions; user carries commit context.

    Returns a list of messages appropriate for Chat Completions APIs.
    """
    system_text = prompt_template
    user_text = (
        f"<!-- Commit: {commit_hash} -->\n"
        "Here is the commit extraction JSON you must base the tests on:\n\n"
        f"```json\n{full_json_text}\n```\n"
    )
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]


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
        # Determine provider. Prefer explicit arg/env, then available API keys (OpenRouter → OpenAI → Anthropic → Bedrock)
        self.provider = (
            provider
            or os.getenv("LLM_PROVIDER")
            or ("openrouter" if os.getenv("OPENROUTER_API_KEY") 
                else ("openai" if os.getenv("OPENAI_API_KEY") 
                      else ("anthropic" if os.getenv("ANTHROPIC_API_KEY") 
                            else "bedrock")))
        )

        # Choose a sensible default model based on provider, while allowing overrides via args/env
        if self.provider == "openrouter":
            default_model = os.getenv("OPENROUTER_MODEL") or "anthropic/claude-opus-4.1"
        else:
            # Use GPT-5 for simple tasks (if available), otherwise standard OpenAI defaults
            default_model = "gpt-5-2025-08-07"
            if os.getenv("FORCE_GPT5"):
                default_model = "gpt-5-2025-08-07"

        self.model = (
            model
            or os.getenv("OPENAI_MODEL")
            or os.getenv("ANTHROPIC_MODEL")
            or default_model
        )
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
        self._openrouter_client = None
        self._anthropic_client = None
        self._bedrock_client = None
        if self.provider == "openai" and OpenAI is not None and os.getenv("OPENAI_API_KEY"):
            self._openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if self.provider == "openrouter" and OpenAI is not None and os.getenv("OPENROUTER_API_KEY"):
            base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
            default_headers = {}
            # Optional but recommended headers per OpenRouter best practices
            if os.getenv("OPENROUTER_REFERRER"):
                default_headers["HTTP-Referer"] = os.getenv("OPENROUTER_REFERRER", "")
            if os.getenv("OPENROUTER_APP_TITLE"):
                default_headers["X-Title"] = os.getenv("OPENROUTER_APP_TITLE", "")
            self._openrouter_client = OpenAI(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url=base_url,
                default_headers=default_headers or None,
            )
        if self.provider == "anthropic" and anthropic is not None and os.getenv("ANTHROPIC_API_KEY"):
            self._anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        if self.provider == "bedrock" and AnthropicBedrock is not None:
            # Initialize Bedrock client - it uses default AWS credential providers automatically
            try:
                aws_region = os.getenv("AWS_REGION", "us-east-1")
                self._bedrock_client = AnthropicBedrock(aws_region=aws_region)
            except Exception as e:
                print(f"Warning: Failed to initialize Bedrock client: {e}")
                self._bedrock_client = None

    def generate(self, prompt_or_messages) -> str:
        """Generate a completion.

        Accepts either:
        - str: treated as a single user message
        - list[{"role": str, "content": str}]: passed as-is
        """
        print(f"LLMClient.generate called with provider: {self.provider}, model: {self.model}")
        try:
            if isinstance(prompt_or_messages, str):
                approx_len = len(prompt_or_messages)
            else:
                approx_len = sum(len(m.get("content", "")) for m in prompt_or_messages)
        except Exception:
            approx_len = 0
        print(f"Prompt/messages approx length: {approx_len} characters")

        if self.provider == "openai":
            result = self._call_openai(prompt_or_messages)
            print(f"OpenAI response length: {len(result)} characters")
            return result
        if self.provider == "openrouter":
            result = self._call_openrouter(prompt_or_messages)
            print(f"OpenRouter response length: {len(result)} characters")
            return result
        if self.provider == "anthropic":
            result = self._call_anthropic(prompt_or_messages)
            print(f"Anthropic response length: {len(result)} characters")
            return result
        if self.provider == "bedrock":
            result = self._call_bedrock(prompt_or_messages)
            print(f"Bedrock response length: {len(result)} characters")
            return result
        # Fallbacks
        if self._openai_client is not None:
            print("Falling back to OpenAI")
            result = self._call_openai(prompt_or_messages)
            print(f"OpenAI fallback response length: {len(result)} characters")
            return result
        if self._openrouter_client is not None:
            print("Falling back to OpenRouter")
            result = self._call_openrouter(prompt_or_messages)
            print(f"OpenRouter fallback response length: {len(result)} characters")
            return result
        if self._anthropic_client is not None:
            print("Falling back to Anthropic")
            result = self._call_anthropic(prompt_or_messages)
            print(f"Anthropic fallback response length: {len(result)} characters")
            return result
        if self._bedrock_client is not None:
            print("Falling back to Bedrock")
            result = self._call_bedrock(prompt_or_messages)
            print(f"Bedrock fallback response length: {len(result)} characters")
            return result
        raise RuntimeError("No usable LLM provider configured. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or configure AWS credentials for Bedrock.")

    def _call_openai(self, prompt_or_messages) -> str:
        if self._openai_client is None:
            print("OpenAI client not initialized")
            return ""

        try:
            print(f"Attempting to use model: {self.model}")

            # Check if this is GPT-5 which uses different parameter names
            if "gpt-5" in self.model:
                print("Using GPT-5 specific parameters")
                # Build messages
                if isinstance(prompt_or_messages, str):
                    messages = [{"role": "user", "content": prompt_or_messages}]
                else:
                    messages = prompt_or_messages
                response = self._openai_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    # GPT-5 doesn't support temperature parameter, only default (1)
                    max_completion_tokens=self.max_tokens,
                )
            else:
                # Standard GPT models
                if isinstance(prompt_or_messages, str):
                    messages = [{"role": "user", "content": prompt_or_messages}]
                else:
                    messages = prompt_or_messages
                response = self._openai_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
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

    def _call_anthropic(self, prompt_or_messages) -> str:
        if self._anthropic_client is None:
            return ""
        try:
            if isinstance(prompt_or_messages, str):
                messages = [{"role": "user", "content": prompt_or_messages}]
            else:
                messages = prompt_or_messages
            message = self._anthropic_client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=messages,
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

    def _call_openrouter(self, prompt_or_messages) -> str:
        if self._openrouter_client is None:
            print("OpenRouter client not initialized")
            return ""
        try:
            if isinstance(prompt_or_messages, str):
                messages = [{"role": "user", "content": prompt_or_messages}]
            else:
                messages = prompt_or_messages
            response = self._openrouter_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            content = response.choices[0].message.content
            return content or ""
        except Exception as e:
            print(f"Error calling OpenRouter with model {self.model}: {e}")
            return ""

    def _call_bedrock(self, prompt_or_messages) -> str:
        if self._bedrock_client is None:
            print("Bedrock client not initialized")
            return ""
        try:
            if isinstance(prompt_or_messages, str):
                messages = [{"role": "user", "content": prompt_or_messages}]
            else:
                messages = prompt_or_messages
            
            # Use appropriate Bedrock model name
            bedrock_model = self.model
            if not (bedrock_model.startswith("anthropic.") or bedrock_model.startswith("us.")):
                # Map common model names to Bedrock model IDs (using inference profiles for Claude 4+ models)
                model_mapping = {
                    "claude-3-opus": "anthropic.claude-3-opus-20240229-v1:0",
                    "claude-3-sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
                    "claude-3-haiku": "anthropic.claude-3-5-haiku-20241022-v1:0",
                    "claude-3.5-sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
                    "claude-3.5-haiku": "anthropic.claude-3-5-haiku-20241022-v1:0",
                    "claude-4": "us.anthropic.claude-sonnet-4-20250514-v1:0",
                    "claude-4-sonnet": "us.anthropic.claude-sonnet-4-20250514-v1:0",
                    "claude-4-opus": "us.anthropic.claude-opus-4-20250514-v1:0",
                    "claude-4.1-opus": "us.anthropic.claude-opus-4-1-20250805-v1:0",
                }
                # Default to Claude Opus 4.1 for the highest quality (using inference profile)
                bedrock_model = model_mapping.get(bedrock_model, "us.anthropic.claude-opus-4-1-20250805-v1:0")
            
            print(f"Using Bedrock model: {bedrock_model}")
            
            # Handle system messages for Bedrock API - extract system content if present
            system_content = None
            filtered_messages = []
            for msg in messages:
                if msg.get("role") == "system":
                    system_content = msg.get("content", "")
                else:
                    filtered_messages.append(msg)
            
            # Use streaming for long requests to avoid 10-minute timeout
            create_kwargs = {
                "model": bedrock_model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": filtered_messages,
                "stream": True,
            }
            if system_content:
                create_kwargs["system"] = system_content
            
            stream = self._bedrock_client.messages.create(**create_kwargs)
            
            # Collect the streamed response
            full_text = ""
            for chunk in stream:
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                    full_text += chunk.delta.text
            
            return full_text
        except Exception as e:
            print(f"Error calling Bedrock with model {bedrock_model}: {e}")
            print(f"Make sure AWS credentials are configured and you have access to Bedrock models")
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

    # Build prompt messages (system + user) and write them to eg_test_generator.txt for transparency.
    template_text = read_prompt_template()
    messages = build_system_and_user_messages(template_text, full_json_text, commit_hash)
    try:
        with open("./eg_test_generator.txt", "w") as f:
            f.write("--- system ---\n")
            f.write(messages[0]["content"])  # system
            f.write("\n\n--- user ---\n")
            f.write(messages[1]["content"])  # user
    except Exception as e:
        print(f"Warning: failed to write prompt to eg_test_generator.txt: {e}")
    print(f"Generating LLM response for commit {commit_hash[:8]}")
    llm_text = client.generate(messages)
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
            regen_text = client.generate(messages)
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
            # Create a forced user message appended to the prior messages
            forced_messages = list(messages)
            forced_messages.append({
                "role": "user",
                "content": (
                    "Your previous response was empty or incomplete. "
                    "Output ONLY a complete, executable Python file implementing the requested test-case generator."
                ),
            })
            forced_text = client.generate(forced_messages)
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
            # Send repair as user while preserving original system context
            repair_messages = [messages[0], {"role": "user", "content": repair_prompt}]
            repaired_text = client.generate(repair_messages)
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
    parser.add_argument("--provider", choices=["openai", "anthropic", "openrouter", "bedrock"], default=None, help="LLM provider")
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=65536)
    args = parser.parse_args()

    # Check for API keys or AWS credentials for Bedrock
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))
    has_aws_creds = (
        bool(os.getenv("AWS_ACCESS_KEY_ID")) or 
        os.path.exists(os.path.expanduser("~/.aws/credentials")) or
        os.path.exists(os.path.expanduser("~/.aws/config"))
    )
    
    if not (has_openai or has_anthropic or has_openrouter or has_aws_creds):
        print("Please set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY, or configure AWS credentials for Bedrock")
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



