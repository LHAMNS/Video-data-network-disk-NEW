#!/usr/bin/env python3
"""
Mercury 2 Auto-Fixer
Automatically scans code files, identifies issues, and uses Mercury 2
(diffusion model from Inception Labs) to generate fixes.

Features:
- Reads source files and known issues
- Splits large files into chunks to stay within 128K context window
- Sends code + issues to Mercury 2 with highest reasoning level
- Applies fixes automatically
- Provides a summary report

Usage:
    export INCEPTION_API_KEY="sk_..."
    python3 mercury_autofix.py [--dry-run] [--file specific_file.py]
"""

import json
import os
import sys
import re
import time
import argparse
import logging
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("INCEPTION_API_KEY", "sk_7a760ef3d2444f756cb947511d21b9d5")
MERCURY_URL = "https://api.inceptionlabs.ai/v1/chat/completions"
MODEL = "mercury-2"
REASONING_EFFORT = "high"
MAX_TOKENS = 16000
# Mercury 2 context window is ~128K tokens; keep prompts under ~80K to leave room
MAX_CODE_CHARS = 60000  # ~15K tokens worth of code per request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("mercury-autofix")

PROJECT_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class CodeIssue:
    file: str
    line: Optional[int]
    severity: int  # 1-5
    description: str
    category: str = ""


@dataclass
class FixResult:
    file: str
    success: bool
    original_code: str
    fixed_code: str
    issues_fixed: list = field(default_factory=list)
    reasoning_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""


# ---------------------------------------------------------------------------
# Known issues registry (from AGENTS.md analysis)
# ---------------------------------------------------------------------------
KNOWN_ISSUES = [
    # main.py
    CodeIssue("main.py", 24, 5, "Duplicate setup_enhanced_logging function definition (lines 24 and 201)", "duplicate"),
    CodeIssue("main.py", 135, 3, "Port number has no range validation (1-65535)", "validation"),
    CodeIssue("main.py", 71, 3, "subprocess.run without timeout can hang", "timeout"),

    # converter/error_correction.py
    CodeIssue("converter/error_correction.py", 7, 5, "import reedsolo without fallback - crashes if not installed", "import"),

    # converter/gpu_error_correction.py
    CodeIssue("converter/gpu_error_correction.py", 7, 4, "import cupy without fallback - crashes if no GPU", "import"),

    # converter/gpu_frame_generator.py
    CodeIssue("converter/gpu_frame_generator.py", 7, 4, "import cupy without fallback - crashes if no GPU", "import"),

    # converter/utils.py
    CodeIssue("converter/utils.py", 200, 4, "_calculate_file_hash crashes on files < 8192 bytes (seek -8192 from end fails)", "crash"),
    CodeIssue("converter/utils.py", 221, 3, "subprocess.run without timeout in is_nvenc_available/is_qsv_available", "timeout"),

    # web_ui/server.py
    CodeIssue("web_ui/server.py", 25, 5, "Unconditional GPU module imports without try-except fallback", "import"),
    CodeIssue("web_ui/server.py", 915, 4, "Path traversal vulnerability in download_file_by_name", "security"),
    CodeIssue("web_ui/server.py", 70, 3, "MAX_CONTENT_LENGTH 16GB allows memory exhaustion", "memory"),

    # converter/frame_generator.py
    CodeIssue("converter/frame_generator.py", 88, 3, "Invalid color_count silently defaults to 16 without warning", "validation"),
    CodeIssue("converter/frame_generator.py", 201, 3, "Array bounds risk - can write beyond color_indices array", "bounds"),
    CodeIssue("converter/frame_generator.py", 426, 3, "Memory inefficiency: bytearray().extend() causes repeated reallocations", "performance"),

    # converter/decoder.py
    CodeIssue("converter/decoder.py", 448, 3, "Threading Lock created per iteration instead of shared", "threading"),

    # converter/pipeline.py
    CodeIssue("converter/pipeline.py", 97, 4, "Imports error_correction at module level which may fail", "import"),

    # requirements.txt
    CodeIssue("requirements.txt", 0, 4, "reedsolo commented out but still needed for CPU fallback; duplicate/conflicting numba versions", "dependency"),
]


# ---------------------------------------------------------------------------
# Mercury 2 API caller
# ---------------------------------------------------------------------------

def call_mercury(prompt: str, max_tokens: int = MAX_TOKENS) -> tuple[str, dict]:
    """
    Call Mercury 2 API and return (content, usage_dict).
    Handles context compression by truncating if needed.
    """
    # Estimate tokens: ~4 chars per token
    estimated_tokens = len(prompt) // 4
    if estimated_tokens > 100000:
        log.warning(f"Prompt too large ({estimated_tokens} est. tokens), truncating...")
        # Keep first and last portions
        half = MAX_CODE_CHARS // 2
        prompt = prompt[:half] + "\n\n... [TRUNCATED FOR CONTEXT LIMIT] ...\n\n" + prompt[-half:]

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "reasoning_effort": REASONING_EFFORT,
        "temperature": 0.5,
    }

    data = json.dumps(payload).encode("utf-8")
    req = Request(MERCURY_URL, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {API_KEY}")

    retries = 3
    for attempt in range(retries):
        try:
            with urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
                content = result["choices"][0]["message"]["content"] or ""
                usage = result.get("usage", {})
                return content, usage
        except (HTTPError, URLError) as e:
            log.warning(f"API error (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return "", {"error": str(e)}

    return "", {"error": "max retries exceeded"}


# ---------------------------------------------------------------------------
# File processor
# ---------------------------------------------------------------------------

def split_file_into_chunks(code: str, max_chars: int = MAX_CODE_CHARS) -> list[str]:
    """Split large files into chunks, preserving function/class boundaries."""
    if len(code) <= max_chars:
        return [code]

    chunks = []
    lines = code.split("\n")
    current_chunk = []
    current_size = 0

    for line in lines:
        line_size = len(line) + 1
        if current_size + line_size > max_chars and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_size = 0
        current_chunk.append(line)
        current_size += line_size

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def fix_file(filepath: str, issues: list[CodeIssue], dry_run: bool = False) -> FixResult:
    """Use Mercury 2 to fix issues in a single file."""
    full_path = PROJECT_ROOT / filepath
    if not full_path.exists():
        return FixResult(filepath, False, "", "", error=f"File not found: {filepath}")

    original_code = full_path.read_text(encoding="utf-8", errors="replace")

    # Build issue description
    issue_desc = "\n".join(
        f"  {i+1}. [Line {iss.line or '?'}] (severity {iss.severity}) {iss.description}"
        for i, iss in enumerate(issues)
    )

    # Split into chunks if needed
    chunks = split_file_into_chunks(original_code)
    fixed_chunks = []

    for chunk_idx, chunk in enumerate(chunks):
        chunk_label = f" (chunk {chunk_idx+1}/{len(chunks)})" if len(chunks) > 1 else ""
        log.info(f"  Sending {filepath}{chunk_label} to Mercury 2...")

        prompt = f"""Fix the following Python code. Known issues:
{issue_desc}

Rules:
- Return ONLY the complete fixed code, no markdown fences, no explanations
- Keep all existing functionality intact
- Only fix the listed issues, don't refactor unrelated code
- Preserve all comments (Chinese and English)
- If an import may fail, wrap it in try-except with a fallback

Code:
{chunk}"""

        fixed_code, usage = call_mercury(prompt)
        reasoning = usage.get("reasoning_tokens", 0)
        completion = usage.get("completion_tokens", 0)

        log.info(f"  Mercury 2: reasoning={reasoning}, completion={completion}")

        if not fixed_code.strip():
            log.warning(f"  Empty response for {filepath}{chunk_label}, keeping original")
            fixed_chunks.append(chunk)
            continue

        # Clean markdown fences if Mercury added them
        fixed_code = re.sub(r'^```\w*\n', '', fixed_code)
        fixed_code = re.sub(r'\n```\s*$', '', fixed_code)

        fixed_chunks.append(fixed_code)

    final_code = "\n".join(fixed_chunks)

    result = FixResult(
        file=filepath,
        success=True,
        original_code=original_code,
        fixed_code=final_code,
        issues_fixed=[iss.description for iss in issues],
        reasoning_tokens=usage.get("reasoning_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
    )

    if not dry_run and final_code != original_code:
        full_path.write_text(final_code, encoding="utf-8")
        log.info(f"  Applied fixes to {filepath}")
    elif dry_run:
        log.info(f"  [DRY RUN] Would apply fixes to {filepath}")

    return result


def fix_requirements(dry_run: bool = False) -> FixResult:
    """Fix requirements.txt specifically."""
    filepath = "requirements.txt"
    full_path = PROJECT_ROOT / filepath
    original = full_path.read_text()

    prompt = f"""Fix this Python requirements.txt file. Issues:
1. reedsolo is commented out but needed for CPU fallback
2. numba appears twice with different versions (0.55.1 and 0.59.1)
3. opencv-python appears twice with different minimum versions

Rules:
- Return ONLY the fixed requirements.txt content
- No markdown fences, no explanations
- Keep reedsolo uncommented
- Remove duplicate entries, keep the newer version
- Keep GPU packages (cupy) as optional/commented

{original}"""

    fixed, usage = call_mercury(prompt, max_tokens=2000)
    log.info(f"  requirements.txt: reasoning={usage.get('reasoning_tokens', 0)}, completion={usage.get('completion_tokens', 0)}")

    if not fixed.strip():
        return FixResult(filepath, False, original, original, error="Empty response")

    fixed = re.sub(r'^```\w*\n', '', fixed)
    fixed = re.sub(r'\n```\s*$', '', fixed)

    result = FixResult(
        file=filepath,
        success=True,
        original_code=original,
        fixed_code=fixed,
        issues_fixed=["dependency conflicts"],
    )

    if not dry_run and fixed != original:
        full_path.write_text(fixed)
        log.info(f"  Applied fixes to {filepath}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mercury 2 Auto-Fixer")
    parser.add_argument("--dry-run", action="store_true", help="Show fixes without applying")
    parser.add_argument("--file", help="Fix specific file only")
    parser.add_argument("--min-severity", type=int, default=3, help="Minimum severity to fix (default: 3)")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Mercury 2 Auto-Fixer")
    log.info(f"Model: {MODEL} | Reasoning: {REASONING_EFFORT}")
    log.info(f"Project: {PROJECT_ROOT}")
    log.info("=" * 60)

    # Group issues by file
    issues_by_file: dict[str, list[CodeIssue]] = {}
    for issue in KNOWN_ISSUES:
        if args.file and issue.file != args.file:
            continue
        if issue.severity < args.min_severity:
            continue
        issues_by_file.setdefault(issue.file, []).append(issue)

    log.info(f"Files to fix: {len(issues_by_file)}")
    log.info(f"Total issues: {sum(len(v) for v in issues_by_file.values())}")
    log.info("")

    results = []
    total_reasoning = 0
    total_completion = 0

    for filepath, issues in sorted(issues_by_file.items(), key=lambda x: -max(i.severity for i in x[1])):
        max_sev = max(i.severity for i in issues)
        log.info(f"[{filepath}] {len(issues)} issues (max severity: {max_sev})")

        if filepath == "requirements.txt":
            result = fix_requirements(dry_run=args.dry_run)
        else:
            result = fix_file(filepath, issues, dry_run=args.dry_run)

        results.append(result)
        total_reasoning += result.reasoning_tokens
        total_completion += result.completion_tokens
        log.info("")

    # Summary
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    success = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)
    log.info(f"Files processed: {len(results)}")
    log.info(f"Successful: {success}")
    log.info(f"Failed: {failed}")
    log.info(f"Total reasoning tokens: {total_reasoning}")
    log.info(f"Total completion tokens: {total_completion}")

    for r in results:
        status = "OK" if r.success else f"FAILED: {r.error}"
        changed = r.original_code != r.fixed_code if r.success else False
        log.info(f"  {r.file}: {status} {'(modified)' if changed else '(unchanged)'}")

    if args.dry_run:
        log.info("\n[DRY RUN] No files were modified.")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
