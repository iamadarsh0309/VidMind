"""Validate and clean code snippets extracted from OCR or LLM output."""

import ast
import re
from dataclasses import dataclass


@dataclass
class CodeValidationResult:
    code: str
    language: str
    valid: bool
    error: str | None = None


def _detect_language(code: str) -> str:
    if re.search(r"\b(def |import |class |async def |@app\.)", code):
        return "python"
    if re.search(r"\b(function |const |let |var |=>|export )", code):
        return "javascript"
    if re.search(r"\b(public |private |void |class |System\.out)", code):
        return "java"
    if re.search(r"^\s*(FROM|SELECT|INSERT|CREATE)\b", code, re.MULTILINE | re.IGNORECASE):
        return "sql"
    if re.search(r"^(docker |kubectl |npm |pip |curl )", code, re.MULTILINE | re.IGNORECASE):
        return "bash"
    return "text"


def validate_python(code: str) -> CodeValidationResult:
    try:
        ast.parse(code)
        return CodeValidationResult(code=code, language="python", valid=True)
    except SyntaxError as exc:
        return CodeValidationResult(
            code=code,
            language="python",
            valid=False,
            error=str(exc.msg),
        )


def validate_code_snippet(code: str, language: str | None = None) -> CodeValidationResult:
    lang = language or _detect_language(code)
    if lang == "python":
        return validate_python(code)
    # For other languages, basic heuristic only (no compiler required)
    if len(code.strip()) < 3:
        return CodeValidationResult(code=code, language=lang, valid=False, error="too short")
    return CodeValidationResult(code=code, language=lang, valid=True)


def validate_code_in_text(text: str) -> str:
    """
    Find fenced code blocks in markdown-ish text, validate Python blocks,
    and append a warning comment for invalid snippets.
    """
    pattern = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)

    def _replace(match: re.Match) -> str:
        lang = match.group(1) or ""
        code = match.group(2).strip()
        result = validate_code_snippet(code, language=lang or None)
        if result.valid:
            return f"```{result.language}\n{result.code}\n```"
        warning = f"<!-- OCR/validation warning: {result.error} -->"
        return f"{warning}\n```{result.language or 'text'}\n{result.code}\n```"

    return pattern.sub(_replace, text)


def validate_raw_code(code: str) -> str:
    """Validate a raw OCR code string; return best-effort cleaned version."""
    if not code.strip():
        return code
    result = validate_code_snippet(code)
    return result.code
