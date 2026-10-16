"""Single LLM provider layer: Claude CLI or Codex CLI, same JSON-in/JSON-out contract.

Configuration, highest priority first:
  1. function arguments (``provider=...``)
  2. environment (or <repo>/.env): REEL_LLM_PROVIDER (auto|claude|codex), REEL_CLAUDE_MODEL, REEL_CODEX_MODEL
  3. optional ``reel_factory.config.json`` next to app.py
     {"provider": "auto", "claude_model": "opus", "codex_model": null}
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from . import runtime_paths
from .pipeline import PipelineError

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "reel_factory.config.json"
PROVIDERS = ("claude", "codex")
DEFAULT_CLAUDE_MODEL = "opus"
CLAUDE_EFFORTS = {"low", "medium", "high", "xhigh", "max"}
CODEX_EFFORTS = {"low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high"}
_STATUS_TTL = 20.0
_status_cache: tuple[float, dict] | None = None


class LLMError(PipelineError):
    pass


# --------------------------------------------------------------------- config
def _config() -> dict:
    try:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError):
        return {}


def _clean(value) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def claude_model() -> str:
    return _clean(runtime_paths.env_value("REEL_CLAUDE_MODEL")) or _clean(_config().get("claude_model")) \
        or DEFAULT_CLAUDE_MODEL


def codex_model() -> str | None:
    return _clean(runtime_paths.env_value("REEL_CODEX_MODEL")) or _clean(_config().get("codex_model"))


def _default_provider() -> str:
    value = (_clean(runtime_paths.env_value("REEL_LLM_PROVIDER")) or _clean(_config().get("provider")) or "auto").lower()
    return value if value in {"auto", *PROVIDERS} else "auto"


# --------------------------------------------------------------------- status
def _claude_ready(path: str) -> bool:
    try:
        result = subprocess.run(
            [path, "auth", "status"], capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=15,
        )
    except Exception:
        return False
    if result.returncode != 0:
        return False
    try:
        return bool(json.loads(result.stdout).get("loggedIn"))
    except (ValueError, AttributeError):
        # Older CLIs print plain text; a zero exit code is the best signal we have.
        return "not logged in" not in (result.stdout or "").lower()


def _codex_ready(command: list[str]) -> bool:
    try:
        from .codex_cli import codex_environment
        result = subprocess.run(
            command + ["login", "status"], cwd=ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=15, env=codex_environment(),
        )
        return result.returncode == 0
    except Exception:
        return False


def provider_status(use_cache: bool = True) -> dict:
    global _status_cache
    now = time.monotonic()
    if use_cache and _status_cache and now - _status_cache[0] < _STATUS_TTL:
        return json.loads(json.dumps(_status_cache[1]))
    claude = runtime_paths.claude_path()
    codex = runtime_paths.codex_command()
    status = {
        "claude": {
            "installed": bool(claude),
            "ready": bool(claude) and _claude_ready(claude),
            "model": claude_model(),
        },
        "codex": {
            "installed": bool(codex),
            "ready": bool(codex) and _codex_ready(codex),
            "model": codex_model(),
        },
    }
    _status_cache = (now, status)
    return json.loads(json.dumps(status))


def reset_status_cache() -> None:
    global _status_cache
    _status_cache = None


def resolve_provider(requested: str | None = None) -> str:
    wanted = (requested or "auto").strip().lower()
    if wanted not in {"auto", *PROVIDERS}:
        raise LLMError(f"Неизвестный ИИ-провайдер: {requested!r}. Доступны: auto, claude, codex.")
    status = provider_status()
    if wanted in PROVIDERS:
        if not status[wanted]["ready"]:
            raise LLMError(_not_ready_message(wanted, status))
        return wanted
    default = _default_provider()
    if default in PROVIDERS and status[default]["ready"]:
        return default
    for name in PROVIDERS:
        if status[name]["ready"]:
            return name
    raise LLMError(
        "Нет доступного ИИ-провайдера. " + _not_ready_message("claude", status) + " "
        + _not_ready_message("codex", status)
    )


def _not_ready_message(name: str, status: dict) -> str:
    info = status[name]
    if name == "claude":
        if not info["installed"]:
            return "Claude CLI не найден (установите Claude Code и выполните `claude auth login`)."
        return "Claude CLI не авторизован (выполните `claude auth login`)."
    if not info["installed"]:
        return "Codex CLI не найден (установите `npm i -g @openai/codex` или положите его в .tools/codex-cli)."
    return "Codex CLI не авторизован (выполните `codex login`)."


# ------------------------------------------------------------------ validation
_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "null": lambda v: v is None,
}


def validate_schema(value, schema: dict, path: str = "$") -> None:
    """Small recursive JSON-schema subset validator; raises ValueError on mismatch."""
    expected = schema.get("type")
    if expected is not None:
        types = expected if isinstance(expected, list) else [expected]
        if not any(_TYPE_CHECKS.get(item, lambda _v: True)(value) for item in types):
            raise ValueError(f"{path}: ожидался тип {'/'.join(types)}, получено {type(value).__name__}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: значение {value!r} не из {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path}: {value} меньше минимума {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path}: {value} больше максимума {schema['maximum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            raise ValueError(f"{path}: строка короче {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ValueError(f"{path}: строка длиннее {schema['maxLength']}")
    if isinstance(value, dict):
        for key in schema.get("required") or []:
            if key not in value:
                raise ValueError(f"{path}: отсутствует обязательное поле {key!r}")
        for key, sub in (schema.get("properties") or {}).items():
            if key in value and isinstance(sub, dict):
                validate_schema(value[key], sub, f"{path}.{key}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise ValueError(f"{path}: элементов меньше {schema['minItems']}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ValueError(f"{path}: элементов больше {schema['maxItems']}")
        items = schema.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(value):
                validate_schema(item, items, f"{path}[{index}]")


# --------------------------------------------------------------------- parsing
def extract_json_object(text: str) -> dict:
    """Pull the outermost JSON object out of free text (handles ``` fences and chatter)."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    decoder = json.JSONDecoder()
    index = cleaned.find("{")
    while index != -1:
        try:
            value, _ = decoder.raw_decode(cleaned, index)
            if isinstance(value, dict):
                return value
        except ValueError:
            pass
        index = cleaned.find("{", index + 1)
    raise ValueError("в ответе не найден JSON-объект")


def _prompt_with_schema(prompt: str, schema: dict) -> str:
    return (
        prompt.rstrip()
        + "\n\nОтветь ОДНИМ JSON-объектом, строго соответствующим JSON-схеме ниже. "
        "Без пояснений, без markdown и без текста до или после JSON.\n"
        "JSON-схема:\n" + json.dumps(schema, ensure_ascii=False)
    )


def _image_hashes(images: list[Path] | None) -> list[str]:
    """SHA-256 содержимого каждого изображения (в порядке передачи): часть ключа кэша."""
    hashes = []
    for image in images or []:
        try:
            hashes.append(hashlib.sha256(Path(image).read_bytes()).hexdigest())
        except OSError as exc:
            raise LLMError(f"Не удалось прочитать изображение {image}: {exc}") from exc
    return hashes


def _fingerprint(prompt: str, schema: dict, provider: str, model: str | None, effort: str,
                 image_hashes: list[str] | None = None) -> str:
    payload = {"prompt": prompt, "schema": schema, "provider": provider, "model": model, "effort": effort}
    if image_hashes:   # без картинок ключ прежний: уже накопленный кэш текстовых запросов остаётся действительным
        payload["images"] = image_hashes
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _prompt_with_images(prompt: str, images: list[Path], provider: str) -> str:
    listing = "\n".join(f"{n}. {Path(p).resolve()}" for n, p in enumerate(images, 1))
    if provider == "claude":
        note = ("К запросу приложены изображения (контактные листы кадров). ПЕРЕД ответом прочитай КАЖДЫЙ файл "
                "инструментом Read (все вызовы Read сделай сразу) и опирайся на то, что на них видно:\n" + listing)
    else:
        note = ("К запросу приложены изображения (контактные листы кадров), в том же порядке, что и список; "
                "опирайся на то, что на них видно:\n" + listing)
    return prompt.rstrip() + "\n\n" + note


# --------------------------------------------------------------------- backends
def _run_claude(prompt: str, schema: dict, work_dir: Path, model: str, effort: str,
                timeout: int, log: list[str], images: list[Path] | None = None) -> str | dict:
    claude = runtime_paths.claude_path()
    if not claude:
        raise LLMError("Claude CLI не найден.")
    command = [
        claude, "-p", "--output-format", "json", "--model", model,
        "--tools", "Read" if images else "", "--no-session-persistence",
        "--effort", effort if effort in CLAUDE_EFFORTS else "high",
        "--json-schema", json.dumps(schema, ensure_ascii=False),
    ]
    if images:  # единственный инструмент — Read, и только каталоги с картинками
        for directory in dict.fromkeys(str(Path(p).resolve().parent) for p in images):
            command += ["--add-dir", directory]
    try:
        result = subprocess.run(
            command, cwd=work_dir, input=prompt, text=True, encoding="utf-8", errors="replace",
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise LLMError(f"Claude не ответил за {timeout} с.") from exc
    log.append(f"$ claude (exit {result.returncode})\n--- stdout\n{result.stdout}\n--- stderr\n{result.stderr}")
    envelope = None
    try:
        envelope = json.loads(result.stdout)
    except ValueError:
        pass
    if result.returncode != 0 or (isinstance(envelope, dict) and envelope.get("is_error")):
        detail = ""
        if isinstance(envelope, dict) and isinstance(envelope.get("result"), str):
            detail = envelope["result"]
        detail = (detail or result.stderr or result.stdout or "").strip()[-600:]
        hint = " Выполните `claude auth login`." if "login" in detail.lower() else ""
        raise LLMError(f"Claude CLI завершился с ошибкой: {detail}{hint}")
    if not isinstance(envelope, dict):
        return result.stdout
    structured = envelope.get("structured_output")
    if isinstance(structured, dict):
        return structured
    return envelope.get("result") if isinstance(envelope.get("result"), str) else result.stdout


def _run_codex(prompt: str, schema: dict, work_dir: Path, task: str, model: str | None,
               effort: str, timeout: int, log: list[str], images: list[Path] | None = None) -> str:
    command_prefix = runtime_paths.codex_command()
    if not command_prefix:
        raise LLMError("Codex CLI не найден.")
    from .codex_cli import codex_environment
    schema_path = work_dir / f"{task}.codex.schema.json"
    output_path = work_dir / f"{task}.codex.output.json"
    schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    output_path.unlink(missing_ok=True)
    command = command_prefix + [
        "--ask-for-approval", "never", "exec", "-",
        "--skip-git-repo-check", "--ephemeral", "--sandbox", "read-only",
        "--ignore-rules", "--config", f'model_reasoning_effort="{CODEX_EFFORTS.get(effort, "high")}"',
        "--output-schema", str(schema_path),
        "--output-last-message", str(output_path), "--color", "never",
    ]
    if model:
        command += ["--model", model]
    for image in images or []:   # -i повторяется; стоит среди флагов, а не рядом с позиционным «-»
        command[command.index("--sandbox"):command.index("--sandbox")] = ["-i", str(Path(image).resolve())]
    try:
        env = codex_environment()
    except RuntimeError as exc:
        raise LLMError(f"Codex CLI не авторизован: {exc}") from exc
    try:
        result = subprocess.run(
            command, cwd=work_dir, input=prompt, text=True, encoding="utf-8", errors="replace",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise LLMError(f"Codex не ответил за {timeout} с.") from exc
    log.append(f"$ codex (exit {result.returncode})\n{result.stdout}")
    if result.returncode != 0 or not output_path.is_file():
        tail = (result.stdout or "")[-5000:].lower()
        if any(marker in tail for marker in (
            "failed to connect", "error sending request", "stream disconnected", "socket",
            "os error 10013", "network",
        )):
            raise LLMError("Codex не смог подключиться к сети. Проверьте интернет и сетевые разрешения.")
        raise LLMError("Codex не смог обработать запрос. Проверьте вход (`codex login`). " + (result.stdout or "")[-400:])
    return output_path.read_text(encoding="utf-8")


# ------------------------------------------------------------------------ API
def run_llm_json(prompt: str, schema: dict, work_dir: Path, *, task: str,
                 provider: str | None = None, effort: str = "high",
                 timeout: int = 1800, images: list[Path] | None = None) -> dict:
    """Run the prompt on the chosen provider and return a schema-validated JSON object.

    ``images`` — необязательные изображения (контактные листы кадров): Claude CLI получает инструмент Read и
    доступ к их каталогам, Codex — ``exec -i <img>``. Хэши изображений входят в ключ кэша.
    """
    chosen = resolve_provider(provider)
    model = claude_model() if chosen == "claude" else codex_model()
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    images = [Path(p) for p in images or []]
    key = _fingerprint(prompt, schema, chosen, model, effort, _image_hashes(images))
    cache_path = work_dir / f"{task}.{chosen}.json"
    log_path = work_dir / f"{task}.{chosen}.log"
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("key") == key and isinstance(cached.get("result"), dict):
                return cached["result"]
        except (OSError, ValueError, AttributeError):
            pass

    full_prompt = _prompt_with_schema(_prompt_with_images(prompt, images, chosen) if images else prompt, schema)
    log: list[str] = []
    last_error = ""
    payload: dict | None = None
    try:
        for attempt in (1, 2):
            attempt_prompt = full_prompt
            if attempt == 2:
                attempt_prompt += (
                    f"\n\nПредыдущий ответ не прошёл проверку: {last_error}. "
                    "Исправь и верни только корректный JSON-объект по схеме."
                )
            if chosen == "claude":
                raw = _run_claude(attempt_prompt, schema, work_dir, model or DEFAULT_CLAUDE_MODEL,
                                  effort, timeout, log, images)
            else:
                raw = _run_codex(attempt_prompt, schema, work_dir, task, model, effort, timeout, log, images)
            try:
                candidate = raw if isinstance(raw, dict) else extract_json_object(raw)
                validate_schema(candidate, schema)
                payload = candidate
                break
            except ValueError as exc:
                last_error = str(exc)
                log.append(f"[attempt {attempt}] невалидный ответ: {last_error}")
    finally:
        try:
            log_path.write_text("\n\n".join(log), encoding="utf-8")
        except OSError:
            pass
    if payload is None:
        raise LLMError(
            f"{'Claude' if chosen == 'claude' else 'Codex'} вернул некорректный JSON: {last_error}. "
            f"Подробности в {log_path.name}."
        )
    cache_path.write_text(json.dumps({"key": key, "result": payload}, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    return payload
