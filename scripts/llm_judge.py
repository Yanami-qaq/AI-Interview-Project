import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ALLOWED_ACTIONS = {"continue", "switch_question", "finish"}
DIMENSION_KEYS = {
    "technical_accuracy",
    "key_point_coverage",
    "logic_structure",
    "practical_application",
    "communication_clarity",
}
ENV_LOADED = False


@dataclass(frozen=True)
class LlmJudgeConfig:
    enabled: bool = False
    api_url: str = ""
    api_key: str = ""
    model: str = ""
    judge_mode: str = "full"
    timeout_seconds: float = 20.0
    max_score_delta: float | None = None


def load_env_file(env_file: str | Path | None = None) -> Path | None:
    global ENV_LOADED
    if ENV_LOADED:
        return None

    candidates: list[Path] = []
    if env_file:
        candidates.append(Path(env_file))
    else:
        candidates.extend(
            [
                Path.cwd() / ".env",
                Path(__file__).resolve().parent.parent / ".env",
            ]
        )

    loaded_path: Path | None = None
    for candidate in candidates:
        if not candidate.exists() or not candidate.is_file():
            continue
        for raw_line in candidate.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        loaded_path = candidate
        break

    ENV_LOADED = True
    return loaded_path


def load_llm_judge_config(
    enabled: bool | None = None,
    api_url: str = "",
    api_key: str = "",
    api_key_env: str = "OPENAI_API_KEY",
    model: str = "",
    judge_mode: str = "",
    timeout_seconds: float = 45.0,
    max_score_delta: float | None = None,
) -> LlmJudgeConfig:
    load_env_file()
    env_enabled = os.getenv("LLM_JUDGE_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}
    resolved_enabled = env_enabled if enabled is None else enabled
    resolved_api_url = api_url or os.getenv("LLM_JUDGE_API_URL", "").strip()
    resolved_model = model or os.getenv("LLM_JUDGE_MODEL", "").strip()
    resolved_mode = (judge_mode or os.getenv("LLM_JUDGE_MODE", "full")).strip().lower()
    if resolved_mode not in {"conservative", "balanced", "full"}:
        resolved_mode = "full"
    resolved_api_key = api_key or os.getenv("LLM_JUDGE_API_KEY", "").strip()
    if not resolved_api_key and api_key_env:
        resolved_api_key = os.getenv(api_key_env, "").strip()
    if not resolved_api_key:
        resolved_api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if _is_placeholder_secret(resolved_api_key):
        resolved_api_key = ""
    if max_score_delta is None:
        env_delta = os.getenv("LLM_JUDGE_MAX_SCORE_DELTA", "").strip()
        max_score_delta = float(env_delta) if env_delta else _default_max_score_delta(resolved_mode)
    return LlmJudgeConfig(
        enabled=bool(resolved_enabled and resolved_api_url and resolved_model),
        api_url=resolved_api_url,
        api_key=resolved_api_key,
        model=resolved_model,
        judge_mode=resolved_mode,
        timeout_seconds=timeout_seconds,
        max_score_delta=max(0.0, max_score_delta),
    )


def public_llm_judge_status(config: LlmJudgeConfig) -> dict[str, Any]:
    return {
        "enabled": config.enabled,
        "configured": bool(config.api_url and config.model),
        "provider": "openai_compatible",
        "model": config.model,
        "mode": config.judge_mode,
        "api_url": config.api_url,
        "has_api_key": bool(config.api_key),
        "max_score_delta": config.max_score_delta,
    }


def _slim_question(question: dict[str, Any], answer_len_hint: int = 0) -> dict[str, Any]:
    std = question.get("standard_answer") or ""
    return {
        "question_id": question.get("question_id"),
        "question": question.get("question"),
        "keywords": question.get("keywords"),
        "score_points": question.get("score_points"),
        "standard_answer": std[:400] if len(std) > 400 else std,
    }


def _slim_evaluation(rule_evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": rule_evaluation.get("score"),
        "dimensions": rule_evaluation.get("dimensions"),
        "confidence": rule_evaluation.get("confidence"),
        "score_point_hits": rule_evaluation.get("score_point_hits"),
        "keyword_hits": rule_evaluation.get("keyword_hits"),
        "relevance": rule_evaluation.get("relevance"),
    }


def _slim_next_question(next_question: dict[str, Any] | None) -> dict[str, Any] | None:
    if not next_question:
        return None
    return {
        "question_id": next_question.get("question_id"),
        "question": next_question.get("question"),
        "topic": next_question.get("topic"),
        "difficulty": next_question.get("difficulty"),
    }


def enhance_interview_result(
    *,
    question: dict[str, Any],
    session: dict[str, Any],
    answer: str,
    rule_evaluation: dict[str, Any],
    rule_decision: dict[str, Any],
    follow_up: str,
    next_question: dict[str, Any] | None,
    config: LlmJudgeConfig,
) -> dict[str, Any]:
    if not config.enabled:
        return {"evaluation": rule_evaluation, "decision": rule_decision, "meta": {"enabled": False}}

    prompt_payload = {
        "question": _slim_question(question),
        "candidate_answer": answer,
        "session_context": {
            "role": session.get("role"),
            "role_label": session.get("role_label"),
            "history_count": len(session.get("history", [])),
            "current_follow_ups": session.get("current_follow_ups", []),
            "max_questions": session.get("max_questions", 3),
        },
        "rule_evaluation": _slim_evaluation(rule_evaluation),
        "rule_decision": rule_decision,
        "rule_follow_up": follow_up,
        "next_question": _slim_next_question(next_question),
    }

    try:
        llm_payload = _call_openai_compatible(config, prompt_payload)
        evaluation = _merge_evaluation(rule_evaluation, llm_payload.get("evaluation", {}), config)
        decision = _merge_decision(rule_decision, llm_payload.get("decision", {}), next_question)
        return {
            "evaluation": evaluation,
            "decision": decision,
            "meta": {
                "enabled": True,
                "provider": "openai_compatible",
                "model": config.model,
                "mode": config.judge_mode,
                "status": "ok",
            },
        }
    except Exception as exc:
        return {
            "evaluation": rule_evaluation,
            "decision": rule_decision,
            "meta": {"enabled": True, "status": "fallback", "error": str(exc)},
        }


def generate_interview_question(
    *,
    role: str,
    role_label: str,
    topic: str,
    difficulty: str,
    focus: str,
    config: LlmJudgeConfig,
) -> dict[str, Any]:
    if not config.enabled:
        raise RuntimeError("LLM judge is not enabled; cannot generate fallback question.")

    payload = {
        "role": role,
        "role_label": role_label,
        "requested_topic": topic,
        "requested_difficulty": difficulty or "medium",
        "requested_focus": focus,
        "required_schema": {
            "title": "string",
            "question_type": "technical_knowledge|scenario|project_deep_dive|behavioral",
            "topic": "short topic key",
            "difficulty": "easy|medium|hard",
            "keywords": ["3-6 core concepts"],
            "question": "one clear interview question in Chinese",
            "standard_answer": "structured reference answer in Chinese",
            "follow_ups": ["2-4 follow-up questions"],
            "score_points": ["4-6 concrete scoring points"],
            "common_mistakes": ["2-4 common mistakes"],
        },
    }
    request_body = {
        "model": config.model,
        "temperature": 0.3,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是中文技术面试题库设计助手。只能输出 JSON。"
                    "请按给定 schema 生成一题可用于真实模拟面试的题目。"
                    "题目必须聚焦用户请求的主题和考点，不能泛泛而谈。"
                    "评分点必须可用于判断候选人回答是否命中。"
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    }
    response_payload = _post_openai_compatible(config, request_body)
    content = response_payload["choices"][0]["message"]["content"]
    generated = _parse_json_content(content)
    return _normalize_generated_question(generated, payload)


def _call_openai_compatible(config: LlmJudgeConfig, payload: dict[str, Any]) -> dict[str, Any]:
    request_body = {
        "model": config.model,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是中文技术面试官、评分官与追问策略助手。"
                    "只能输出 JSON，不要输出 Markdown。"
                    "规则评分只是参考证据，不是最终答案。"
                    "你需要结合标准答案、评分点、候选人回答和上下文给出独立评价。"
                    "评分前必须先判断回答是否回应了当前题目。"
                    "如果回答与题目无关或明显跑题，总分必须在 0-2 分，技术准确性和要点覆盖度必须很低。"
                    "不要凭空认定候选人说过没有出现的内容，不要改变题目。"
                    "decision.message 是你作为面试官说出的话，必须像真实面试官一样自然、有温度。"
                    "先用 1-2 句直接回应候选人的具体内容（肯定说对的点、指出遗漏、或纠正错误），"
                    "再自然地引出追问或下一题，语气要像在真实对话，不要用'好的下一题'这样的机械过渡。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "请基于输入 JSON 返回："
                    "{\"evaluation\":{\"score\":0-10,\"dimensions\":{},\"strengths\":[],\"improvements\":[],"
                    "\"confidence\":\"low|medium|high\",\"relevance\":{\"level\":\"relevant|partial|weak|off_topic|empty\","
                    "\"max_score\":0-10,\"reason\":\"\"},\"llm_review\":\"\",\"llm_reason\":\"\"},"
                    "\"decision\":{\"action\":\"continue|switch_question|finish\",\"message\":\"\","
                    "\"should_finish\":false,\"reason\":\"\"}}。\n"
                    f"当前 judge_mode={config.judge_mode}。"
                    "conservative 表示尽量贴近规则分；balanced 表示可纠正规则遗漏并给出更像真人面试官的评价；"
                    "full 表示以你的专业判断为最终评分依据，规则评分只作为证据参考和异常兜底。"
                    "如果 candidate_answer 没有回答当前 question，relevance.level 必须是 off_topic 或 weak，"
                    "score 不能因为表达流畅、长度足够或出现业务场景词而升高。"
                    "decision.message 写法要求：先用 1-2 句针对候选人实际回答内容给出回应（可以是认可、补充、纠正），"
                    "再自然地引出追问或下一题，不要把题目原文直接粘贴进去作为过渡，不要使用'好的，下一题'这类机械模板。"
                    "如果 action=switch_question，过渡要自然，让候选人感觉是对话而不是系统在切换题目。\n"
                    f"输入 JSON：{json.dumps(payload, ensure_ascii=False)}"
                ),
            },
        ],
    }
    response_payload = _post_openai_compatible(config, request_body)
    content = response_payload["choices"][0]["message"]["content"]
    return _parse_json_content(content)


def _post_openai_compatible(config: LlmJudgeConfig, request_body: dict[str, Any]) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    request = urllib.request.Request(
        config.api_url,
        data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"LLM judge HTTP {exc.code}: {detail[:300]}") from exc

    response_payload = json.loads(raw)
    return response_payload


def _merge_evaluation(
    rule_evaluation: dict[str, Any],
    llm_evaluation: dict[str, Any],
    config: LlmJudgeConfig,
) -> dict[str, Any]:
    merged = dict(rule_evaluation)
    rule_score = float(rule_evaluation.get("score", 0))
    if "score" in llm_evaluation:
        merged["score"] = _bounded_score(float(llm_evaluation["score"]), rule_score, config.max_score_delta or 0.0)

    llm_dimensions = llm_evaluation.get("dimensions")
    if isinstance(llm_dimensions, dict):
        dimensions = dict(rule_evaluation.get("dimensions", {}))
        for key, value in llm_dimensions.items():
            if key in DIMENSION_KEYS and key in dimensions:
                dimensions[key] = _bounded_score(float(value), float(dimensions[key]), config.max_score_delta or 0.0)
        merged["dimensions"] = dimensions

    for key in ("strengths", "improvements"):
        if isinstance(llm_evaluation.get(key), list) and llm_evaluation[key]:
            merged[key] = [str(item) for item in llm_evaluation[key]][:4]
    if llm_evaluation.get("confidence") in {"low", "medium", "high"}:
        merged["confidence"] = llm_evaluation["confidence"]
    if llm_evaluation.get("llm_reason"):
        merged["llm_reason"] = str(llm_evaluation["llm_reason"])[:500]
    if llm_evaluation.get("llm_review"):
        merged["llm_review"] = str(llm_evaluation["llm_review"])[:500]
    elif llm_evaluation.get("llm_reason"):
        merged["llm_review"] = str(llm_evaluation["llm_reason"])[:500]
    _merge_relevance(merged, llm_evaluation)
    merged["judge_source"] = f"rule_with_llm_{config.judge_mode}"
    return merged


def _merge_decision(
    rule_decision: dict[str, Any],
    llm_decision: dict[str, Any],
    next_question: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(rule_decision)
    action = llm_decision.get("action")
    if action in ALLOWED_ACTIONS:
        if action != "switch_question" or next_question:
            merged["action"] = action
    if isinstance(llm_decision.get("message"), str) and llm_decision["message"].strip():
        merged["message"] = llm_decision["message"].strip()
    if isinstance(llm_decision.get("reason"), str) and llm_decision["reason"].strip():
        merged["reason"] = llm_decision["reason"].strip()
    merged["should_finish"] = merged["action"] == "finish"
    if merged["action"] == "switch_question" and next_question:
        merged["next_question"] = next_question
    else:
        merged.pop("next_question", None)
    merged["judge_source"] = "rule_with_llm_decision"
    return merged


def _bounded_score(value: float, baseline: float, max_delta: float) -> float:
    lower = max(0.0, baseline - max_delta)
    upper = min(10.0, baseline + max_delta)
    return round(min(max(value, lower), upper), 1)


def _merge_relevance(merged: dict[str, Any], llm_evaluation: dict[str, Any]) -> None:
    rule_relevance = merged.get("relevance") if isinstance(merged.get("relevance"), dict) else {}
    llm_relevance = llm_evaluation.get("relevance") if isinstance(llm_evaluation.get("relevance"), dict) else {}
    level = str(llm_relevance.get("level") or rule_relevance.get("level") or "relevant")
    max_score = _safe_float(llm_relevance.get("max_score"), _safe_float(rule_relevance.get("max_score"), 10.0))
    reason = str(llm_relevance.get("reason") or rule_relevance.get("reason") or "")
    if level not in {"relevant", "partial", "weak", "off_topic", "empty"}:
        level = "relevant"

    if level == "empty":
        max_score = min(max_score, 0.0)
    elif level == "off_topic":
        max_score = min(max_score, 2.0)
    elif level == "weak":
        max_score = min(max_score, 3.5)
    elif level == "partial":
        max_score = min(max_score, 6.0)

    if max_score < 10:
        merged["score"] = round(min(float(merged.get("score", 0)), max_score), 1)
        dimensions = merged.get("dimensions")
        if isinstance(dimensions, dict):
            merged["dimensions"] = {key: round(min(float(value), max_score), 1) for key, value in dimensions.items()}
        if level in {"empty", "off_topic", "weak"}:
            merged["confidence"] = "low"
            improvements = list(merged.get("improvements", []))
            reminder = "回答与当前问题关联度不足，请先正面回应题目中的核心概念。"
            if reminder not in improvements:
                improvements.insert(0, reminder)
            merged["improvements"] = improvements[:4]

    merged["relevance"] = {
        **rule_relevance,
        **llm_relevance,
        "level": level,
        "max_score": round(max_score, 1),
        "reason": reason,
    }


def _safe_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _parse_json_content(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _normalize_generated_question(generated: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    difficulty = str(generated.get("difficulty") or payload["requested_difficulty"] or "medium").strip().lower()
    if difficulty not in {"easy", "medium", "hard"}:
        difficulty = "medium"
    question_type = str(generated.get("question_type") or "technical_knowledge").strip()
    if question_type not in {"technical_knowledge", "scenario", "project_deep_dive", "behavioral"}:
        question_type = "technical_knowledge"

    return {
        "title": str(generated.get("title") or payload["requested_focus"] or payload["requested_topic"] or "LLM 生成题").strip(),
        "question_type": question_type,
        "topic": str(generated.get("topic") or payload["requested_topic"] or "llm_generated").strip(),
        "difficulty": difficulty,
        "keywords": _string_list(generated.get("keywords"), 6),
        "question": str(generated.get("question") or "").strip(),
        "standard_answer": str(generated.get("standard_answer") or "").strip(),
        "follow_ups": _string_list(generated.get("follow_ups"), 4),
        "score_points": _string_list(generated.get("score_points"), 6),
        "common_mistakes": _string_list(generated.get("common_mistakes"), 4),
    }


def _string_list(value: Any, limit: int) -> list[str]:
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
    elif isinstance(value, str) and value.strip():
        items = [part.strip() for part in re.split(r"[\n;；]+", value) if part.strip()]
    else:
        items = []
    return items[:limit]


def _default_max_score_delta(judge_mode: str) -> float:
    if judge_mode == "conservative":
        return 1.0
    if judge_mode == "full":
        return 10.0
    return 2.5


def _is_placeholder_secret(value: str) -> bool:
    lowered = value.strip().lower()
    return not lowered or lowered.startswith("replace-with-") or lowered in {"your-api-key", "你的真实密钥"}
