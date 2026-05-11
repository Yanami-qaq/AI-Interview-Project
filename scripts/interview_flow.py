import argparse
import json
import os
import random
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_judge import (
    LlmJudgeConfig,
    enhance_interview_result,
    generate_interview_question,
    load_llm_judge_config,
    public_llm_judge_status,
)


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_DB_DIR = ROOT_DIR / "db"
DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
DEFAULT_RUNTIME_DIR = ROOT_DIR / "runtime" / "sessions"
DEFAULT_REPORT_DIR = ROOT_DIR / "runtime" / "reports"
SELF_INTRO_QUESTION_ID = "SELF_INTRO"


ROLE_CONFIGS = {
    "java_backend": {
        "role_label": "Java后端开发工程师",
        "main_file": DEFAULT_DATA_DIR / "java_backend" / "Java后端主知识库-标准化面试题库.md",
        "collection": "java_interview_main",
    },
    "web_frontend": {
        "role_label": "Web前端开发工程师",
        "main_file": DEFAULT_DATA_DIR / "web_frontend" / "Web前端主知识库-标准化面试题库.md",
        "collection": "frontend_interview_main",
    },
}

SCORING_RUBRIC = {
    "technical_accuracy": {"label": "技术准确性", "weight": 0.30},
    "key_point_coverage": {"label": "要点覆盖度", "weight": 0.25},
    "logic_structure": {"label": "逻辑结构", "weight": 0.15},
    "practical_application": {"label": "项目场景结合", "weight": 0.15},
    "communication_clarity": {"label": "表达清晰度", "weight": 0.15},
}

LOGIC_MARKERS = ("首先", "其次", "然后", "最后", "一方面", "另一方面", "因为", "所以", "例如", "总结")
PRACTICAL_MARKERS = (
    "项目",
    "业务",
    "线上",
    "生产",
    "实际",
    "场景",
    "经验",
    "排查",
    "监控",
    "性能",
    "压测",
    "日志",
    "指标",
    "落地",
)
COMMUNICATION_MARKERS = ("核心", "关键", "本质", "区别", "优点", "缺点", "风险", "边界", "取舍", "结论")
STOP_UNITS = {
    "的",
    "了",
    "是",
    "和",
    "与",
    "及",
    "或",
    "在",
    "中",
    "对",
    "把",
    "被",
    "会",
    "可以",
    "需要",
    "能",
    "说明",
    "讲出",
    "提到",
    "通过",
    "一个",
    "进行",
}


@dataclass
class InterviewQuestion:
    question_id: str
    title: str
    job_role: str
    role_label: str
    question_type: str
    topic: str
    difficulty: str
    keywords: list[str]
    question: str
    standard_answer: str
    follow_ups: list[str]
    score_points: list[str]
    common_mistakes: list[str]
    source_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal text interview flow engine.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    shared_parent = argparse.ArgumentParser(add_help=False)
    shared_parent.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Knowledge base root directory.")
    shared_parent.add_argument("--db-dir", default=str(DEFAULT_DB_DIR), help="Chroma persistence directory.")
    shared_parent.add_argument(
        "--embedding-model",
        default=DEFAULT_MODEL,
        help="Embedding model name or local path.",
    )
    shared_parent.add_argument(
        "--local-model-only",
        action="store_true",
        help="Only load embedding model from local files.",
    )
    shared_parent.add_argument(
        "--runtime-dir",
        default=str(DEFAULT_RUNTIME_DIR),
        help="Directory for local interview session state.",
    )
    shared_parent.add_argument(
        "--llm-judge-enabled",
        action="store_true",
        default=None,
        help="Enable optional LLM judge calibration. Can also be enabled with LLM_JUDGE_ENABLED=1.",
    )
    shared_parent.add_argument(
        "--llm-judge-api-url",
        default="",
        help="OpenAI-compatible chat completions endpoint for judge calibration.",
    )
    shared_parent.add_argument(
        "--llm-judge-api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable that stores the judge API key.",
    )
    shared_parent.add_argument(
        "--llm-judge-model",
        default="",
        help="Model name used by the optional judge.",
    )
    shared_parent.add_argument(
        "--llm-judge-mode",
        default="",
        choices=["", "conservative", "balanced", "full"],
        help="LLM judge authority: conservative, balanced, or full. Default can be set by LLM_JUDGE_MODE.",
    )

    start_parser = subparsers.add_parser("start", parents=[shared_parent], help="Start a new interview session.")
    start_parser.add_argument("--role", required=True, choices=sorted(ROLE_CONFIGS.keys()), help="Interview role.")
    start_parser.add_argument("--topic", default="", help="Optional topic filter.")
    start_parser.add_argument("--difficulty", default="", help="Optional difficulty filter.")
    start_parser.add_argument("--question-query", default="", help="Optional seed query for selecting the first question.")

    answer_parser = subparsers.add_parser("answer", parents=[shared_parent], help="Submit an answer for current round.")
    answer_parser.add_argument("--session-id", required=True, help="Interview session id.")
    answer_parser.add_argument("--answer", required=True, help="Candidate answer text.")

    finish_parser = subparsers.add_parser("finish", parents=[shared_parent], help="Finish a session and print summary.")
    finish_parser.add_argument("--session-id", required=True, help="Interview session id.")

    report_parser = subparsers.add_parser("report", parents=[shared_parent], help="Generate a structured report.")
    report_parser.add_argument("--session-id", required=True, help="Interview session id.")
    report_parser.add_argument(
        "--report-dir",
        default=str(DEFAULT_REPORT_DIR),
        help="Directory for generated markdown reports.",
    )

    inspect_parser = subparsers.add_parser("inspect", parents=[shared_parent], help="Inspect current session state.")
    inspect_parser.add_argument("--session-id", required=True, help="Interview session id.")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_runtime_dir(runtime_dir: Path) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)


def sanitize_filename(text: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", text).strip("_") or "report"


def read_markdown_file(file_path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return file_path.read_text(encoding="utf-8", errors="ignore")


def normalize_topic_name(value: str) -> str:
    mapping = {
        "java基础": "java_basic",
        "集合": "collection",
        "并发": "concurrency",
        "jvm": "jvm",
        "mysql": "mysql",
        "redis": "redis",
        "接口设计": "backend_design",
        "系统设计": "backend_design",
        "项目": "project",
        "项目深挖": "project",
        "行为题": "behavioral",
        "前端基础": "frontend_basic",
        "浏览器": "browser",
        "浏览器原理": "browser",
        "react": "frontend_framework",
        "vue": "frontend_framework",
        "框架": "frontend_framework",
        "工程化": "frontend_engineering",
        "前端工程化": "frontend_engineering",
        "前端性能优化": "frontend_performance",
    }
    key = value.strip().lower()
    return mapping.get(key, key.replace(" ", "_"))


def normalize_question_type(value: str) -> str:
    mapping = {
        "技术知识": "technical_knowledge",
        "场景题": "scenario",
        "项目深挖": "project_deep_dive",
        "行为题": "behavioral",
    }
    return mapping.get(value.strip(), value.strip().lower().replace(" ", "_"))


def normalize_difficulty(value: str) -> str:
    mapping = {
        "初级": "easy",
        "中级": "medium",
        "中高级": "hard",
        "高级": "hard",
    }
    return mapping.get(value.strip(), value.strip().lower())


def extract_section(text: str, heading: str) -> str:
    pattern = rf"^###\s+{re.escape(heading)}\s*$"
    lines = text.splitlines()
    capture = False
    collected: list[str] = []
    for line in lines:
        stripped = line.strip()
        if re.match(pattern, stripped):
            capture = True
            continue
        if capture and stripped.startswith("### "):
            break
        if capture:
            collected.append(line)
    return "\n".join(collected).strip()


def parse_bullet_lines(section_text: str) -> list[str]:
    items: list[str] = []
    for raw_line in section_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("-"):
            continue
        item = line.lstrip("-").strip().strip("`")
        if item:
            items.append(item)
    return items


def parse_metadata(section_text: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for raw_line in section_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("-"):
            continue
        cleaned = line.lstrip("-").strip()
        if "：" in cleaned:
            key, value = cleaned.split("：", 1)
        elif ":" in cleaned:
            key, value = cleaned.split(":", 1)
        else:
            continue
        metadata[key.strip()] = value.strip()
    return metadata


def split_questions(markdown_text: str) -> list[str]:
    matches = list(re.finditer(r"^##\s+(Q\d+\s+.+)$", markdown_text, flags=re.MULTILINE))
    sections: list[str] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown_text)
        sections.append(markdown_text[start:end].strip())
    return sections


_question_cache: dict[str, tuple[float, list["InterviewQuestion"]]] = {}


def load_questions_for_role(role: str) -> list[InterviewQuestion]:
    config = ROLE_CONFIGS[role]
    file_path = Path(config["main_file"])
    if not file_path.exists():
        return []

    mtime = file_path.stat().st_mtime
    cached = _question_cache.get(role)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    text = read_markdown_file(file_path)
    questions: list[InterviewQuestion] = []
    for section in split_questions(text):
        title_match = re.search(r"^##\s+(Q\d+)\s+(.+)$", section, flags=re.MULTILINE)
        if not title_match:
            continue
        question_id = title_match.group(1).strip()
        title = title_match.group(2).strip()
        metadata = parse_metadata(extract_section(section, "元数据"))
        question_text = extract_section(section, "面试题")
        standard_answer = extract_section(section, "标准回答")
        follow_ups = parse_bullet_lines(extract_section(section, "追问点"))
        score_points = parse_bullet_lines(extract_section(section, "评分点"))
        common_mistakes = parse_bullet_lines(extract_section(section, "常见失分点"))
        raw_keywords = metadata.get("关键词", "")
        keywords = [part.strip("` ").strip() for part in re.split(r"[,\s，、]+", raw_keywords) if part.strip("` ").strip()]
        questions.append(
            InterviewQuestion(
                question_id=question_id,
                title=title,
                job_role=role,
                role_label=metadata.get("岗位", config["role_label"]),
                question_type=normalize_question_type(metadata.get("题型", "技术知识")),
                topic=normalize_topic_name(metadata.get("主题", "")),
                difficulty=normalize_difficulty(metadata.get("难度", "")),
                keywords=keywords,
                question=question_text,
                standard_answer=standard_answer,
                follow_ups=follow_ups,
                score_points=score_points,
                common_mistakes=common_mistakes,
                source_path=str(file_path),
            )
        )
    _question_cache[role] = (mtime, questions)
    return questions


def tokenize(text: str) -> list[str]:
    return [token for token in re.split(r"[\s,，。；：:、\-\(\)（）`]+", text.lower()) if token]


def text_units(text: str) -> set[str]:
    units = {token for token in tokenize(text) if token not in STOP_UNITS}
    normalized = text.lower()
    cjk_sequences = re.findall(r"[\u4e00-\u9fff]{2,}", normalized)
    for sequence in cjk_sequences:
        if sequence not in STOP_UNITS:
            units.add(sequence)
        for size in (2, 3):
            for index in range(0, max(0, len(sequence) - size + 1)):
                gram = sequence[index : index + size]
                if gram not in STOP_UNITS:
                    units.add(gram)
    return units


def clamp_score(value: float) -> float:
    return round(max(0.0, min(10.0, value)), 1)


def marker_count(text: str, markers: tuple[str, ...]) -> int:
    return sum(1 for marker in markers if marker in text)


def sentence_count(text: str) -> int:
    return len([part for part in re.split(r"[。！？!?；;\n]+", text) if part.strip()])


def find_evidence(answer: str, terms: list[str] | set[str]) -> str:
    normalized_terms = [term.lower() for term in terms if term]
    candidates = [part.strip() for part in re.split(r"[。！？!?；;\n]+", answer) if part.strip()]
    for candidate in candidates:
        lowered = candidate.lower()
        if any(term in lowered for term in normalized_terms):
            return candidate[:120]
    return candidates[0][:120] if candidates else ""


def evaluate_score_points(question: InterviewQuestion, answer: str) -> list[dict[str, Any]]:
    answer_units = text_units(answer)
    normalized_answer = answer.lower()
    checks: list[dict[str, Any]] = []
    for point in question.score_points:
        point_units = text_units(point)
        overlap = point_units & answer_units
        denominator = max(1, min(len(point_units), 14))
        ratio = len(overlap) / denominator
        direct_hit = point.lower() in normalized_answer
        keyword_overlap = {keyword for keyword in question.keywords if keyword.lower() in normalized_answer}
        matched = direct_hit or ratio >= 0.35 or (len(overlap) >= 3 and bool(keyword_overlap))
        checks.append(
            {
                "point": point,
                "matched": matched,
                "overlap_ratio": round(min(1.0, ratio), 2),
                "matched_terms": sorted(overlap, key=len, reverse=True)[:6],
                "evidence": find_evidence(answer, overlap) if matched else "",
                "reason": "已覆盖该评分点" if matched else "回答中未明确覆盖该评分点",
            }
        )
    return checks


def evaluate_answer_relevance(question: InterviewQuestion, answer: str, point_checks: list[dict[str, Any]]) -> dict[str, Any]:
    normalized_answer = answer.strip().lower()
    answer_units = text_units(answer)
    question_units = text_units(question.question)
    standard_units = text_units(question.standard_answer)
    keyword_hits = [keyword for keyword in question.keywords if keyword.lower() in normalized_answer]
    matched_points = [item for item in point_checks if item["matched"]]

    question_overlap = question_units & answer_units
    standard_overlap = standard_units & answer_units
    concept_hits = len(keyword_hits) + len(matched_points)
    overlap_ratio = len((question_units | standard_units) & answer_units) / max(1, min(len(question_units | standard_units), 28))

    if not answer.strip():
        level = "empty"
        max_score = 0.0
        reason = "未作答。"
    elif concept_hits == 0 and len(question_overlap) < 2 and len(standard_overlap) < 2:
        level = "off_topic"
        max_score = 1.5
        reason = "回答与当前问题及标准要点基本无关。"
    elif concept_hits == 0 and overlap_ratio < 0.12:
        level = "weak"
        max_score = 3.0
        reason = "回答只有少量题面词重合，缺少关键技术概念。"
    elif not matched_points and len(keyword_hits) <= 1:
        level = "partial"
        max_score = 5.0
        reason = "回答部分相关，但没有明确覆盖核心评分点。"
    else:
        level = "relevant"
        max_score = 10.0
        reason = "回答与当前问题相关。"

    return {
        "level": level,
        "max_score": max_score,
        "reason": reason,
        "keyword_hits": keyword_hits,
        "question_overlap": sorted(question_overlap, key=len, reverse=True)[:8],
        "standard_overlap": sorted(standard_overlap, key=len, reverse=True)[:8],
        "overlap_ratio": round(min(1.0, overlap_ratio), 2),
    }


def confidence_level(answer: str, point_checks: list[dict[str, Any]]) -> str:
    if len(answer.strip()) < 40:
        return "low"
    if not point_checks:
        return "medium"
    matched_count = sum(1 for item in point_checks if item["matched"])
    if matched_count == 0:
        return "low"
    if matched_count / len(point_checks) >= 0.6 and len(answer.strip()) >= 120:
        return "high"
    return "medium"


def choose_question_from_local(
    questions: list[InterviewQuestion],
    topic: str,
    difficulty: str,
    question_query: str,
) -> InterviewQuestion:
    filtered = questions
    normalized_topic = normalize_topic_name(topic) if topic else ""
    normalized_difficulty = normalize_difficulty(difficulty) if difficulty else ""

    if normalized_topic:
        topic_filtered = [question for question in filtered if question.topic == normalized_topic]
        if topic_filtered:
            filtered = topic_filtered
    if normalized_difficulty:
        difficulty_filtered = [question for question in filtered if question.difficulty == normalized_difficulty]
        if difficulty_filtered:
            filtered = difficulty_filtered

    candidates = filtered or questions
    if not question_query:
        return random.choice(candidates)

    query_tokens = set(tokenize(question_query))
    best_score = -1
    best_questions: list[InterviewQuestion] = []
    for question in candidates:
        corpus = " ".join(
            [question.title, question.question, question.standard_answer, " ".join(question.keywords), question.topic]
        )
        score = sum(1 for token in query_tokens if token and token in corpus.lower())
        if score > best_score:
            best_score = score
            best_questions = [question]
        elif score == best_score:
            best_questions.append(question)
    return random.choice(best_questions or candidates)


def topic_exists(questions: list[InterviewQuestion], topic: str) -> bool:
    return not topic or any(question.topic == normalize_topic_name(topic) for question in questions)


def keyword_matches_question(question: InterviewQuestion, question_query: str) -> bool:
    if not question_query.strip():
        return True
    query_tokens = set(tokenize(question_query))
    corpus = " ".join(
        [question.title, question.question, question.standard_answer, " ".join(question.keywords), question.topic]
    ).lower()
    return any(token and token in corpus for token in query_tokens)


def keyword_exists(questions: list[InterviewQuestion], question_query: str) -> bool:
    return not question_query.strip() or any(keyword_matches_question(question, question_query) for question in questions)


def build_llm_question(
    role: str,
    topic: str,
    difficulty: str,
    question_query: str,
    llm_judge_config: LlmJudgeConfig | None,
) -> InterviewQuestion:
    config = llm_judge_config or load_llm_judge_config()
    generated = generate_interview_question(
        role=role,
        role_label=ROLE_CONFIGS[role]["role_label"],
        topic=topic,
        difficulty=difficulty,
        focus=question_query,
        config=config,
    )
    if not generated["question"] or not generated["standard_answer"] or not generated["score_points"]:
        raise RuntimeError("LLM fallback question is incomplete.")
    return InterviewQuestion(
        question_id=f"LLM-{uuid.uuid4().hex[:8]}",
        title=generated["title"],
        job_role=role,
        role_label=ROLE_CONFIGS[role]["role_label"],
        question_type=generated["question_type"],
        topic=normalize_topic_name(generated["topic"]),
        difficulty=normalize_difficulty(generated["difficulty"]),
        keywords=generated["keywords"],
        question=generated["question"],
        standard_answer=generated["standard_answer"],
        follow_ups=generated["follow_ups"],
        score_points=generated["score_points"],
        common_mistakes=generated["common_mistakes"],
        source_path="llm_generated",
    )


def build_resume_context_question(
    role: str,
    session: dict[str, Any],
    self_intro: str,
    local_questions: list[InterviewQuestion],
    llm_judge_config: LlmJudgeConfig | None,
) -> InterviewQuestion:
    profile = session.get("profile", {})
    resume_text = profile.get("resume_text", "")
    requested_topic = profile.get("requested_topic", "")
    requested_difficulty = profile.get("requested_difficulty", "")
    requested_focus = profile.get("requested_focus", "")
    config = llm_judge_config or load_llm_judge_config()
    if config.enabled and (resume_text.strip() or self_intro.strip()):
        focus = "\n".join(
            part
            for part in [
                f"候选人简历：{resume_text[:1600]}",
                f"候选人自我介绍：{self_intro[:800]}",
                f"候选人希望考察的考点：{requested_focus}",
                "请生成一题贴合候选人经历和应聘岗位的正式面试问题。",
            ]
            if part.strip()
        )
        try:
            return build_llm_question(
                role=role,
                topic=requested_topic or "resume_project",
                difficulty=requested_difficulty or "medium",
                question_query=focus,
                llm_judge_config=config,
            )
        except Exception:
            pass

    return choose_question_from_local(
        questions=local_questions,
        topic=requested_topic,
        difficulty=requested_difficulty,
        question_query=requested_focus,
    )


def choose_next_question_for_session(
    questions: list[InterviewQuestion],
    session: dict[str, Any],
) -> InterviewQuestion | None:
    asked_ids = {session.get("question_id")}
    asked_ids.update(item.get("question_id") for item in session.get("history", []) if item.get("question_id"))
    candidates = [question for question in questions if question.question_id not in asked_ids]
    if not candidates:
        return None

    current_topic = normalize_topic_name(session.get("topic", ""))
    current_difficulty = normalize_difficulty(session.get("difficulty", ""))
    same_context = [
        question
        for question in candidates
        if (not current_topic or question.topic == current_topic)
        and (not current_difficulty or question.difficulty == current_difficulty)
    ]
    return random.choice(same_context or candidates)


def choose_question_via_chroma(
    role: str,
    db_dir: Path,
    embedding_model: str,
    local_model_only: bool,
    topic: str,
    difficulty: str,
    question_query: str,
) -> dict[str, Any] | None:
    config = ROLE_CONFIGS[role]
    query_text = question_query or f"{config['role_label']} 面试题"
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
    except Exception:
        return None

    model_path = Path(embedding_model)
    if not model_path.exists() and not local_model_only:
        return None
    model_kwargs = {"local_files_only": local_model_only or model_path.exists()}
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=str(model_path if model_path.exists() else embedding_model),
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": True},
        )
        search_filter: dict[str, Any] = {"job_role": role}
        if topic:
            search_filter["topic"] = normalize_topic_name(topic)
        if difficulty:
            search_filter["difficulty"] = normalize_difficulty(difficulty)
        db = Chroma(
            persist_directory=str(db_dir.resolve()),
            embedding_function=embeddings,
            collection_name=config["collection"],
        )
        results = db.similarity_search(query_text, k=1, filter=search_filter)
        if not results:
            return None
        result = results[0]
        return {
            "question_id": result.metadata.get("question_id", ""),
            "question": result.metadata.get("question") or result.metadata.get("title") or "",
            "topic": result.metadata.get("topic", ""),
            "difficulty": result.metadata.get("difficulty", ""),
            "question_type": result.metadata.get("question_type", ""),
            "source": result.metadata.get("source", ""),
            "page_content": result.page_content,
            "metadata": result.metadata,
        }
    except Exception:
        return None


def get_session_file(runtime_dir: Path, session_id: str) -> Path:
    return runtime_dir / f"{session_id}.json"


def save_session(runtime_dir: Path, session_id: str, payload: dict[str, Any]) -> Path:
    ensure_runtime_dir(runtime_dir)
    session_path = get_session_file(runtime_dir, session_id)
    session_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return session_path


def load_session(runtime_dir: Path, session_id: str) -> dict[str, Any]:
    session_path = get_session_file(runtime_dir, session_id)
    if not session_path.exists():
        raise FileNotFoundError(f"Session not found: {session_path}")
    return json.loads(session_path.read_text(encoding="utf-8"))


def serialize_question(question: InterviewQuestion) -> dict[str, Any]:
    return {
        "question_id": question.question_id,
        "title": question.title,
        "job_role": question.job_role,
        "role_label": question.role_label,
        "question_type": question.question_type,
        "topic": question.topic,
        "difficulty": question.difficulty,
        "keywords": question.keywords,
        "question": question.question,
        "standard_answer": question.standard_answer,
        "follow_ups": question.follow_ups,
        "score_points": question.score_points,
        "common_mistakes": question.common_mistakes,
        "source_path": question.source_path,
    }


def score_answer(question: InterviewQuestion, answer: str) -> dict[str, Any]:
    normalized_answer = answer.strip()
    keyword_hits = [keyword for keyword in question.keywords if keyword.lower() in normalized_answer.lower()]
    point_checks = evaluate_score_points(question, normalized_answer)
    relevance = evaluate_answer_relevance(question, normalized_answer, point_checks)
    score_point_hits = [item["point"] for item in point_checks if item["matched"]]
    missing_points = [item["point"] for item in point_checks if not item["matched"]][:3]

    total_points = max(1, len(question.score_points))
    point_coverage_ratio = len(score_point_hits) / total_points
    keyword_ratio = len(keyword_hits) / max(1, len(question.keywords))
    answer_length = len(normalized_answer)
    sentences = sentence_count(normalized_answer)
    logic_hits = marker_count(normalized_answer, LOGIC_MARKERS)
    practical_hits = marker_count(normalized_answer, PRACTICAL_MARKERS)
    communication_hits = marker_count(normalized_answer, COMMUNICATION_MARKERS)

    technical_score = clamp_score(2.0 + point_coverage_ratio * 5.7 + min(keyword_ratio, 1.0) * 2.3)
    key_point_coverage_score = clamp_score(1.5 + point_coverage_ratio * 8.5)
    logic_structure_score = clamp_score(
        3.0 + min(logic_hits, 4) * 1.1 + min(sentences, 4) * 0.55 + (1.0 if answer_length >= 120 else 0)
    )
    practical_application_score = clamp_score(
        3.0 + min(practical_hits, 4) * 1.25 + (1.0 if answer_length >= 100 else 0)
    )
    communication_clarity_score = clamp_score(
        3.0
        + min(sentences, 5) * 0.55
        + min(communication_hits, 4) * 0.75
        + (1.0 if 80 <= answer_length <= 900 else 0)
    )

    dimensions = {
        "technical_accuracy": technical_score,
        "key_point_coverage": key_point_coverage_score,
        "logic_structure": logic_structure_score,
        "practical_application": practical_application_score,
        "communication_clarity": communication_clarity_score,
    }
    summary_score = clamp_score(
        sum(dimensions[key] * config["weight"] for key, config in SCORING_RUBRIC.items())
    )
    if relevance["max_score"] < 10:
        dimensions = {key: min(value, relevance["max_score"]) for key, value in dimensions.items()}
        summary_score = min(summary_score, relevance["max_score"])
    strengths: list[str] = []
    if relevance["level"] in {"off_topic", "empty"}:
        strengths.append("当前回答未能回应题目核心。")
    elif keyword_hits:
        strengths.append(f"覆盖了关键词：{', '.join(keyword_hits[:4])}")
    if score_point_hits:
        strengths.append(f"命中了核心评分点：{'; '.join(score_point_hits[:2])}")
    if practical_hits:
        strengths.append("回答中包含一定的项目或场景意识。")
    if not strengths:
        strengths.append("回答与题目主题基本相关。")

    improvements: list[str] = []
    if relevance["level"] in {"off_topic", "empty", "weak"}:
        improvements.append("请先回到当前题目本身，围绕题目中的核心概念作答。")
        improvements.append(f"本题需要重点回答：{question.question}")
    if missing_points:
        improvements.append(f"建议补充这些评分点：{'; '.join(missing_points)}")
    if len(normalized_answer) < 80:
        improvements.append("回答偏短，可以补充关键原因、适用场景和结论。")
    if logic_structure_score < 6:
        improvements.append("表达结构还不够清晰，可以按结论、原因、场景、风险的顺序组织。")
    if practical_application_score < 6:
        improvements.append("可以加入项目经验、线上问题或工程取舍，让回答更贴近岗位。")

    return {
        "score": summary_score,
        "dimensions": dimensions,
        "rubric": SCORING_RUBRIC,
        "keyword_hits": keyword_hits,
        "score_point_hits": score_point_hits,
        "point_checks": point_checks,
        "evidence": [item["evidence"] for item in point_checks if item.get("evidence")][:3],
        "confidence": confidence_level(normalized_answer, point_checks),
        "relevance": relevance,
        "strengths": strengths,
        "improvements": improvements or ["可以进一步补充细节和工程案例，使回答更完整。"],
    }


def choose_follow_up(question: InterviewQuestion, asked_follow_ups: list[str], answer: str) -> str:
    for follow_up in question.follow_ups:
        if follow_up not in asked_follow_ups:
            return follow_up
    answer_lower = answer.lower()
    mentioned = [kw for kw in question.keywords if kw.lower() in answer_lower]
    if mentioned:
        return f"你提到了{mentioned[0]}，能具体说说它在实际项目中会遇到哪些问题，你是怎么处理的吗？"
    not_mentioned = [kw for kw in question.keywords if kw.lower() not in answer_lower]
    if not_mentioned:
        return f"这道题还有一个核心考点你还没提到——{not_mentioned[0]}，能展开讲讲吗？"
    return "能结合你实际做过的项目，说说这块你遇到过什么问题、怎么解决的吗？"


_SWITCH_TRANSITION_TEMPLATES = [
    "好，这道题先聊到这里。我换一个方向问你：{question}",
    "明白了，我们换一题。{question}",
    "这块先到这里，下面我想问你另一个问题：{question}",
    "好的，接下来我们换一个考点：{question}",
]


def _pick_switch_transition(next_question: "InterviewQuestion") -> str:
    template = random.choice(_SWITCH_TRANSITION_TEMPLATES)
    return template.format(question=next_question.question)


def _build_self_intro_transition(self_intro: str, next_question: "InterviewQuestion") -> str:
    intro_lower = self_intro.lower()
    project_keywords = ["项目", "负责", "开发", "实习", "参与", "做过"]
    has_project = any(kw in intro_lower for kw in project_keywords)
    if has_project:
        return f"好，背景了解了，看起来你有一些实际项目经验。那我们进入技术问题——{next_question.question}"
    return f"好，了解了你的基本情况。接下来我问你一个技术问题：{next_question.question}"


def make_interviewer_decision(
    question: InterviewQuestion,
    session: dict[str, Any],
    evaluation: dict[str, Any],
    follow_up: str,
    next_question: InterviewQuestion | None = None,
) -> dict[str, Any]:
    history_count = len(session.get("history", []))
    answered_question_count = len({item.get("question_id") for item in session.get("history", []) if item.get("question_id")})
    asked_count = len(session.get("current_follow_ups", [])) + 1
    matched_count = len(evaluation.get("score_point_hits", []))
    total_points = max(1, len(question.score_points))
    coverage_ratio = matched_count / total_points
    score = float(evaluation.get("score", 0))
    confidence = evaluation.get("confidence", "medium")

    min_rounds = 2
    max_rounds = 6
    max_questions = int(session.get("max_questions", 3))
    enough_quality = score >= 7.2 and coverage_ratio >= 0.65 and confidence != "low"
    exhausted_followups = asked_count >= max(2, min(max_rounds, len(question.follow_ups) or max_rounds))
    current_question_done = enough_quality or exhausted_followups
    should_switch = current_question_done and next_question is not None and answered_question_count < max_questions
    should_finish = history_count >= min_rounds and (
        enough_quality or history_count >= max_rounds or exhausted_followups
    )

    if should_switch:
        transition = _pick_switch_transition(next_question)
        return {
            "action": "switch_question",
            "message": transition,
            "should_finish": False,
            "reason": "当前题追问已完成，进入下一题。",
            "next_question": serialize_question(next_question),
        }

    if should_finish:
        if enough_quality:
            reason = "当前题核心要点覆盖较充分，可以进入总结。"
            finish_message = "这道题你回答得比较完整，我们就到这里。稍等，我来整理一下这轮面试的总体情况。"
        elif history_count >= max_rounds:
            reason = "本轮面试追问已达到上限，可以进入总结。"
            finish_message = "好，这轮我们聊了不少，我先整理一下反馈给你。"
        else:
            reason = "当前题追问已完成，可以进入总结。"
            finish_message = "这道题先到这里。接下来我给你总结一下本轮的整体表现。"
        return {
            "action": "finish",
            "message": finish_message,
            "should_finish": True,
            "reason": reason,
        }

    return {
        "action": "continue",
        "message": follow_up,
        "should_finish": False,
        "reason": "当前回答仍有可继续追问的空间。",
    }


def start_interview(
    role: str,
    topic: str = "",
    difficulty: str = "",
    question_query: str = "",
    resume_text: str = "",
    db_dir: Path = DEFAULT_DB_DIR,
    embedding_model: str = DEFAULT_MODEL,
    local_model_only: bool = False,
    runtime_dir: Path = DEFAULT_RUNTIME_DIR,
    llm_judge_config: LlmJudgeConfig | None = None,
) -> dict[str, Any]:
    runtime_dir = Path(runtime_dir).resolve()
    local_questions = load_questions_for_role(role)
    if not local_questions:
        raise FileNotFoundError(f"Knowledge base not found for role={role}")

    normalized_topic = normalize_topic_name(topic) if topic else ""
    topic_hit = topic_exists(local_questions, topic)
    keyword_hit = keyword_exists(local_questions, question_query)
    should_generate_question = False

    chroma_result = None if should_generate_question else choose_question_via_chroma(
        role=role,
        db_dir=Path(db_dir),
        embedding_model=embedding_model,
        local_model_only=local_model_only,
        topic=topic,
        difficulty=difficulty,
        question_query=question_query,
    )

    question_source = "markdown_fallback"
    fallback_reason = ""
    if should_generate_question:
        local_question = build_llm_question(role, normalized_topic or topic, difficulty, question_query, llm_judge_config)
        question_source = "llm_generated_question"
        fallback_reason = "用户选择的主题和考点在题库中均未命中，已由 LLM 按题库结构生成临时题。"
    else:
        local_question = choose_question_from_local(
            questions=local_questions,
            topic=topic,
            difficulty=difficulty,
            question_query=question_query,
        )
        question_source = "chroma" if chroma_result else "markdown_fallback"
    session_id = uuid.uuid4().hex[:12]
    self_intro_question = build_self_intro_question(role, resume_text)
    opening_message = build_opening_message_for_profile(role, resume_text)
    pending_question = serialize_question(local_question)
    session = {
        "session_id": session_id,
        "role": role,
        "role_label": ROLE_CONFIGS[role]["role_label"],
        "created_at": now_iso(),
        "status": "active",
        "interview_stage": "self_intro",
        "question_id": SELF_INTRO_QUESTION_ID,
        "topic": "profile",
        "difficulty": "",
        "question_type": "self_intro",
        "current_question": self_intro_question["question"],
        "opening_message": opening_message,
        "pending_question": pending_question,
        "profile": {
            "resume_text": resume_text.strip(),
            "requested_topic": topic,
            "requested_difficulty": difficulty,
            "requested_focus": question_query,
        },
        "current_follow_ups": [],
        "history": [],
        "retrieval": {
            "source": question_source,
            "chroma_question": chroma_result["question"] if chroma_result else "",
            "chroma_metadata": chroma_result["metadata"] if chroma_result else {},
            "topic_hit": topic_hit,
            "keyword_hit": keyword_hit,
            "fallback_reason": fallback_reason,
        },
    }
    session_path = save_session(runtime_dir, session_id, session)
    return {
        "session_id": session_id,
        "session_file": str(session_path),
        "role": session["role"],
        "role_label": session["role_label"],
        "retrieval_source": session["retrieval"]["source"],
        "fallback_reason": fallback_reason,
        "opening_message": opening_message,
        "question": self_intro_question,
        "guidance": "请先做一段简短自我介绍，重点说明你的背景、项目经历和应聘方向。",
        "session": session,
    }


def build_self_intro_question(role: str, resume_text: str = "") -> dict[str, Any]:
    role_label = ROLE_CONFIGS[role]["role_label"]
    resume_hint = "我已经收到了你的简历。" if resume_text.strip() else "如果没有填写简历，也可以在自我介绍里补充你的项目和技术背景。"
    return {
        "question_id": SELF_INTRO_QUESTION_ID,
        "title": "自我介绍",
        "job_role": role,
        "role_label": role_label,
        "question_type": "self_intro",
        "topic": "profile",
        "difficulty": "",
        "keywords": ["自我介绍", "项目经历", "应聘岗位"],
        "question": f"{resume_hint}请你先做一个 1 分钟左右的自我介绍，重点围绕你和 {role_label} 岗位相关的经历展开。",
        "standard_answer": "",
        "follow_ups": [],
        "score_points": [],
        "common_mistakes": [],
        "source_path": "interview_flow",
    }


def build_opening_message_for_profile(role: str, resume_text: str = "") -> str:
    role_label = ROLE_CONFIGS[role]["role_label"]
    resume_text = resume_text.strip()
    if resume_text:
        return (
            f"你好，我是本轮 {role_label} 模拟面试官。我会先结合你提交的简历了解你的背景，"
            "然后围绕岗位要求、项目经历和技术基础逐步提问。我们先从自我介绍开始。"
        )
    return (
        f"你好，我是本轮 {role_label} 模拟面试官。你还没有提交详细简历，"
        "所以我会先通过自我介绍了解你的背景，再根据岗位要求继续提问。"
    )


def build_opening_message(question: InterviewQuestion, has_filters: bool, fallback_reason: str = "") -> str:
    topic_text = question.topic.replace("_", " ")
    difficulty_text = {"easy": "偏基础", "medium": "中等难度", "hard": "偏深入"}.get(question.difficulty, "适中难度")
    if fallback_reason:
        source_text = "你选择的方向题库里没有完全匹配的题，我会先用一题临时生成题来考察这个方向。"
    elif has_filters:
        source_text = "我会根据你选择的方向开始提问。"
    else:
        source_text = "你没有限定专题，我会随机从题库中抽取问题，模拟真实面试中的开放出题。"
    return (
        f"你好，我是本轮 {question.role_label} 模拟面试官。{source_text}"
        f"我们会先从 {topic_text} 方向的一道{difficulty_text}问题开始。"
        "你可以像真实面试一样先给结论，再展开原因、场景和取舍。"
    )


def start_session(args: argparse.Namespace) -> None:
    result = start_interview(
        role=args.role,
        topic=args.topic,
        difficulty=args.difficulty,
        question_query=args.question_query,
        db_dir=Path(args.db_dir),
        embedding_model=args.embedding_model,
        local_model_only=args.local_model_only,
        runtime_dir=Path(args.runtime_dir),
        llm_judge_config=load_llm_judge_config(
            enabled=args.llm_judge_enabled,
            api_url=args.llm_judge_api_url,
            api_key_env=args.llm_judge_api_key_env,
            model=args.llm_judge_model,
            judge_mode=args.llm_judge_mode,
        ),
    )

    print(f"session_id={result['session_id']}")
    print(f"session_file={result['session_file']}")
    print(f"role={result['role_label']}")
    print(f"retrieval_source={result['retrieval_source']}")
    print(f"question_id={result['question']['question_id']}")
    print(
        f"topic={result['question']['topic']} | difficulty={result['question']['difficulty']} | "
        f"type={result['question']['question_type']}"
    )
    print("\n=== 当前面试题 ===")
    print(result["question"]["question"])
    print("\n=== 回答建议 ===")
    print(result["guidance"])


def answer_interview(
    session_id: str,
    answer: str,
    runtime_dir: Path = DEFAULT_RUNTIME_DIR,
    answer_source: str = "text",
    speech_result: dict[str, Any] | None = None,
    expression_analysis: dict[str, Any] | None = None,
    llm_judge_config: LlmJudgeConfig | None = None,
) -> dict[str, Any]:
    runtime_dir = Path(runtime_dir).resolve()
    session = load_session(runtime_dir, session_id)
    role = session["role"]
    questions = {item.question_id: item for item in load_questions_for_role(role)}
    if session.get("interview_stage") == "self_intro":
        local_questions = list(questions.values())
        next_question = build_resume_context_question(
            role=role,
            session=session,
            self_intro=answer,
            local_questions=local_questions,
            llm_judge_config=llm_judge_config,
        )
        next_question_payload = serialize_question(next_question)
        decision = {
            "action": "switch_question",
            "message": _build_self_intro_transition(answer, next_question),
            "should_finish": False,
            "reason": "自我介绍已完成，进入岗位与简历相关的正式面试问题。",
            "next_question": next_question_payload,
            "judge_source": "profile_stage",
        }
        evaluation = {
            "score": None,
            "dimensions": {},
            "rubric": SCORING_RUBRIC,
            "keyword_hits": [],
            "score_point_hits": [],
            "point_checks": [],
            "evidence": [],
            "confidence": "medium",
            "strengths": ["已完成自我介绍，作为后续追问的背景材料。"],
            "improvements": [],
            "stage": "self_intro",
            "scored": False,
        }
        history_item = {
            "timestamp": now_iso(),
            "question_id": SELF_INTRO_QUESTION_ID,
            "question": session.get("current_question", ""),
            "answer": answer,
            "answer_source": answer_source,
            "evaluation": evaluation,
            "follow_up": decision["message"],
            "interviewer_decision": decision,
            "judge_meta": {"enabled": bool((llm_judge_config or load_llm_judge_config()).enabled), "status": "profile_stage"},
            "scored": False,
        }
        if speech_result:
            history_item["speech_result"] = speech_result
        if expression_analysis:
            history_item["expression_analysis"] = expression_analysis
        session["history"].append(history_item)
        session["interview_stage"] = "technical"
        session["question_id"] = next_question.question_id
        session["topic"] = next_question.topic
        session["difficulty"] = next_question.difficulty
        session["question_type"] = next_question.question_type
        session["current_question"] = next_question.question
        session["pending_question"] = next_question_payload
        session["current_follow_ups"] = []
        session["updated_at"] = now_iso()
        save_session(runtime_dir, session_id, session)
        return {
            "session_id": session_id,
            "question": next_question_payload,
            "evaluation": evaluation,
            "follow_up": decision["message"],
            "interviewer_decision": decision,
            "judge_meta": history_item["judge_meta"],
            "session": session,
        }

    question = questions.get(session["question_id"])
    if not question:
        pending = session.get("pending_question")
        if isinstance(pending, dict) and pending.get("question_id") == session["question_id"]:
            question = InterviewQuestion(
                question_id=pending["question_id"],
                title=pending.get("title", ""),
                job_role=pending.get("job_role", role),
                role_label=pending.get("role_label", session.get("role_label", "")),
                question_type=pending.get("question_type", "technical_knowledge"),
                topic=pending.get("topic", ""),
                difficulty=pending.get("difficulty", ""),
                keywords=pending.get("keywords", []),
                question=pending.get("question", ""),
                standard_answer=pending.get("standard_answer", ""),
                follow_ups=pending.get("follow_ups", []),
                score_points=pending.get("score_points", []),
                common_mistakes=pending.get("common_mistakes", []),
                source_path=pending.get("source_path", ""),
            )
        else:
            raise KeyError(f"Question not found for session: {session['question_id']}")

    evaluation = score_answer(question, answer)
    follow_up = choose_follow_up(question, session.get("current_follow_ups", []), answer)
    session["current_follow_ups"] = session.get("current_follow_ups", []) + [follow_up]
    next_question = choose_next_question_for_session(list(questions.values()), session)
    decision = make_interviewer_decision(question, session, evaluation, follow_up, next_question=next_question)
    judge_result = enhance_interview_result(
        question=serialize_question(question),
        session=session,
        answer=answer,
        rule_evaluation=evaluation,
        rule_decision=decision,
        follow_up=follow_up,
        next_question=serialize_question(next_question) if next_question else None,
        config=llm_judge_config or load_llm_judge_config(),
    )
    evaluation = judge_result["evaluation"]
    decision = judge_result["decision"]
    history_item = {
        "timestamp": now_iso(),
        "question_id": question.question_id,
        "question": question.question,
        "answer": answer,
        "answer_source": answer_source,
        "evaluation": evaluation,
        "follow_up": decision["message"],
        "interviewer_decision": decision,
        "judge_meta": judge_result["meta"],
        "scored": True,
    }
    if speech_result:
        history_item["speech_result"] = speech_result
    if expression_analysis:
        history_item["expression_analysis"] = expression_analysis
    session["history"].append(history_item)
    if decision["action"] == "switch_question" and next_question:
        session["question_id"] = next_question.question_id
        session["topic"] = next_question.topic
        session["difficulty"] = next_question.difficulty
        session["question_type"] = next_question.question_type
        session["current_question"] = next_question.question
        session["pending_question"] = serialize_question(next_question)
        session["current_follow_ups"] = []
    session["updated_at"] = now_iso()
    save_session(runtime_dir, session_id, session)
    return {
        "session_id": session_id,
        "question": serialize_question(next_question if decision["action"] == "switch_question" and next_question else question),
        "evaluation": evaluation,
        "follow_up": decision["message"],
        "interviewer_decision": decision,
        "judge_meta": judge_result["meta"],
        "session": session,
    }


def answer_interview_stream(
    session_id: str,
    answer: str,
    runtime_dir: Path = DEFAULT_RUNTIME_DIR,
    answer_source: str = "text",
    speech_result: dict[str, Any] | None = None,
    expression_analysis: dict[str, Any] | None = None,
    llm_judge_config: LlmJudgeConfig | None = None,
):
    """Generator that yields {"type":"preliminary",...} immediately (rule-based),
    then {"type":"final",...} after LLM enhancement completes."""
    runtime_dir = Path(runtime_dir).resolve()
    session = load_session(runtime_dir, session_id)
    role = session["role"]
    questions = {item.question_id: item for item in load_questions_for_role(role)}

    if session.get("interview_stage") == "self_intro":
        result = answer_interview(
            session_id=session_id,
            answer=answer,
            runtime_dir=runtime_dir,
            answer_source=answer_source,
            speech_result=speech_result,
            expression_analysis=expression_analysis,
            llm_judge_config=llm_judge_config,
        )
        yield {**result, "type": "final"}
        return

    question = questions.get(session["question_id"])
    if not question:
        pending = session.get("pending_question")
        if isinstance(pending, dict) and pending.get("question_id") == session["question_id"]:
            question = InterviewQuestion(
                question_id=pending["question_id"],
                title=pending.get("title", ""),
                job_role=pending.get("job_role", role),
                role_label=pending.get("role_label", session.get("role_label", "")),
                question_type=pending.get("question_type", "technical_knowledge"),
                topic=pending.get("topic", ""),
                difficulty=pending.get("difficulty", ""),
                keywords=pending.get("keywords", []),
                question=pending.get("question", ""),
                standard_answer=pending.get("standard_answer", ""),
                follow_ups=pending.get("follow_ups", []),
                score_points=pending.get("score_points", []),
                common_mistakes=pending.get("common_mistakes", []),
                source_path=pending.get("source_path", ""),
            )
        else:
            raise KeyError(f"Question not found for session: {session['question_id']}")

    evaluation = score_answer(question, answer)
    follow_up = choose_follow_up(question, session.get("current_follow_ups", []), answer)
    session["current_follow_ups"] = session.get("current_follow_ups", []) + [follow_up]
    next_question = choose_next_question_for_session(list(questions.values()), session)
    decision = make_interviewer_decision(question, session, evaluation, follow_up, next_question=next_question)

    preliminary_q = serialize_question(
        next_question if decision["action"] == "switch_question" and next_question else question
    )
    yield {
        "type": "preliminary",
        "session_id": session_id,
        "question": preliminary_q,
        "evaluation": evaluation,
        "follow_up": decision["message"],
        "interviewer_decision": decision,
        "judge_meta": {"enabled": bool((llm_judge_config or load_llm_judge_config()).enabled), "status": "pending"},
    }

    judge_result = enhance_interview_result(
        question=serialize_question(question),
        session=session,
        answer=answer,
        rule_evaluation=evaluation,
        rule_decision=decision,
        follow_up=follow_up,
        next_question=serialize_question(next_question) if next_question else None,
        config=llm_judge_config or load_llm_judge_config(),
    )
    evaluation = judge_result["evaluation"]
    decision = judge_result["decision"]

    history_item = {
        "timestamp": now_iso(),
        "question_id": question.question_id,
        "question": question.question,
        "answer": answer,
        "answer_source": answer_source,
        "evaluation": evaluation,
        "follow_up": decision["message"],
        "interviewer_decision": decision,
        "judge_meta": judge_result["meta"],
        "scored": True,
    }
    if speech_result:
        history_item["speech_result"] = speech_result
    if expression_analysis:
        history_item["expression_analysis"] = expression_analysis
    session["history"].append(history_item)
    if decision["action"] == "switch_question" and next_question:
        session["question_id"] = next_question.question_id
        session["topic"] = next_question.topic
        session["difficulty"] = next_question.difficulty
        session["question_type"] = next_question.question_type
        session["current_question"] = next_question.question
        session["pending_question"] = serialize_question(next_question)
        session["current_follow_ups"] = []
    session["updated_at"] = now_iso()
    save_session(runtime_dir, session_id, session)

    final_q = serialize_question(
        next_question if decision["action"] == "switch_question" and next_question else question
    )
    yield {
        "type": "final",
        "session_id": session_id,
        "question": final_q,
        "evaluation": evaluation,
        "follow_up": decision["message"],
        "interviewer_decision": decision,
        "judge_meta": judge_result["meta"],
        "session": session,
    }


def answer_session(args: argparse.Namespace) -> None:
    result = answer_interview(
        session_id=args.session_id,
        answer=args.answer,
        runtime_dir=Path(args.runtime_dir),
        llm_judge_config=load_llm_judge_config(
            enabled=args.llm_judge_enabled,
            api_url=args.llm_judge_api_url,
            api_key_env=args.llm_judge_api_key_env,
            model=args.llm_judge_model,
            judge_mode=args.llm_judge_mode,
        ),
    )

    print(f"session_id={result['session_id']}")
    print(f"question_id={result['question']['question_id']}")
    print("\n=== 回答评估 ===")
    print(f"overall_score={result['evaluation']['score']}/10")
    for name, value in result["evaluation"]["dimensions"].items():
        print(f"{name}={value}/10")

    print("\n=== 亮点 ===")
    for item in result["evaluation"]["strengths"]:
        print(f"- {item}")

    print("\n=== 改进建议 ===")
    for item in result["evaluation"]["improvements"]:
        print(f"- {item}")

    print("\n=== 追问 ===")
    print(result["follow_up"])
    print(f"\njudge_meta={result['judge_meta']}")


def finish_interview(session_id: str, runtime_dir: Path = DEFAULT_RUNTIME_DIR) -> dict[str, Any]:
    runtime_dir = Path(runtime_dir).resolve()
    session = load_session(runtime_dir, session_id)
    history = session.get("history", [])
    if not history:
        return {
            "session_id": session_id,
            "message": "该会话还没有回答记录。",
            "session": session,
        }
    scored_history = [item for item in history if item.get("scored", True) and item.get("evaluation", {}).get("score") is not None]
    if not scored_history:
        return {
            "session_id": session_id,
            "message": "该会话还没有正式评分记录。",
            "session": session,
        }

    total_score = round(sum(item["evaluation"]["score"] for item in scored_history) / len(scored_history), 1)
    weak_dimensions: dict[str, list[int]] = {}
    for item in scored_history:
        for name, value in item["evaluation"]["dimensions"].items():
            weak_dimensions.setdefault(name, []).append(value)

    average_dimensions = {
        name: round(sum(values) / len(values), 1) for name, values in weak_dimensions.items()
    }
    weakest = sorted(average_dimensions.items(), key=lambda item: item[1])[:2]
    session["status"] = "finished"
    session["finished_at"] = now_iso()
    save_session(runtime_dir, session_id, session)
    improvement_plan: list[str] = []
    for name, _ in weakest:
        if name == "technical_accuracy":
            improvement_plan.append("优先补齐核心概念、原理边界和关键机制，避免只描述表层现象。")
        elif name == "key_point_coverage":
            improvement_plan.append("对照每道题的评分点复盘，确认回答覆盖定义、原理、场景和风险。")
        elif name == "logic_structure":
            improvement_plan.append("练习先给结论，再说明原因、步骤、场景和风险，让回答层次更稳定。")
        elif name == "practical_application":
            improvement_plan.append("回答中加入项目经验、线上问题、监控指标或工程取舍。")
        elif name == "communication_clarity":
            improvement_plan.append("控制回答长度，减少跳跃表达，使用更清晰的关键词和小结。")
    return {
        "session_id": session_id,
        "overall_score": total_score,
        "rounds": len(scored_history),
        "average_dimensions": average_dimensions,
        "weakest_dimensions": [{"name": name, "score": value} for name, value in weakest],
        "improvement_plan": improvement_plan,
        "session": session,
    }


def finish_session(args: argparse.Namespace) -> None:
    result = finish_interview(session_id=args.session_id, runtime_dir=Path(args.runtime_dir))
    print(f"session_id={args.session_id}")
    if "message" in result:
        print(result["message"])
        return
    print("\n=== 面试总结 ===")
    print(f"overall_score={result['overall_score']}/10")
    print(f"rounds={result['rounds']}")
    print("average_dimensions=")
    for name, value in result["average_dimensions"].items():
        print(f"- {name}: {value}/10")

    print("\n=== 当前短板 ===")
    for item in result["weakest_dimensions"]:
        print(f"- {item['name']}: {item['score']}/10")

    print("\n=== 建议下一步练习 ===")
    for item in result["improvement_plan"]:
        print(f"- {item}")


def inspect_interview(session_id: str, runtime_dir: Path = DEFAULT_RUNTIME_DIR) -> dict[str, Any]:
    runtime_dir = Path(runtime_dir).resolve()
    return load_session(runtime_dir, session_id)


def build_recommendations_from_topics(role: str, topics: list[str], weak_dimensions: list[dict[str, Any]]) -> list[dict[str, str]]:
    unique_topics = []
    for topic in topics:
        if topic and topic not in unique_topics:
            unique_topics.append(topic)

    resources: list[dict[str, str]] = []
    for topic in unique_topics[:3]:
        if role == "java_backend":
            topic_map = {
                "collection": "优先复习 Java 集合体系、HashMap/ConcurrentHashMap、扩容与冲突处理。",
                "concurrency": "优先复习 volatile、CAS、AQS、线程池和常见并发问题。",
                "jvm": "优先复习 JVM 内存区域、GC、类加载与常见排障命令。",
                "mysql": "优先复习索引、事务隔离级别、MVCC 和 SQL 优化案例。",
                "backend_design": "优先复习接口设计、幂等、限流、降级与性能优化。",
                "project": "整理一段可量化的项目经历，准备职责、难点、方案和结果。",
            }
        else:
            topic_map = {
                "frontend_basic": "优先复习 JavaScript 基础、闭包、作用域、事件循环与类型判断。",
                "browser": "优先复习浏览器渲染流程、缓存、跨域、重排与重绘。",
                "frontend_framework": "优先复习 React/Vue 核心概念、状态管理与组件设计。",
                "frontend_engineering": "优先复习 Webpack/Vite、构建流程、代码分割与工程规范。",
                "frontend_performance": "优先复习首屏优化、懒加载、缓存与性能指标分析。",
                "project": "整理一段可量化的前端项目经历，准备性能优化或工程化案例。",
            }
        suggestion = topic_map.get(topic)
        if suggestion:
            resources.append({"topic": topic, "suggestion": suggestion})

    for item in weak_dimensions:
        if item["name"] == "logic_clarity":
            resources.append({"topic": "expression", "suggestion": "练习用“背景、原理、方案、结果”四段式回答，提升表达结构。"})
        if item["name"] == "completeness":
            resources.append({"topic": "completeness", "suggestion": "回答时主动补充使用场景、边界条件、优缺点和工程实践。"})

    deduped: list[dict[str, str]] = []
    seen = set()
    for item in resources:
        key = (item["topic"], item["suggestion"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[:5]


def build_expression_summary(expression_items: list[dict[str, Any]]) -> dict[str, Any]:
    if not expression_items:
        return {
            "enabled": False,
            "rounds": 0,
            "average_metrics": {},
            "suggestions": [],
        }

    metric_values: dict[str, list[float]] = {}
    suggestions: list[str] = []
    for item in expression_items:
        for name, value in item.get("metrics", {}).items():
            metric_values.setdefault(name, []).append(float(value))
        for suggestion in item.get("suggestions", []):
            if suggestion not in suggestions:
                suggestions.append(suggestion)

    average_metrics = {
        name: round(sum(values) / len(values), 1)
        for name, values in metric_values.items()
        if values
    }

    return {
        "enabled": True,
        "rounds": len(expression_items),
        "average_metrics": average_metrics,
        "suggestions": suggestions[:5],
    }


def generate_interview_report(
    session_id: str,
    runtime_dir: Path = DEFAULT_RUNTIME_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> dict[str, Any]:
    runtime_dir = Path(runtime_dir).resolve()
    report_dir = Path(report_dir).resolve()
    finish_result = finish_interview(session_id=session_id, runtime_dir=runtime_dir)
    session = finish_result["session"]
    if "message" in finish_result:
        return {
            "session_id": session_id,
            "message": finish_result["message"],
            "session": session,
        }
    history = session.get("history", [])
    if not history:
        return {
            "session_id": session_id,
            "message": "该会话还没有回答记录，无法生成报告。",
            "session": session,
        }

    strengths: list[str] = []
    improvements: list[str] = []
    question_summaries: list[dict[str, Any]] = []
    expression_items: list[dict[str, Any]] = []
    topics: list[str] = []
    scored_history = [item for item in history if item.get("scored", True) and item.get("evaluation", {}).get("score") is not None]
    for round_item in scored_history:
        evaluation = round_item["evaluation"]
        for item in evaluation.get("strengths", []):
            if item not in strengths:
                strengths.append(item)
        for item in evaluation.get("improvements", []):
            if item not in improvements:
                improvements.append(item)
        question_summaries.append(
            {
                "question_id": round_item["question_id"],
                "question": round_item["question"],
                "score": evaluation["score"],
                "follow_up": round_item["follow_up"],
                "strengths": evaluation.get("strengths", []),
                "improvements": evaluation.get("improvements", []),
                "answer_source": round_item.get("answer_source", "text"),
                "expression_analysis": round_item.get("expression_analysis", {}),
            }
        )
        if round_item.get("expression_analysis"):
            expression_items.append(round_item["expression_analysis"])

    role = session["role"]
    all_questions = {item.question_id: item for item in load_questions_for_role(role)}
    for round_item in scored_history:
        question = all_questions.get(round_item["question_id"])
        if question:
            topics.append(question.topic)

    resources = build_recommendations_from_topics(
        role=role,
        topics=topics,
        weak_dimensions=finish_result["weakest_dimensions"],
    )
    expression_summary = build_expression_summary(expression_items)
    report = {
        "session_id": session_id,
        "generated_at": now_iso(),
        "role": session["role"],
        "role_label": session["role_label"],
        "overall_score": finish_result["overall_score"],
        "rounds": finish_result["rounds"],
        "average_dimensions": finish_result["average_dimensions"],
        "weakest_dimensions": finish_result["weakest_dimensions"],
        "highlight_summary": strengths[:5] or ["整体回答与岗位主题相关，具备继续提升基础。"],
        "improvement_summary": improvements[:5] or ["建议继续补充技术细节和结构化表达。"],
        "question_summaries": question_summaries,
        "expression_summary": expression_summary,
        "improvement_plan": finish_result["improvement_plan"],
        "recommended_resources": resources,
        "session": session,
    }

    ensure_runtime_dir(report_dir)
    report_filename = sanitize_filename(f"{session_id}_{session['role']}_report.md")
    report_path = report_dir / report_filename
    report_markdown = render_report_markdown(report)
    report_path.write_text(report_markdown, encoding="utf-8")
    report["report_file"] = str(report_path)
    report["report_markdown"] = report_markdown
    return report


def build_report_view_model(report: dict[str, Any]) -> dict[str, Any]:
    score_labels = {
        key: config["label"] for key, config in SCORING_RUBRIC.items()
    }
    summary_text = (
        f"本次 {report['role_label']} 模拟面试共完成 {report['rounds']} 轮，"
        f"综合得分 {report['overall_score']}/10。"
    )
    if report.get("weakest_dimensions"):
        weakest_names = "、".join(score_labels.get(item["name"], item["name"]) for item in report["weakest_dimensions"])
        summary_text += f" 当前最需要提升的维度是：{weakest_names}。"

    score_cards = []
    for key, value in report["average_dimensions"].items():
        score_cards.append(
            {
                "key": key,
                "label": score_labels.get(key, key),
                "score": value,
                "max_score": 10,
                "weight": SCORING_RUBRIC.get(key, {}).get("weight", 0),
            }
        )

    radar_indicators = [
        {
            "name": score_labels.get(key, key),
            "key": key,
            "value": value,
            "max": 10,
        }
        for key, value in report["average_dimensions"].items()
    ]

    highlight_cards = [{"text": item, "type": "highlight"} for item in report["highlight_summary"]]
    improvement_cards = [{"text": item, "type": "improvement"} for item in report["improvement_summary"]]
    report_evaluation = {
        "score": report["overall_score"],
        "dimensions": report["average_dimensions"],
        "rubric": SCORING_RUBRIC,
        "confidence": "high" if report["rounds"] >= 2 else "medium",
        "strengths": report["highlight_summary"],
        "improvements": report["improvement_summary"],
        "evidence": [
            f"评分基于 {report['rounds']} 轮回答的 LLM 参与评价与五维 Rubric 汇总。",
        ],
    }

    rounds = []
    for item in report["question_summaries"]:
        rounds.append(
            {
                "question_id": item["question_id"],
                "question": item["question"],
                "score": item["score"],
                "follow_up": item["follow_up"],
                "highlights": item["strengths"],
                "improvements": item["improvements"],
            }
        )

    action_plan = [{"step": index + 1, "text": item} for index, item in enumerate(report["improvement_plan"])]
    recommended_resources = [
        {
            "topic": item["topic"],
            "title": item["topic"],
            "description": item["suggestion"],
        }
        for item in report["recommended_resources"]
    ]

    conversation = []
    for item in report["session"].get("history", []):
        msg = (item.get("interviewer_decision") or {}).get("message") or item.get("follow_up", "")
        conv_item = {
            "question_id": item.get("question_id", ""),
            "question": item.get("question", ""),
            "answer": item.get("answer", ""),
            "interviewer_message": msg,
            "score": item.get("evaluation", {}).get("score"),
            "scored": item.get("scored", False),
        }
        if conv_item["question"] or conv_item["answer"]:
            conversation.append(conv_item)

    return {
        "session_id": report["session_id"],
        "role": report["role"],
        "role_label": report["role_label"],
        "generated_at": report["generated_at"],
        "report_file": report.get("report_file", ""),
        "summary": {
            "overall_score": report["overall_score"],
            "rounds": report["rounds"],
            "text": summary_text,
        },
        "score_cards": score_cards,
        "radar": {
            "indicators": radar_indicators,
        },
        "expression": report.get("expression_summary", {}),
        "highlights": highlight_cards,
        "improvements": improvement_cards,
        "report_evaluation": report_evaluation,
        "rounds_detail": rounds,
        "action_plan": action_plan,
        "recommended_resources": recommended_resources,
        "conversation": conversation,
    }


def render_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# 面试评估报告 - {report['role_label']}",
        "",
        f"- 会话 ID：`{report['session_id']}`",
        f"- 生成时间：`{report['generated_at']}`",
        f"- 综合得分：`{report['overall_score']}/10`",
        f"- 面试轮次：`{report['rounds']}`",
        "",
        "## 一、维度得分",
        "",
    ]
    for name, value in report["average_dimensions"].items():
        lines.append(f"- {name}: `{value}/10`")

    lines.extend(["", "## 二、表现亮点", ""])
    for item in report["highlight_summary"]:
        lines.append(f"- {item}")

    lines.extend(["", "## 三、待改进点", ""])
    for item in report["improvement_summary"]:
        lines.append(f"- {item}")

    lines.extend(["", "## 四、逐题总结", ""])
    for item in report["question_summaries"]:
        lines.append(f"### {item['question_id']}")
        lines.append("")
        lines.append(f"- 题目：{item['question']}")
        lines.append(f"- 本题得分：`{item['score']}/10`")
        lines.append(f"- 追问：{item['follow_up']}")
        if item["strengths"]:
            lines.append("- 亮点：")
            for strength in item["strengths"]:
                lines.append(f"  - {strength}")
        if item["improvements"]:
            lines.append("- 改进：")
            for improvement in item["improvements"]:
                lines.append(f"  - {improvement}")
        lines.append("")

    lines.extend(["## 五、下一步练习计划", ""])
    for item in report["improvement_plan"]:
        lines.append(f"- {item}")

    lines.extend(["", "## 六、推荐补强方向", ""])
    for item in report["recommended_resources"]:
        lines.append(f"- {item['topic']}: {item['suggestion']}")
    lines.append("")
    return "\n".join(lines)


def inspect_session(args: argparse.Namespace) -> None:
    session = inspect_interview(session_id=args.session_id, runtime_dir=Path(args.runtime_dir))
    print(json.dumps(session, ensure_ascii=False, indent=2))


def report_session(args: argparse.Namespace) -> None:
    report = generate_interview_report(
        session_id=args.session_id,
        runtime_dir=Path(args.runtime_dir),
        report_dir=Path(args.report_dir),
    )
    print(f"session_id={args.session_id}")
    if "message" in report:
        print(report["message"])
        return
    print(f"report_file={report['report_file']}")
    print("\n=== 报告摘要 ===")
    print(f"overall_score={report['overall_score']}/10")
    print("weakest_dimensions=")
    for item in report["weakest_dimensions"]:
        print(f"- {item['name']}: {item['score']}/10")
    print("\n=== 建议 ===")
    for item in report["improvement_plan"]:
        print(f"- {item}")


def main() -> None:
    args = parse_args()
    if args.command == "start":
        start_session(args)
    elif args.command == "answer":
        answer_session(args)
    elif args.command == "finish":
        finish_session(args)
    elif args.command == "inspect":
        inspect_session(args)
    elif args.command == "report":
        report_session(args)


if __name__ == "__main__":
    main()
