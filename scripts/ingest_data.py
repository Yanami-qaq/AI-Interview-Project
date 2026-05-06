import argparse
import os
import re
from pathlib import Path
from typing import Iterable

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_DB_DIR = Path(__file__).resolve().parent.parent / "db"
DEFAULT_COLLECTION = "interview_knowledge"
DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
DEFAULT_JOB_ROLES = ("java_backend",)

TOPIC_RULES = {
    "jvm": ("jvm", "内存区域", "垃圾回收", "类加载", "双亲委派", "gc", "oom", "jstack"),
    "java_basic": ("java基础", "equals", "hashcode", "final", "反射", "泛型", "枚举", "序列化", "异常"),
    "collection": ("集合", "arraylist", "linkedlist", "hashmap", "concurrenthashmap", "map", "list", "set"),
    "concurrency": ("并发", "线程", "线程池", "锁", "cas", "volatile", "死锁", "aqs"),
    "mysql": ("mysql", "sql", "索引", "事务", "mvcc", "order by", "explain"),
    "redis": ("redis", "缓存", "分布式锁", "缓存击穿", "缓存雪崩"),
    "backend_design": ("接口设计", "系统设计", "幂等", "性能优化", "日志", "安全", "限流", "降级"),
    "project": ("项目", "实践", "排查", "复盘", "案例", "落地"),
    "behavioral": ("沟通", "协作", "冲突", "压力", "成长", "复盘"),
    "frontend_basic": ("前端基础", "html", "css", "javascript", "js", "dom", "bom", "事件循环"),
    "browser": ("浏览器", "渲染", "重排", "重绘", "缓存", "跨域", "同源策略"),
    "frontend_framework": ("react", "vue", "组件化", "hooks", "状态管理", "虚拟dom"),
    "frontend_engineering": ("工程化", "webpack", "vite", "babel", "打包", "构建", "npm", "pnpm"),
    "frontend_performance": ("性能优化", "首屏", "懒加载", "code split", "代码分割", "性能指标", "优化"),
}

QUESTION_TYPE_RULES = {
    "technical_knowledge": ("是什么", "原理", "区别", "为什么", "怎么实现", "解析", "讲一下"),
    "scenario": ("如何", "排查", "优化", "故障", "设计", "处理", "解决"),
    "project_deep_dive": ("项目", "实践", "复盘", "落地", "指标"),
    "behavioral": ("沟通", "协作", "冲突", "压力", "成长"),
}

DIFFICULTY_RULES = (
    ("hard", ("jvm", "并发", "死锁", "类加载", "内存", "索引失效", "分布式锁", "系统设计")),
    ("medium", ("线程池", "事务", "hashmap", "redis", "sql", "反射", "序列化", "接口设计")),
)

TOPIC_LABEL_MAP = {
    "java基础": "java_basic",
    "集合": "collection",
    "java集合": "collection",
    "并发": "concurrency",
    "java并发": "concurrency",
    "jvm": "jvm",
    "mysql": "mysql",
    "redis": "redis",
    "接口设计": "backend_design",
    "系统设计": "backend_design",
    "性能优化": "backend_design",
    "项目深挖": "project",
    "项目经验": "project",
    "项目": "project",
    "行为题": "behavioral",
    "前端基础": "frontend_basic",
    "浏览器": "browser",
    "浏览器原理": "browser",
    "框架": "frontend_framework",
    "react": "frontend_framework",
    "vue": "frontend_framework",
    "前端工程化": "frontend_engineering",
    "工程化": "frontend_engineering",
    "前端性能优化": "frontend_performance",
    "性能优化": "frontend_performance",
}

QUESTION_TYPE_LABEL_MAP = {
    "技术知识": "technical_knowledge",
    "八股题": "technical_knowledge",
    "场景题": "scenario",
    "系统设计": "scenario",
    "项目深挖": "project_deep_dive",
    "项目题": "project_deep_dive",
    "行为题": "behavioral",
}

DIFFICULTY_LABEL_MAP = {
    "初级": "easy",
    "中级": "medium",
    "中高级": "hard",
    "高级": "hard",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest interview knowledge into Chroma.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Knowledge base root directory.")
    parser.add_argument("--db-dir", default=str(DEFAULT_DB_DIR), help="Chroma persistence directory.")
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION,
        help="Chroma collection name. Use different names for different experiments.",
    )
    parser.add_argument(
        "--roles",
        nargs="+",
        default=list(DEFAULT_JOB_ROLES),
        help="Job roles to ingest, for example: java_backend web_frontend",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=[],
        help="Optional specific markdown files to ingest. Paths can be relative to data-dir or absolute.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_MODEL,
        help="Embedding model name or local path. Local path is recommended for offline demos.",
    )
    parser.add_argument(
        "--local-model-only",
        action="store_true",
        help="Only load the embedding model from local files. Recommended for offline delivery.",
    )
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size for recursive splitting.")
    parser.add_argument("--chunk-overlap", type=int, default=80, help="Chunk overlap for recursive splitting.")
    parser.add_argument(
        "--clear-collection",
        action="store_true",
        help="Delete the target collection before ingesting new data.",
    )
    return parser.parse_args()


def resolve_embeddings(model_name_or_path: str, local_only: bool) -> HuggingFaceEmbeddings:
    model_path = Path(model_name_or_path)
    model_kwargs = {"local_files_only": local_only or model_path.exists()}
    return HuggingFaceEmbeddings(
        model_name=str(model_path if model_path.exists() else model_name_or_path),
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},
    )


def normalize_text(text: str) -> str:
    return text.lower().replace("_", " ").replace("-", " ")


def normalize_label(value: str) -> str:
    return value.strip().replace("`", "").replace("：", ":")


def infer_topic(text: str) -> str:
    normalized = normalize_text(text)
    for topic, keywords in TOPIC_RULES.items():
        if any(keyword.lower() in normalized for keyword in keywords):
            return topic
    return "general_java_backend"


def infer_question_type(text: str) -> str:
    normalized = normalize_text(text)
    for question_type, keywords in QUESTION_TYPE_RULES.items():
        if any(keyword.lower() in normalized for keyword in keywords):
            return question_type
    return "technical_knowledge"


def infer_difficulty(text: str) -> str:
    normalized = normalize_text(text)
    for level, keywords in DIFFICULTY_RULES:
        if any(keyword.lower() in normalized for keyword in keywords):
            return level
    return "easy"


def extract_keywords(text: str) -> list[str]:
    normalized = normalize_text(text)
    hits: list[str] = []
    for keywords in TOPIC_RULES.values():
        for keyword in keywords:
            if keyword.lower() in normalized:
                hits.append(keyword)
    unique_hits: list[str] = []
    for item in hits:
        if item not in unique_hits:
            unique_hits.append(item)
    return unique_hits[:8]


def extract_section_content(text: str, heading: str) -> str:
    target = f"### {heading}"
    lines = text.splitlines()
    capture = False
    collected: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == target:
            capture = True
            continue
        if capture and (stripped.startswith("### ") or stripped.startswith("## ")):
            break
        if capture:
            collected.append(line)
    return "\n".join(collected).strip()


def extract_metadata_block(text: str) -> dict[str, str]:
    metadata_text = extract_section_content(text, "元数据")
    metadata: dict[str, str] = {}
    for raw_line in metadata_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("-"):
            continue
        clean_line = line.lstrip("-").strip()
        if "：" in clean_line:
            key, value = clean_line.split("：", 1)
        elif ":" in clean_line:
            key, value = clean_line.split(":", 1)
        else:
            continue
        metadata[normalize_label(key)] = normalize_label(value)
    return metadata


def normalize_topic(raw_value: str, fallback_text: str) -> str:
    raw = normalize_label(raw_value).lower()
    if raw in TOPIC_LABEL_MAP:
        return TOPIC_LABEL_MAP[raw]
    return infer_topic(fallback_text)


def normalize_question_type(raw_value: str, fallback_text: str) -> str:
    raw = normalize_label(raw_value)
    if raw in QUESTION_TYPE_LABEL_MAP:
        return QUESTION_TYPE_LABEL_MAP[raw]
    return infer_question_type(fallback_text)


def normalize_difficulty(raw_value: str, fallback_text: str) -> str:
    raw = normalize_label(raw_value)
    if raw in DIFFICULTY_LABEL_MAP:
        return DIFFICULTY_LABEL_MAP[raw]
    return infer_difficulty(fallback_text)


def normalize_keywords(raw_value: str, fallback_text: str) -> str:
    raw = normalize_label(raw_value)
    if raw:
        parts = [segment.strip("` ").strip() for segment in re.split(r"[,\s，、]+", raw) if segment.strip("` ").strip()]
        if parts:
            return ",".join(parts[:8])
    return ",".join(extract_keywords(fallback_text))


def build_base_metadata(doc: Document, role: str, file_path: Path, data_dir: Path) -> dict[str, str]:
    title = file_path.stem
    source_text = f"{title}\n{doc.page_content[:1200]}"
    relative_path = file_path.relative_to(data_dir).as_posix()
    return {
        "job_role": role,
        "source": file_path.name,
        "source_path": relative_path,
        "title": title,
        "topic": infer_topic(source_text),
        "question_type": infer_question_type(source_text),
        "difficulty": infer_difficulty(source_text),
        "keywords": ",".join(extract_keywords(source_text)),
    }


def build_chunk_metadata(chunk: Document, base_metadata: dict[str, str], role: str) -> dict[str, str]:
    chunk_text = chunk.page_content
    section_text = chunk.metadata.get("_section_text", chunk_text)
    structured_metadata = extract_metadata_block(section_text)
    question_text = chunk.metadata.get("_section_question") or extract_section_content(section_text, "面试题")
    topic = normalize_topic(
        chunk.metadata.get("_section_topic_raw", structured_metadata.get("主题", "")),
        section_text,
    )
    question_type = normalize_question_type(
        chunk.metadata.get("_section_question_type_raw", structured_metadata.get("题型", "")),
        section_text,
    )
    difficulty = normalize_difficulty(
        chunk.metadata.get("_section_difficulty_raw", structured_metadata.get("难度", "")),
        section_text,
    )
    keywords = normalize_keywords(
        chunk.metadata.get("_section_keywords_raw", structured_metadata.get("关键词", "")),
        section_text,
    )

    metadata = dict(base_metadata)
    metadata.update(
        {
            "job_role": role,
            "title": question_text[:80] if question_text else base_metadata["title"],
            "question": question_text,
            "topic": topic,
            "question_type": question_type,
            "difficulty": difficulty,
            "keywords": keywords,
        }
    )
    if structured_metadata.get("岗位"):
        metadata["role_label"] = structured_metadata["岗位"]
    if chunk.metadata.get("h2"):
        metadata["question_id"] = chunk.metadata["h2"]
    return metadata


def split_markdown(doc: Document, chunk_size: int, chunk_overlap: int) -> list[Document]:
    title_match = re.search(r"^#\s+(.+)$", doc.page_content, flags=re.MULTILINE)
    h1_title = title_match.group(1).strip() if title_match else ""

    question_matches = list(re.finditer(r"^##\s+(.+)$", doc.page_content, flags=re.MULTILINE))
    header_docs: list[Document] = []
    if question_matches:
        for index, match in enumerate(question_matches):
            start = match.start()
            end = question_matches[index + 1].start() if index + 1 < len(question_matches) else len(doc.page_content)
            section_text = doc.page_content[start:end].strip()
            header_docs.append(
                Document(
                    page_content=section_text,
                    metadata={"h1": h1_title, "h2": match.group(1).strip()},
                )
            )
    else:
        header_docs = [Document(page_content=doc.page_content, metadata={"h1": h1_title})]

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n### ", "\n## ", "\n\n", "\n", "。", "，", "？", " ", ""],
    )

    chunks: list[Document] = []
    for header_doc in header_docs:
        section_metadata = extract_metadata_block(header_doc.page_content)
        section_question = extract_section_content(header_doc.page_content, "面试题")
        pieces = recursive_splitter.split_documents(
            [Document(page_content=header_doc.page_content, metadata=header_doc.metadata)]
        )
        for piece in pieces:
            piece.metadata["_section_text"] = header_doc.page_content
            piece.metadata["_section_question"] = section_question
            piece.metadata["_section_topic_raw"] = section_metadata.get("主题", "")
            piece.metadata["_section_question_type_raw"] = section_metadata.get("题型", "")
            piece.metadata["_section_difficulty_raw"] = section_metadata.get("难度", "")
            piece.metadata["_section_keywords_raw"] = section_metadata.get("关键词", "")
        chunks.extend(pieces)
    return chunks


def read_markdown_file(file_path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return file_path.read_text(encoding="utf-8", errors="ignore")


def resolve_target_files(data_dir: Path, roles: list[str], file_args: list[str]) -> list[tuple[str, Path]]:
    if file_args:
        resolved_files: list[tuple[str, Path]] = []
        for raw_path in file_args:
            candidate = Path(raw_path)
            file_path = candidate if candidate.is_absolute() else (data_dir / candidate)
            file_path = file_path.resolve()
            if not file_path.exists():
                print(f"Skip missing file: {file_path}")
                continue
            role = file_path.parent.name
            resolved_files.append((role, file_path))
        return resolved_files

    resolved_files = []
    for role in roles:
        role_dir = data_dir / role
        if not role_dir.exists():
            print(f"Skip missing role directory: {role_dir}")
            continue
        markdown_files = list(sorted(role_dir.glob("*.md"))) + list(sorted(role_dir.glob("*.MD")))
        for file_path in markdown_files:
            resolved_files.append((role, file_path))
    return resolved_files


def load_documents(
    data_dir: Path,
    roles: list[str],
    file_args: list[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    all_chunks: list[Document] = []
    target_files = resolve_target_files(data_dir=data_dir, roles=roles, file_args=file_args)

    for role, file_path in target_files:
        file_chunk_count = 0
        loaded_doc = Document(page_content=read_markdown_file(file_path), metadata={})
        base_metadata = build_base_metadata(loaded_doc, role, file_path, data_dir)
        split_docs = split_markdown(loaded_doc, chunk_size, chunk_overlap)
        for index, chunk in enumerate(split_docs):
            chunk.metadata.update(build_chunk_metadata(chunk, base_metadata, role))
            chunk.metadata.pop("_section_text", None)
            chunk.metadata.pop("_section_question", None)
            chunk.metadata.pop("_section_topic_raw", None)
            chunk.metadata.pop("_section_question_type_raw", None)
            chunk.metadata.pop("_section_difficulty_raw", None)
            chunk.metadata.pop("_section_keywords_raw", None)
            chunk.metadata["chunk_index"] = index
            chunk.metadata["section"] = (
                chunk.metadata.get("h2")
                or chunk.metadata.get("h3")
                or chunk.metadata.get("h1")
                or chunk.metadata["title"]
            )
            all_chunks.append(chunk)
            file_chunk_count += 1

        print(f"Prepared {role}/{file_path.name}: {file_chunk_count} chunks")

    return all_chunks


def clear_collection(vector_db: Chroma) -> None:
    try:
        vector_db.delete_collection()
        print("Cleared target collection.")
    except Exception as exc:
        print(f"Collection clear skipped: {exc}")


def ingest_documents(
    documents: Iterable[Document],
    embeddings: HuggingFaceEmbeddings,
    db_dir: Path,
    collection_name: str,
    clear_first: bool,
) -> None:
    db_dir.mkdir(parents=True, exist_ok=True)
    vector_db = Chroma(
        persist_directory=str(db_dir),
        embedding_function=embeddings,
        collection_name=collection_name,
    )

    if clear_first:
        clear_collection(vector_db)

    docs = list(documents)
    if not docs:
        print("No documents found. Nothing was ingested.")
        return

    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(db_dir),
        collection_name=collection_name,
    )
    print(f"Ingest success: collection={collection_name}, chunks={len(docs)}")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    db_dir = Path(args.db_dir).resolve()

    print(f"Embedding model: {args.embedding_model}")
    print(f"Local only: {args.local_model_only}")
    embeddings = resolve_embeddings(args.embedding_model, args.local_model_only)

    all_documents = load_documents(
        data_dir=data_dir,
        roles=args.roles,
        file_args=args.files,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    ingest_documents(
        documents=all_documents,
        embeddings=embeddings,
        db_dir=db_dir,
        collection_name=args.collection_name,
        clear_first=args.clear_collection,
    )


if __name__ == "__main__":
    main()
