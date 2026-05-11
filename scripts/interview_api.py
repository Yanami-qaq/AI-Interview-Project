import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from interview_flow import (
    build_report_view_model,
    DEFAULT_DB_DIR,
    DEFAULT_MODEL,
    DEFAULT_REPORT_DIR,
    DEFAULT_RUNTIME_DIR,
    ROLE_CONFIGS,
    answer_interview,
    answer_interview_stream,
    finish_interview,
    generate_interview_report,
    inspect_interview,
    load_llm_judge_config,
    public_llm_judge_status,
    start_interview,
)
from speech_io import (
    DEFAULT_AUDIO_DIR,
    analyze_expression_stub,
    speech_capabilities,
    synthesize_speech_stub,
    transcribe_audio_stub,
)

WEBAPP_DIR = Path(__file__).resolve().parent.parent / "webapp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal interview API server.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8010, help="Bind port.")
    parser.add_argument("--db-dir", default=str(DEFAULT_DB_DIR), help="Chroma db dir.")
    parser.add_argument("--runtime-dir", default=str(DEFAULT_RUNTIME_DIR), help="Session runtime dir.")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR), help="Generated report dir.")
    parser.add_argument("--audio-dir", default=str(DEFAULT_AUDIO_DIR), help="Speech input/output runtime dir.")
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL, help="Embedding model path or name.")
    parser.add_argument("--local-model-only", action="store_true", help="Only load local embedding model files.")
    parser.add_argument(
        "--llm-judge-enabled",
        action="store_true",
        default=None,
        help="Enable optional LLM judge calibration. Can also be enabled with LLM_JUDGE_ENABLED=1.",
    )
    parser.add_argument(
        "--llm-judge-api-url",
        default="",
        help="OpenAI-compatible chat completions endpoint for judge calibration.",
    )
    parser.add_argument("--llm-judge-api-key-env", default="OPENAI_API_KEY", help="Judge API key env var.")
    parser.add_argument("--llm-judge-model", default="", help="Model name used by the optional judge.")
    parser.add_argument(
        "--llm-judge-mode",
        default="",
        choices=["", "conservative", "balanced", "full"],
        help="LLM judge authority: conservative, balanced, or full. Default can be set by LLM_JUDGE_MODE.",
    )
    return parser.parse_args()


class InterviewApiHandler(BaseHTTPRequestHandler):
    server_version = "InterviewAPI/0.1"

    def _sse_stream(self, generator) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        try:
            for event in generator:
                line = f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            try:
                err = json.dumps({"type": "error", "error": str(exc)}, ensure_ascii=False)
                self.wfile.write(f"data: {err}\n\n".encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass

    def _json_response(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _runtime_dir(self) -> Path:
        return Path(self.server.runtime_dir).resolve()

    def _db_dir(self) -> Path:
        return Path(self.server.db_dir).resolve()

    def _embedding_model(self) -> str:
        return self.server.embedding_model

    def _local_model_only(self) -> bool:
        return self.server.local_model_only

    def _report_dir(self) -> Path:
        return Path(self.server.report_dir).resolve()

    def _audio_dir(self) -> Path:
        return Path(self.server.audio_dir).resolve()

    def _llm_judge_config(self):
        return load_llm_judge_config(
            enabled=self.server.llm_judge_enabled,
            api_url=self.server.llm_judge_api_url,
            api_key_env=self.server.llm_judge_api_key_env,
            model=self.server.llm_judge_model,
            judge_mode=self.server.llm_judge_mode,
        )

    def _webapp_dir(self) -> Path:
        return Path(self.server.webapp_dir).resolve()

    def _serve_file(self, file_path: Path) -> None:
        if not file_path.exists() or not file_path.is_file():
            self._json_response({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return
        mime_type, _ = mimetypes.guess_type(str(file_path))
        body = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{mime_type or 'application/octet-stream'}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._json_response({"ok": True})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        try:
            if path == "/health":
                self._json_response(
                    {
                        "ok": True,
                        "service": "minimal_interview_api",
                        "roles": sorted(ROLE_CONFIGS.keys()),
                        "speech": speech_capabilities(),
                        "llm_judge": public_llm_judge_status(self._llm_judge_config()),
                    }
                )
                return

            if path == "/llm/status":
                self._json_response({"llm_judge": public_llm_judge_status(self._llm_judge_config())})
                return

            if path == "/roles":
                self._json_response(
                    {
                        "roles": [
                            {
                                "role": role,
                                "role_label": config["role_label"],
                                "collection": config["collection"],
                                "main_file": str(config["main_file"]),
                            }
                            for role, config in ROLE_CONFIGS.items()
                        ]
                    }
                )
                return

            if path == "/speech/capabilities":
                self._json_response(speech_capabilities())
                return

            if path in ("/", "/app"):
                self._serve_file(self._webapp_dir() / "index.html")
                return

            if path.startswith("/static/"):
                relative_path = path.removeprefix("/static/")
                safe_path = (self._webapp_dir() / relative_path).resolve()
                if self._webapp_dir() not in safe_path.parents and safe_path != self._webapp_dir():
                    self._json_response({"error": "Invalid static path"}, status=HTTPStatus.BAD_REQUEST)
                    return
                self._serve_file(safe_path)
                return

            if path.startswith("/interview/session/"):
                session_id = path.split("/")[-1]
                session = inspect_interview(session_id=session_id, runtime_dir=self._runtime_dir())
                self._json_response({"session": session})
                return

            if path.startswith("/report/"):
                parts = [part for part in path.split("/") if part]
                if len(parts) >= 3 and parts[1] == "view":
                    session_id = parts[2]
                    report = generate_interview_report(
                        session_id=session_id,
                        runtime_dir=self._runtime_dir(),
                        report_dir=self._report_dir(),
                    )
                    self._json_response({"view_model": build_report_view_model(report), "report": report})
                    return
                session_id = parts[-1]
                report = generate_interview_report(
                    session_id=session_id,
                    runtime_dir=self._runtime_dir(),
                    report_dir=self._report_dir(),
                )
                self._json_response(report)
                return

            if path == "/history":
                sessions = []
                runtime_path = self._runtime_dir()
                if runtime_path.exists():
                    for f in sorted(runtime_path.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:30]:
                        try:
                            data = json.loads(f.read_text(encoding="utf-8"))
                            history = data.get("history", [])
                            scored = [h for h in history if h.get("scored")]
                            scores = [
                                h["evaluation"]["score"]
                                for h in scored
                                if isinstance(h.get("evaluation", {}).get("score"), (int, float))
                            ]
                            sessions.append({
                                "session_id": data.get("session_id", f.stem),
                                "role_label": data.get("role_label", ""),
                                "role": data.get("role", ""),
                                "created_at": data.get("created_at", ""),
                                "status": data.get("status", "unknown"),
                                "rounds": len(scored),
                                "score": round(sum(scores) / len(scores), 1) if scores else None,
                            })
                        except Exception:
                            continue
                self._json_response({"sessions": sessions})
                return

            self._json_response({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
        except FileNotFoundError as exc:
            self._json_response({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._json_response({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        try:
            payload = self._read_json()

            if path == "/interview/start":
                result = start_interview(
                    role=payload["role"],
                    topic=payload.get("topic", ""),
                    difficulty=payload.get("difficulty", ""),
                    question_query=payload.get("question_query", ""),
                    resume_text=payload.get("resume_text", ""),
                    db_dir=self._db_dir(),
                    embedding_model=self._embedding_model(),
                    local_model_only=self._local_model_only(),
                    runtime_dir=self._runtime_dir(),
                    llm_judge_config=self._llm_judge_config(),
                )
                self._json_response(result, status=HTTPStatus.CREATED)
                return

            if path == "/interview/answer":
                result = answer_interview(
                    session_id=payload["session_id"],
                    answer=payload["answer"],
                    runtime_dir=self._runtime_dir(),
                    llm_judge_config=self._llm_judge_config(),
                )
                self._json_response(result)
                return

            if path == "/interview/answer/stream":
                gen = answer_interview_stream(
                    session_id=payload["session_id"],
                    answer=payload["answer"],
                    runtime_dir=self._runtime_dir(),
                    llm_judge_config=self._llm_judge_config(),
                )
                self._sse_stream(gen)
                return

            if path == "/interview/answer-audio":
                speech_result = transcribe_audio_stub(payload, audio_dir=self._audio_dir())
                transcript = speech_result.get("transcript", "")
                if not transcript:
                    self._json_response(
                        {
                            "error": "Empty transcript",
                            "speech_result": speech_result,
                        },
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                expression_analysis = analyze_expression_stub(
                    {
                        "transcript": transcript,
                        "duration_ms": speech_result.get("duration_ms", 0),
                        "input_type": "audio",
                    }
                )
                result = answer_interview(
                    session_id=payload["session_id"],
                    answer=transcript,
                    runtime_dir=self._runtime_dir(),
                    answer_source="audio",
                    speech_result=speech_result,
                    expression_analysis=expression_analysis,
                    llm_judge_config=self._llm_judge_config(),
                )
                result["speech_result"] = speech_result
                result["expression_analysis"] = expression_analysis
                self._json_response(result)
                return

            if path == "/speech/transcribe":
                result = transcribe_audio_stub(payload, audio_dir=self._audio_dir())
                self._json_response(result)
                return

            if path == "/speech/synthesize":
                result = synthesize_speech_stub(payload, audio_dir=self._audio_dir())
                self._json_response(result)
                return

            if path == "/speech/analyze-expression":
                result = analyze_expression_stub(payload)
                self._json_response(result)
                return

            if path == "/interview/finish":
                result = finish_interview(
                    session_id=payload["session_id"],
                    runtime_dir=self._runtime_dir(),
                )
                self._json_response(result)
                return

            if path == "/report/generate":
                result = generate_interview_report(
                    session_id=payload["session_id"],
                    runtime_dir=self._runtime_dir(),
                    report_dir=self._report_dir(),
                )
                self._json_response(result)
                return

            if path == "/report/view":
                report = generate_interview_report(
                    session_id=payload["session_id"],
                    runtime_dir=self._runtime_dir(),
                    report_dir=self._report_dir(),
                )
                self._json_response({"view_model": build_report_view_model(report), "report": report})
                return

            self._json_response({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
        except KeyError as exc:
            self._json_response({"error": f"Missing field: {exc}"}, status=HTTPStatus.BAD_REQUEST)
        except FileNotFoundError as exc:
            self._json_response({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
        except json.JSONDecodeError:
            self._json_response({"error": "Invalid JSON body"}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._json_response({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)


def build_server(args: argparse.Namespace) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((args.host, args.port), InterviewApiHandler)
    server.db_dir = args.db_dir
    server.runtime_dir = args.runtime_dir
    server.report_dir = args.report_dir
    server.audio_dir = args.audio_dir
    server.webapp_dir = str(WEBAPP_DIR)
    server.embedding_model = args.embedding_model
    server.local_model_only = args.local_model_only
    server.llm_judge_enabled = args.llm_judge_enabled
    server.llm_judge_api_url = args.llm_judge_api_url
    server.llm_judge_api_key_env = args.llm_judge_api_key_env
    server.llm_judge_model = args.llm_judge_model
    server.llm_judge_mode = args.llm_judge_mode
    return server


def main() -> None:
    args = parse_args()
    server = build_server(args)
    print(f"Interview API listening on http://{args.host}:{args.port}")
    print(f"runtime_dir={Path(args.runtime_dir).resolve()}")
    print(f"report_dir={Path(args.report_dir).resolve()}")
    print(f"audio_dir={Path(args.audio_dir).resolve()}")
    print(f"db_dir={Path(args.db_dir).resolve()}")
    print(f"local_model_only={args.local_model_only}")
    print(f"llm_judge_enabled={args.llm_judge_enabled}")
    print(f"llm_judge_mode={args.llm_judge_mode or 'balanced'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down API server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
