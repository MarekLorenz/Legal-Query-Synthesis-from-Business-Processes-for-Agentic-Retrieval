from __future__ import annotations

import shutil
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bpmn_parser import build_process_model, load_optional_process_json  # noqa: E402
from main import clean_text  # noqa: E402
from retrieval.retrieval_bm25 import BM25Index, Document, Query, load_corpus  # noqa: E402
from retrieval.sota_retrieval import SotaRetriever  # noqa: E402

UPLOAD_ROOT = Path(tempfile.gettempdir()) / "regulatory_process_annotator"

app = FastAPI(title="Regulatory Process Annotator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnnotationRequest(BaseModel):
    session_id: str
    top_k: int = 5
    include_cross_encoder: bool = True


@dataclass
class UploadedSession:
    session_id: str
    work_dir: Path
    corpus_path: Path
    diagram_xml: str
    process_model: dict[str, Any]
    documents: list[Document] = field(default_factory=list)
    bm25_index: BM25Index | None = None
    sota_retriever: SotaRetriever | None = None


SESSIONS: dict[str, UploadedSession] = {}


def get_session(session_id: str) -> UploadedSession:
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Upload session not found. Load your files again.")
    return session


def load_documents_from_path(path: Path) -> list[Document]:
    return load_corpus(str(path))


def build_bm25_from_documents(documents: list[Document]) -> BM25Index:
    return BM25Index(documents)


def annotation_targets(model: dict[str, Any]) -> list[dict[str, Any]]:
    targets = [model["topLevel"]]
    for subprocess in model["subprocesses"]:
        targets.append(subprocess)
        targets.extend(subprocess["children"])
    return targets


def annotate_text(session: UploadedSession, text: str, top_k: int, include_cross_encoder: bool) -> list[dict[str, Any]]:
    if include_cross_encoder:
        if session.sota_retriever is None:
            session.sota_retriever = SotaRetriever([document.text for document in session.documents])
        passages = session.sota_retriever.search_bm25_ce(n=top_k, query=text, query_variant="baseline")
        return [
            {
                "text": passage.rel_text,
                "score": passage.score,
                "method": passage.method,
            }
            for passage in passages
        ]

    if session.bm25_index is None:
        session.bm25_index = build_bm25_from_documents(session.documents)
    results = session.bm25_index.rank(Query(text=text), top_k=top_k)
    return [
        {
            "text": session.documents[result.document.doc_id].text.replace("\n", " "),
            "score": result.score,
            "method": "bm25",
        }
        for result in results
    ]


async def read_upload_file(upload: UploadFile, label: str) -> bytes:
    if upload is None:
        raise HTTPException(status_code=400, detail=f"Missing file: {label}")
    content = await upload.read()
    if not content:
        raise HTTPException(status_code=400, detail=f"Uploaded file is empty: {label}")
    return content


@app.post("/api/upload")
async def upload_process(
    diagram_xml: UploadFile = File(..., description="BPMN 2.0 diagram XML"),
    corpus_xlsx: UploadFile = File(..., description="Regulatory corpus Excel file"),
    process_json: UploadFile | None = File(None, description="Optional process description JSON"),
    session_label: str = Form("custom"),
) -> dict[str, Any]:
    diagram_name = (diagram_xml.filename or "").lower()
    corpus_name = (corpus_xlsx.filename or "").lower()
    if not diagram_name.endswith(".xml") and not diagram_name.endswith(".bpmn"):
        raise HTTPException(status_code=400, detail="Process file must be a BPMN XML (.xml or .bpmn).")
    if not corpus_name.endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Corpus file must be an Excel workbook (.xlsx).")

    diagram_bytes = await read_upload_file(diagram_xml, "diagram_xml")
    corpus_bytes = await read_upload_file(corpus_xlsx, "corpus_xlsx")
    process_json_bytes = await process_json.read() if process_json is not None else b""

    try:
        diagram_text = diagram_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="BPMN XML must be UTF-8 encoded.") from exc

    try:
        optional_json = load_optional_process_json(process_json_bytes)
        process_model = build_process_model(diagram_text, optional_json)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse uploaded files: {exc}") from exc

    try:
        corpus_frame = pd.read_excel(BytesIO(corpus_bytes))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Corpus Excel file is invalid: {exc}") from exc
    if "requirement_text" not in corpus_frame.columns:
        raise HTTPException(
            status_code=400,
            detail="Corpus Excel must include a 'requirement_text' column.",
        )

    session_id = str(uuid.uuid4())
    work_dir = UPLOAD_ROOT / session_id
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = work_dir / "corpus.xlsx"
    corpus_path.write_bytes(corpus_bytes)
    (work_dir / "diagram.xml").write_text(diagram_text, encoding="utf-8")

    documents = load_documents_from_path(corpus_path)
    session = UploadedSession(
        session_id=session_id,
        work_dir=work_dir,
        corpus_path=corpus_path,
        diagram_xml=diagram_text,
        process_model=process_model,
        documents=documents,
    )
    SESSIONS[session_id] = session

    return {
        "sessionId": session_id,
        "label": session_label,
        "process": process_model,
        "diagram": {"xml": diagram_text},
        "corpusDocuments": len(documents),
        "subprocessCount": len(process_model.get("subprocesses", [])),
        "taskCount": sum(len(item.get("children", [])) for item in process_model.get("subprocesses", [])),
    }


@app.get("/api/sessions/{session_id}/process")
def get_session_process(session_id: str) -> dict[str, Any]:
    return get_session(session_id).process_model


@app.get("/api/sessions/{session_id}/diagram")
def get_session_diagram(session_id: str) -> dict[str, str]:
    session = get_session(session_id)
    return {"xml": session.diagram_xml}


@app.post("/api/annotate")
def annotate(request: AnnotationRequest) -> dict[str, Any]:
    session = get_session(request.session_id)
    model = session.process_model
    top_k = max(1, min(request.top_k, 20))
    annotations = {}

    for target in annotation_targets(model):
        annotations[target["id"]] = annotate_text(
            session=session,
            text=clean_text(target["description"]),
            top_k=top_k,
            include_cross_encoder=request.include_cross_encoder,
        )

    return {
        "sessionId": session.session_id,
        "topK": top_k,
        "annotations": annotations,
    }


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, str]:
    session = SESSIONS.pop(session_id, None)
    if session is not None and session.work_dir.exists():
        shutil.rmtree(session.work_dir, ignore_errors=True)
    return {"status": "deleted"}


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
