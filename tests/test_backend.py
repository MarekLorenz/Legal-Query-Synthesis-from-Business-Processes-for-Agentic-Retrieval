import sys
import types
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
WEBAPP = ROOT / "webapp"
if str(WEBAPP) not in sys.path:
    sys.path.insert(0, str(WEBAPP))

_fake_main = types.ModuleType("main")
_fake_main.clean_text = lambda text: " ".join(str(text).split())
sys.modules["main"] = _fake_main

import backend  # noqa: E402
from backend import UploadedSession, get_session  # noqa: E402


@pytest.fixture
def client():
    backend.SESSIONS.clear()
    return TestClient(backend.app)


@pytest.fixture
def sample_process_model():
    return {
        "topLevel": {"id": "process", "description": "Store customer data"},
        "subprocesses": [
            {
                "id": "sub-1",
                "description": "Collect consent",
                "children": [
                    {"id": "task-1", "description": "Record opt-in"},
                ],
            }
        ],
    }


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_get_session_raises_404_for_unknown_session():
    backend.SESSIONS.clear()
    with pytest.raises(HTTPException) as exc_info:
        get_session("missing-session-id")
    assert exc_info.value.status_code == 404


def test_annotate_endpoint_uses_mocked_retrieval(client, sample_process_model, monkeypatch):
    session = UploadedSession(
        session_id="annotate-session",
        work_dir=Path("/tmp/annotate-session"),
        corpus_path=Path("/tmp/annotate-session/corpus.xlsx"),
        diagram_xml="<bpmn/>",
        process_model=sample_process_model,
        documents=[],
    )
    backend.SESSIONS[session.session_id] = session
    monkeypatch.setattr(
        backend,
        "annotate_text",
        lambda *_args, **_kwargs: [{"text": "mock hit", "score": 0.9, "method": "bm25"}],
    )

    response = client.post(
        "/api/annotate",
        json={"session_id": session.session_id, "top_k": 3, "include_cross_encoder": False},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["sessionId"] == session.session_id
    assert payload["topK"] == 3
    assert set(payload["annotations"]) == {"process", "sub-1", "task-1"}
    assert payload["annotations"]["process"][0]["text"] == "mock hit"
