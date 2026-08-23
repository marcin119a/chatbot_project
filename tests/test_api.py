from __future__ import annotations

from types import SimpleNamespace

import api
from tests.conftest import FAQ_ANSWER

def _fake_openai_client(chunks: list[bytes]) -> SimpleNamespace:
    """Stands in for app.state.openai_client, shaped like the real one:
    client.audio.speech.with_streaming_response.create(...) used in
    api.ask_speech.
    """

    class FakeStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def iter_bytes(self):
            for chunk in chunks:
                yield chunk

    return SimpleNamespace(
        audio=SimpleNamespace(
            speech=SimpleNamespace(
                with_streaming_response=SimpleNamespace(create=lambda **kw: FakeStream())
            )
        )
    )


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ask_empty_question_returns_422(client):
   r = client.post("/ask", json={"question": ""})
   assert r.status_code == 422


def test_ask_happy_path_and_cache_hit(client, fake_faq_model):
    with api.app.state.agent.override(model=fake_faq_model):
        r1 = client.post("/ask", json={"question": "Ile kosztuje nadbagaż?"})
        assert r1.status_code == 200
        assert r1.headers["x-cache"] == "MISS"
        assert r1.json() == {"answer": FAQ_ANSWER}

        r2 = client.post("/ask", json={"question": "Ile kosztuje nadbagaż?"})
        assert r2.status_code == 200
        assert r2.headers["x-cache"] == "HIT"
        assert r2.json() == r1.json()


def test_chat_then_delete_session(client, mock_handoff_agents):
    r = client.post("/chat", json={"question": "Ile kosztuje nadbagaż?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == FAQ_ANSWER
    session_id = body["session_id"]

    assert client.delete(f"/chat/{session_id}").status_code == 204
    assert client.delete(f"/chat/{session_id}").status_code == 404


def test_delete_unknown_session_returns_404(client):
    r = client.delete("/chat/does-not-exist")
    assert r.status_code == 404
    assert r.json() == {"detail": "Unknown session_id"}