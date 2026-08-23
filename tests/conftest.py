from __future__ import annotations

from collections import OrderedDict

import pytest
from fastapi.testclient import TestClient
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

import agents.handoff as handoff_mod
import api
import storage
import utils
from agents.guardian.agent import GuardianVerdict
from agents.triage.agent import Triage

FAQ_ANSWER = "Nadbagaż kosztuje dodatkowo, sprawdź FAQ."



@pytest.fixture
def client(tmp_path, monkeypatch):
    """Isolated TestClient: fake API key, empty cache, throwaway sessions DB
    and qa_log — none of this should ever touch the real files on disk
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    monkeypatch.setattr(storage, "_DB_PATH", tmp_path / "sessions.db")
    monkeypatch.setattr(storage, "_ask_cache", dict())
    monkeypatch.setattr(utils, "_QA_LOG_PATH", tmp_path / "qa_log.jsonl")
    with TestClient(api.app) as c:
        yield c


@pytest.fixture
def fake_faq_model() -> TestModel:
    """Stands in for the real OpenAI model behind app.state.agent."""
    return TestModel(custom_output_text=FAQ_ANSWER)

@pytest.fixture
def mock_handoff_agents(monkeypatch):
    """Replaces the guardian/triage/FAQ agents that /chat builds per-request.

    Patched on `agents.handoff` (where the names are looked up at call
    time), not on the modules that define them 
    """

    def fake_create_guardian_agent(settings):
        return Agent(
            TestModel(custom_output_args={"triggered": False, "reason": "ok"}),
            output_type=GuardianVerdict,
        )

    def fake_create_triage_agent(settings):
        return Agent(
            TestModel(custom_output_args={"target": "faq", "reason": "pytanie ogólne"}),
            output_type=Triage,
        )

    def fake_create_agent(settings):
        return Agent(TestModel(custom_output_text=FAQ_ANSWER))

    monkeypatch.setattr(handoff_mod, "create_guardian_agent", fake_create_guardian_agent)
    monkeypatch.setattr(handoff_mod, "create_triage_agent", fake_create_triage_agent)
    monkeypatch.setattr(handoff_mod, "create_agent", fake_create_agent)