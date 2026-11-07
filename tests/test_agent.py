"""エージェントの配線テスト。

TestModel は実際のLLMを呼ばない偽モデルで、指定したツール（既定では全部）を呼んでから
ダミーの回答を返す。APIキーも利用枠も使わずに、
「ツールが登録され、呼ばれ、結果がモデルに戻るか」を確かめられる。
"""

import pytest
from pydantic_ai import DeferredToolRequests, DeferredToolResults, ToolDenied
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from app import tools
from app.agent import agent

BASE_TOOLS = {"search_glossary", "get_lecture_summary"}


def _parts(result, kind):
    return [p for m in result.all_messages() for p in m.parts if isinstance(p, kind)]


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    """本物の learning_log/ を汚さないよう、保存先を一時フォルダに差し替える。"""
    d = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", d)
    return d


def test_base_tools_are_registered_and_called():
    # 承認が必要な save_learning_log は呼ばせない（下の専用テストで確かめる）
    model = TestModel(call_tools=sorted(BASE_TOOLS))
    with agent.override(model=model):
        result = agent.run_sync("LoRAとは？")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert BASE_TOOLS <= offered  # 演習でツールを足しても壊れないよう「含む」で判定

    called = {p.tool_name for p in _parts(result, ToolCallPart)}
    assert BASE_TOOLS <= called
    assert len(_parts(result, ToolReturnPart)) == len(_parts(result, ToolCallPart))
    assert result.output


def test_glossary_result_reaches_the_model():
    model = TestModel(call_tools=["search_glossary"])
    with agent.override(model=model):
        result = agent.run_sync("RAGとは？")
    returns = _parts(result, ToolReturnPart)
    assert returns and returns[0].tool_name == "search_glossary"


def test_save_learning_log_is_offered_by_test_model(log_dir):
    model = TestModel(call_tools=["save_learning_log"])
    with agent.override(model=model):
        result = agent.run_sync("LoRAについて学習したことを保存して")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert "save_learning_log" in offered
    # 承認前は回答ではなく「承認待ち」が返り、ファイルはまだ作られない
    assert isinstance(result.output, DeferredToolRequests)
    assert [c.tool_name for c in result.output.approvals] == ["save_learning_log"]
    assert not log_dir.exists()


def _run_with_decision(decision):
    """1回目で承認待ちを受け取り、decision（True か ToolDenied）を渡して続きを実行する。"""
    model = TestModel(call_tools=["save_learning_log"])
    with agent.override(model=model):
        first = agent.run_sync("LoRAについて学習したことを保存して")
        results = DeferredToolResults()
        for call in first.output.approvals:
            results.approvals[call.tool_call_id] = decision
        return agent.run_sync(message_history=first.all_messages(), deferred_tool_results=results)


def test_save_learning_log_saves_when_approved(log_dir):
    result = _run_with_decision(True)
    assert isinstance(result.output, str)
    assert len(list(log_dir.glob("*.md"))) == 1


def test_save_learning_log_does_not_save_when_denied(log_dir):
    result = _run_with_decision(ToolDenied("利用者が許可しなかったため、保存していません。"))
    assert isinstance(result.output, str)
    assert not log_dir.exists()
