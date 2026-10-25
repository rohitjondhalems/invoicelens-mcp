"""
The LangGraph agent against the real MCP server (in-memory), with a scripted
LLM so routing is tested deterministically and offline. The fake model only
decides *which* tool calls to make; every tool actually runs on the server.
"""
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp.shared.memory import create_connected_server_and_client_session

from agent.graph import build_graph
from invoicelens.mcp_server import build_server
from invoicelens.service import doc_id_for

from .conftest import ACME, INITECH, sample


class ScriptedLLM(BaseChatModel):
    """Returns the next scripted AIMessage on each call; records what it was shown."""
    script: list
    seen: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        return ChatResult(generations=[ChatGeneration(message=self.script.pop(0))])


def call(name, **args):
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"call_{len(args)}_{name[-6:]}"}])


async def run(loaded_service, script, question):
    llm = ScriptedLLM(script=script, seen=[])
    async with create_connected_server_and_client_session(build_server(loaded_service)._mcp_server) as session:
        graph = build_graph(llm, await load_mcp_tools(session))
        state = await graph.ainvoke({"messages": [HumanMessage(question)]})
    return state, llm


async def test_valid_invoice_flows_straight_to_answer(loaded):
    acme = doc_id_for(sample(ACME))
    state, llm = await run(loaded, [
        call("invoicelens_search_documents", query="Acme invoice total", k=3),
        call("invoicelens_extract_invoice", doc_id=acme),
        AIMessage("The Acme invoice total is EUR 3,369.49 (acme_supplies_2026-03.pdf, page 1)."),
    ], "What's the total on the Acme invoice?")
    assert state["messages"][-1].content.startswith("The Acme invoice total is EUR 3,369.49")
    assert not any("[validator]" in str(m.content) for m in state["messages"])
    assert state["review_reason"] is None and len(llm.seen) == 3


async def test_failed_validation_feeds_back_then_flags_for_review(loaded):
    initech = doc_id_for(sample(INITECH))
    state, llm = await run(loaded, [
        call("invoicelens_extract_invoice", doc_id=initech),
        call("invoicelens_extract_invoice", doc_id=initech, feedback="Line items sum to 5,700.00 ..."),
        AIMessage("should never be reached"),
    ], "What's the total on the Initech invoice?")

    # 1st failure: the validator's exact error went back to the agent, with instructions
    second_call_context = llm.seen[1]
    feedback = [m for m in second_call_context if "[validator]" in str(m.content)]
    assert feedback and "Line items sum to 5,700.00 but subtotal is 5,900.00" in feedback[0].content
    assert f"doc_id='{initech}'" in feedback[0].content

    # 2nd failure: hard stop — the LLM is not asked again, the graph flags review
    assert len(llm.seen) == 2
    final = state["messages"][-1].content
    assert final.startswith("This invoice needs human review")
    assert "initech_software_2026-06.pdf" in final and state["extraction_attempts"] == {initech: 2}
