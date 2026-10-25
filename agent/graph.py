"""
The InvoiceLens agent graph:

    START -> agent --(tool calls)--> tools --(extract_invoice?)--> validate --> agent
               |                       '--(other tools)------------------------> agent
               '--(no tool calls)--> END            validate --(attempts exhausted)--> flag_review -> END

- agent      LLM with the three MCP tools bound. Decides what to call, writes the answer.
- tools      Executes tool calls against the MCP server (LangGraph ToolNode).
- validate   Plain code, no LLM. Reads the server's validation report on each
             extract_invoice result. On failure it hands the exact errors back to the
             agent and asks for a re-extraction with `feedback`; after
             `max_extraction_attempts` failures for a document it stops the loop.
- flag_review  Ends with an explicit "needs human review" answer instead of letting
             the agent present numbers that don't add up.

The validate node is the harness-engineering piece: a sensor inside the agent's
own loop, so the agent self-corrects before anyone sees the output, and a hard
stop when it can't.
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

EXTRACT_TOOL = "invoicelens_extract_invoice"

SYSTEM_PROMPT = """You answer questions about invoices using the InvoiceLens tools.
- To find which invoice answers a question, use invoicelens_search_documents.
- For exact amounts, dates or line items, call invoicelens_extract_invoice with the doc_id;
  don't read numbers off search snippets.
- To add a new PDF, use invoicelens_ingest_document.
- Cite filename and page for every fact. If a tool reports a validation failure, follow
  the validator's instructions. Never adjust numbers yourself."""


class AgentState(MessagesState):
    extraction_attempts: dict[str, int]     # doc_id -> failed validation count
    review_reason: str | None


def _tool_payload(msg: ToolMessage) -> dict[str, Any] | None:
    """MCP tool results arrive as text blocks containing JSON; return the parsed dict."""
    content = msg.content
    blocks = content if isinstance(content, list) else [content]
    for b in blocks:
        text = b.get("text") if isinstance(b, dict) else b
        if isinstance(text, str):
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
    return None


def build_graph(llm: BaseChatModel, tools: list[BaseTool], max_extraction_attempts: int = 2):
    llm_with_tools = llm.bind_tools(tools)

    def agent(state: AgentState) -> dict:
        reply = llm_with_tools.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])
        return {"messages": [reply]}

    def after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

    def _latest_tool_messages(state: AgentState) -> list[ToolMessage]:
        out = []
        for m in reversed(state["messages"]):
            if not isinstance(m, ToolMessage):
                break
            out.append(m)
        return out

    def after_tools(state: AgentState) -> str:
        return ("validate" if any(m.name == EXTRACT_TOOL for m in _latest_tool_messages(state))
                else "agent")

    def validate(state: AgentState) -> dict:
        attempts = dict(state.get("extraction_attempts") or {})
        notes, review = [], None
        for m in _latest_tool_messages(state):
            if m.name != EXTRACT_TOOL:
                continue
            data = _tool_payload(m)
            if not data or "validation" not in data:
                continue                       # tool error: the agent already sees it
            report, doc = data["validation"], data.get("doc_id") or "?"
            if report.get("valid"):
                continue
            attempts[doc] = attempts.get(doc, 0) + 1
            errors = "; ".join(report.get("errors", []))
            if attempts[doc] >= max_extraction_attempts:
                review = (f"{data.get('filename') or doc}: failed validation {attempts[doc]} times "
                          f"— {errors}")
            else:
                notes.append(f"Validation failed for doc_id={doc}: {errors}. Call {EXTRACT_TOOL} "
                             f"again with doc_id='{doc}' and feedback set to these errors.")
        update: dict[str, Any] = {"extraction_attempts": attempts, "review_reason": review}
        if notes and not review:
            update["messages"] = [HumanMessage("[validator] " + " ".join(notes))]
        return update

    def after_validate(state: AgentState) -> str:
        return "flag_review" if state.get("review_reason") else "agent"

    def flag_review(state: AgentState) -> dict:
        return {"messages": [AIMessage(
            "This invoice needs human review: the extracted numbers don't add up, and a "
            f"re-read gave the same result, so the document itself is likely wrong.\n"
            f"{state['review_reason']}")]}

    g = StateGraph(AgentState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(tools))
    g.add_node("validate", validate)
    g.add_node("flag_review", flag_review)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", after_agent, ["tools", END])
    g.add_conditional_edges("tools", after_tools, ["validate", "agent"])
    g.add_conditional_edges("validate", after_validate, ["flag_review", "agent"])
    g.add_edge("flag_review", END)
    return g.compile()
