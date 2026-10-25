"""
Ask the InvoiceLens agent a question from the command line.

    # server running separately (invoicelens-serve), Azure OpenAI chat model:
    python -m agent.run "What is the total on the Acme invoice?"

    # or let the agent launch the server itself over stdio:
    python -m agent.run --stdio "Which invoice has a late payment penalty?"

Env: INVOICELENS_MCP_URL (default http://127.0.0.1:8765/mcp), INVOICELENS_API_TOKEN,
     AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_CHAT_DEPLOYMENT,
     AZURE_OPENAI_API_VERSION.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.graph import build_graph


def connection(stdio: bool) -> dict:
    if stdio:
        return {"transport": "stdio", "command": sys.executable,
                "args": ["-m", "invoicelens.mcp_server", "--transport", "stdio"]}
    conn = {"transport": "streamable_http",
            "url": os.getenv("INVOICELENS_MCP_URL", "http://127.0.0.1:8765/mcp")}
    if token := os.getenv("INVOICELENS_API_TOKEN"):
        conn["headers"] = {"Authorization": f"Bearer {token}"}
    return conn


def make_llm():
    from langchain_openai import AzureChatOpenAI

    return AzureChatOpenAI(azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o"),
                           api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
                           temperature=0)


async def ask(question: str, stdio: bool = False) -> str:
    client = MultiServerMCPClient({"invoicelens": connection(stdio)})
    graph = build_graph(make_llm(), await client.get_tools())
    result = await graph.ainvoke({"messages": [("user", question)]})
    return result["messages"][-1].content


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--stdio", action="store_true", help="launch the MCP server as a subprocess")
    args = ap.parse_args()
    print(asyncio.run(ask(args.question, args.stdio)))


if __name__ == "__main__":
    main()
