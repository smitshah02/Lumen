"""
MCP smoke client
================
Launches the server over stdio and exercises every tool — the same
transport Claude Desktop uses, without needing Node.

    LUMEN_DATA_PLANE=demo     python -m src.mcp_server.test_client
    LUMEN_DATA_PLANE=research python -m src.mcp_server.test_client --subject 10000032
"""

from __future__ import annotations

import os
import sys
import json
import asyncio
import argparse

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def show(title: str, result) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
    for block in result.content:
        txt = getattr(block, "text", str(block))
        try:
            parsed = json.loads(txt)
            print(json.dumps(parsed, indent=2)[:1800])
        except Exception:
            print(txt[:1800])


async def run(subject: int, lab: str) -> None:
    plane = os.environ.get("LUMEN_DATA_PLANE", "research")
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.mcp_server.server"],
        env={**os.environ, "LUMEN_DATA_PLANE": plane},
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print(f"\n  plane={plane}   {len(tools.tools)} tools registered:")
            for t in tools.tools:
                print(f"    - {t.name}")

            show("search_patient_notes", await session.call_tool(
                "search_patient_notes",
                {"query": "recent potassium and kidney function", "subject_id": subject, "top_k": 3}))

            show("get_lab_trend", await session.call_tool(
                "get_lab_trend", {"subject_id": subject, "lab_name": lab, "limit": 8}))

            show("get_patient_timeline", await session.call_tool(
                "get_patient_timeline", {"subject_id": subject, "limit": 12}))

            if plane != "demo":
                show("search_guidelines", await session.call_tool(
                    "search_guidelines",
                    {"query": "potassium monitoring in chronic kidney disease", "top_k": 2}))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=None)
    ap.add_argument("--lab", default="potassium")
    args = ap.parse_args()
    subject = args.subject if args.subject is not None else (
        90001 if os.environ.get("LUMEN_DATA_PLANE") == "demo" else 10000032)
    asyncio.run(run(subject, args.lab))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())