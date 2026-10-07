"""The optional MCP adapter must stay on the same data/service path as the API."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_mcp_has_no_second_synthetic_corpus_or_hard_coded_lab_ids():
    server = (ROOT / "src/mcp_server/server.py").read_text()
    planes = (ROOT / "src/mcp_server/planes.py").read_text()
    assert "DEMO_NOTES" not in server + planes
    assert "DEMO_LABS" not in server + planes
    assert "DEMO_TIMELINE" not in server + planes
    assert "LAB_ITEMIDS" not in server


def test_mcp_reuses_retrieval_and_structured_lab_services():
    server = (ROOT / "src/mcp_server/server.py").read_text()
    assert "HybridRetriever" in server
    assert "LabResolver" in server
    assert "resolver.fetch" in server


def test_mcp_server_starts_and_lists_its_tools_over_stdio():
    """Live smoke test: spawn the server, complete the MCP handshake, list tools.
    Needs no database and loads no model: tools are only listed, and the one call is to a
    tool that does not exist."""
    import asyncio
    import os
    import sys

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def session():
        params = StdioServerParameters(command=sys.executable, args=["-m", "src.mcp_server.server"],
                                       env={**os.environ, "LUMEN_DATA_PLANE": "demo"}, cwd=str(ROOT))
        with open(os.devnull, "w") as quiet:
            async with stdio_client(params, errlog=quiet) as (read, write):
                async with ClientSession(read, write) as client:
                    init = await client.initialize()
                    tools = await client.list_tools()
                    unknown = await client.call_tool("no_such_tool", {})
                    return init.server_info.name, {t.name: t.input_schema for t in tools.tools}, unknown.is_error

    name, tools, unknown_is_error = asyncio.run(asyncio.wait_for(session(), timeout=60))
    assert name == "lumen"
    assert set(tools) == {"search_patient_notes", "search_guidelines", "get_lab_trend", "get_patient_timeline"}
    assert tools["get_lab_trend"]["required"] == ["subject_id", "lab_name"]
    assert unknown_is_error is True                      # an error comes back through MCP; the server keeps running


def test_research_plane_refuses_any_transport_but_stdio(monkeypatch):
    import pytest
    from src.mcp_server import planes

    monkeypatch.setattr(planes, "PLANE", "research")
    planes.enforce_transport("stdio")
    for transport in ("sse", "streamable-http"):
        with pytest.raises(planes.PlaneViolation):
            planes.enforce_transport(transport)
    with pytest.raises(planes.PlaneViolation):
        planes.enforce_transport("stdio", host="0.0.0.0")
