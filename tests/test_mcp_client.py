"""MCP 客户端传输层的单元测试。

这个模块从未在真实服务上跑过，全部行为都是从 MCP Streamable HTTP 文档推出来的，
所以测试要钉住三件事：

1. 服务端两种返回格式（``application/json`` 与 ``text/event-stream``）都能解析；
2. 错误不被吞掉——JSON-RPC error、``isError``、HTTP 401/429 都要变成可展示的 ``McpError``；
3. ``Mcp-Session-Id`` 只在首个响应里出现一次，后续请求必须回带，否则服务端会当新会话。

全部用例用 ``unittest.mock`` 替换 ``urllib.request.urlopen``，不发任何真实网络请求。
"""

from __future__ import annotations

import io
import json
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcpareto.errors import McpError
from mcpareto.mcp_client import McpClient

MCP_URL = "https://mcp.example.test/mcp"
TOKEN = "test-token"


class FakeResponse:
    """足够像 ``http.client.HTTPResponse``：支持 context manager、headers.get、read。"""

    def __init__(self, body: str = "", headers: dict[str, str] | None = None) -> None:
        self._body = body.encode("utf-8")
        self.headers = headers or {}

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc_info) -> bool:
        return False


def json_response(result: dict, *, headers: dict[str, str] | None = None) -> FakeResponse:
    """构造一条 JSON-RPC 成功响应。"""
    envelope = {"jsonrpc": "2.0", "id": 1, "result": result}
    return FakeResponse(json.dumps(envelope, ensure_ascii=False), headers)


def text_content(*texts: str) -> dict:
    """构造 ``tools/call`` 的 result，每个字符串是一条 text content。"""
    return {"content": [{"type": "text", "text": t} for t in texts]}


def http_error(code: int, body: str = "") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        MCP_URL, code, "server error", {}, io.BytesIO(body.encode("utf-8"))
    )


def header_of(request, name: str):
    """urllib 会把头名重写成 ``Xxx-yyy`` 形式，比较时统一小写。"""
    return {k.lower(): v for k, v in request.headers.items()}.get(name.lower())


class McpTestCase(unittest.TestCase):
    """公共夹具：关掉节流，避免用例里出现真实 sleep。"""

    def setUp(self) -> None:
        self.client = McpClient(MCP_URL, TOKEN, timeout=5.0)
        self.client._min_interval = 0.0  # 节流不在本文件覆盖范围，避免 0.11s/次的空转

    def call_tool_with(self, response: FakeResponse):
        with mock.patch("urllib.request.urlopen", return_value=response):
            return self.client.call_tool("query-meals")


class TestParseSse(McpTestCase):
    """SSE 解析：取最后一个携带 result/error 的事件。"""

    def test_single_data_event_parsed(self):
        raw = 'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"tools":[]}}\n\n'
        message = McpClient._parse_sse(raw)
        self.assertEqual(message["result"], {"tools": []})

    def test_multiple_events_last_result_wins(self):
        raw = (
            'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"stage":"first"}}\n\n'
            'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"stage":"last"}}\n\n'
        )
        message = McpClient._parse_sse(raw)
        assert message is not None
        self.assertEqual(message["result"], {"stage": "last"})

    def test_event_without_result_does_not_overwrite(self):
        """中间夹一个 notification 事件时，不能把它当成最终结果。"""
        raw = (
            'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n\n'
            'event: message\ndata: {"jsonrpc":"2.0","method":"notifications/progress"}\n\n'
        )
        message = McpClient._parse_sse(raw)
        assert message is not None
        self.assertEqual(message["result"], {"ok": True})

    def test_done_sentinel_skipped(self):
        raw = (
            'event: message\ndata: {"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n\n'
            "data: [DONE]\n"
        )
        message = McpClient._parse_sse(raw)
        assert message is not None
        self.assertEqual(message["result"], {"ok": True})

    def test_non_json_data_line_skipped(self):
        raw = 'event: ping\ndata: not-a-json\n\n' 'data: {"jsonrpc":"2.0","id":1,"result":{"ok":true}}\n'
        message = McpClient._parse_sse(raw)
        assert message is not None
        self.assertEqual(message["result"], {"ok": True})

    def test_no_valid_result_raises(self):
        for raw in ("", "event: ping\n\n", 'data: {"jsonrpc":"2.0","id":1}\n', "data: [DONE]\n"):
            with self.subTest(raw=raw):
                with self.assertRaises(McpError):
                    McpClient._parse_sse(raw)

    def test_sse_transport_response_is_parsed(self):
        """Content-Type 为 text/event-stream 时，_post 应走 SSE 分支而不是 json.loads。"""
        body = 'event: message\ndata: ' + json.dumps(
            {"jsonrpc": "2.0", "id": 1, "result": text_content('{"items": [1]}')},
            ensure_ascii=False,
        ) + "\n"
        out = self.call_tool_with(FakeResponse(body, headers={"Content-Type": "text/event-stream"}))
        self.assertEqual(out, {"items": [1]})


class TestRpc(McpTestCase):
    """JSON-RPC 层的错误与通知语义。"""

    def test_error_field_raises_with_server_message(self):
        envelope = {
            "jsonrpc": "2.0",
            "id": 1,
            "error": {"code": -32601, "message": "Method not found"},
        }
        with mock.patch("urllib.request.urlopen", return_value=FakeResponse(json.dumps(envelope))):
            with self.assertRaises(McpError) as ctx:
                self.client.list_tools()
        self.assertIn("Method not found", str(ctx.exception))

    def test_initialize_sends_notification_without_id(self):
        """通知报文不带 id，也不等待响应；这是握手成功的必要条件。"""
        responses = [
            json_response({"protocolVersion": "2025-03-26"}, headers={"Mcp-Session-Id": "sess-1"}),
            FakeResponse(""),
        ]
        with mock.patch("urllib.request.urlopen", side_effect=responses) as urlopen:
            self.client.initialize()

        bodies = [json.loads(call.args[0].data.decode("utf-8")) for call in urlopen.call_args_list]
        self.assertEqual([b["method"] for b in bodies], ["initialize", "notifications/initialized"])
        self.assertNotIn("id", bodies[1])


class TestCallTool(McpTestCase):
    """tool 结果的四种形态。"""

    def test_single_json_text_returns_parsed_dict(self):
        out = self.call_tool_with(json_response(text_content('{"meals": [{"name": "汉堡"}]}')))
        self.assertEqual(out, {"meals": [{"name": "汉堡"}]})

    def test_single_non_json_text_wrapped_as_raw(self):
        out = self.call_tool_with(json_response(text_content("调用成功")))
        self.assertEqual(out, {"_raw": "调用成功"})

    def test_multiple_texts_returned_as_items(self):
        out = self.call_tool_with(json_response(text_content('{"a": 1}', "plain")))
        self.assertEqual(out, {"_items": [{"a": 1}, "plain"]})

    def test_is_error_raises(self):
        result = {"isError": True, "content": [{"type": "text", "text": "store not found"}]}
        with self.assertRaises(McpError) as ctx:
            self.call_tool_with(json_response(result))
        self.assertIn("query-meals", str(ctx.exception))

    def test_result_without_text_content_returned_as_is(self):
        result = {"content": [{"type": "image", "data": "base64..."}]}
        out = self.call_tool_with(json_response(result))
        self.assertEqual(out, result)


class TestHttpErrorMapping(McpTestCase):
    """HTTP 状态码要映射成用户能照着做的提示。"""

    def test_401_points_to_token_renewal(self):
        with mock.patch(
            "urllib.request.urlopen", side_effect=http_error(401, '{"message":"expired"}')
        ):
            with self.assertRaises(McpError) as ctx:
                self.client.call_tool("query-meals")
        self.assertIn("重新获取", str(ctx.exception))

    def test_429_points_to_rate_limit(self):
        with mock.patch("urllib.request.urlopen", side_effect=http_error(429, "too many")):
            with self.assertRaises(McpError) as ctx:
                self.client.call_tool("query-meals")
        self.assertIn("限流", str(ctx.exception))

    def test_other_status_keeps_code_and_body(self):
        with mock.patch("urllib.request.urlopen", side_effect=http_error(500, "boom")):
            with self.assertRaises(McpError) as ctx:
                self.client.call_tool("query-meals")
        message = str(ctx.exception)
        self.assertIn("500", message)
        self.assertIn("boom", message)


class TestSessionAndRequestShape(McpTestCase):
    """会话保持与请求报文形态。"""

    def test_session_id_sent_on_subsequent_requests(self):
        """首个响应下发 Mcp-Session-Id 后，后续请求必须回带，否则每次都是新会话。"""
        responses = [
            json_response(text_content('{"ok": true}'), headers={"Mcp-Session-Id": "sess-abc"}),
            json_response(text_content('{"ok": true}')),
        ]
        with mock.patch("urllib.request.urlopen", side_effect=responses) as urlopen:
            self.client.call_tool("query-meals")
            self.client.call_tool("query-meals")

        first, second = (call.args[0] for call in urlopen.call_args_list)
        self.assertIsNone(header_of(first, "Mcp-Session-Id"))
        self.assertEqual(header_of(second, "Mcp-Session-Id"), "sess-abc")
        self.assertEqual(self.client.session_id, "sess-abc")

    def test_request_is_jsonrpc_post_with_bearer_token(self):
        with mock.patch(
            "urllib.request.urlopen", return_value=json_response(text_content('{"ok": true}'))
        ) as urlopen:
            self.client.call_tool("calculate-price", {"storeCode": "S1"})

        request = urlopen.call_args.args[0]
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.full_url, MCP_URL)
        self.assertEqual(header_of(request, "Authorization"), f"Bearer {TOKEN}")
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["jsonrpc"], "2.0")
        self.assertEqual(body["method"], "tools/call")
        self.assertEqual(
            body["params"], {"name": "calculate-price", "arguments": {"storeCode": "S1"}}
        )
        self.assertIsInstance(body["id"], int)


if __name__ == "__main__":
    unittest.main(verbosity=2)
