"""零依赖的 MCP Streamable HTTP 客户端。

只实现本项目需要的部分：initialize → notifications/initialized → tools/list →
tools/call。不引入第三方 SDK，评审者克隆仓库后无需 pip install 任何依赖即可运行。

协议要点（MCP 2025-03-26 Streamable HTTP）：
* 单一端点，POST JSON-RPC 2.0 报文；
* 服务端可能以 ``application/json`` 直接返回，也可能返回 ``text/event-stream``；
* 首个响应可能带 ``Mcp-Session-Id``，后续请求需原样带上。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from .errors import McpError


class McpClient:
    """面向麦当劳 MCP 服务的最小客户端。"""

    def __init__(self, url: str, token: str, *, timeout: float = 30.0) -> None:
        self.url = url
        self._token = token
        self.timeout = timeout
        self.session_id: str | None = None
        self._msg_id = 0
        self._last_call_at = 0.0
        self._min_interval = 0.11  # 留足余量，控制在 600 次/分钟以内

    # ---------------------------------------------------------------- 传输层

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call_at
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call_at = time.monotonic()

    def _post(self, payload: dict, *, expect_response: bool = True) -> dict | None:
        self._throttle()
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {self._token}",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id

        req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                sid = resp.headers.get("Mcp-Session-Id")
                if sid:
                    self.session_id = sid
                ctype = resp.headers.get("Content-Type", "")
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")[:400]
            if exc.code == 401:
                raise McpError(
                    "MCP Token 无效或已过期，请到 open.mcd.cn/mcp 控制台重新获取"
                ) from exc
            if exc.code == 429:
                raise McpError("触发限流（600 次/分钟），请降低调用频率") from exc
            raise McpError(f"HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise McpError(f"网络不可达: {exc.reason}") from exc

        if not expect_response or not raw.strip():
            return None
        if "text/event-stream" in ctype:
            return self._parse_sse(raw)
        return json.loads(raw)

    @staticmethod
    def _parse_sse(raw: str) -> dict | None:
        """从 SSE 流中取出最后一个携带 result 或 error 的事件。"""
        result: dict | None = None
        for line in raw.splitlines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                msg = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if "result" in msg or "error" in msg:
                result = msg
        if result is None:
            raise McpError("SSE 响应中未找到有效 JSON-RPC 结果")
        return result

    def _rpc(self, method: str, params: dict | None = None, *, notify: bool = False) -> dict:
        payload: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        if not notify:
            payload["id"] = self._next_id()

        resp = self._post(payload, expect_response=not notify)
        if notify:
            return {}
        assert resp is not None
        if "error" in resp:
            err = resp["error"]
            raise McpError(f"{method} 失败: {err.get('message', err)}")
        return resp.get("result", {})

    # ---------------------------------------------------------------- 协议层

    def initialize(self) -> dict:
        result = self._rpc(
            "initialize",
            {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "mcpareto", "version": "0.1.0"},
            },
        )
        self._rpc("notifications/initialized", notify=True)
        return result

    def list_tools(self) -> list[dict]:
        result = self._rpc("tools/list")
        return result.get("tools", [])

    def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        """调用工具并返回解析后的业务数据。

        MCP 的 tool 结果通常形如 ``{"content": [{"type": "text", "text": "{...}"}]}``。
        不同服务端可能直接给结构化 content，这里的处理顺序是：
        单条 text 且可解析为 JSON → 返回该 JSON；否则返回原始 content 列表。
        """
        result = self._rpc("tools/call", {"name": name, "arguments": arguments or {}})
        if result.get("isError"):
            raise McpError(f"工具 {name} 返回错误: {result}")

        content = result.get("content", [])
        texts = [c.get("text", "") for c in content if c.get("type") == "text"]
        if len(texts) == 1:
            try:
                return json.loads(texts[0])
            except json.JSONDecodeError:
                return {"_raw": texts[0]}
        if texts:
            parsed = []
            for t in texts:
                try:
                    parsed.append(json.loads(t))
                except json.JSONDecodeError:
                    parsed.append(t)
            return {"_items": parsed}
        return result
