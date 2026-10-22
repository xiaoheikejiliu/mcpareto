#!/usr/bin/env python3
"""从麦当劳 MCP 采集真实菜单与营养数据，落地为 data/menu.json。

用法::

    export MCD_MCP_TOKEN="你的 token"
    python3 scripts/collect.py --address "深圳北站" --out data/menu.json

设计取舍：餐厅 MCP 的 Tool 入参在不同版本间会调整，因此参数不是硬编码的，
而是从 ``tools/list`` 返回的 inputSchema 里推断必填项并自动填充；遇到无法推断
的必填字段时明确报错并提示用 ``--extra-args`` 补，而不是带着错误参数静默跑完。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcpareto.errors import McpError
from mcpareto.mcp_client import McpClient
from mcpareto.models import MenuItem
from mcpareto.normalize import find_records, merge_nutrition, parse_item

DEFAULT_URL = "https://mcp.mcd.cn"


def build_arguments(schema: dict, known: dict[str, object]) -> tuple[dict, list[str]]:
    """按 schema 的必填字段组装参数，返回 (参数, 未满足的必填字段)。"""
    props = (schema or {}).get("properties", {}) or {}
    required = (schema or {}).get("required", []) or []
    args: dict[str, object] = dict(known)
    missing: list[str] = []
    for field in required:
        if field in args:
            continue
        lowered = field.lower()
        if "store" in lowered or "restaurant" in lowered:
            if known.get("store_code"):
                args[field] = known["store_code"]
            else:
                missing.append(field)
        elif "address" in lowered or "location" in lowered:
            if known.get("address"):
                args[field] = known["address"]
            else:
                missing.append(field)
        elif "city" in lowered and known.get("city"):
            args[field] = known["city"]
        elif "type" in lowered or "scene" in lowered:
            args[field] = known.get("be_type", 1)
        else:
            missing.append(field)
    # 保留 schema 里出现、且我们已知道值的可选字段
    for prop in props:
        if prop in known and prop not in args:
            args[prop] = known[prop]
    return args, missing


def call_with_schema(
    client: McpClient,
    tools: dict[str, dict],
    name: str,
    known: dict[str, object],
    extra: dict | None = None,
) -> dict | None:
    if name not in tools:
        print(f"  ! 服务端未提供工具 {name}，跳过", flush=True)
        return None
    merged = dict(known)
    if extra:
        merged.update(extra)
    args, missing = build_arguments(tools[name].get("inputSchema", {}), merged)
    if missing:
        raise McpError(
            f"{name} 有无法自动推断的必填参数 {missing}。"
            f"请用 --extra-args 显式指定，例如 "
            f'''--extra-args '{{"{missing[0]}": "值"}}' '''
        )
    print(f"  → {name} {json.dumps(args, ensure_ascii=False)}", flush=True)
    return client.call_tool(name, args)


def main() -> int:
    parser = argparse.ArgumentParser(description="从麦当劳 MCP 采集菜单与营养数据")
    parser.add_argument("--url", default=os.environ.get("MCD_MCP_URL", DEFAULT_URL))
    parser.add_argument("--token", default=os.environ.get("MCD_MCP_TOKEN", ""))
    parser.add_argument("--address", required=True, help="用于查询附近门店的地址关键词")
    parser.add_argument("--store-code", default=None, help="直接指定门店编码，跳过搜索")
    parser.add_argument("--out", default="data/menu.json")
    parser.add_argument(
        "--extra-args",
        default="{}",
        help="JSON 字符串，补充无法自动推断的工具参数",
    )
    parser.add_argument(
        "--skip-price",
        action="store_true",
        help="跳过逐项算价（速度快，但价格来自菜单接口，可能不含优惠）",
    )
    args = parser.parse_args()

    if not args.token:
        print(
            "缺少 MCP Token。请先到 https://open.mcd.cn/mcp 登录，"
            "在控制台点击「激活」领取，然后 export MCD_MCP_TOKEN=...",
            file=sys.stderr,
        )
        return 2

    extra = json.loads(args.extra_args)
    client = McpClient(args.url, args.token)

    print("1/5 初始化 MCP 会话", flush=True)
    info = client.initialize()
    print(f"    服务端: {json.dumps(info.get('serverInfo', {}), ensure_ascii=False)}", flush=True)

    print("2/5 读取工具清单", flush=True)
    tools = {t["name"]: t for t in client.list_tools()}
    print(f"    可用工具 {len(tools)} 个: {', '.join(sorted(tools))}", flush=True)

    known: dict[str, object] = {"address": args.address}
    if args.store_code:
        known["store_code"] = args.store_code
    else:
        print("3/5 查询附近门店", flush=True)
        stores = call_with_schema(client, tools, "query-nearby-stores", known, extra)
        store_records = find_records(stores) if stores else []
        if not store_records:
            raise McpError("未查到可用门店，请换一个更具体的地址，或用 --store-code 指定")
        first = store_records[0]
        store_code = (
            first.get("storeCode")
            or first.get("store_code")
            or first.get("code")
            or first.get("id")
        )
        known["store_code"] = store_code
        known["city"] = first.get("city") or first.get("cityName")
        print(
            f"    选中门店: {first.get('storeName') or first.get('name')} ({store_code})",
            flush=True,
        )

    print("4/5 拉取菜单与营养表", flush=True)
    meals_payload = call_with_schema(client, tools, "query-meals", known, extra)
    items: list[MenuItem] = []
    for record in find_records(meals_payload):
        parsed = parse_item(record)
        if parsed is not None:
            items.append(parsed)
    print(f"    菜单条目 {len(items)} 个", flush=True)

    nutrition_payload = call_with_schema(client, tools, "list-nutrition-foods", known, extra)
    if nutrition_payload is not None:
        items, unmatched = merge_nutrition(items, nutrition_payload)
        print(
            f"    营养匹配 {len(items) - len(unmatched)}/{len(items)}"
            f"{f'，未匹配 {len(unmatched)} 个' if unmatched else ''}",
            flush=True,
        )

    if not args.skip_price:
        print("5/5 逐项试算价格（可加 --skip-price 跳过）", flush=True)
        priced = 0
        for index, item in enumerate(items, 1):
            try:
                quote = client.call_tool(
                    "calculate-price",
                    {
                        **{k: v for k, v in known.items() if k in ("store_code", "address")},
                        "items": [{"itemCode": item.item_code, "quantity": 1}],
                    },
                )
                total = find_records(quote)
                amount = None
                if total:
                    amount = total[0].get("totalPrice") or total[0].get("payPrice")
                if amount:
                    item.price_cents = int(amount)
                    priced += 1
            except McpError as exc:
                if index == 1:
                    print(f"    算价失败（{exc}），保留菜单接口价格", flush=True)
                    break
            if index % 20 == 0:
                print(f"    已处理 {index}/{len(items)}", flush=True)
        print(f"    成功算价 {priced} 项", flush=True)
    else:
        print("5/5 已跳过算价", flush=True)

    payload = {
        "schema_version": 1,
        "data_source": "mcd-mcp",
        "collected_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "store": {
            "code": known.get("store_code"),
            "city": known.get("city"),
            "address_query": args.address,
        },
        "items": [
            {
                "item_code": i.item_code,
                "name": i.name,
                "category": i.category,
                "price_cents": i.price_cents,
                "tags": i.tags,
                "nutrition": {
                    "energy_kcal": i.nutrition.energy_kcal,
                    "protein_g": i.nutrition.protein_g,
                    "fat_g": i.nutrition.fat_g,
                    "carb_g": i.nutrition.carb_g,
                    "sodium_mg": i.nutrition.sodium_mg,
                    "calcium_mg": i.nutrition.calcium_mg,
                },
            }
            for i in items
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {out_path}（{len(items)} 项）", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except McpError as exc:
        print(f"\n采集失败: {exc}", file=sys.stderr)
        raise SystemExit(1)
