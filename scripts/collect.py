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
from mcpareto.normalize import (
    PRICE_UNIT_CENT,
    PRICE_UNIT_YUAN,
    find_records,
    infer_energy_unit,
    infer_price_unit,
    merge_nutrition,
    parse_item,
    read_price,
)

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
        else:
            # 这里**不发明默认值**。``mealType`` / ``orderType`` 之类的字段一旦
            # 被随便填个 1，接口只会返回某个子集菜单：采集「成功」、页面正常、
            # 结论却建立在残缺数据上，是最难发现的一类错误。宁可报错让人显式
            # 用 --extra-args 指定。
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


def quote_amount_cents(record: dict) -> int | None:
    """从 ``calculate-price`` 的返回记录里取出总价（分）。

    返回结构可能是裸数字，也可能包成 ``{"amount": 2000, "currency": "CNY"}``。
    早期实现直接 ``int(amount)``，遇到后者抛 TypeError 逃出 ``except McpError``，
    整段采集直接崩；即便被吞掉，也只会表现为「价格来自菜单」，没有任何提示。
    """
    raw = record.get("totalPrice") or record.get("payPrice") or record.get("amount")
    if isinstance(raw, dict):
        raw = raw.get("amount") or raw.get("value") or raw.get("cent") or raw.get("price")
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return int(round(float(raw)))
    if isinstance(raw, str):
        try:
            return int(round(float(raw.replace(",", "").strip())))
        except ValueError:
            return None
    return None


def nutrition_coverage(items: list[MenuItem]) -> dict[str, int]:
    """各营养字段真正拿到值的条数，用来暴露「匹配 N/N 但其实全空」这种情况。"""
    return {
        "energy_kcal": sum(1 for i in items if i.nutrition.energy_kcal is not None),
        "protein_g": sum(1 for i in items if i.nutrition.protein_g is not None),
        "fat_g": sum(1 for i in items if i.nutrition.fat_g is not None),
        "carb_g": sum(1 for i in items if i.nutrition.carb_g is not None),
        "sodium_mg": sum(1 for i in items if i.nutrition.sodium_mg is not None),
        "calcium_mg": sum(1 for i in items if i.nutrition.calcium_mg is not None),
    }


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
    records = find_records(meals_payload)
    readings = [r for r in (read_price(rec) for rec in records) if r is not None]
    # 单位是接口级全局属性，必须整表判定后再逐条换算（审查 S-4 / S-5）。
    price_unit = infer_price_unit(readings)
    energy_unit = infer_energy_unit(records)
    print(
        f"    单位判定：价格裸整数按{'分' if price_unit == PRICE_UNIT_CENT else '元'}，"
        f"无单位能量按{'千焦' if energy_unit == 'kj' else '千卡'}",
        flush=True,
    )

    items: list[MenuItem] = []
    dropped_names = 0
    for record in records:
        parsed = parse_item(record, price_unit=price_unit, energy_unit=energy_unit)
        if parsed is None:
            dropped_names += 1
        else:
            items.append(parsed)
    print(f"    菜单条目 {len(items)} 个{f'（丢弃 {dropped_names} 条无名称记录）' if dropped_names else ''}", flush=True)
    if not items:
        raise McpError(
            "query-meals 没有解析出任何可用餐品（记录 "
            f"{len(records)} 条）。拒绝写出空菜单：空文件会被下游当成"
            "「这家店真的没东西卖」，比失败更难发现。"
        )

    nutrition_payload = call_with_schema(client, tools, "list-nutrition-foods", known, extra)
    if nutrition_payload is not None:
        items, unmatched = merge_nutrition(items, nutrition_payload)
        print(
            f"    营养匹配 {len(items) - len(unmatched)}/{len(items)}"
            f"{f'，未匹配 {len(unmatched)} 个' if unmatched else ''}",
            flush=True,
        )

    priced = 0
    price_failed: list[str] = []
    if not args.skip_price:
        print("5/5 逐项试算价格（可加 --skip-price 跳过）", flush=True)
        for index, item in enumerate(items, 1):
            try:
                quote = client.call_tool(
                    "calculate-price",
                    {
                        **{k: v for k, v in known.items() if k in ("store_code", "address")},
                        "items": [{"itemCode": item.item_code, "quantity": 1}],
                    },
                )
            except McpError as exc:
                # 单条失败不能中断整轮：早期实现只在第 1 项 break，从第 2 项起
                # 静默跳过，最终「成功算价」的数字看着正常，实际混着一批标价。
                if not price_failed:
                    print(f"    算价失败（{exc}），该项保留菜单标价", flush=True)
                price_failed.append(item.name)
                continue

            total = find_records(quote)
            amount_cents = quote_amount_cents(total[0]) if total else None
            if amount_cents is None:
                price_failed.append(item.name)
            else:
                item.price_cents = amount_cents
                priced += 1
            if index % 20 == 0:
                print(f"    已处理 {index}/{len(items)}", flush=True)
        print(f"    成功算价 {priced} 项", flush=True)
        if price_failed:
            print(
                f"    ! 警告：{len(price_failed)} 项使用标价，与折扣价口径不一致"
                f"（如 {', '.join(price_failed[:3])}{' 等' if len(price_failed) > 3 else ''}）",
                file=sys.stderr,
                flush=True,
            )
    else:
        print("5/5 已跳过算价", flush=True)

    missing_price = sum(1 for i in items if i.price_cents is None)
    if args.skip_price or priced == 0:
        price_source = "menu"
    elif priced == len(items):
        price_source = "calculate-price"
    else:
        price_source = "mixed"

    quality = {
        "nutrition_coverage": nutrition_coverage(items),
        "price_source": price_source,
        "price_unit": {
            "resolved": price_unit,
            "explicit_yuan": sum(1 for r in readings if r.unit == PRICE_UNIT_YUAN),
            "explicit_cent": sum(1 for r in readings if r.unit == PRICE_UNIT_CENT),
            "ambiguous": sum(1 for r in readings if r.unit is None),
        },
        "excluded_missing_price": missing_price,
        "dropped_without_name": dropped_names,
    }
    if missing_price:
        print(
            f"    ! 警告：{missing_price} 项没有价格，会被前沿计算跳过（不会按 0 元处理）",
            file=sys.stderr,
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "data_source": "mcd-mcp",
        "collected_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "store": {
            "code": known.get("store_code"),
            "city": known.get("city"),
            "address_query": args.address,
        },
        "quality": quality,
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
