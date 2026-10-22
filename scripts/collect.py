#!/usr/bin/env python3
"""从麦当劳 MCP 采集真实菜单与营养数据，落地为 data/menu.json。

用法::

    export MCD_MCP_TOKEN="你的 token"
    python3 scripts/collect.py --city "深圳" --keyword "深圳北站" --out data/menu.json

关于参数：真实 MCP schema 要求 ``query-nearby-stores`` 同时传 ``city`` 与 ``keyword``，
只传其一会返回 600058「城市名或者关键词不能为空」。查不到门店时可用
``--store-code`` 直接指定，跳过搜索。

关于营养匹配：菜单里大量是套餐组合（「巨无霸三件套」）与规格未定的单品（「可乐」），
而官方营养表只有具体单品。用单品营养冒充套餐会严重低估热量，猜规格会直接算错，
因此这两类一律**不匹配**——宁可样本变小，也不给出错误结论。匹配情况写进产物的
``quality`` 字段，页面上也会展示覆盖率。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcpareto.errors import McpError
from mcpareto.mcp_client import McpClient

DEFAULT_URL = "https://mcp.mcd.cn"

# 套餐关键词：含这些词的菜单项不做营养匹配
COMBO_MARKERS = (
    "套餐", "三件套", "四件套", "组合", "双人", "多人", "拼盘",
    "桶", "餐盒", "随心配", "1+1", "任选", "开心乐园餐", "单品精选",
)
# 非食品：周边、玩具等，本来就不该进营养分析
NON_FOOD_MARKERS = ("帽", "捶", "痒痒挠", "玩具", "联名", "周边", "盲盒")


def extract_original(text: str) -> str:
    """MCP 的 tool 返回是「字段说明 + ## Original Response + JSON」的混合文本。

    直接把整段丢给 json.loads 会失败，必须先切出最后那段 JSON。
    """
    marker = "## Original Response"
    if marker in text:
        text = text.split(marker, 1)[1]
    return text.strip()


def unwrap(payload: dict) -> dict:
    """把 call_tool 的返回统一成业务 JSON。

    注意 JSON 后面常常还跟着一段展示说明（形如 ``## 展示结果时，...``），
    所以不能用 ``json.loads`` 直接吃到末尾，否则报 Extra data。
    ``raw_decode`` 只消费一个完整 JSON 值并返回结束位置，正好适配。
    """
    if "_raw" not in payload:
        return payload
    text = extract_original(payload["_raw"])
    start = text.find('{"success"')
    if start < 0:
        raise McpError(f"响应中找不到 JSON 主体：{text[:200]}")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    return obj


def parse_compact_table(text: str) -> list[dict]:
    """解析 ``[N]{col1,col2,...}:`` 后接每行一条记录的紧凑表。

    这是 ``list-nutrition-foods`` 的真实返回格式，不是标准 JSON 数组。
    """
    body_match = re.match(r"^\[(\d+)\]\{([^}]*)\}:\s*\n?(.*)$", text.strip(), re.S)
    if not body_match:
        return []
    declared = int(body_match.group(1))
    columns = [c.strip() for c in body_match.group(2).split(",")]

    rows: list[dict] = []
    for line in body_match.group(3).splitlines():
        line = line.strip()
        if not line:
            continue
        cells = [c.strip() for c in line.split(",")]
        if len(cells) < len(columns):
            cells.extend([""] * (len(columns) - len(cells)))
        record: dict = {}
        for index, column in enumerate(columns):
            cell = cells[index] if index < len(cells) else ""
            if cell in ("", "null", "NULL", "None"):
                record[column] = None
                continue
            try:
                record[column] = float(cell) if "." in cell else int(cell)
            except ValueError:
                record[column] = cell
        rows.append(record)
    return rows[:declared]


def normalize_name(name: str) -> str:
    """名称归一化：全角括号转半角、去掉引号与所有空白。"""
    text = str(name or "").strip().replace("（", "(").replace("）", ")")
    return re.sub(r"\s+", "", text.replace("“", "").replace("”", ""))


def match_nutrition(
    menu_name: str, nutrition_index: dict[str, dict]
) -> tuple[dict | None, str | None]:
    """把菜单项匹配到营养记录，返回 (记录, 匹配方式)。

    三种允许的方式：

    * ``exact`` —— 归一化后完全相同
    * ``stripped`` —— 去掉括号补充说明后相同（如「那么大鸡排（椒盐风味）」）
    * ``size_default`` —— 候选里存在唯一「中杯」的饮品，按中杯默认规格处理

    套餐与非食品直接返回 ``(None, None)``，不做任何猜测。
    """
    if any(marker in menu_name for marker in COMBO_MARKERS):
        return None, None
    if any(marker in menu_name for marker in NON_FOOD_MARKERS):
        return None, None

    key = normalize_name(menu_name)
    if key in nutrition_index:
        return nutrition_index[key], "exact"

    base = re.sub(r"\([^)]*\)", "", key).strip()
    if base and base != key and base in nutrition_index:
        return nutrition_index[base], "stripped"

    if not base:
        return None, None
    candidates = [(k, v) for k, v in nutrition_index.items() if base in k]
    medium = [(k, v) for k, v in candidates if "中杯" in k]
    medium_names = {v.get("productName") for _, v in medium}
    if len(medium_names) == 1:
        return medium[0][1], "size_default"
    if not medium and len(candidates) == 1:
        return candidates[0][1], "size_default"
    return None, None


def main() -> int:
    parser = argparse.ArgumentParser(description="从麦当劳 MCP 采集菜单与营养数据")
    parser.add_argument("--url", default=os.environ.get("MCD_MCP_URL", DEFAULT_URL))
    parser.add_argument("--token", default=os.environ.get("MCD_MCP_TOKEN", ""))
    parser.add_argument("--city", default="", help="城市名，与 --keyword 须同时提供")
    parser.add_argument("--keyword", default="", help="门店名称或地标关键词")
    parser.add_argument("--store-code", default="", help="已知门店编码，跳过搜索")
    parser.add_argument("--order-type", type=int, default=1, help="1=到店 2=外送")
    parser.add_argument("--be-type", type=int, default=1, help="1=到店取餐 5=得来速 6=团餐")
    parser.add_argument("--out", default="data/menu.json")
    args = parser.parse_args()

    if not args.token:
        print(
            "缺少 MCP Token。到 https://open.mcd.cn/mcp 登录，控制台「激活」领取，"
            "然后 export MCD_MCP_TOKEN=...",
            file=sys.stderr,
        )
        return 2
    if not args.store_code and not (args.city and args.keyword):
        print(
            "需要 --city 与 --keyword 同时提供（真实 schema 要求），"
            "或用 --store-code 指定门店。",
            file=sys.stderr,
        )
        return 2

    client = McpClient(args.url, args.token)

    print("1/4 初始化会话", flush=True)
    info = client.initialize()
    print(f"    {info.get('serverInfo', {})}", flush=True)

    store_code, city = args.store_code, args.city
    if not store_code:
        print("2/4 查询附近门店", flush=True)
        stores = unwrap(
            client.call_tool(
                "query-nearby-stores",
                {
                    "beType": args.be_type,
                    "searchType": 2,
                    "city": args.city,
                    "keyword": args.keyword,
                },
            )
        )
        if not stores.get("success"):
            raise McpError(f"门店查询失败: {stores.get('message')}")
        records = stores.get("data") or []
        if not records:
            raise McpError("未查到门店，请换关键词或用 --store-code 指定")
        first = records[0]
        store_code = str(first.get("storeCode") or "")
        city = city or str(first.get("city") or "")
        print(f"    {first.get('storeName')} ({store_code})", flush=True)
    else:
        print("2/4 已指定门店，跳过搜索", flush=True)

    print("3/4 拉取菜单与营养表", flush=True)
    meals_outer = unwrap(
        client.call_tool(
            "query-meals",
            {"storeCode": store_code, "orderType": args.order_type, "beType": args.be_type},
        )
    )
    if not meals_outer.get("success"):
        raise McpError(f"菜单查询失败: {meals_outer.get('message')}")
    data = meals_outer.get("data") or {}
    meals = data.get("meals") or {}
    if not meals:
        raise McpError(
            "query-meals 没有返回任何餐品。拒绝写出空菜单：空文件会被下游当成"
            "「这家店真的没东西卖」，比失败更难发现。"
        )

    code_to_category: dict[str, str] = {}
    for category in data.get("categories") or []:
        for entry in category.get("meals") or []:
            code = entry.get("code") if isinstance(entry, dict) else entry
            if code:
                code_to_category[str(code)] = str(category.get("name", "")).replace("\n", "")

    nutrition_outer = unwrap(client.call_tool("list-nutrition-foods", {}))
    nutrition_rows = parse_compact_table(nutrition_outer.get("data") or "")
    nutrition_index: dict[str, dict] = {}
    for row in nutrition_rows:
        name = row.get("productName")
        if name:
            nutrition_index.setdefault(normalize_name(name), row)
    print(f"    菜单 {len(meals)} 项 / 营养表 {len(nutrition_rows)} 条", flush=True)

    print("4/4 匹配营养并写出", flush=True)
    items: list[dict] = []
    stats = {"exact": 0, "stripped": 0, "size_default": 0}
    for code, meal in meals.items():
        name = str(meal.get("name") or "").strip()
        if not name:
            continue
        try:
            price_cents = int(round(float(meal.get("currentPrice")) * 100))
        except (TypeError, ValueError):
            price_cents = None

        nutrition, how = match_nutrition(name, nutrition_index)
        if how:
            stats[how] += 1

        items.append(
            {
                "item_code": str(code),
                "name": name,
                "category": code_to_category.get(str(code), "未分类"),
                "price_cents": price_cents,
                "tags": [f"match:{how}"] if how else [],
                "nutrition": {
                    "energy_kcal": nutrition.get("energyKcal") if nutrition else None,
                    "protein_g": nutrition.get("protein") if nutrition else None,
                    "fat_g": nutrition.get("fat") if nutrition else None,
                    "carb_g": nutrition.get("carbohydrate") if nutrition else None,
                    "sodium_mg": nutrition.get("sodium") if nutrition else None,
                    "calcium_mg": nutrition.get("calcium") if nutrition else None,
                },
            }
        )

    with_energy = sum(1 for i in items if i["nutrition"]["energy_kcal"] is not None)
    missing_price = sum(1 for i in items if i["price_cents"] is None)
    print(
        f"    匹配 {with_energy}/{len(items)} 项"
        f"（精确 {stats['exact']} / 去括号 {stats['stripped']} / 中杯默认 {stats['size_default']}）",
        flush=True,
    )
    if missing_price:
        print(f"    ! {missing_price} 项缺价格，会被前沿计算跳过（不会按 0 元处理）", flush=True)

    payload = {
        "schema_version": 1,
        "data_source": "mcd-mcp",
        "collected_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "store": {"code": store_code, "city": city, "address_query": args.keyword or city},
        "quality": {
            "items_total": len(items),
            "nutrition_coverage": {
                "with_energy": with_energy,
                "with_protein": sum(
                    1 for i in items if i["nutrition"]["protein_g"] is not None
                ),
                "with_sodium": sum(
                    1 for i in items if i["nutrition"]["sodium_mg"] is not None
                ),
            },
            "match_breakdown": stats,
            "price_source": "menu",
            "excluded_missing_price": missing_price,
            "unmatched_reason": "套餐组合与规格歧义项不匹配，避免用单品营养冒充套餐",
        },
        "items": items,
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
