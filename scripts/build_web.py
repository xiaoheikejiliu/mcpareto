#!/usr/bin/env python3
"""把采集到的数据编译成一个自包含的 HTML 应用。

产物是单文件：内联数据、内联样式、内联脚本，无任何外部依赖。
可以直接双击打开、可以丢到任意静态托管、也可以通过 WorkBuddy 发布成在线链接。

用法::

    python3 scripts/build_web.py --data data/sample_menu.json --out web/index.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcpareto.models import METRICS, PERSPECTIVES, MenuItem, Nutrition
from mcpareto.pareto import compute_front


def load_items(path: Path) -> tuple[list[MenuItem], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items: list[MenuItem] = []
    for raw in payload.get("items", []):
        n = raw.get("nutrition") or {}
        items.append(
            MenuItem(
                item_code=raw["item_code"],
                name=raw["name"],
                category=raw.get("category", "未分类"),
                price_cents=int(raw.get("price_cents") or 0),
                nutrition=Nutrition(
                    energy_kcal=n.get("energy_kcal"),
                    protein_g=n.get("protein_g"),
                    fat_g=n.get("fat_g"),
                    carb_g=n.get("carb_g"),
                    sodium_mg=n.get("sodium_mg"),
                    calcium_mg=n.get("calcium_mg"),
                ),
                tags=raw.get("tags") or [],
            )
        )
    meta = {
        "data_source": payload.get("data_source", "unknown"),
        "sample_note": payload.get("sample_note", ""),
        "collected_at": payload.get("collected_at", ""),
        "store": payload.get("store", {}),
        "count": len(items),
    }
    return items, meta


def build_payload(items: list[MenuItem], meta: dict) -> dict:
    perspectives: dict[str, dict] = {}
    for key, cfg in PERSPECTIVES.items():
        x_axis = METRICS[cfg["x"]]
        y_axis = METRICS[cfg["y"]]
        result = compute_front(items, (x_axis, y_axis))

        points = []
        for item in items:
            xv = x_axis.value(item)
            yv = y_axis.value(item)
            if xv is None or yv is None:
                continue
            points.append(
                {
                    "code": item.item_code,
                    "name": item.name,
                    "cat": item.category,
                    "x": round(xv, 2),
                    "y": round(yv, 1),
                    "isFront": item.item_code in {f.item_code for f in result.front},
                    "rank": result.ranks.get(item.item_code, 0),
                    "price": round(item.price_yuan, 2),
                    "kcal": item.nutrition.energy_kcal,
                    "protein": item.nutrition.protein_g,
                    "sodium": item.nutrition.sodium_mg,
                }
            )

        beaters = {}
        for code, lst in result.dominators.items():
            beaters[code] = [
                {
                    "code": b.item_code,
                    "name": b.name,
                    "x": round(x_axis.value(b) or 0, 2),
                    "y": round(y_axis.value(b) or 0, 1),
                    "price": round(b.price_yuan, 2),
                }
                for b in sorted(lst, key=lambda i: (i.price_cents, i.name))[:4]
            ]

        perspectives[key] = {
            "name": cfg["name"],
            "note": cfg["note"],
            "x": {
                "label": x_axis.label,
                "unit": x_axis.unit,
                "dir": x_axis.direction.value,
            },
            "y": {
                "label": y_axis.label,
                "unit": y_axis.unit,
                "dir": y_axis.direction.value,
            },
            "points": points,
            "frontCount": len(result.front),
            "dominatedCount": len(result.dominated),
            "dominatedRatio": round(result.dominated_ratio, 4),
            "beaters": beaters,
        }

    return {"meta": meta, "perspectives": perspectives}


def render(payload: dict) -> str:
    template = (ROOT / "web" / "template.html").read_text(encoding="utf-8")
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return template.replace("/*__PAYLOAD__*/null", data)


def main() -> int:
    parser = argparse.ArgumentParser(description="编译自包含 HTML 应用")
    parser.add_argument("--data", default="data/menu.json")
    # 产物落在 docs/，便于直接用 GitHub Pages（只接受 / 或 /docs）发布成在线链接
    parser.add_argument("--out", default="docs/index.html")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists() and args.data == "data/menu.json":
        fallback = ROOT / "data" / "sample_menu.json"
        print(f"{args.data} 不存在，回退到 {fallback.relative_to(ROOT)}（演示样本）")
        data_path = fallback

    items, meta = load_items(data_path)
    if not items:
        print("数据文件中没有任何餐品，无法生成页面", file=sys.stderr)
        return 1

    payload = build_payload(items, meta)
    html = render(payload)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    stats = {k: v["dominatedRatio"] for k, v in payload["perspectives"].items()}
    print(f"已生成 {out_path}（{len(items)} 项餐品，{out_path.stat().st_size / 1024:.0f} KB）")
    for key, ratio in stats.items():
        print(f"  {payload['perspectives'][key]['name']}: 劣解占比 {ratio:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
