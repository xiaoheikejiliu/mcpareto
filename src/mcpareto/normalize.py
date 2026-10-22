"""把 MCP 返回的异构结构归一化成领域模型。

为什么需要这一层：麦当劳 MCP 不同 Tool、不同版本的返回字段命名并不完全一致
（``energy`` / ``energyKcal`` / ``能量``，``price`` 可能是元也可能是分）。
把兼容逻辑集中在这里，上层的算法与展示就能保持干净。

归一化遵循两个原则：
1. **不猜数据**。取不到就是 None，绝不用 0 或均值填充，否则前沿会被污染。
2. **单位必须对齐**。能量统一成 kcal（麦当劳官网常用千焦），价格统一成分。
"""

from __future__ import annotations

import re
from typing import Any

from .models import MenuItem, Nutrition

_KJ_PER_KCAL = 4.184

# 单份餐品超过 1500 kcal 在麦当劳菜单中极罕见（双层安格斯厚牛堡约 984 kcal）。
# 超过这个量级时，数值更可能以千焦给出。
_KCAL_SANITY_LIMIT = 1500


def _norm_key(key: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", str(key).lower())


def _to_float(value: Any) -> float | None:
    """从数字、数字字符串或带单位的字符串中取出数值。"""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").replace("，", "").strip()
        match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
    return None


def pick(obj: Any, *candidates: str) -> Any:
    """按候选字段名取值，忽略大小写与分隔符差异。"""
    if not isinstance(obj, dict):
        return None
    index = {_norm_key(k): v for k, v in obj.items()}
    for candidate in candidates:
        value = index.get(_norm_key(candidate))
        if value not in (None, "", [], {}):
            return value
    return None


def find_records(payload: Any) -> list[dict]:
    """在任意嵌套结构中定位最像「记录集合」的那一层。

    MCP 返回可能形如 ``{"data": {"meals": [...]}}``，也可能直接是数组。
    这里收集所有「字典组成的列表」，优先选元素普遍带名称字段、且长度最大的那个。
    """
    candidates: list[list[dict]] = []

    def walk(node: Any, depth: int = 0) -> None:
        if depth > 6:
            return
        if isinstance(node, list):
            dicts = [x for x in node if isinstance(x, dict)]
            if dicts:
                candidates.append(dicts)
            for item in node:
                walk(item, depth + 1)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value, depth + 1)

    walk(payload)
    if not candidates:
        return []

    def score(records: list[dict]) -> tuple[int, int]:
        named = sum(
            1
            for r in records
            if pick(r, "name", "mealName", "productName", "title", "餐品名称", "名称")
        )
        return (named, len(records))

    return max(candidates, key=score)


def parse_energy_kcal(record: Any) -> float | None:
    """能量统一换算为千卡。"""
    explicit_kcal = pick(record, "energyKcal", "kcal", "calorie", "calories", "energy_kcal")
    value = _to_float(explicit_kcal)
    if value is not None:
        return value

    explicit_kj = pick(record, "energyKj", "kj", "kilojoule", "energy_kj", "千焦")
    value = _to_float(explicit_kj)
    if value is not None:
        return round(value / _KJ_PER_KCAL, 1)

    ambiguous = _to_float(pick(record, "energy", "能量", "热量"))
    if ambiguous is None:
        return None
    if ambiguous > _KCAL_SANITY_LIMIT:
        return round(ambiguous / _KJ_PER_KCAL, 1)
    return ambiguous


def parse_nutrition(record: Any) -> Nutrition:
    return Nutrition(
        energy_kcal=parse_energy_kcal(record),
        protein_g=_to_float(pick(record, "protein", "proteinG", "protein_g", "蛋白质")),
        fat_g=_to_float(pick(record, "fat", "fatG", "fat_g", "脂肪")),
        carb_g=_to_float(
            pick(record, "carbohydrate", "carbohydrates", "carb", "carbG", "碳水化合物")
        ),
        sodium_mg=_to_float(pick(record, "sodium", "sodiumMg", "sodium_mg", "钠")),
        calcium_mg=_to_float(pick(record, "calcium", "calciumMg", "calcium_mg", "钙")),
    )


def parse_price_cents(record: Any) -> int:
    """价格统一换算为「分」。

    麦当劳 ``calculate-price`` 以分返回；菜单接口有时以元返回。区分规则：
    单价超过 200 元的餐品不存在，因此数值 ≥ 200 且为整数时按分处理。
    """
    explicit = pick(record, "priceCent", "priceCents", "price_cent", "pricecent")
    value = _to_float(explicit)
    if value is not None:
        return int(round(value))

    raw = pick(record, "price", "salePrice", "amount", "currentPrice", "价格", "单价")
    value = _to_float(raw)
    if value is None:
        return 0

    text = str(raw)
    is_yuan_explicit = ("." in text) or ("¥" in text) or ("元" in text)
    if is_yuan_explicit:
        return int(round(value * 100))
    if value >= 200 and float(value).is_integer():
        return int(value)
    return int(round(value * 100))


def parse_item(record: Any) -> MenuItem | None:
    """把一条餐品记录转成 MenuItem；缺少名称的条目直接丢弃。"""
    name = pick(record, "name", "mealName", "productName", "title", "餐品名称", "名称")
    if not name:
        return None

    code = pick(record, "code", "itemCode", "mealCode", "productCode", "sku", "id", "餐品编码")
    category = pick(record, "category", "categoryName", "type", "分类", "餐品分类")

    tags_raw = pick(record, "tags", "tag", "labels", "标签")
    if isinstance(tags_raw, list):
        tags = [str(t) for t in tags_raw]
    elif isinstance(tags_raw, str):
        tags = [t.strip() for t in re.split(r"[,，|]", tags_raw) if t.strip()]
    else:
        tags = []

    return MenuItem(
        item_code=str(code) if code else str(name),
        name=str(name).strip(),
        category=str(category) if category else "未分类",
        price_cents=parse_price_cents(record),
        nutrition=parse_nutrition(record),
        tags=tags,
    )


def merge_nutrition(
    items: list[MenuItem], nutrition_payload: Any
) -> tuple[list[MenuItem], list[str]]:
    """把营养表按餐品名称回填到菜单项上。

    ``list-nutrition-foods`` 与 ``query-meals`` 的编码体系不保证一致，所以主键
    优先用 item_code，失败时退回名称精确匹配。返回未能匹配上的名称清单，
    便于使用者判断数据覆盖度。
    """
    records = find_records(nutrition_payload)
    by_code: dict[str, Nutrition] = {}
    by_name: dict[str, Nutrition] = {}
    for rec in records:
        nutrition = parse_nutrition(rec)
        code = pick(rec, "code", "itemCode", "mealCode", "productCode", "sku", "id", "餐品编码")
        name = pick(rec, "name", "mealName", "productName", "title", "餐品名称", "名称")
        if code:
            by_code[str(code)] = nutrition
        if name:
            by_name[str(name).strip()] = nutrition

    unmatched: list[str] = []
    for item in items:
        nutrition = by_code.get(item.item_code) or by_name.get(item.name)
        if nutrition is None:
            unmatched.append(item.name)
        elif item.nutrition.protein_g is None:
            item.nutrition = nutrition
    return items, unmatched
