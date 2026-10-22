"""把 MCP 返回的异构结构归一化成领域模型。

为什么需要这一层：麦当劳 MCP 不同 Tool、不同版本的返回字段命名并不完全一致
（``energy`` / ``energyKcal`` / ``能量``，``price`` 可能是元也可能是分）。
把兼容逻辑集中在这里，上层的算法与展示就能保持干净。

归一化遵循两个原则：
1. **不猜数据**。取不到就是 None，绝不用 0 或均值填充，否则前沿会被污染。
2. **单位必须对齐**。能量统一成 kcal（麦当劳官网常用千焦），价格统一成分。
   单位和字段含义一样是**接口级全局属性**，只能整表判定（``infer_energy_unit`` /
   ``infer_price_unit``）：同一份菜单不会这条按 kcal、下一条按 kJ。逐条比大小
   猜单位，必然会把小份餐品或大额套餐换算错。
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Any, Iterable

from .models import MenuItem, Nutrition

_KJ_PER_KCAL = 4.184

# 单位是**接口级全局属性**，不是单条记录的属性：同一份菜单不可能这条按 kcal 给、
# 下一条按 kJ 给。因此只按整表中位数判断一次，绝不逐条比大小 —— 单条判断会让
# 苹果片、玉米杯、饮品类的小数值被当成 kcal，低估 4.18 倍。
# 中位数超过这个量级时，整表更可能是千焦（真实菜单里单份餐品极少超过 1500 kcal）。
_ENERGY_TABLE_LIMIT = 1500

# 价格单位同样只能整表判定：裸整数（``26`` / ``2650``）单条无法区分元与分，
# 逐条用阈值猜会同时制造两种错误 —— ¥200 的家庭桶被读成 ¥2.00，
# 2650 分被读成 ¥26.50。这里只负责**记录原始读数**，判定交给 infer_price_unit。
PRICE_UNIT_YUAN = "yuan"
PRICE_UNIT_CENT = "cent"

# 整表判定裸整数单位时的分界线：单位是元时菜单均价不可能到 200 元；
# 单位是分时 200 分 = ¥2.00，同样不可能。中位数落在 200 以上的整表按分处理。
_PRICE_CENT_MEDIAN_THRESHOLD = 200


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


_ENERGY_KCAL_FIELDS = ("energyKcal", "kcal", "calorie", "calories", "energy_kcal")
_ENERGY_KJ_FIELDS = ("energyKj", "kj", "kilojoule", "energy_kj", "千焦")
_ENERGY_BARE_FIELDS = ("energy", "能量", "热量")


def ambiguous_energy_value(record: Any) -> float | None:
    """取「没写单位」的能量读数；显式声明了 kcal / kJ 的记录不参与单位投票。"""
    if pick(record, *_ENERGY_KCAL_FIELDS) is not None:
        return None
    if pick(record, *_ENERGY_KJ_FIELDS) is not None:
        return None
    return _to_float(pick(record, *_ENERGY_BARE_FIELDS))


def infer_energy_unit(records: Iterable[Any]) -> str:
    """对整表判定「无单位能量」是千卡还是千焦，返回 ``"kcal"`` 或 ``"kj"``。

    只看中位数：单位是接口的全局约定，任何逐条判断都会把数值小的餐品误判成
    kcal。中位数为空（表里没有无单位读数）时按 kcal 处理，不做假设性换算。
    """
    values = [v for v in (ambiguous_energy_value(r) for r in records) if v is not None]
    if not values:
        return "kcal"
    return "kj" if statistics.median(values) > _ENERGY_TABLE_LIMIT else "kcal"


def parse_energy_kcal(record: Any, *, ambiguous_unit: str = "kcal") -> float | None:
    """能量统一换算为千卡。

    ``ambiguous_unit`` 是调用方对整表「无单位能量」的单位判定（见
    ``infer_energy_unit``）。单条记录不自己猜单位：同一份菜单里不会有两种单位。
    """
    explicit_kcal = pick(record, *_ENERGY_KCAL_FIELDS)
    value = _to_float(explicit_kcal)
    if value is not None:
        return value

    explicit_kj = pick(record, *_ENERGY_KJ_FIELDS)
    value = _to_float(explicit_kj)
    if value is not None:
        return round(value / _KJ_PER_KCAL, 1)

    ambiguous = ambiguous_energy_value(record)
    if ambiguous is None:
        return None
    if ambiguous_unit == "kj":
        return round(ambiguous / _KJ_PER_KCAL, 1)
    return ambiguous


def parse_nutrition(record: Any, *, energy_unit: str = "kcal") -> Nutrition:
    return Nutrition(
        energy_kcal=parse_energy_kcal(record, ambiguous_unit=energy_unit),
        protein_g=_to_float(pick(record, "protein", "proteinG", "protein_g", "蛋白质")),
        fat_g=_to_float(pick(record, "fat", "fatG", "fat_g", "脂肪")),
        carb_g=_to_float(
            pick(record, "carbohydrate", "carbohydrates", "carb", "carbG", "碳水化合物")
        ),
        sodium_mg=_to_float(pick(record, "sodium", "sodiumMg", "sodium_mg", "钠")),
        calcium_mg=_to_float(pick(record, "calcium", "calciumMg", "calcium_mg", "钙")),
    )


@dataclass(frozen=True)
class PriceReading:
    """一条记录的价格**原始读数**：数值 + 已能确定的单位。

    ``unit`` 为 None 表示「裸整数」——单条记录无法判断是元还是分，必须由
    ``infer_price_unit`` 对整表投票后再换算。
    """

    value: float
    unit: str | None


_PRICE_CENT_FIELDS = ("priceCent", "priceCents", "price_cent", "pricecent")
_PRICE_BARE_FIELDS = ("price", "salePrice", "amount", "currentPrice", "价格", "单价")


def read_price(record: Any) -> PriceReading | None:
    """读出价格的原始数值与可判定的单位，不做任何换算。"""
    explicit = pick(record, *_PRICE_CENT_FIELDS)
    value = _to_float(explicit)
    if value is not None:
        return PriceReading(value, PRICE_UNIT_CENT)

    raw = pick(record, *_PRICE_BARE_FIELDS)
    value = _to_float(raw)
    if value is None:
        return None

    # 浮点数按元：``2650.0`` 是 ¥2650 的家庭桶，不是 265000 分。带 ¥ / 元 /
    # 小数点的字符串同理。这两类读数不需要整表投票。
    if isinstance(raw, float):
        return PriceReading(value, PRICE_UNIT_YUAN)
    text = str(raw)
    if ("." in text) or ("¥" in text) or ("元" in text):
        return PriceReading(value, PRICE_UNIT_YUAN)
    return PriceReading(value, None)


def infer_price_unit(readings: Iterable[PriceReading]) -> str:
    """对整表判定「裸整数」价格是元还是分，返回 ``"yuan"`` 或 ``"cent"``。

    单位是接口级约定，只能整表判定：菜单均价不可能是 ¥200，也不可能是 ¥2.00，
    所以裸整数中位数 ≥ 200 时整表按分处理，否则按元。单条阈值猜测会同时制造
    「¥200 读成 ¥2.00」和「2650 分读成 ¥26.50」两种错误，两种都真实发生过。
    """
    ambiguous = [r.value for r in readings if r.unit is None]
    if not ambiguous:
        return PRICE_UNIT_YUAN
    return (
        PRICE_UNIT_CENT
        if statistics.median(ambiguous) >= _PRICE_CENT_MEDIAN_THRESHOLD
        else PRICE_UNIT_YUAN
    )


def parse_price_cents(record: Any, *, unit: str | None = None) -> int | None:
    """价格统一换算为「分」；取不到返回 None（绝不返回 0）。

    ``unit`` 是 ``infer_price_unit`` 对整表裸整数的判定结果。遇到裸整数却没给
    判定结果时直接报错：宁可中断，也不要悄悄按错的单位生成一张金额差 100 倍的
    菜单。
    """
    reading = read_price(record)
    if reading is None:
        return None

    resolved = reading.unit or unit
    if resolved is None:
        raise ValueError(
            f"价格 {reading.value:g} 是裸整数，单条记录无法判定单位；"
            "请先用 infer_price_unit() 对整表判定，再传入 unit="
        )
    if resolved == PRICE_UNIT_CENT:
        return int(round(reading.value))
    return int(round(reading.value * 100))


def parse_item(
    record: Any,
    *,
    price_unit: str | None = None,
    energy_unit: str = "kcal",
) -> MenuItem | None:
    """把一条餐品记录转成 MenuItem；名称缺失或全为空白时直接丢弃。

    ``price_unit`` / ``energy_unit`` 是整表级的单位判定结果，见
    ``infer_price_unit`` 与 ``infer_energy_unit``。
    """
    name = pick(record, "name", "mealName", "productName", "title", "餐品名称", "名称")
    if name is None or not str(name).strip():
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
        price_cents=parse_price_cents(record, unit=price_unit),
        nutrition=parse_nutrition(record, energy_unit=energy_unit),
        tags=tags,
    )


def _is_empty_nutrition(nutrition: Nutrition) -> bool:
    """六个字段全为 None 的记录没有任何信息，视为无效。"""
    return all(
        value is None
        for value in (
            nutrition.energy_kcal,
            nutrition.protein_g,
            nutrition.fat_g,
            nutrition.carb_g,
            nutrition.sodium_mg,
            nutrition.calcium_mg,
        )
    )


def _fill_missing(base: Nutrition, extra: Nutrition) -> Nutrition:
    """逐字段合并：只用 ``extra`` 补 ``base`` 里为 None 的字段，绝不覆盖已有值。"""
    return Nutrition(
        energy_kcal=base.energy_kcal if base.energy_kcal is not None else extra.energy_kcal,
        protein_g=base.protein_g if base.protein_g is not None else extra.protein_g,
        fat_g=base.fat_g if base.fat_g is not None else extra.fat_g,
        carb_g=base.carb_g if base.carb_g is not None else extra.carb_g,
        sodium_mg=base.sodium_mg if base.sodium_mg is not None else extra.sodium_mg,
        calcium_mg=base.calcium_mg if base.calcium_mg is not None else extra.calcium_mg,
    )


def merge_nutrition(
    items: list[MenuItem], nutrition_payload: Any
) -> tuple[list[MenuItem], list[str]]:
    """把营养表按餐品名称回填到菜单项上。

    ``list-nutrition-foods`` 与 ``query-meals`` 的编码体系不保证一致，所以主键
    优先用 item_code，失败时退回名称精确匹配。返回未能匹配上的名称清单，
    便于使用者判断数据覆盖度。

    合并是**逐字段**的：菜单接口自带的蛋白质不会被营养表的空值冲掉，营养表里
    菜单缺的脂肪/碳水/钙也能补进来。营养表里六个字段全为 None 的记录视为无效，
    既不参与合并，也不计入匹配数 —— 否则日志会报「匹配 N/N」，而实际什么都没拿到。
    """
    records = find_records(nutrition_payload)
    energy_unit = infer_energy_unit(records)
    by_code: dict[str, Nutrition] = {}
    by_name: dict[str, Nutrition] = {}
    for rec in records:
        nutrition = parse_nutrition(rec, energy_unit=energy_unit)
        if _is_empty_nutrition(nutrition):
            continue
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
        else:
            item.nutrition = _fill_missing(item.nutrition, nutrition)
    return items, unmatched
