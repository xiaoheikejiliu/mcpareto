"""领域模型：餐品、营养、指标轴。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class Direction(str, Enum):
    """指标方向。MAX 表示越大越好，MIN 表示越小越好。"""

    MAX = "max"
    MIN = "min"


@dataclass(frozen=True)
class Nutrition:
    """单个餐品的营养构成。

    单位对齐麦当劳中国 MCP `list-nutrition-foods` 的返回口径：
    能量 kcal、蛋白质 g、脂肪 g、碳水化合物 g、钠 mg、钙 mg。
    缺失字段用 None，计算时会跳过该维度而不是当成 0。
    """

    energy_kcal: float | None = None
    protein_g: float | None = None
    fat_g: float | None = None
    carb_g: float | None = None
    sodium_mg: float | None = None
    calcium_mg: float | None = None


@dataclass
class MenuItem:
    """一个可比较的餐品。

    price_cents 用「分」存整数，避免浮点误差 —— 麦当劳 `calculate-price`
    返回的就是分，直接对齐可以做到分毫不差。取不到价格时是 None，**不是 0**：
    0 是合法价格，用它当哨兵会让缺价餐品以「免费」身份支配整张菜单。
    """

    item_code: str
    name: str
    category: str
    price_cents: int | None
    nutrition: Nutrition
    tags: list[str] = field(default_factory=list)

    @property
    def price_yuan(self) -> float | None:
        if self.price_cents is None:
            return None
        return self.price_cents / 100.0

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        price = "价格缺失" if self.price_cents is None else f"¥{self.price_cents / 100:.2f}"
        return f"<MenuItem {self.name} {price}>"


@dataclass(frozen=True)
class Metric:
    """一条可比较的指标轴。

    getter 接收 MenuItem，返回浮点数；返回 None 表示该餐品缺这个维度的数据，
    在帕累托比较中会被跳过（不做假数据填充）。
    """

    key: str
    label: str
    unit: str
    direction: Direction
    getter: Callable[[MenuItem], float | None]

    def value(self, item: MenuItem) -> float | None:
        return self.getter(item)


def _per_yuan(n: float | None, item: MenuItem) -> float | None:
    """把绝对营养量折算成「每元获得量」。价格必须为正。"""
    if n is None or item.price_cents is None or item.price_cents <= 0:
        return None
    return n / (item.price_cents / 100.0)


def _price(item: MenuItem) -> float | None:
    """价格轴。

    缺失或非正都必须返回 None：价格是「越小越好」的轴，若把缺价当 0，缺价餐品
    会在所有视角里以「免费且给得最多」的姿态支配全表，把正常餐品的可省金额
    算成原价 —— 这是最安静也最致命的一种数据错误。
    """
    if item.price_cents is None or item.price_cents <= 0:
        return None
    return item.price_cents / 100.0


METRICS: dict[str, Metric] = {
    "price": Metric(
        key="price",
        label="价格",
        unit="元",
        direction=Direction.MIN,
        getter=_price,
    ),
    "kcal_per_yuan": Metric(
        key="kcal_per_yuan",
        label="每元热量",
        unit="kcal/元",
        direction=Direction.MAX,
        getter=lambda i: _per_yuan(i.nutrition.energy_kcal, i),
    ),
    "protein_per_yuan": Metric(
        key="protein_per_yuan",
        label="每元蛋白质",
        unit="g/元",
        direction=Direction.MAX,
        getter=lambda i: _per_yuan(i.nutrition.protein_g, i),
    ),
    "sodium": Metric(
        key="sodium",
        label="钠含量",
        unit="mg",
        direction=Direction.MIN,
        getter=lambda i: i.nutrition.sodium_mg,
    ),
    "sodium_per_yuan": Metric(
        key="sodium_per_yuan",
        label="每元钠摄入",
        unit="mg/元",
        direction=Direction.MIN,
        getter=lambda i: _per_yuan(i.nutrition.sodium_mg, i),
    ),
    "kcal": Metric(
        key="kcal",
        label="热量",
        unit="kcal",
        # 在「吃饱优先」语境下热量是收益，越高越好。若把这里写成 MIN，
        # 最便宜的低热量单品会支配掉所有更贵的高热量餐品，结论不成立。
        direction=Direction.MAX,
        getter=lambda i: i.nutrition.energy_kcal,
    ),
    "protein": Metric(
        key="protein",
        label="蛋白质",
        unit="g",
        direction=Direction.MAX,
        getter=lambda i: i.nutrition.protein_g,
    ),
}


# 预置的分析视角：不同人群在乎的东西不一样，前沿形状也不一样。
#
# 注意：两个维度必须是独立目标。若把「价格」和「每元蛋白质」并列，价格已经
# 内含在后者分母里，前沿会退化成单点，是比较设计的错误。正确做法是拿
# **绝对收益**（热量 / 蛋白质 / 钠）对价格做权衡，前沿才是真正的效率边界。
# 每元指标保留在 METRICS 里，仅作为结果展示时的辅助说明。
PERSPECTIVES: dict[str, dict[str, str]] = {
    "saver": {
        "name": "吃饱优先",
        "x": "price",
        "y": "kcal",
        "note": "同样的钱，谁给的热量更多",
    },
    "gym": {
        "name": "蛋白质优先",
        "x": "price",
        "y": "protein",
        "note": "同样的钱，谁给的蛋白质更多",
    },
    "low_sodium": {
        "name": "控盐优先",
        "x": "price",
        "y": "sodium",
        "note": "同样的钱，谁的钠负担更小",
    },
}
