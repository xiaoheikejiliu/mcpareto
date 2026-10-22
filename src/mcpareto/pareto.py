"""帕累托支配与前沿求解。

术语约定（与经济学 / 多目标优化的标准定义一致）：

* **支配（dominate）**：若 A 在所有维度上都不劣于 B，且至少在一个维度上严格
  更好，则称 A 支配 B。被支配的选项，理性决策者永远没有理由选它 —— 除非他
  的偏好不在这些维度上。
* **帕累托前沿（Pareto front）**：不被任何其他选项支配的选项集合。它们之间
  无法再分高下，只能按个人偏好取舍。

本模块把「越小越好」的维度统一取负，转成「越大越好」后复用同一套比较逻辑，
避免在支配判断里出现方向分支导致的错误。
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Direction, MenuItem, Metric

# 浮点比较容差。价格是整数分、营养是一位小数，1e-9 足够区分任何真实差异，
# 又能吸收 0.1 + 0.2 这类表示误差。
EPS = 1e-9


def to_max_vector(item: MenuItem, axes: tuple[Metric, ...]) -> tuple[float, ...] | None:
    """把餐品投影到「全部最大化」的坐标空间。

    任一维度缺数据则返回 None —— 缺数据的餐品不应参与前沿计算，否则会被当成
    表现极差（若当 0）或表现极好（若当无穷）而污染结果。
    """
    values: list[float] = []
    for axis in axes:
        v = axis.value(item)
        if v is None:
            return None
        values.append(v if axis.direction is Direction.MAX else -v)
    return tuple(values)


def dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    """判断向量 a 是否支配向量 b（两者均已转为最大化空间）。"""
    if len(a) != len(b):
        raise ValueError("支配比较要求两个向量维度一致")
    not_worse = all(x >= y - EPS for x, y in zip(a, b))
    if not not_worse:
        return False
    return any(x > y + EPS for x, y in zip(a, b))


@dataclass
class FrontResult:
    """一次前沿求解的完整结果。"""

    front: list[MenuItem]
    """第一层前沿：不被任何餐品支配。"""

    dominated: list[MenuItem]
    """被支配的餐品，即数学意义上的劣解。"""

    skipped: list[MenuItem]
    """因缺数据无法比较的餐品。"""

    ranks: dict[str, int]
    """每个餐品的前沿层级，1 表示第一层。数值越小越优。"""

    dominators: dict[str, list[MenuItem]]
    """item_code -> 支配它的餐品列表（仅第一层即可淘汰它时给出）。"""

    @property
    def total(self) -> int:
        return len(self.front) + len(self.dominated) + len(self.skipped)

    @property
    def dominated_ratio(self) -> float:
        """被支配比例，只对参与比较的餐品计算。"""
        comparable = len(self.front) + len(self.dominated)
        if comparable == 0:
            return 0.0
        return len(self.dominated) / comparable


def compute_front(
    items: list[MenuItem],
    axes: tuple[Metric, ...],
    *,
    collect_dominators: bool = True,
) -> FrontResult:
    """求解帕累托前沿并给出全部被支配餐品的「碾压者」。

    复杂度 O(n²)。158 个餐品的规模下约 2.5 万次比较，毫秒级完成，
    不需要引入 k-d tree 之类的结构。
    """
    if len(axes) < 2:
        raise ValueError("帕累托前沿至少需要两个维度")

    projected: list[tuple[MenuItem, tuple[float, ...]]] = []
    skipped: list[MenuItem] = []
    for item in items:
        vec = to_max_vector(item, axes)
        if vec is None:
            skipped.append(item)
        else:
            projected.append((item, vec))

    front: list[MenuItem] = []
    dominated: list[MenuItem] = []
    ranks: dict[str, int] = {}
    dominators: dict[str, list[MenuItem]] = {}

    for item, vec in projected:
        beaters = [other for other, ovec in projected if other is not item and dominates(ovec, vec)]
        # 数据完全相同的餐品互不支配，它们会一起留在前沿上，这是正确行为。
        if beaters:
            dominated.append(item)
            dominators[item.item_code] = beaters
        else:
            front.append(item)
            ranks[item.item_code] = 1

    # 分层：逐层剥离第一层，得到第 2、3… 层，用于回答「如果最优的都不想要，
    # 次优是谁」。层数在实践中很少超过 6，直接迭代即可。
    remaining = list(dominated)
    layer = 2
    while remaining:
        vecs = {i.item_code: to_max_vector(i, axes) for i in remaining}
        current_layer: list[MenuItem] = []
        rest: list[MenuItem] = []
        for item in remaining:
            vec = vecs[item.item_code]
            assert vec is not None
            beaten_by_remaining = any(
                other is not item and dominates(vecs[other.item_code], vec)  # type: ignore[arg-type]
                for other in remaining
            )
            if beaten_by_remaining:
                rest.append(item)
            else:
                current_layer.append(item)
        if not current_layer:  # 理论上不会发生，防御性退出避免死循环
            break
        for item in current_layer:
            ranks[item.item_code] = layer
        remaining = rest
        layer += 1

    if not collect_dominators:
        dominators = {}

    return FrontResult(
        front=front,
        dominated=dominated,
        skipped=skipped,
        ranks=ranks,
        dominators=dominators,
    )


def strict_dominator(
    target: MenuItem,
    candidates: list[MenuItem],
    axes: tuple[Metric, ...],
) -> MenuItem | None:
    """返回一个能「全方位碾压」目标餐品的选项，优先挑节省金额最大的。

    这是产品里最直接的一句话：「你想点的这个，被另一个完全替代了，
    还便宜 X 元」。
    """
    tvec = to_max_vector(target, axes)
    if tvec is None:
        return None

    beaters: list[MenuItem] = []
    for cand in candidates:
        if cand.item_code == target.item_code:
            continue
        cvec = to_max_vector(cand, axes)
        if cvec is None:
            continue
        if dominates(cvec, tvec):
            beaters.append(cand)

    if not beaters:
        return None
    return min(beaters, key=lambda i: (i.price_cents, i.name))
