"""McPareto —— 麦当劳菜单的性价比帕累托分析。

把门店全量菜单投影到「价格 × 营养密度」平面上，求帕累托前沿，并指出哪些
餐品在任何合理偏好下都不该被选中（被支配解），以及具体是谁碾压了它。
"""

from .models import METRICS, PERSPECTIVES, Direction, MenuItem, Metric, Nutrition
from .pareto import FrontResult, compute_front, dominates, strict_dominator
from .normalize import find_records, merge_nutrition, parse_item, parse_nutrition

__version__ = "0.1.0"

__all__ = [
    "METRICS",
    "PERSPECTIVES",
    "Direction",
    "MenuItem",
    "Metric",
    "Nutrition",
    "FrontResult",
    "compute_front",
    "dominates",
    "strict_dominator",
    "find_records",
    "merge_nutrition",
    "parse_item",
    "parse_nutrition",
    "__version__",
]
