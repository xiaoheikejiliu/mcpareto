"""帕累托引擎的单元测试。

重点覆盖容易出错的边界：完全相同的数据、缺维度、单元素集合，以及
「不严格更优就不算支配」这个判定。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcpareto.models import METRICS, Direction, MenuItem, Metric, Nutrition
from mcpareto.pareto import compute_front, dominates, strict_dominator, to_max_vector


def make(name: str, price_yuan: float, kcal: float | None, protein: float | None) -> MenuItem:
    return MenuItem(
        item_code=name,
        name=name,
        category="测试",
        price_cents=int(round(price_yuan * 100)),
        nutrition=Nutrition(energy_kcal=kcal, protein_g=protein),
    )


PRICE = METRICS["price"]
PROTEIN = METRICS["protein"]
KCAL = METRICS["kcal"]
AXES = (PRICE, PROTEIN)


class TestDominates(unittest.TestCase):
    def test_strictly_better_on_all_axes(self):
        self.assertTrue(dominates((1.0, 1.0), (0.0, 0.0)))

    def test_equal_on_one_axis_better_on_other(self):
        self.assertTrue(dominates((1.0, 0.0), (0.0, 0.0)))
        self.assertTrue(dominates((0.0, 1.0), (0.0, 0.0)))

    def test_identical_vectors_do_not_dominate(self):
        self.assertFalse(dominates((1.0, 1.0), (1.0, 1.0)))

    def test_trade_off_does_not_dominate(self):
        self.assertFalse(dominates((2.0, 0.0), (0.0, 2.0)))
        self.assertFalse(dominates((0.0, 2.0), (2.0, 0.0)))

    def test_dimension_mismatch_raises(self):
        with self.assertRaises(ValueError):
            dominates((1.0,), (1.0, 2.0))


class TestVectors(unittest.TestCase):
    def test_min_axis_is_negated(self):
        item = make("A", price_yuan=10.0, kcal=100, protein=5)
        vec = to_max_vector(item, AXES)
        assert vec is not None
        self.assertAlmostEqual(vec[0], -10.0)
        self.assertAlmostEqual(vec[1], 5.0)

    def test_missing_dimension_returns_none(self):
        item = make("A", price_yuan=10.0, kcal=100, protein=None)
        self.assertIsNone(to_max_vector(item, AXES))


class TestComputeFront(unittest.TestCase):
    def test_cheaper_and_more_protein_wins(self):
        cheap = make("便宜且高蛋白", 10.0, 400, 20)
        pricey = make("贵且低蛋白", 20.0, 400, 15)
        result = compute_front([cheap, pricey], AXES)
        self.assertEqual([i.name for i in result.front], ["便宜且高蛋白"])
        self.assertEqual([i.name for i in result.dominated], ["贵且低蛋白"])

    def test_trade_off_keeps_both_on_front(self):
        cheap = make("便宜", 10.0, 300, 10)
        rich = make("营养高", 30.0, 700, 35)
        result = compute_front([cheap, rich], AXES)
        self.assertEqual(len(result.front), 2)
        self.assertEqual(len(result.dominated), 0)

    def test_equal_products_stay_on_front_together(self):
        a = make("甲", 20.0, 500, 25)
        b = make("乙", 20.0, 500, 25)
        result = compute_front([a, b], AXES)
        self.assertEqual(len(result.front), 2)
        self.assertEqual(len(result.dominated), 0)

    def test_missing_data_is_skipped_not_judged(self):
        good = make("A", 10.0, 500, 25)
        broken = make("B", 5.0, 100, None)
        result = compute_front([good, broken], AXES)
        self.assertEqual(len(result.skipped), 1)
        self.assertEqual(result.skipped[0].name, "B")
        self.assertEqual(len(result.front), 1)

    def test_single_item_is_front(self):
        result = compute_front([make("A", 10.0, 500, 25)], AXES)
        self.assertEqual(len(result.front), 1)
        self.assertEqual(result.dominated_ratio, 0.0)

    def test_requires_two_axes(self):
        with self.assertRaises(ValueError):
            compute_front([make("A", 10.0, 500, 25)], (PRICE,))

    def test_ranks_are_layered(self):
        best = make("最优", 5.0, 900, 40)
        mid = make("次优", 10.0, 500, 20)
        worst = make("最差", 20.0, 300, 10)
        result = compute_front([best, mid, worst], AXES)
        self.assertEqual(result.ranks["最优"], 1)
        self.assertEqual(result.ranks["次优"], 2)
        self.assertEqual(result.ranks["最差"], 3)

    def test_dominators_are_first_layer_only(self):
        """审查 S-8：推荐替代品必须来自第一层前沿。

        乙 虽然也支配 丙，但它自己是被 甲 支配的劣解。旧实现把 乙 一起放进来，
        页面可能推荐「换成一个同样被支配的选项」，用户照做只是从一个劣解
        跳到另一个劣解。
        """
        a = make("甲", 10.0, 600, 30)
        b = make("乙", 15.0, 550, 28)
        c = make("丙", 30.0, 400, 12)
        result = compute_front([a, b, c], AXES)
        self.assertEqual(result.ranks["乙"], 2)
        names = {i.name for i in result.dominators["丙"]}
        self.assertEqual(names, {"甲"})

    def test_every_dominator_is_on_the_front(self):
        """任意数据集里，dominators 出现的每个 code 必须 ranks == 1。"""
        items = [
            make("A", 5.0, 900, 40),
            make("B", 10.0, 500, 20),
            make("C", 20.0, 300, 10),
            make("D", 8.0, 700, 30),
            make("E", 30.0, 200, 5),
        ]
        result = compute_front(items, AXES)
        codes = {i.item_code for i in result.front}
        for code, beaters in result.dominators.items():
            for beater in beaters:
                self.assertIn(beater.item_code, codes)
                self.assertEqual(result.ranks[beater.item_code], 1)

    def test_kcal_axis_treats_more_as_better(self):
        """「吃饱优先」下热量是收益，便宜的低热量单品不得支配贵的高热量餐品。

        这条曾经是真 bug：kcal 的方向被写成 MIN，导致 ¥15 的鸡排把 ¥39 的
        汉堡判成劣解，结论完全反了。
        """
        cheap_low = make("便宜低热量", 10.0, 300, 10)
        pricey_high = make("贵高热量", 30.0, 700, 30)
        result = compute_front([cheap_low, pricey_high], (PRICE, KCAL))
        self.assertEqual(len(result.front), 2)
        self.assertEqual(len(result.dominated), 0)

    def test_sodium_axis_treats_less_as_better(self):
        """钠越低越好：低钠的同价位餐品应支配高钠的。"""
        axes = (PRICE, METRICS["sodium"])
        low = MenuItem(
            item_code="low", name="低钠", category="测试", price_cents=1000,
            nutrition=Nutrition(sodium_mg=500),
        )
        high = MenuItem(
            item_code="high", name="高钠", category="测试", price_cents=1000,
            nutrition=Nutrition(sodium_mg=1200),
        )
        result = compute_front([low, high], axes)
        self.assertEqual([i.name for i in result.front], ["低钠"])
        self.assertEqual([i.name for i in result.dominated], ["高钠"])


class TestStrictDominator(unittest.TestCase):
    def test_picks_cheapest_beater(self):
        target = make("目标", 30.0, 400, 15)
        cheap = make("便宜替补", 15.0, 450, 20)
        also_cheap = make("另一个替补", 18.0, 500, 22)
        picked = strict_dominator(target, [target, cheap, also_cheap], AXES)
        assert picked is not None
        self.assertEqual(picked.name, "便宜替补")

    def test_returns_none_when_nothing_beats(self):
        target = make("目标", 10.0, 900, 40)
        other = make("更差", 20.0, 100, 5)
        self.assertIsNone(strict_dominator(target, [target, other], AXES))

    def test_returns_none_for_incomplete_target(self):
        target = make("目标", 10.0, 900, None)
        other = make("其他", 20.0, 100, 5)
        self.assertIsNone(strict_dominator(target, [target, other], AXES))


class TestSampleDataset(unittest.TestCase):
    """用仓库内的演示数据做一次端到端校验，防止重构把主流程改坏。"""

    def test_sample_data_produces_front(self):
        import json

        path = Path(__file__).resolve().parents[1] / "data" / "sample_menu.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = []
        for raw in payload["items"]:
            n = raw["nutrition"]
            items.append(
                MenuItem(
                    item_code=raw["item_code"],
                    name=raw["name"],
                    category=raw["category"],
                    price_cents=raw["price_cents"],
                    nutrition=Nutrition(
                        energy_kcal=n["energy_kcal"],
                        protein_g=n["protein_g"],
                        sodium_mg=n["sodium_mg"],
                    ),
                )
            )
        result = compute_front(items, AXES)
        self.assertEqual(len(items), 13)
        self.assertGreater(len(result.front), 0)
        self.assertGreater(len(result.dominated), 0)
        # 前沿上的餐品必须真的没被任何餐品支配
        front_codes = {i.item_code for i in result.front}
        for code in front_codes:
            self.assertNotIn(code, result.dominators)
        self.assertAlmostEqual(result.dominated_ratio, len(result.dominated) / 13, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
