"""归一化层的单元测试。

normalize 是「MCP 异构返回 → 领域模型」的唯一收口，字段名与单位都在这里做兼容，
所以测试重点不是 happy path，而是那些一旦猜错就会污染帕累托前沿的地方：

* 单位：kJ/kcal 混用、元/分混用；
* 缺数据：必须保持 None / 0，绝不能静默填成 0 或均值；
* 字段名：``energyKcal`` / ``energy_kcal`` / ``能量`` 必须等价；
* 记录定位：``{"data": {"meals": [...]}}`` 这类包装层要能穿透。

测试只覆盖公开行为与少量必要的私有辅助（``_to_float``），不复制实现逻辑。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcpareto.normalize import (
    _to_float,  # 单位清理的核心，单独测；故意测私有函数以便定位到具体环节
    find_records,
    infer_energy_unit,
    infer_price_unit,
    merge_nutrition,
    parse_energy_kcal,
    parse_item,
    parse_nutrition,
    parse_price_cents,
    pick,
    read_price,
)

KJ_PER_KCAL = 4.184


class TestToFloat(unittest.TestCase):
    """数值提取：数字、千分位、带单位字符串、以及「不能被当成数」的输入。"""

    def test_int_becomes_float(self):
        self.assertEqual(_to_float(385), 385.0)

    def test_float_passes_through(self):
        self.assertEqual(_to_float(26.5), 26.5)

    def test_thousands_separator_removed(self):
        """``"1,208"`` 是 MCP 里真实的千分位写法，逗号必须去掉而不是截断。"""
        self.assertEqual(_to_float("1,208"), 1208.0)

    def test_unit_suffix_ignored(self):
        self.assertEqual(_to_float("1208 mg"), 1208.0)

    def test_none_returns_none(self):
        self.assertIsNone(_to_float(None))

    def test_bool_returns_none_instead_of_one_or_zero(self):
        """``True`` 是 int 子类，若漏判会变成 1.0，凭空造出一个营养值。"""
        self.assertIsNone(_to_float(True))
        self.assertIsNone(_to_float(False))

    def test_string_without_digits_returns_none(self):
        self.assertIsNone(_to_float("暂无数据"))

    def test_empty_string_returns_none(self):
        self.assertIsNone(_to_float(""))

    def test_container_returns_none(self):
        self.assertIsNone(_to_float(["385"]))
        self.assertIsNone(_to_float({"value": 385}))


class TestPick(unittest.TestCase):
    """字段名变体取值。"""

    def test_field_name_variants_hit(self):
        self.assertEqual(pick({"energyKcal": 385}, "energy_kcal"), 385)
        self.assertEqual(pick({"energy_kcal": 385}, "energyKcal"), 385)
        self.assertEqual(pick({"能量": 385}, "能量"), 385)

    def test_case_and_separator_insensitive(self):
        self.assertEqual(pick({"ENERGY_KCAL": 385}, "energyKcal"), 385)
        self.assertEqual(pick({"Energy-Kcal": 385}, "energy_kcal"), 385)

    def test_empty_string_is_treated_as_missing(self):
        """空串应视为未命中并继续尝试后面的候选名，而不是把空串当有效值。"""
        self.assertEqual(pick({"a": "", "b": 7}, "a", "b"), 7)

    def test_empty_list_is_treated_as_missing(self):
        self.assertEqual(pick({"tags": [], "name": "x"}, "tags", "name"), "x")

    def test_empty_dict_is_treated_as_missing(self):
        self.assertIsNone(pick({"nutrition": {}}, "nutrition"))

    def test_zero_is_kept_as_real_value(self):
        """0 是合法数值（例如钠为 0），不能被当成「没取到」。"""
        self.assertEqual(pick({"sodium": 0}, "sodium"), 0)

    def test_non_dict_returns_none(self):
        self.assertIsNone(pick(None, "name"))
        self.assertIsNone(pick([{"name": "x"}], "name"))


class TestFindRecords(unittest.TestCase):
    """在任意嵌套结构里定位记录集合。"""

    def test_payload_is_already_an_array(self):
        records = [{"name": "汉堡"}, {"name": "薯条"}]
        self.assertEqual(find_records(records), records)

    def test_nested_under_data_meals(self):
        payload = {"data": {"meals": [{"name": "汉堡"}, {"name": "薯条"}]}}
        self.assertEqual(find_records(payload), payload["data"]["meals"])

    def test_multi_level_nesting(self):
        payload = {"result": {"payload": {"list": {"foods": [{"name": "薯条"}]}}}}
        self.assertEqual(find_records(payload), [{"name": "薯条"}])

    def test_empty_structures_return_empty_list(self):
        for payload in (None, [], {}, {"data": {"meals": []}}, {"data": {}}):
            with self.subTest(payload=payload):
                self.assertEqual(find_records(payload), [])

    def test_prefers_records_with_name_field(self):
        """带名称字段的短列表，胜过更长但无名称的列表——名字决定了能否与菜单对齐。"""
        payload = {
            "data": {
                "meta": [{"foo": 1}, {"bar": 2}],
                "items": [{"name": "汉堡"}],
            }
        }
        self.assertEqual(find_records(payload), [{"name": "汉堡"}])

    def test_prefers_longer_list_when_named_counts_equal(self):
        payload = {
            "a": [{"name": "甲"}],
            "b": [{"name": "乙"}, {"name": "丙"}, {"name": "丁"}],
        }
        self.assertEqual([r["name"] for r in find_records(payload)], ["乙", "丙", "丁"])

    def test_non_dict_elements_are_ignored(self):
        payload = [1, "x", {"name": "汉堡"}, None]
        self.assertEqual(find_records(payload), [{"name": "汉堡"}])


class TestParseEnergyKcal(unittest.TestCase):
    """能量统一成 kcal。显式字段直接采信；无单位读数由整表判定后再换算。"""

    def test_explicit_kcal_field_used_directly(self):
        self.assertEqual(parse_energy_kcal({"energyKcal": 385}), 385.0)

    def test_explicit_kcal_with_unit_string(self):
        self.assertEqual(parse_energy_kcal({"energy_kcal": "385 kcal"}), 385.0)

    def test_explicit_kj_field_converted(self):
        """``2029 kJ`` ≈ 485 kcal；客户端必须自行换算，否则数值虚高 4 倍。"""
        value = parse_energy_kcal({"energyKj": 2029})
        assert value is not None
        self.assertAlmostEqual(value, 2029 / KJ_PER_KCAL, delta=0.05)
        self.assertAlmostEqual(value, 485.0, delta=0.2)

    def test_ambiguous_energy_defaults_to_kcal(self):
        """没有整表判定结果时按 kcal 原样返回，不做单条猜测。"""
        self.assertEqual(parse_energy_kcal({"energy": 385}), 385.0)
        self.assertEqual(parse_energy_kcal({"energy": 2029}), 2029.0)

    def test_ambiguous_energy_converted_when_table_is_kj(self):
        """整表判定为千焦后才换算：2029 不再由单条数值大小决定。"""
        value = parse_energy_kcal({"energy": 2029}, ambiguous_unit="kj")
        assert value is not None
        self.assertAlmostEqual(value, 2029 / KJ_PER_KCAL, delta=0.05)
        self.assertLess(value, 1500)

    def test_ambiguous_chinese_fields(self):
        self.assertEqual(parse_energy_kcal({"能量": 300}), 300.0)
        self.assertAlmostEqual(
            parse_energy_kcal({"热量": 2029}, ambiguous_unit="kj") or 0,
            2029 / KJ_PER_KCAL,
            delta=0.05,
        )

    def test_explicit_kcal_wins_over_ambiguous_energy(self):
        self.assertEqual(parse_energy_kcal({"energyKcal": 400, "energy": 2029}), 400.0)

    def test_missing_energy_returns_none(self):
        self.assertIsNone(parse_energy_kcal({"protein": 20}))
        self.assertIsNone(parse_energy_kcal(None))

    def test_non_numeric_energy_returns_none(self):
        self.assertIsNone(parse_energy_kcal({"energy": "暂无"}))


class TestInferEnergyUnit(unittest.TestCase):
    """能量单位是接口级全局属性，只能整表判定（审查 S-4）。"""

    def test_median_above_limit_means_kilojoule(self):
        records = [{"energy": 2029}, {"energy": 2100}, {"energy": 2600}]
        self.assertEqual(infer_energy_unit(records), "kj")

    def test_median_at_or_below_limit_means_kcal(self):
        self.assertEqual(infer_energy_unit([{"energy": 385}, {"energy": 230}]), "kcal")
        self.assertEqual(infer_energy_unit([{"energy": 1500}, {"energy": 1500}]), "kcal")

    def test_small_items_follow_the_table_decision(self):
        """苹果片 251 kJ、玉米杯 377 kJ：逐条阈值会把它们当 kcal 而低估 4.18 倍，
        整表判定为千焦后它们跟着一起换算（真实菜单里主食把中位数顶到 1500 以上）。"""
        records = [{"energy": 251}, {"energy": 377}, {"energy": 2029}, {"energy": 2100}, {"energy": 2600}]
        self.assertEqual(infer_energy_unit(records), "kj")
        self.assertAlmostEqual(parse_energy_kcal({"energy": 251}, ambiguous_unit="kj"), 60.0, delta=0.1)

    def test_explicit_unit_records_do_not_vote(self):
        records = [{"energyKcal": 385}, {"energyKj": 2029}, {"energy": 2029}]
        # 只有第三条参与投票，中位数 2029 > 1500
        self.assertEqual(infer_energy_unit(records), "kj")

    def test_no_ambiguous_values_defaults_to_kcal(self):
        self.assertEqual(infer_energy_unit([{"energyKcal": 385}]), "kcal")
        self.assertEqual(infer_energy_unit([]), "kcal")


class TestParsePriceCents(unittest.TestCase):
    """价格统一成分。核心风险：把「分」再乘一次 100，金额直接放大 100 倍。"""

    def test_explicit_price_cent_used_directly(self):
        self.assertEqual(parse_price_cents({"priceCent": 2650}), 2650)

    def test_yuan_string_with_symbol_converted(self):
        self.assertEqual(parse_price_cents({"price": "¥26.50"}), 2650)

    def test_float_yuan_converted(self):
        self.assertEqual(parse_price_cents({"price": 26.5}), 2650)

    def test_float_large_value_is_yuan_not_cents(self):
        """测试 D2：``2650.0`` 是 ¥2650（家庭桶），不是 265000 分，也不是 ¥26.50。"""
        self.assertEqual(parse_price_cents({"price": 2650.0}), 265000)

    def test_bare_integer_needs_table_unit(self):
        """裸整数单条不可判：不再用 200 阈值猜，缺上下文就显式报错。"""
        with self.assertRaises(ValueError):
            parse_price_cents({"price": 26})
        with self.assertRaises(ValueError):
            parse_price_cents({"price": 2650})

    def test_table_unit_yuan_applied_to_bare_integer(self):
        self.assertEqual(parse_price_cents({"price": 26}, unit="yuan"), 2600)

    def test_table_unit_cent_applied_to_bare_integer(self):
        self.assertEqual(parse_price_cents({"price": 2650}, unit="cent"), 2650)

    def test_yuan_suffix_string_converted(self):
        self.assertEqual(parse_price_cents({"价格": "26.50 元"}), 2650)

    def test_explicit_cent_field_wins_over_yuan_field(self):
        self.assertEqual(parse_price_cents({"priceCent": 2650, "price": 26.5}), 2650)

    def test_missing_or_unparseable_price_returns_none(self):
        """审查 S-1：取不到价格必须是 None。用 0 当哨兵会让缺价餐品变成「免费」，
        以最优身份支配全表，把别人的可省金额算成原价。"""
        self.assertIsNone(parse_price_cents({}))
        self.assertIsNone(parse_price_cents({"price": "面议"}))


class TestReadPriceAndUnit(unittest.TestCase):
    """价格读数与整表单位判定（审查 S-5）。"""

    def test_read_price_records_explicit_units(self):
        self.assertEqual(read_price({"priceCent": 2650}).unit, "cent")
        self.assertEqual(read_price({"price": "¥26.50"}).unit, "yuan")
        self.assertEqual(read_price({"price": 26.5}).unit, "yuan")

    def test_read_price_marks_bare_integer_ambiguous(self):
        self.assertIsNone(read_price({"price": 26}).unit)
        self.assertIsNone(read_price({"price": "2650"}).unit)

    def test_read_price_missing_returns_none(self):
        self.assertIsNone(read_price({}))

    def test_table_of_yuan_integers_is_yuan(self):
        """审查 S-5 正面案例：¥200 的家庭桶在一张以元标价的菜单里必须读成 ¥200。"""
        readings = [read_price({"price": p}) for p in (200, 15, 26, 32, 12)]
        self.assertEqual(infer_price_unit(r for r in readings if r is not None), "yuan")

    def test_table_of_cent_integers_is_cent(self):
        readings = [read_price({"price": p}) for p in (2650, 1500, 2600, 1200, 3200)]
        self.assertEqual(infer_price_unit(r for r in readings if r is not None), "cent")

    def test_no_ambiguous_readings_defaults_to_yuan(self):
        self.assertEqual(infer_price_unit([]), "yuan")
        self.assertEqual(infer_price_unit([read_price({"price": "¥26.50"})]), "yuan")


class TestParseItem(unittest.TestCase):
    """单条记录 → MenuItem。缺名称必须返回 None（宁缺勿造）。"""

    def test_missing_name_returns_none(self):
        self.assertIsNone(parse_item({"priceCent": 2650, "energyKcal": 385}))
        self.assertIsNone(parse_item(None))

    def test_empty_name_returns_none(self):
        """空串名称由 ``pick`` 视为未命中，条目应被丢弃而不是带着空名字进入前沿。"""
        self.assertIsNone(parse_item({"name": ""}))

    def test_whitespace_only_name_returns_none(self):
        """测试 D3：``"   "`` 能通过 ``pick``，trim 之后却是空名，必须丢弃。"""
        self.assertIsNone(parse_item({"name": "   "}))
        self.assertIsNone(parse_item({"name": "\t\n "}))

    def test_tags_from_list(self):
        item = parse_item({"name": "麦辣鸡腿堡", "tags": ["新品", "辣"]})
        assert item is not None
        self.assertEqual(item.tags, ["新品", "辣"])

    def test_tags_from_delimited_string(self):
        item = parse_item({"name": "薯条", "tags": "新品, 辣|限定"})
        assert item is not None
        self.assertEqual(item.tags, ["新品", "辣", "限定"])

    def test_tags_missing_yields_empty_list(self):
        item = parse_item({"name": "薯条"})
        assert item is not None
        self.assertEqual(item.tags, [])

    def test_missing_code_falls_back_to_name(self):
        item = parse_item({"name": "薯条"})
        assert item is not None
        self.assertEqual(item.item_code, "薯条")

    def test_code_used_when_present(self):
        item = parse_item({"name": "薯条", "code": "M001"})
        assert item is not None
        self.assertEqual(item.item_code, "M001")

    def test_name_is_stripped_and_category_defaults(self):
        item = parse_item({"name": "  麦辣鸡腿堡  "})
        assert item is not None
        self.assertEqual(item.name, "麦辣鸡腿堡")
        self.assertEqual(item.category, "未分类")

    def test_price_and_nutrition_are_wired_in(self):
        item = parse_item(
            {"name": "麦辣鸡腿堡", "price": "¥26.50", "energyKcal": 385, "protein": 20}
        )
        assert item is not None
        self.assertEqual(item.price_cents, 2650)
        self.assertEqual(item.nutrition.energy_kcal, 385.0)
        self.assertEqual(item.nutrition.protein_g, 20.0)


class TestParseNutrition(unittest.TestCase):
    """营养结构：缺字段保持 None，不做 0 填充。"""

    def test_missing_fields_stay_none(self):
        nutrition = parse_nutrition({"energyKcal": 385})
        self.assertIsNone(nutrition.protein_g)
        self.assertIsNone(nutrition.sodium_mg)

    def test_chinese_field_names_parsed(self):
        nutrition = parse_nutrition(
            {"能量": 385, "蛋白质": "20 g", "脂肪": "12.5 g", "碳水化合物": 40, "钠": "800 mg"}
        )
        self.assertEqual(nutrition.energy_kcal, 385.0)
        self.assertEqual(nutrition.protein_g, 20.0)
        self.assertEqual(nutrition.fat_g, 12.5)
        self.assertEqual(nutrition.carb_g, 40.0)
        self.assertEqual(nutrition.sodium_mg, 800.0)


class TestMergeNutrition(unittest.TestCase):
    """营养表回填：编码优先、名称兜底、失败要可见。"""

    def test_matches_by_item_code(self):
        items = [parse_item({"name": "汉堡", "code": "M001"})]
        payload = {"data": {"foods": [{"code": "M001", "energyKcal": 385, "protein": 20}]}}
        merged, unmatched = merge_nutrition(items, payload)
        self.assertEqual(merged[0].nutrition.energy_kcal, 385.0)
        self.assertEqual(unmatched, [])

    def test_falls_back_to_name_when_code_differs(self):
        """两套编码体系不一致时按名称精确匹配，这是文档承诺的兜底路径。"""
        items = [parse_item({"name": "薯条", "code": "X9"})]
        payload = {"data": {"foods": [{"code": "OTHER", "name": "薯条", "energyKcal": 230}]}}
        merged, _ = merge_nutrition(items, payload)
        self.assertEqual(merged[0].nutrition.energy_kcal, 230.0)

    def test_unmatched_names_reported(self):
        items = [parse_item({"name": "不存在的餐品"})]
        payload = [{"code": "M001", "name": "薯条", "energyKcal": 230}]
        _, unmatched = merge_nutrition(items, payload)
        self.assertEqual(unmatched, ["不存在的餐品"])

    def test_mixed_batch_partitions_matched_and_unmatched(self):
        items = [
            parse_item({"name": "汉堡", "code": "M001"}),
            parse_item({"name": "薯条", "code": "X9"}),
            parse_item({"name": "孤儿餐品"}),
        ]
        payload = {
            "data": {
                "foods": [
                    {"code": "M001", "name": "汉堡", "energyKcal": 385},
                    {"code": "OTHER", "name": "薯条", "energyKcal": 230},
                ]
            }
        }
        merged, unmatched = merge_nutrition(items, payload)
        self.assertEqual(unmatched, ["孤儿餐品"])
        self.assertEqual([i.nutrition.energy_kcal for i in merged], [385.0, 230.0, None])

    def test_existing_fields_are_kept_and_missing_ones_filled(self):
        """审查 S-9：菜单已带 protein、缺能量/脂肪时，营养表的完整数据必须能补进来。

        旧实现只要 ``protein_g`` 非 None 就整条跳过，营养表等于没用。
        """
        items = [parse_item({"name": "汉堡", "code": "M1", "protein": 20})]
        payload = [{"code": "M1", "energyKcal": 385, "fat": 12, "sodium": 800, "protein": 99}]
        merged, unmatched = merge_nutrition(items, payload)
        nutrition = merged[0].nutrition
        self.assertEqual(unmatched, [])
        self.assertEqual(nutrition.protein_g, 20.0)  # 已有值不被覆盖
        self.assertEqual(nutrition.energy_kcal, 385.0)
        self.assertEqual(nutrition.fat_g, 12.0)
        self.assertEqual(nutrition.sodium_mg, 800.0)

    def test_all_empty_record_does_not_wipe_existing_data(self):
        """测试 D1：全是 None 的记录既不能冲掉已有热量，也不能计入匹配数。

        旧实现会把 ``energy=500`` 冲成 None，该餐品随即从前沿消失，日志还报「匹配 N/N」。
        """
        items = [parse_item({"name": "汉堡", "code": "M1", "energyKcal": 500, "protein": 25})]
        payload = [{"code": "M1", "name": "汉堡"}]
        merged, unmatched = merge_nutrition(items, payload)
        self.assertEqual(merged[0].nutrition.energy_kcal, 500.0)
        self.assertEqual(merged[0].nutrition.protein_g, 25.0)
        self.assertEqual(unmatched, ["汉堡"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
