"""Small end-to-end fixtures for profile semantics and accounting invariants.

Run from the project directory: python3 -m unittest discover -s tests -v
"""

import csv
import json
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import build_profiles as builder


class ProfilesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    @staticmethod
    def company(uid, **values):
        row = dict.fromkeys(builder.COMPANY_FIELDS, "")
        row["SYNTHETIC_U"] = uid
        row.update(values)
        return row

    @staticmethod
    def payment(payer, beneficiary, amount="1", count="1"):
        return {
            "PAYER": payer,
            "BENEFICIARY": beneficiary,
            "PAY_AMT": amount,
            "PAY_CNT": count,
        }

    @staticmethod
    def person(fid, uid, city="Москва", region="МОСКОВСКИЙ"):
        return {
            "SYNTHETIC_F": fid,
            "SYNTHETIC_U": uid,
            "CITY_NAME": city,
            "RC_NAME": region,
        }

    @staticmethod
    def write_csv(path, fields, rows, encoding="utf-8"):
        with path.open("w", encoding=encoding, newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter=";")
            writer.writeheader()
            writer.writerows(rows)

    def build(self, companies, payments=(), persons=(), top_n=10, encoding="utf-8"):
        paths = [self.root / name for name in ("companies.csv", "payments.csv", "persons.csv")]
        for path, fields, rows in zip(
            paths,
            (builder.COMPANY_FIELDS, builder.PAYMENT_FIELDS, builder.PERSON_FIELDS),
            (companies, payments, persons),
        ):
            self.write_csv(path, fields, rows, encoding=encoding)
        out_dir = self.root / "result"
        before = {path: path.read_bytes() for path in paths}
        summary = builder.build_profiles(*paths, out_dir=out_dir, top_n=top_n)
        self.assertEqual(before, {path: path.read_bytes() for path in paths})

        def read_jsonl(name):
            with (out_dir / name).open(encoding="utf-8") as stream:
                return [json.loads(line) for line in stream if line.strip()]

        result = {
            "companies": {r["synthetic_u"]: r for r in read_jsonl("company_profiles.jsonl")},
            "groups": {r["group_id"]: r for r in read_jsonl("group_profiles.jsonl")},
            "edges": read_jsonl("payment_edges.jsonl"),
            "issues": read_jsonl("data_issues.jsonl"),
            "summary": json.loads((out_dir / "summary.json").read_text(encoding="utf-8")),
            "out_dir": out_dir,
        }
        self.assertEqual(summary, result["summary"])
        return result

    def test_ordinary_company_and_group_preserve_semantics_and_coverage(self):
        result = self.build([
            self.company(
                "U_001", GROUP_ID="G_001", MAINOKVED_CODE="01.11",
                MAINOKVED_NAME="Выращивание зерновых", REGION="Регион А",
                CITY="Город А", CL_PRODUCT_LOA_MARK="1.0",
                NEGATIVEMARK="1.0", REVENUE_COMPANY="10.10",
                AMOUNT_WORKERS_FNS_GRADATION="Менее 15",
                AMOUNT_WORKERS_SPARK_GRADATION="16-100",
                COMPANY_AGE_YEARS_GRADATION="16-25",
            ),
            self.company(
                "U_002", GROUP_ID="G_001", MAINOKVED_CODE="02.20",
                MAINOKVED_NAME="Лесозаготовки", REGION="Регион Б",
                CL_PRODUCT_LOA_MARK="0.0", REVENUE_COMPANY="20.20",
            ),
            self.company("U_003", GROUP_ID="G_001"),
        ])
        company = result["companies"]["U_001"]
        self.assertEqual(company["status"], "ok")
        normalized = company["facts"]["normalized"]
        self.assertEqual(normalized["MAINOKVED_CODE"], "01.11")
        self.assertIs(normalized["NEGATIVEMARK"], True)
        self.assertEqual(normalized["AMOUNT_WORKERS_FNS_GRADATION"], "Менее 15")
        self.assertEqual(normalized["AMOUNT_WORKERS_SPARK_GRADATION"], "16-100")
        self.assertEqual(normalized["COMPANY_AGE_YEARS_GRADATION"], "16-25")
        self.assertEqual(company["facts"]["raw_records"][0]["values"]["NEGATIVEMARK"], "1.0")
        self.assertEqual(company["facts"]["raw_records"][0]["source"]["record"], 1)
        self.assertIn(
            {"kind": "key_owner_flag", "value": True,
             "basis": "facts.normalized.NEGATIVEMARK"},
            company["labels"],
        )
        for label in company["labels"]:
            self.assertTrue(label["basis"].startswith("facts.normalized."))
            self.assertIn(label["basis"].split(".")[-1], normalized)
            self.assertNotIn("risk", label["kind"])
        group = result["groups"]["G_001"]
        self.assertEqual(group["members"], ["U_001", "U_002", "U_003"])
        self.assertEqual(group["derived"]["member_count"], 3)
        product = group["derived"]["products"]["CL_PRODUCT_LOA_MARK"]
        self.assertEqual(
            (product["true_count"], product["false_count"], product["unknown_count"]),
            (1, 1, 1),
        )
        self.assertEqual(product["members_with_product"], ["U_001"])
        revenue = group["derived"]["members_revenue_sum"]
        self.assertEqual(Decimal(revenue["value"]), Decimal("30.30"))
        self.assertEqual((revenue["known_members"], revenue["total_members"]), (2, 3))
        for key in ("year", "unit", "currency"):
            self.assertIsNone(revenue[key])
        industries = {r["code"]: r["members"] for r in group["derived"]["industries"]}
        self.assertEqual(industries, {"01.11": ["U_001"], "02.20": ["U_002"]})

    def test_multiple_countries_are_trimmed_unique_and_member_specific(self):
        result = self.build([
            self.company("U_A", GROUP_ID="G", FTS_FLAG="1.0",
                         FTS_COUNTRY_NAME=" Китай, Индия,Китай,  , Индия "),
            self.company("U_B", GROUP_ID="G", FTS_FLAG="1.0",
                         FTS_COUNTRY_NAME="Китай"),
        ])
        countries = result["companies"]["U_A"]["facts"]["normalized"]["FTS_COUNTRY_NAME"]
        self.assertEqual(countries, sorted(["Китай", "Индия"]))
        group_countries = {
            row["value"]: row["members"]
            for row in result["groups"]["G"]["derived"]["fts_countries"]
        }
        self.assertEqual(group_countries, {"Индия": ["U_A"], "Китай": ["U_A", "U_B"]})

    def test_conflicting_id_has_variants_and_no_group_characteristics(self):
        result = self.build([
            self.company("U_X", GROUP_ID="G1", REVENUE_COMPANY="10", REGION="А"),
            self.company("U_X", GROUP_ID="G2", REVENUE_COMPANY="999", REGION="Б"),
            self.company("U_Y", GROUP_ID="G1", REVENUE_COMPANY="20"),
            self.company("U_Z", GROUP_ID="G1", REVENUE_COMPANY="30", REGION="А"),
            self.company("U_Z", GROUP_ID="G1", REVENUE_COMPANY="30", REGION="Б"),
        ])
        conflict = result["companies"]["U_X"]
        self.assertEqual(conflict["status"], "conflict")
        self.assertIsNone(conflict["facts"]["normalized"])
        self.assertIsNone(conflict["derived"]["group_id"])
        self.assertEqual(conflict["labels"], [])
        self.assertEqual(len(conflict["facts"]["raw_records"]), 2)
        self.assertEqual(result["companies"]["U_Z"]["status"], "conflict")
        self.assertIsNone(result["companies"]["U_Z"]["derived"]["group_id"])
        self.assertEqual(result["groups"]["G1"]["members"], ["U_Y"])
        self.assertNotIn("G2", result["groups"])
        self.assertEqual(
            Decimal(result["groups"]["G1"]["derived"]["members_revenue_sum"]["value"]),
            Decimal("20"),
        )
        issue_text = json.dumps(result["issues"], ensure_ascii=False)
        self.assertIn("U_X", issue_text)
        self.assertIn("999", issue_text)
        self.assertIn("G2", issue_text)

    def test_identical_company_rows_preserve_provenance_without_conflict(self):
        row = self.company("U_A", GROUP_ID="G", REVENUE_COMPANY="12.5")
        result = self.build([row, row.copy()])
        company = result["companies"]["U_A"]
        self.assertEqual(company["status"], "ok")
        self.assertEqual(len(company["facts"]["raw_records"]), 2)
        self.assertEqual(result["groups"]["G"]["members"], ["U_A"])
        self.assertEqual(
            Decimal(result["groups"]["G"]["derived"]["members_revenue_sum"]["value"]),
            Decimal("12.5"),
        )

    def test_unknown_counterparties_and_unknown_to_unknown_edges_survive(self):
        result = self.build(
            [self.company("U_A")],
            [
                self.payment("U_A", "U_UNKNOWN", "7", "3"),
                self.payment("U_UNKNOWN2", "U_A", "4", "2"),
                self.payment("U_UNKNOWN2", "U_UNKNOWN", "9", "5"),
            ],
        )
        self.assertEqual(len(result["edges"]), 3)
        self.assertEqual(set(result["companies"]), {"U_A"})
        payments = result["companies"]["U_A"]["derived"]["payments"]
        for direction in ("incoming", "outgoing"):
            self.assertEqual(payments[direction]["coverage"]["unknown_counterparties"], 1)
            self.assertEqual(payments[direction]["coverage"]["known_counterparties"], 0)
            self.assertEqual(payments[direction]["top_counterparties"][0]["company_status"], "unknown")
        edge = next(e for e in result["edges"] if e["payer"] == "U_A")
        self.assertEqual(edge["beneficiary_status"], "unknown")

    def test_person_links_do_not_multiply_payments_or_imply_group(self):
        result = self.build(
            [self.company("U_A"), self.company("U_B")],
            [self.payment("U_A", "U_B", "5.25", "2")],
            [self.person("F_1", "U_A"), self.person("F_2", "U_A"), self.person("F_1", "U_B")],
        )
        company = result["companies"]["U_A"]
        self.assertEqual(len(company["derived"]["persons"]), 2)
        self.assertEqual(Decimal(company["derived"]["payments"]["outgoing"]["amount"]), Decimal("5.25"))
        self.assertEqual(company["derived"]["payments"]["outgoing"]["count"], 2)
        self.assertEqual(len(result["edges"]), 1)
        self.assertEqual(result["groups"], {})
        observation = company["derived"]["persons"][0]["observations"][0]
        self.assertEqual(observation["raw"]["CITY_NAME"], "Москва")
        self.assertEqual(observation["raw"]["RC_NAME"], "МОСКОВСКИЙ")

    def test_self_payments_are_separate_from_normal_flows_and_rankings(self):
        result = self.build(
            [self.company("U_A"), self.company("U_B")],
            [self.payment("U_A", "U_A", "100", "7"),
             self.payment("U_A", "U_B", "3", "2")],
        )
        payments = result["companies"]["U_A"]["derived"]["payments"]
        self.assertEqual(Decimal(payments["self_payments"]["amount"]), Decimal("100"))
        self.assertEqual(payments["self_payments"]["count"], 7)
        self.assertEqual(Decimal(payments["outgoing"]["amount"]), Decimal("3"))
        self.assertEqual(payments["outgoing"]["unique_counterparties"], 1)
        self.assertEqual(payments["outgoing"]["top_counterparties"][0]["synthetic_u"], "U_B")
        self.assertEqual(payments["incoming"]["unique_counterparties"], 0)
        self.assertEqual(sum(e["is_self_payment"] for e in result["edges"]), 1)

    def test_repeated_pair_and_identical_payment_rows_are_all_counted(self):
        repeated = self.payment("U_A", "U_B", "0.10", "2")
        result = self.build(
            [self.company("U_A"), self.company("U_B")],
            [repeated, repeated.copy(), self.payment("U_A", "U_B", "0.20", "3")],
        )
        self.assertEqual(len(result["edges"]), 1)
        edge = result["edges"][0]
        self.assertEqual(Decimal(edge["amount"]), Decimal("0.40"))
        self.assertEqual(edge["count"], 7)
        self.assertEqual(edge["source_rows"], 3)
        self.assertEqual(edge["accepted_rows"], 3)
        self.assertEqual(edge["source"]["records"], [1, 2, 3])
        self.assertEqual(edge["duplicate_extra_rows"], 1)
        self.assertTrue(result["issues"])

    def test_large_pair_provenance_uses_reproducible_selector_without_losing_rows(self):
        repeated = self.payment("U_A", "U_B", "0.10", "1")
        rows = [
            repeated, repeated.copy(), repeated.copy(),
            self.payment(" U_A ", " U_B ", "0.20", "1"),
            self.payment("U_A", "U_B", "0.30", "2"),
        ]
        with patch.object(builder, "PROVENANCE_RECORD_LIMIT", 2):
            result = self.build(
                [self.company("U_A"), self.company("U_B")],
                rows,
            )
        self.assertEqual(len(result["edges"]), 1)
        edge = result["edges"][0]
        self.assertEqual(Decimal(edge["amount"]), Decimal("0.80"))
        self.assertEqual(edge["count"], 6)
        self.assertEqual(edge["accepted_rows"], 5)
        self.assertEqual(edge["source_rows"], 5)
        self.assertEqual(edge["duplicate_extra_rows"], 2)
        self.assertIsNone(edge["source"]["records"])
        self.assertEqual(edge["source"]["record_count"], 5)
        selector = edge["source"]["selector"]
        self.assertEqual(selector, {
            "PAYER": "U_A", "BENEFICIARY": "U_B",
            "comparison": "trimmed_endpoint_strings",
        })
        recovered = [
            row for row in rows
            if row["PAYER"].strip() == selector["PAYER"]
            and row["BENEFICIARY"].strip() == selector["BENEFICIARY"]
        ]
        self.assertEqual(len(recovered), edge["source"]["record_count"])
        duplicates = [issue for issue in result["issues"] if issue["code"] == "identical_payment_rows"]
        self.assertEqual(len(duplicates), 1)
        duplicate = duplicates[0]
        self.assertIsNone(duplicate["source"]["records"])
        self.assertEqual(duplicate["source"]["record_count"], 3)
        self.assertEqual(duplicate["extra_rows"], 2)
        self.assertEqual(duplicate["source"]["selector"]["exact_raw_fields"], repeated)
        self.assertEqual(
            sum(row == duplicate["source"]["selector"]["exact_raw_fields"] for row in rows),
            duplicate["source"]["record_count"],
        )
        self.assertTrue(result["summary"]["all_checks_passed"])

    def test_person_location_variants_and_unknown_company_links_are_preserved(self):
        persons = [
            self.person("F_1", "U_A", "Москва", "МОСКОВСКИЙ"),
            self.person("F_1", "U_B", "Санкт-Петербург", "СЕВЕРО-ЗАПАДНЫЙ"),
            self.person("F_1", "U_A", "Москва", "МОСКОВСКИЙ"),
            self.person("F_2", "U_UNKNOWN", "Омск", "СИБИРСКИЙ"),
        ]
        with patch.object(builder, "PROVENANCE_RECORD_LIMIT", 2):
            result = self.build(
                [self.company("U_A"), self.company("U_B")],
                persons=persons,
            )
        for uid in ("U_A", "U_B"):
            self.assertEqual(
                result["companies"][uid]["quality"]["person_ids_with_location_variants"],
                ["F_1"],
            )
        a_observations = result["companies"]["U_A"]["derived"]["persons"][0]["observations"]
        self.assertEqual(len(a_observations), 2)
        self.assertEqual([r["source"]["record"] for r in a_observations], [1, 3])
        variants = [issue for issue in result["issues"] if issue["code"] == "person_location_variants"]
        self.assertEqual(len(variants), 1)
        self.assertEqual(variants[0]["synthetic_f"], "F_1")
        self.assertEqual(variants[0]["source"]["selector"], {"SYNTHETIC_F": "F_1"})
        self.assertEqual(variants[0]["source"]["record_count"], 3)
        self.assertIsNone(variants[0]["source"]["records"])
        self.assertEqual(variants[0]["locations"], [
            {"CITY_NAME": "Москва", "RC_NAME": "МОСКОВСКИЙ"},
            {"CITY_NAME": "Санкт-Петербург", "RC_NAME": "СЕВЕРО-ЗАПАДНЫЙ"},
        ])
        unknown = [issue for issue in result["issues"] if issue["code"] == "person_unknown_company"]
        self.assertEqual(len(unknown), 1)
        self.assertEqual(unknown[0]["observation"]["raw"], persons[3])
        self.assertEqual(unknown[0]["observation"]["source"]["record"], 4)
        self.assertEqual(set(result["companies"]), {"U_A", "U_B"})
        self.assertEqual(result["summary"]["persons"]["unknown_company_rows"], 1)
        self.assertEqual(result["summary"]["persons"]["persons_with_location_variants"], 1)
        self.assertEqual(result["summary"]["persons"]["accepted_rows"], 4)

    def test_money_is_exact_beyond_default_decimal_precision(self):
        result = self.build(
            [self.company("U_A"), self.company("U_B")],
            [
                self.payment("U_A", "U_B", "123456789012345678901234567890.123456789"),
                self.payment("U_A", "U_B", "0.000000001"),
            ],
        )
        expected = Decimal("123456789012345678901234567890.123456790")
        edge = result["edges"][0]
        self.assertIsInstance(edge["amount"], str)
        self.assertEqual(Decimal(edge["amount"]), expected)
        outgoing = result["companies"]["U_A"]["derived"]["payments"]["outgoing"]
        self.assertEqual(Decimal(outgoing["amount"]), expected)

    def test_missing_and_invalid_payment_numbers_are_excluded_not_zeroed(self):
        result = self.build(
            [self.company("U_A"), self.company("U_B")],
            [
                self.payment("U_A", "U_B", "0", "0"),
                self.payment("U_A", "U_B", "1.25", "2"),
                self.payment("U_A", "U_B", "", "1"),
                self.payment("U_A", "U_B", "NaN", "1"),
                self.payment("U_A", "U_B", "abc", "1"),
                self.payment("U_A", "U_B", "6", ""),
                self.payment("U_A", "U_B", "7", "-1"),
                self.payment("U_A", "U_B", "8", "1.2"),
            ],
        )
        edge = result["edges"][0]
        self.assertEqual((edge["source_rows"], edge["accepted_rows"], edge["excluded_rows"]), (8, 2, 6))
        self.assertEqual(Decimal(edge["amount"]), Decimal("1.25"))
        self.assertEqual(edge["count"], 2)
        issues = result["issues"]
        reported_records = {
            issue["source"]["record"]
            for issue in issues
            if isinstance(issue.get("source"), dict) and "record" in issue["source"]
            and issue["source"]["file"].endswith("payments.csv")
        }
        self.assertTrue(set(range(3, 9)).issubset(reported_records))

    def test_finite_negative_amount_is_preserved_without_inventing_its_meaning(self):
        result = self.build(
            [self.company("U_A"), self.company("U_B")],
            [
                self.payment("U_A", "U_B", "10.20", "2"),
                self.payment("U_A", "U_B", "-0.10", "1"),
            ],
        )
        edge = result["edges"][0]
        self.assertEqual(Decimal(edge["amount"]), Decimal("10.10"))
        self.assertEqual(edge["count"], 3)
        self.assertEqual(edge["accepted_rows"], 2)
        self.assertEqual(edge["excluded_rows"], 0)

    def test_all_invalid_edge_keeps_unknown_totals(self):
        result = self.build(
            [self.company("U_A"), self.company("U_B")],
            [self.payment("U_A", "U_B", "", "1")],
        )
        edge = result["edges"][0]
        self.assertIsNone(edge["amount"])
        self.assertIsNone(edge["count"])
        self.assertEqual(edge["accepted_rows"], 0)
        self.assertEqual(edge["excluded_rows"], 1)
        outgoing = result["companies"]["U_A"]["derived"]["payments"]["outgoing"]
        self.assertIsNone(outgoing["amount"])
        self.assertIsNone(outgoing["count"])

    def test_missing_counterparty_is_an_excluded_observation_for_known_endpoint(self):
        result = self.build(
            [self.company("U_A"), self.company("U_B")],
            [
                self.payment("U_A", "", "10", "2"),
                self.payment("", "U_A", "20", "3"),
                self.payment("U_A", "", "", "1"),
                self.payment("", "", "30", "4"),
            ],
        )
        self.assertEqual(result["edges"], [])
        payments = result["companies"]["U_A"]["derived"]["payments"]
        self.assertEqual(payments["observation_status"], "only_excluded")
        self.assertEqual(payments["unpairable_rows"], 3)
        self.assertEqual(
            payments["unpairable_source"],
            {"file": "payments.csv", "records": [1, 2, 3]},
        )
        for direction, expected_rows in (("incoming", 1), ("outgoing", 2)):
            flow = payments[direction]
            self.assertEqual(flow["source_rows"], expected_rows)
            self.assertEqual(flow["excluded_rows"], expected_rows)
            self.assertEqual(flow["accepted_rows"], 0)
            self.assertEqual(flow["unique_counterparties"], 0)
            self.assertEqual(flow["top_counterparties"], [])
            self.assertIsNone(flow["amount"])
            self.assertIsNone(flow["count"])
        untouched = result["companies"]["U_B"]["derived"]["payments"]
        self.assertEqual(untouched["observation_status"], "no_observations")
        self.assertEqual(untouched["unpairable_rows"], 0)
        self.assertIsNone(untouched["unpairable_source"])
        self.assertEqual(result["summary"]["payments"]["excluded_rows"], 4)
        self.assertTrue(result["summary"]["all_checks_passed"])

    def test_blank_group_ids_are_not_one_group_and_missing_flags_stay_null(self):
        result = self.build([self.company("U_A"), self.company("U_B", GROUP_ID="  ")])
        self.assertEqual(result["groups"], {})
        for company in result["companies"].values():
            self.assertIsNone(company["derived"]["group_id"])
            self.assertIsNone(company["facts"]["normalized"]["CL_PRODUCT_LOA_MARK"])
            self.assertEqual(company["derived"]["payments"]["outgoing"]["source_rows"], 0)
            self.assertEqual(company["derived"]["payments"]["incoming"]["source_rows"], 0)
            self.assertEqual(company["derived"]["payments"]["observation_status"], "no_observations")

    def test_bom_and_unknown_columns_preserve_source_values(self):
        fields = list(builder.COMPANY_FIELDS) + ["EXTRA"]
        row = self.company("U_0001", MAINOKVED_CODE="01.01")
        row["EXTRA"] = "  исходный;текст  "
        paths = [self.root / name for name in ("companies.csv", "payments.csv", "persons.csv")]
        self.write_csv(paths[0], fields, [row], encoding="utf-8-sig")
        self.write_csv(paths[1], builder.PAYMENT_FIELDS, [], encoding="utf-8-sig")
        self.write_csv(paths[2], builder.PERSON_FIELDS, [], encoding="utf-8-sig")
        output = self.root / "result"
        builder.build_profiles(*paths, out_dir=output)
        company = json.loads((output / "company_profiles.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(company["synthetic_u"], "U_0001")
        self.assertEqual(company["facts"]["normalized"]["MAINOKVED_CODE"], "01.01")
        self.assertEqual(company["facts"]["raw_records"][0]["values"]["EXTRA"], "  исходный;текст  ")

    def test_missing_required_column_fails_clearly(self):
        paths = [self.root / name for name in ("companies.csv", "payments.csv", "persons.csv")]
        self.write_csv(paths[0], builder.COMPANY_FIELDS, [])
        self.write_csv(paths[1], ["PAYER", "BENEFICIARY", "PAY_CNT"], [])
        self.write_csv(paths[2], builder.PERSON_FIELDS, [])
        with self.assertRaisesRegex(ValueError, "PAY_AMT"):
            builder.build_profiles(*paths, out_dir=self.root / "result")

    def test_deterministic_outputs_and_top_n_tie_break(self):
        result = self.build(
            [self.company("U_C"), self.company("U_A"), self.company("U_B")],
            [self.payment("U_A", "U_C", "2"), self.payment("U_A", "U_B", "2")],
            top_n=1,
        )
        outgoing = result["companies"]["U_A"]["derived"]["payments"]["outgoing"]
        self.assertEqual(outgoing["unique_counterparties"], 2)
        self.assertEqual(outgoing["top_counterparties"][0]["synthetic_u"], "U_B")
        filenames = [
            "company_profiles.jsonl", "group_profiles.jsonl", "payment_edges.jsonl",
            "data_issues.jsonl", "summary.json",
        ]
        first_bytes = {name: (result["out_dir"] / name).read_bytes() for name in filenames}
        builder.build_profiles(
            self.root / "companies.csv", self.root / "payments.csv",
            self.root / "persons.csv", out_dir=result["out_dir"], top_n=1,
        )
        self.assertEqual(
            first_bytes,
            {name: (result["out_dir"] / name).read_bytes() for name in filenames},
        )
        self.assertEqual(list(result["companies"]), ["U_A", "U_B", "U_C"])


if __name__ == "__main__":
    unittest.main()
