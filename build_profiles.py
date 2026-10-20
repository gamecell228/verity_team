#!/usr/bin/env python3
"""Build auditable company/group profiles with only the Python standard library."""

import argparse
import csv
import hashlib
import itertools
import json
import re
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation, Inexact, localcontext
from pathlib import Path


COMPANY_FIELDS = (
    "SYNTHETIC_U", "ORG_TYPE", "OKFS_NAME", "COMPANY_AGE_YEARS_GRADATION",
    "ECO_MARK_LABEL", "MAINOKVED_CODE", "MAINOKVED_NAME", "NEGATIVEMARK",
    "AMOUNT_WORKERS_FNS_GRADATION", "AMOUNT_WORKERS_SPARK_GRADATION", "CL_TYPE",
    "CL_PRODUCT_RKO_MARK", "CL_PRODUCT_GAR_MARK", "CL_PRODUCT_LOA_MARK",
    "CL_PRODUCT_LEA_MARK", "CL_PRODUCT_DEP_MARK", "CL_PRODUCT_ZPP_MARK",
    "FTS_FLAG", "FTS_COUNTRY_NAME", "FACTORING_MARK", "REVENUE_COMPANY",
    "GROUP_ID", "CITY", "REGION",
)
PAYMENT_FIELDS = ("BENEFICIARY", "PAYER", "PAY_AMT", "PAY_CNT")
PERSON_FIELDS = ("SYNTHETIC_F", "CITY_NAME", "RC_NAME", "SYNTHETIC_U")
PRODUCT_FIELDS = tuple(f for f in COMPANY_FIELDS if f.startswith("CL_PRODUCT_")) + (
    "FACTORING_MARK",
)
FLAG_FIELDS = PRODUCT_FIELDS + ("NEGATIVEMARK", "FTS_FLAG")
OUTPUT_NAMES = (
    "company_profiles.jsonl", "group_profiles.jsonl", "payment_edges.jsonl",
    "data_issues.jsonl", "summary.json",
)
NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
ZERO = Decimal(0)
# A display/provenance bound, never a bound on input size or accepted row count.
PROVENANCE_RECORD_LIMIT = 1000


class InputError(ValueError):
    """An actionable input/schema error, suitable for a CLI message."""


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def write_jsonl(stream, value):
    stream.write(dumps(value) + "\n")


def text(value):
    return value.strip() or None


def decimal_value(value):
    value = value.strip()
    if not value:
        raise ValueError("missing")
    if not NUMBER.fullmatch(value):
        raise ValueError("invalid_decimal")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("invalid_decimal") from exc
    if not result.is_finite():
        raise ValueError("nonfinite_decimal")
    return result


def money(value):
    """Canonical fixed-point string; never rounds and never uses float."""
    result = format(value, "f")
    if "." in result:
        result = result.rstrip("0").rstrip(".")
    return "0" if value == 0 else result


def add_exact(left, right):
    """Decimal's default 28-digit context is insufficient for arbitrary totals."""
    least_exponent = min(left.as_tuple().exponent, right.as_tuple().exponent)
    largest_position = max(left.adjusted(), right.adjusted(), 0)
    with localcontext() as ctx:
        ctx.prec = max(1, largest_position - least_exponent + 3)
        ctx.traps[Inexact] = True
        return left + right


def count_value(value):
    number = decimal_value(value)
    if number < 0 or number != number.to_integral_value():
        raise ValueError("count_must_be_nonnegative_integer")
    return int(number)


def source(path, record):
    return {"file": path.name, "record": record}


def record_reference(path, records, total, selector):
    """Large provenance lists use an exact reproducible source predicate."""
    if total <= PROVENANCE_RECORD_LIMIT:
        return {"file": path.name, "records": records}
    return {"file": path.name, "records": None, "record_count": total,
            "selector": selector}


def csv_rows(path, required):
    """Record numbers count data records, not physical lines or the header."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter=";", strict=True)
            fields = reader.fieldnames
            if not fields or len(fields) != len(set(fields)):
                raise InputError(f"{path}: missing or duplicate CSV headers")
            missing = sorted(set(required) - set(fields))
            if missing:
                raise InputError(f"{path}: missing required columns: {', '.join(missing)}")
            for record, row in enumerate(reader, 1):
                if None in row or any(value is None for value in row.values()):
                    raise InputError(f"{path}: malformed CSV record {record}: column count")
                yield record, row
    except (UnicodeError, csv.Error) as exc:
        raise InputError(f"{path}: cannot read UTF-8 semicolon CSV: {exc}") from exc


def metadata(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        columns = next(csv.reader(stream, delimiter=";"), [])
    return {"file": path.name, "path": str(path.resolve()), "sha256": digest.hexdigest(),
            "bytes": path.stat().st_size, "columns": columns}


class Issues:
    def __init__(self, stream):
        self.stream = stream
        self.counts = Counter()

    def emit(self, code, location, **details):
        self.counts[code] += 1
        write_jsonl(self.stream, {"code": code, "source": location, **details})


def normalize_company(row, location, issues):
    result = {key: text(value) for key, value in row.items()}
    invalid_fields = []
    for field in FLAG_FIELDS:
        if result[field] is None:
            continue
        try:
            parsed = decimal_value(row[field])
            if parsed not in (ZERO, Decimal(1)):
                raise ValueError("flag_must_be_0_or_1")
            result[field] = bool(parsed)
        except ValueError as exc:
            result[field] = None
            invalid_fields.append(field)
            issues.emit("invalid_company_flag", location, field=field,
                        raw_value=row[field], reason=str(exc))
    countries = sorted({v.strip() for v in row["FTS_COUNTRY_NAME"].split(",") if v.strip()})
    result["FTS_COUNTRY_NAME"] = countries if text(row["FTS_COUNTRY_NAME"]) else None
    if result["REVENUE_COMPANY"] is not None:
        try:
            revenue = decimal_value(row["REVENUE_COMPANY"])
            result["REVENUE_COMPANY"] = money(revenue)
            if revenue < 0:
                issues.emit("negative_revenue", location, raw_value=row["REVENUE_COMPANY"],
                            policy="preserved_without_invented_business_rule")
        except ValueError as exc:
            result["REVENUE_COMPANY"] = None
            invalid_fields.append("REVENUE_COMPANY")
            issues.emit("invalid_company_revenue", location,
                        raw_value=row["REVENUE_COMPANY"], reason=str(exc))
    if result["FTS_FLAG"] is False and countries:
        issues.emit("fts_countries_with_false_flag", location,
                    synthetic_u=result["SYNTHETIC_U"], countries=countries)
    elif result["FTS_FLAG"] is True and not countries:
        issues.emit("fts_flag_without_countries", location,
                    synthetic_u=result["SYNTHETIC_U"])
    return result, invalid_fields


def load_companies(db, path, issues):
    db.executescript("""
        CREATE TABLE company_rows(uid TEXT, record INTEGER, raw TEXT, normalized TEXT,
                                  invalid_fields TEXT, comparison TEXT);
        CREATE TABLE companies(uid TEXT PRIMARY KEY, status TEXT, group_id TEXT,
                               payload TEXT);
    """)
    rows = 0
    skipped = 0
    missing = Counter()
    for record, raw in csv_rows(path, COMPANY_FIELDS):
        rows += 1
        missing.update(key for key, value in raw.items() if not value.strip())
        location = source(path, record)
        normalized, invalid = normalize_company(raw, location, issues)
        uid = normalized["SYNTHETIC_U"]
        if uid is None:
            skipped += 1
            issues.emit("company_missing_id", location, raw=raw, excluded=True)
            continue
        # Distinct invalid raw values must not collapse just because both normalize to null.
        comparison = dict(normalized)
        for field in invalid:
            comparison[field] = {"invalid_raw": raw[field]}
        db.execute("INSERT INTO company_rows VALUES(?,?,?,?,?,?)",
                   (uid, record, dumps(raw), dumps(normalized), dumps(invalid), dumps(comparison)))
    db.execute("CREATE INDEX company_rows_uid ON company_rows(uid,record)")
    statuses = {}
    conflicts = []
    duplicate_ids = 0
    duplicate_extra = 0
    cursor = db.execute(
        "SELECT uid,record,raw,normalized,invalid_fields,comparison "
        "FROM company_rows ORDER BY uid,record")
    for uid, batch in itertools.groupby(cursor, key=lambda r: r[0]):
        variants = list(batch)
        comparisons = [json.loads(r[5]) for r in variants]
        conflicting_fields = sorted(
            key for key in comparisons[0]
            if len({dumps(value[key]) for value in comparisons}) > 1)
        status = "conflict" if conflicting_fields else "ok"
        statuses[uid] = status
        raw_records = [{"source": source(path, r[1]), "values": json.loads(r[2])}
                       for r in variants]
        normalized = None if conflicting_fields else json.loads(variants[0][3])
        if len(variants) > 1:
            duplicate_ids += 1
            duplicate_extra += len(variants) - 1
        if conflicting_fields:
            conflicts.append(uid)
            issues.emit("company_conflict", {"file": path.name,
                        "records": [r[1] for r in variants]}, synthetic_u=uid,
                        fields=conflicting_fields, variants=raw_records,
                        excluded_from_groups=True)
        elif len(variants) > 1:
            issues.emit("equivalent_company_rows", {"file": path.name,
                        "records": [r[1] for r in variants]}, synthetic_u=uid,
                        policy="one_profile_all_raw_records_preserved")
        quality = {
            "source_rows": len(variants), "conflicting_fields": conflicting_fields,
            "missing_fields": sorted(k for k, v in normalized.items() if v is None)
                              if normalized is not None else None,
            "missing_fields_by_record": [
                {"record": r[1], "fields": sorted(k for k, v in json.loads(r[2]).items()
                                                if not v.strip())}
                for r in variants],
            "invalid_fields": sorted({f for r in variants for f in json.loads(r[4])}),
            "group_aggregation_eligible": status == "ok" and normalized["GROUP_ID"] is not None,
        }
        profile = {"synthetic_u": uid, "status": status,
                   "facts": {"raw_records": raw_records, "normalized": normalized},
                   "quality": quality}
        db.execute("INSERT INTO companies VALUES(?,?,?,?)",
                   (uid, status, normalized["GROUP_ID"] if normalized else None, dumps(profile)))
    db.execute("CREATE INDEX companies_group ON companies(group_id,uid)")
    db.commit()
    return statuses, {
        "rows": rows, "rows_without_id": skipped, "unique_ids": len(statuses),
        "conflict_ids": conflicts, "conflict_count": len(conflicts),
        "duplicate_ids": duplicate_ids, "duplicate_extra_rows": duplicate_extra,
        "missing_by_field": dict(sorted(missing.items())),
    }


def load_persons(db, path, statuses, issues):
    db.execute("CREATE TABLE persons(uid TEXT, fid TEXT, record INTEGER, payload TEXT)")
    db.execute("CREATE TABLE person_location_variants(fid TEXT PRIMARY KEY)")
    rows = accepted = unknown_rows = 0
    unknown_ids = set()
    fids = set()
    missing = Counter()
    for record, raw in csv_rows(path, PERSON_FIELDS):
        rows += 1
        missing.update(key for key, value in raw.items() if not value.strip())
        normalized = {key: text(value) for key, value in raw.items()}
        uid, fid = normalized["SYNTHETIC_U"], normalized["SYNTHETIC_F"]
        if uid is None or fid is None:
            issues.emit("person_link_missing_id", source(path, record), raw=raw, excluded=True)
            continue
        accepted += 1
        fids.add(fid)
        observation = {"source": source(path, record), "raw": raw, "normalized": normalized}
        db.execute("INSERT INTO persons VALUES(?,?,?,?)", (uid, fid, record, dumps(observation)))
        if uid not in statuses:
            unknown_rows += 1
            unknown_ids.add(uid)
            issues.emit("person_unknown_company", source(path, record),
                        synthetic_u=uid, synthetic_f=fid, observation=observation,
                        policy="link_preserved_here_no_company_characteristics_known")
    db.execute("CREATE INDEX persons_uid ON persons(uid,fid,record)")
    db.execute("CREATE INDEX persons_fid ON persons(fid,record)")
    location_variants = 0
    cursor = db.execute("SELECT fid,record,payload FROM persons ORDER BY fid,record")
    for fid, batch in itertools.groupby(cursor, key=lambda r: r[0]):
        locations = set()
        records = []
        total = 0
        for _, record, payload in batch:
            total += 1
            if len(records) < PROVENANCE_RECORD_LIMIT:
                records.append(record)
            normalized = json.loads(payload)["normalized"]
            location = (normalized["CITY_NAME"], normalized["RC_NAME"])
            if any(location):
                locations.add(location)
        if len(locations) > 1:
            location_variants += 1
            db.execute("INSERT INTO person_location_variants VALUES(?)", (fid,))
            issues.emit("person_location_variants",
                        record_reference(path, records, total, {"SYNTHETIC_F": fid}),
                        synthetic_f=fid,
                        locations=[{"CITY_NAME": city, "RC_NAME": region}
                                   for city, region in sorted(locations,
                                                             key=lambda v: (v[0] or "", v[1] or ""))],
                        policy="all_observations_preserved_no_period_to_resolve_variants")
    db.commit()
    return {"rows": rows, "accepted_rows": accepted, "excluded_rows": rows - accepted,
            "unique_persons": len(fids), "unknown_company_rows": unknown_rows,
            "unknown_company_ids": len(unknown_ids),
            "persons_with_location_variants": location_variants,
            "missing_by_field": dict(sorted(missing.items()))}


def load_payments(db, path, statuses, issues):
    db.execute("""CREATE TABLE payments(
        record INTEGER PRIMARY KEY, payer TEXT, beneficiary TEXT, amount TEXT,
        count TEXT, accepted INTEGER, raw TEXT, duplicate_extra INTEGER DEFAULT 0)""")
    rows = accepted = excluded = 0
    valid_amount = ZERO
    valid_count = 0
    self_amount = ZERO
    self_count = self_rows = 0
    endpoint_ids = set()
    known_rows = unknown_rows = 0
    for record, raw in csv_rows(path, PAYMENT_FIELDS):
        rows += 1
        payer, beneficiary = text(raw["PAYER"]), text(raw["BENEFICIARY"])
        problems = []
        for field, value in (("PAYER", payer), ("BENEFICIARY", beneficiary)):
            if value is None:
                problems.append({"field": field, "reason": "missing"})
        try:
            amount = decimal_value(raw["PAY_AMT"])
        except ValueError as exc:
            amount = None
            problems.append({"field": "PAY_AMT", "reason": str(exc)})
        try:
            count = count_value(raw["PAY_CNT"])
        except ValueError as exc:
            count = None
            problems.append({"field": "PAY_CNT", "reason": str(exc)})
        is_accepted = not problems
        if problems:
            excluded += 1
            issues.emit("payment_row_excluded", source(path, record), raw=raw,
                        problems=problems, excluded_from=["PAY_AMT", "PAY_CNT"],
                        parsed_amount=money(amount) if amount is not None else None,
                        parsed_count=count)
        else:
            accepted += 1
            valid_amount = add_exact(valid_amount, amount)
            valid_count += count
            if payer == beneficiary:
                self_amount = add_exact(self_amount, amount)
                self_count += count
                self_rows += 1
            if amount < 0:
                issues.emit("negative_payment_amount", source(path, record),
                            raw_value=raw["PAY_AMT"], policy="included_as_signed_amount")
        endpoint_ids.update(v for v in (payer, beneficiary) if v is not None)
        if payer in statuses and beneficiary in statuses:
            known_rows += 1
        else:
            unknown_rows += 1
        # Invalid numeric rows still form an observed edge if both endpoints exist.
        db.execute("INSERT INTO payments(record,payer,beneficiary,amount,count,accepted,raw) "
                   "VALUES(?,?,?,?,?,?,?)",
                   (record, payer, beneficiary, money(amount) if amount is not None else None,
                    str(count) if count is not None else None, int(is_accepted), dumps(raw)))
        if rows % 50000 == 0:
            db.commit()
    db.execute("CREATE INDEX payments_pair ON payments(payer,beneficiary,record)")
    db.execute("CREATE INDEX payments_raw ON payments(raw,record)")
    db.commit()
    duplicate_sets = duplicate_extra = 0
    for raw, n in db.execute(
            "SELECT raw,COUNT(*) FROM payments GROUP BY raw HAVING COUNT(*)>1 ORDER BY raw"):
        duplicate_sets += 1
        duplicate_extra += n - 1
        records = [r[0] for r in db.execute(
            "SELECT record FROM payments WHERE raw=? ORDER BY record LIMIT ?",
            (raw, PROVENANCE_RECORD_LIMIT))]
        db.execute("UPDATE payments SET duplicate_extra=1 WHERE raw=? AND record<>?",
                   (raw, records[0]))
        issues.emit("identical_payment_rows",
                    record_reference(path, records, n, {"exact_raw_fields": json.loads(raw)}),
                    raw=json.loads(raw), source_rows=n, extra_rows=n - 1,
                    policy="all_rows_retained_no_operation_id_or_period")
    return {
        "rows": rows, "accepted_rows": accepted, "excluded_rows": excluded,
        "accepted_amount": money(valid_amount), "accepted_count": valid_count,
        "self_accepted_rows": self_rows, "self_amount": money(self_amount), "self_count": self_count,
        "both_endpoints_known_rows": known_rows, "at_least_one_unknown_or_missing_rows": unknown_rows,
        "unique_endpoint_ids": len(endpoint_ids),
        "unknown_endpoint_ids": len(endpoint_ids - statuses.keys()),
        "identical_row_sets": duplicate_sets, "identical_extra_rows": duplicate_extra,
    }, endpoint_ids


def build_edges(db, path, output, statuses):
    db.execute("""CREATE TABLE edges(
        payer TEXT, beneficiary TEXT, amount TEXT, count TEXT,
        accepted_rows INTEGER, source_rows INTEGER, excluded_rows INTEGER,
        duplicate_extra_rows INTEGER, PRIMARY KEY(payer,beneficiary))""")
    totals = {"amount": ZERO, "count": 0, "accepted_rows": 0, "source_rows": 0,
              "edges": 0, "self_edges": 0, "repeated_pairs": 0, "repeated_pair_extra_rows": 0}
    endpoints = set()
    cursor = db.execute(
        "SELECT payer,beneficiary,record,amount,count,accepted,duplicate_extra FROM payments "
        "WHERE payer IS NOT NULL AND beneficiary IS NOT NULL ORDER BY payer,beneficiary,record")
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for pair, batch in itertools.groupby(cursor, key=lambda r: (r[0], r[1])):
            amount = ZERO
            count = accepted = 0
            records = []
            duplicates = 0
            n = 0
            for payer, beneficiary, record, amt, cnt, valid, duplicate_extra in batch:
                n += 1
                if len(records) < PROVENANCE_RECORD_LIMIT:
                    records.append(record)
                duplicates += duplicate_extra
                if valid:
                    amount = add_exact(amount, Decimal(amt))
                    count += int(cnt)
                    accepted += 1
            payer, beneficiary = pair
            endpoints.update(pair)
            edge = {
                "payer": payer, "beneficiary": beneficiary, "is_self_payment": payer == beneficiary,
                "payer_status": statuses.get(payer, "unknown"),
                "beneficiary_status": statuses.get(beneficiary, "unknown"),
                "amount": money(amount) if accepted else None, "count": count if accepted else None,
                "source_rows": n, "accepted_rows": accepted, "excluded_rows": n - accepted,
                "duplicate_extra_rows": duplicates,
                "source": record_reference(path, records, n, {"PAYER": payer, "BENEFICIARY": beneficiary,
                                                            "comparison": "trimmed_endpoint_strings"}),
                "currency": None, "period": None,
            }
            write_jsonl(stream, edge)
            db.execute("INSERT INTO edges VALUES(?,?,?,?,?,?,?,?)", (
                payer, beneficiary, edge["amount"], str(count) if accepted else None,
                accepted, n, n - accepted, duplicates))
            totals["amount"] = add_exact(totals["amount"], amount)
            totals["count"] += count
            totals["accepted_rows"] += accepted
            totals["source_rows"] += n
            totals["edges"] += 1
            totals["self_edges"] += int(payer == beneficiary)
            if n > 1:
                totals["repeated_pairs"] += 1
                totals["repeated_pair_extra_rows"] += n - 1
    db.execute("CREATE INDEX edges_beneficiary ON edges(beneficiary,payer)")
    db.commit()
    totals["amount"] = money(totals["amount"])
    return totals, endpoints


def empty_flow(counterparties=True):
    result = {"amount": None, "count": None, "source_rows": 0,
              "accepted_rows": 0, "excluded_rows": 0}
    if counterparties:
        result.update(unique_counterparties=0, top_counterparties=[],
                      coverage={"known_counterparties": 0, "unknown_counterparties": 0,
                                "conflict_counterparties": 0})
    return result


def empty_payments():
    return {"observation_status": "no_observations", "incoming": empty_flow(),
            "outgoing": empty_flow(), "self_payments": empty_flow(False),
            "unpairable_rows": 0, "unpairable_source": None,
            "currency": None, "period": None}


def add_to_flow(flow, amount, count, accepted, rows, excluded):
    if accepted:
        flow["amount"] = money(add_exact(Decimal(flow["amount"] or "0"), Decimal(amount)))
        flow["count"] = (flow["count"] or 0) + int(count)
    flow["source_rows"] += rows
    flow["accepted_rows"] += accepted
    flow["excluded_rows"] += excluded


def build_payment_summaries(db, statuses, top_n, payment_path):
    db.execute("CREATE TABLE payment_summaries(uid TEXT PRIMARY KEY, payload TEXT)")
    # Sort on disk once; no whole payment scan for each company and no person join.
    cursor = db.execute("""
        SELECT payer AS owner, 'outgoing' AS direction, beneficiary AS other,
               amount,count,accepted_rows,source_rows,excluded_rows,NULL AS unpairable_record
        FROM edges WHERE payer<>beneficiary
        UNION ALL
        SELECT beneficiary AS owner,'incoming' AS direction,payer AS other,
               amount,count,accepted_rows,source_rows,excluded_rows,NULL AS unpairable_record
        FROM edges WHERE payer<>beneficiary
        UNION ALL
        SELECT payer AS owner,'self_payments' AS direction,beneficiary AS other,
               amount,count,accepted_rows,source_rows,excluded_rows,NULL AS unpairable_record
        FROM edges WHERE payer=beneficiary
        UNION ALL
        SELECT payer AS owner,'outgoing' AS direction,NULL AS other,
               NULL,NULL,0,1,1,record AS unpairable_record
        FROM payments WHERE payer IS NOT NULL AND beneficiary IS NULL
        UNION ALL
        SELECT beneficiary AS owner,'incoming' AS direction,NULL AS other,
               NULL,NULL,0,1,1,record AS unpairable_record
        FROM payments WHERE beneficiary IS NOT NULL AND payer IS NULL
        ORDER BY owner,direction,other,unpairable_record
    """)
    totals = {direction: empty_flow(False)
              for direction in ("incoming", "outgoing", "self_payments")}
    owners = 0
    for owner, batch in itertools.groupby(cursor, key=lambda r: r[0]):
        owners += 1
        summary = empty_payments()
        unpairable_records = []
        unpairable_count = 0
        for _, direction, other, amount, count, accepted, rows, excluded, unpairable_record in batch:
            flow = summary[direction]
            add_to_flow(flow, amount, count, accepted, rows, excluded)
            add_to_flow(totals[direction], amount, count, accepted, rows, excluded)
            if unpairable_record is not None:
                unpairable_count += 1
                if len(unpairable_records) < PROVENANCE_RECORD_LIMIT:
                    unpairable_records.append(unpairable_record)
                continue
            if direction == "self_payments":
                continue
            flow["unique_counterparties"] += 1
            status = statuses.get(other, "unknown")
            key = "unknown_counterparties" if status == "unknown" else "known_counterparties"
            flow["coverage"][key] += 1
            if status == "conflict":
                flow["coverage"]["conflict_counterparties"] += 1
            if accepted and top_n:
                item = {"synthetic_u": other, "amount": amount, "count": int(count),
                        "company_status": status, "source_rows": rows, "accepted_rows": accepted,
                        "excluded_rows": excluded,
                        "edge": {"payer": owner if direction == "outgoing" else other,
                                 "beneficiary": other if direction == "outgoing" else owner}}
                # Stable tie-break by ID, then stable exact Decimal descending sort.
                top = flow["top_counterparties"]
                top.append(item)
                top.sort(key=lambda v: v["synthetic_u"])
                top.sort(key=lambda v: Decimal(v["amount"]), reverse=True)
                del top[top_n:]
        accepted_rows = sum(summary[d]["accepted_rows"] for d in totals)
        summary["observation_status"] = "observed" if accepted_rows else "only_excluded"
        summary["unpairable_rows"] = unpairable_count
        if unpairable_count:
            summary["unpairable_source"] = record_reference(
                payment_path, sorted(unpairable_records), unpairable_count,
                {"endpoint_id": owner, "other_endpoint": "missing",
                 "comparison": "trimmed_endpoint_strings"})
        db.execute("INSERT INTO payment_summaries VALUES(?,?)", (owner, dumps(summary)))
    db.commit()
    return {"all_endpoints": totals, "endpoint_summaries": owners}


def company_labels(normalized):
    if normalized is None:
        return []
    labels = []
    fields = {
        "MAINOKVED_CODE": "okved", "REGION": "registration_region",
        "CITY": "registration_city", "ORG_TYPE": "organization_type",
        "OKFS_NAME": "ownership_form", "CL_TYPE": "client_type",
        "ECO_MARK_LABEL": "economic_stage",
        "COMPANY_AGE_YEARS_GRADATION": "company_age_category",
        "AMOUNT_WORKERS_FNS_GRADATION": "workers_fns_category",
        "AMOUNT_WORKERS_SPARK_GRADATION": "workers_spark_category",
    }
    for field, kind in fields.items():
        if normalized.get(field) is not None:
            labels.append({"kind": kind, "value": normalized[field],
                           "basis": f"facts.normalized.{field}"})
    for country in normalized["FTS_COUNTRY_NAME"] or []:
        labels.append({"kind": "fts_country", "value": country,
                       "basis": "facts.normalized.FTS_COUNTRY_NAME"})
    for field in PRODUCT_FIELDS:
        if normalized[field] is True:
            labels.append({"kind": "bank_product", "value": field,
                           "basis": f"facts.normalized.{field}"})
    if normalized["NEGATIVEMARK"] is True:
        labels.append({"kind": "key_owner_flag", "value": True,
                       "basis": "facts.normalized.NEGATIVEMARK"})
    if normalized["FTS_FLAG"] is True:
        labels.append({"kind": "fts_information_flag", "value": True,
                       "basis": "facts.normalized.FTS_FLAG"})
    return sorted(labels, key=lambda item: (item["kind"], dumps(item["value"])))


def write_companies(db, output):
    without_payments = with_persons = 0
    profile_totals = {d: empty_flow(False) for d in ("incoming", "outgoing", "self_payments")}
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for uid, payload in db.execute("SELECT uid,payload FROM companies ORDER BY uid"):
            profile = json.loads(payload)
            normalized = profile["facts"]["normalized"]
            persons = []
            person_cursor = db.execute(
                "SELECT fid,payload FROM persons WHERE uid=? ORDER BY fid,record", (uid,))
            for fid, batch in itertools.groupby(person_cursor, key=lambda r: r[0]):
                persons.append({"synthetic_f": fid, "observations": [json.loads(r[1]) for r in batch]})
            payment_row = db.execute(
                "SELECT payload FROM payment_summaries WHERE uid=?", (uid,)).fetchone()
            payments = json.loads(payment_row[0]) if payment_row else empty_payments()
            without_payments += int(payment_row is None)
            with_persons += int(bool(persons))
            for direction, flow in profile_totals.items():
                p = payments[direction]
                add_to_flow(flow, p["amount"], p["count"], p["accepted_rows"],
                            p["source_rows"], p["excluded_rows"])
            profile["derived"] = {
                "group_id": normalized["GROUP_ID"] if normalized else None,
                "persons": persons, "unique_persons": len(persons), "payments": payments,
                "revenue_metadata": {"currency": None, "unit": None, "year": None,
                                     "period_kind": "annual"},
                "fts_trade_direction": None,
            }
            profile["labels"] = company_labels(normalized)
            profile["quality"]["person_ids_with_location_variants"] = [
                person["synthetic_f"] for person in persons
                if db.execute("SELECT 1 FROM person_location_variants WHERE fid=?",
                              (person["synthetic_f"],)).fetchone()]
            profile["quality"]["fts_consistency"] = (
                "not_assessed_conflict" if normalized is None else
                "countries_with_false_flag" if normalized["FTS_FLAG"] is False
                and normalized["FTS_COUNTRY_NAME"] else
                "flag_without_countries" if normalized["FTS_FLAG"] is True
                and not normalized["FTS_COUNTRY_NAME"] else "no_detected_contradiction")
            write_jsonl(stream, profile)
    return {"without_payment_observations": without_payments, "with_person_links": with_persons,
            "profile_payment_totals": profile_totals}


def write_groups(db, output):
    groups = members_total = 0
    cursor = db.execute(
        "SELECT group_id,uid,payload FROM companies WHERE status='ok' AND group_id IS NOT NULL "
        "ORDER BY group_id,uid")
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        for group_id, batch in itertools.groupby(cursor, key=lambda r: r[0]):
            members = []
            industries = defaultdict(list)
            regions = defaultdict(list)
            countries = defaultdict(list)
            products = {f: {"true_count": 0, "false_count": 0, "unknown_count": 0,
                            "members_with_product": []} for f in PRODUCT_FIELDS}
            revenue = ZERO
            revenue_members = []
            provenance = []
            for _, uid, payload in batch:
                profile = json.loads(payload)
                n = profile["facts"]["normalized"]
                members.append(uid)
                provenance.append({"synthetic_u": uid,
                                   "sources": [r["source"] for r in profile["facts"]["raw_records"]]})
                if n["MAINOKVED_CODE"] is not None or n["MAINOKVED_NAME"] is not None:
                    industries[(n["MAINOKVED_CODE"], n["MAINOKVED_NAME"])].append(uid)
                if n["REGION"] is not None:
                    regions[n["REGION"]].append(uid)
                for country in n["FTS_COUNTRY_NAME"] or []:
                    countries[country].append(uid)
                for field, data in products.items():
                    flag = n[field]
                    data["true_count" if flag is True else
                         "false_count" if flag is False else "unknown_count"] += 1
                    if flag is True:
                        data["members_with_product"].append(uid)
                if n["REVENUE_COMPANY"] is not None:
                    revenue = add_exact(revenue, Decimal(n["REVENUE_COMPANY"]))
                    revenue_members.append(uid)
            groups += 1
            members_total += len(members)
            write_jsonl(stream, {
                "group_id": group_id, "members": members,
                "derived": {
                    "member_count": len(members),
                    "industries": [{"code": k[0], "name": k[1], "members": v}
                                   for k, v in sorted(industries.items(),
                                                      key=lambda x: (x[0][0] or "", x[0][1] or ""))],
                    "registration_regions": [{"value": k, "members": v}
                                             for k, v in sorted(regions.items())],
                    "fts_countries": [{"value": k, "members": v} for k, v in sorted(countries.items())],
                    "products": products,
                    "members_revenue_sum": {
                        "value": money(revenue) if revenue_members else None,
                        "known_members": len(revenue_members), "total_members": len(members),
                        "members_with_known_revenue": revenue_members,
                        "currency": None, "unit": None, "year": None,
                        "is_consolidated": False,
                    },
                },
                "provenance": provenance,
            })
    eligible = db.execute(
        "SELECT COUNT(*) FROM companies WHERE status='ok' AND group_id IS NOT NULL").fetchone()[0]
    empty_groups = db.execute(
        "SELECT COUNT(*) FROM companies WHERE status='ok' AND group_id IS NULL").fetchone()[0]
    return {"groups": groups, "members": members_total, "eligible_members": eligible,
            "companies_without_group": empty_groups}


def build_profiles(companies, payments, persons, out_dir, top_n=10):
    """Build all five files; validate controls before replacing any previous result."""
    if not isinstance(top_n, int) or top_n < 0:
        raise InputError("top_n must be a nonnegative integer")
    companies, payments, persons, out_dir = map(Path, (companies, payments, persons, out_dir))
    inputs = (companies, payments, persons)
    for path in inputs:
        if not path.is_file():
            raise InputError(f"Input file not found: {path}")
    for name in OUTPUT_NAMES:
        target = (out_dir / name).resolve()
        for path in inputs:
            if target == path.resolve() or (target.exists() and target.samefile(path)):
                raise InputError(f"Output would overwrite source: {path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    source_metadata = {key: metadata(p) for key, p in
                       (("companies", companies), ("payments", payments), ("persons", persons))}
    # All intermediate rows and sorts stay on disk; discarded after success or failure.
    with tempfile.TemporaryDirectory(prefix=".build_profiles-", dir=out_dir) as temp:
        stage = Path(temp)
        db = sqlite3.connect(stage / "work.sqlite")
        try:
            db.execute("PRAGMA journal_mode=OFF")
            db.execute("PRAGMA synchronous=OFF")
            db.execute("PRAGMA temp_store=FILE")
            db.execute("PRAGMA cache_size=-32768")
            with (stage / "data_issues.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
                issues = Issues(stream)
                statuses, company_stats = load_companies(db, companies, issues)
                person_stats = load_persons(db, persons, statuses, issues)
                payment_stats, input_endpoints = load_payments(db, payments, statuses, issues)
                edge_stats, output_endpoints = build_edges(
                    db, payments, stage / "payment_edges.jsonl", statuses)
                flow_stats = build_payment_summaries(db, statuses, top_n, payments)
                company_stats.update(write_companies(db, stage / "company_profiles.jsonl"))
                group_stats = write_groups(db, stage / "group_profiles.jsonl")
                # Include references for unknown endpoints without millions of repeated issues.
                for uid in sorted(output_endpoints - statuses.keys()):
                    issues.emit("payment_unknown_company",
                                {"file": payments.name, "endpoint_id": uid},
                                synthetic_u=uid, characteristics=None,
                                provenance="payment_edges.jsonl: payer or beneficiary equals endpoint_id")
            flows = flow_stats["all_endpoints"]
            incoming, outgoing, self_flow = (flows[d] for d in ("incoming", "outgoing", "self_payments"))
            as_decimal = lambda f: Decimal(f["amount"] or "0")
            as_count = lambda f: f["count"] or 0
            paired_input_endpoints = {
                r[0] for r in db.execute(
                    "SELECT payer FROM payments WHERE payer IS NOT NULL AND beneficiary IS NOT NULL "
                    "UNION SELECT beneficiary FROM payments "
                    "WHERE payer IS NOT NULL AND beneficiary IS NOT NULL")}
            invalid_endpoint_rows = db.execute(
                "SELECT COUNT(*) FROM payments WHERE payer IS NULL OR beneficiary IS NULL").fetchone()[0]
            checks = {
                "accepted_amount_equals_edges": Decimal(payment_stats["accepted_amount"]) == Decimal(edge_stats["amount"]),
                "accepted_count_equals_edges": payment_stats["accepted_count"] == edge_stats["count"],
                "accepted_rows_equal_edges": payment_stats["accepted_rows"] == edge_stats["accepted_rows"],
                "amount_equals_outgoing_plus_self": Decimal(payment_stats["accepted_amount"])
                    == add_exact(as_decimal(outgoing), as_decimal(self_flow)),
                "count_equals_outgoing_plus_self": payment_stats["accepted_count"]
                    == as_count(outgoing) + as_count(self_flow),
                "rows_equal_outgoing_plus_self": payment_stats["accepted_rows"]
                    == outgoing["accepted_rows"] + self_flow["accepted_rows"],
                "incoming_equals_outgoing_amount": as_decimal(incoming) == as_decimal(outgoing),
                "incoming_equals_outgoing_count": as_count(incoming) == as_count(outgoing),
                "self_amount_matches_input": as_decimal(self_flow) == Decimal(payment_stats["self_amount"]),
                "self_count_matches_input": as_count(self_flow) == payment_stats["self_count"],
                "all_source_rows_accounted": edge_stats["source_rows"] + invalid_endpoint_rows == payment_stats["rows"],
                "all_rows_accepted_or_excluded": payment_stats["accepted_rows"] + payment_stats["excluded_rows"]
                    == payment_stats["rows"],
                "all_pairable_endpoints_preserved": paired_input_endpoints == output_endpoints,
                "all_unknown_pairable_endpoints_preserved":
                    paired_input_endpoints - statuses.keys() == output_endpoints - statuses.keys(),
                "group_membership_reconciles": group_stats["members"] == group_stats["eligible_members"],
                "company_profile_count_reconciles": company_stats["unique_ids"]
                    == db.execute("SELECT COUNT(*) FROM companies").fetchone()[0],
            }
            summary = {
                "schema_version": "1.0", "sources": source_metadata, "top_n": top_n,
                "companies": company_stats, "persons": person_stats,
                "payments": payment_stats, "edges": edge_stats, "groups": group_stats,
                "payment_flows": flow_stats,
                "coverage": {"unpairable_endpoint_ids": len(input_endpoints - output_endpoints),
                             "invalid_endpoint_rows": invalid_endpoint_rows},
                "issues": {"total": sum(issues.counts.values()), "by_code": dict(sorted(issues.counts.items()))},
                "checks": checks, "all_checks_passed": all(checks.values()),
                "policies": {
                    "money": "exact_decimal_strings", "currency": None, "payment_period": None,
                    "revenue_unit": None, "revenue_year": None,
                    "invalid_payment": "exclude_whole_row_from_both_numeric_totals_keep_observed_edge",
                    "self_payments": "separate_excluded_from_normal_flows_and_rankings",
                    "identical_payments": "retain_all_rows", "company_conflict": "exclude_all_attributes_and_group_membership",
                    "counterparty_counts": "all_observed_pairable_edges_including_excluded_numeric_rows",
                    "top_counterparties": "accepted_amount_descending_then_id_ascending",
                    "missing_observations": "null_totals_do_not_imply_no_business_activity",
                    "record_numbers": "one_based_data_records_excluding_header",
                    "large_provenance": "records_null_with_exact_selector_above_1000_records",
                    "unpairable_payment": "known_side_observation_and_exclusion_preserved_no_directed_edge",
                },
            }
            if not summary["all_checks_passed"]:
                raise RuntimeError(f"Reconciliation failed: {[k for k,v in checks.items() if not v]}")
            with (stage / "summary.json").open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(summary, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n")
        finally:
            db.close()
        for name in OUTPUT_NAMES:
            (stage / name).replace(out_dir / name)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--companies", required=True, type=Path)
    parser.add_argument("--payments", required=True, type=Path)
    parser.add_argument("--persons", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args(argv)
    print("Reading sources and building profiles (temporary SQLite on disk)...", file=sys.stderr)
    try:
        summary = build_profiles(args.companies, args.payments, args.persons, args.out_dir, args.top_n)
    except (OSError, ValueError, sqlite3.Error, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"Companies: {summary['companies']['unique_ids']}; conflicts: "
          f"{summary['companies']['conflict_count']}; groups: {summary['groups']['groups']}; "
          f"payment rows: {summary['payments']['rows']}; edges: {summary['edges']['edges']}; "
          f"excluded payments: {summary['payments']['excluded_rows']}; all controls passed.")
    print(f"Output: {args.out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
