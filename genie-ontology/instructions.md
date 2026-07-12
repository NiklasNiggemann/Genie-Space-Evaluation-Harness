# Genie Space — Text Instructions

> **Purpose**: Document all text instructions configured in the Genie Space.
> Keep this file in sync with the live Space so colleagues can review changes via Git diffs.

---

## Business Definitions

<!-- Add instructions that define business jargon and domain terms -->

- "Revenue" means the `total_amount` column after discounts are applied.
- "Active customer" means a customer with at least one transaction in the last 90 days.
- "Last month" means the previous full calendar month (not the last 30 days).

## Query Behavior Rules

<!-- Add instructions that control how Genie writes SQL -->

- When asked about revenue, always use `SUM(total_amount)` from the `sales_transactions` table.
- Always exclude records where `status = 'cancelled'` unless explicitly asked to include them.
- When grouping by time periods, use `DATE_TRUNC('month', transaction_date)`.

## Column Value Guidance

<!-- Add instructions that clarify ambiguous column values -->

- The `paymentMethod` column uses lowercase values: `'visa'`, `'mastercard'`, `'cash'`, `'amex'`.
- The `region` column uses abbreviations: `'EU'`, `'NA'`, `'APAC'`.

## Formatting Preferences

<!-- Add instructions about how results should be presented -->

- Format currency values with two decimal places.
- When returning top-N results, default to top 10 unless specified.

---

_Last updated: YYYY-MM-DD_
_Corresponds to Genie Space ID: `01f16364-...`_
