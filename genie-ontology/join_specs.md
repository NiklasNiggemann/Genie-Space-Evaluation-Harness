# Genie Space — Join Specifications

> **Purpose**: Document all join relationships configured in the Genie Space.
> These teach Genie how to combine tables when answering multi-table questions.

---

## Configured Joins

### sales_transactions ↔ customers

| Property | Value |
|----------|-------|
| Left table | `samples.bakehouse.sales_transactions` |
| Right table | `samples.bakehouse.customers` |
| Join type | INNER JOIN |
| Join condition | `sales_transactions.customer_id = customers.customer_id` |
| When to use | Any question about customer names, demographics, or segments combined with transaction data |

### sales_transactions ↔ products

| Property | Value |
|----------|-------|
| Left table | `samples.bakehouse.sales_transactions` |
| Right table | `samples.bakehouse.products` |
| Join type | INNER JOIN |
| Join condition | `sales_transactions.product_id = products.product_id` |
| When to use | Any question about product names, categories, or attributes combined with sales data |

---

## Tables in Scope

| Table | Description | Primary Key |
|-------|-------------|-------------|
| `samples.bakehouse.sales_transactions` | Fact table with all transactions | `transaction_id` |
| `samples.bakehouse.customers` | Customer dimension | `customer_id` |
| `samples.bakehouse.products` | Product catalog dimension | `product_id` |

---

## Notes

- Genie will use these join specs even if the user doesn't explicitly mention both tables.
- If a question could be answered from a single table, Genie should prefer the simpler query (no unnecessary joins).
- Foreign keys defined in Unity Catalog are automatically picked up; these specs cover additional or overridden relationships.

---

_Last updated: YYYY-MM-DD_
_Corresponds to Genie Space ID: `01f16364-...`_
