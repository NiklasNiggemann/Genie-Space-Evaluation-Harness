# Genie Space — Join Specifications

> **Purpose**: Document all join relationships configured in the Genie Space.

---

## Configured Joins

### sales_transactions ↔ customers

| Property | Value |
|----------|-------|
| Left table | `samples.bakehouse.sales_transactions` |
| Right table | `samples.bakehouse.customers` |
| Join type | INNER JOIN |
| Join condition | `sales_transactions.customer_id = customers.customer_id` |
| When to use | Questions about customer names or demographics combined with transactions |

### sales_transactions ↔ products

| Property | Value |
|----------|-------|
| Left table | `samples.bakehouse.sales_transactions` |
| Right table | `samples.bakehouse.products` |
| Join type | INNER JOIN |
| Join condition | `sales_transactions.product_id = products.product_id` |
| When to use | Questions about product names or categories combined with sales data |

---

_Last updated: YYYY-MM-DD_
_Corresponds to Genie Space ID: `01f16364-...`_
