-- ============================================================================
-- Genie Space — Example SQL Queries
-- ============================================================================

-- Example 1: Total revenue
-- Question: "What is our total revenue?"
SELECT SUM(total_amount) AS total_revenue
FROM samples.bakehouse.sales_transactions
WHERE status != 'cancelled';

-- Example 2: Revenue by payment method
-- Question: "Break down revenue by payment method"
SELECT paymentMethod, SUM(total_amount) AS revenue, COUNT(*) AS transaction_count
FROM samples.bakehouse.sales_transactions
WHERE status != 'cancelled'
GROUP BY paymentMethod
ORDER BY revenue DESC;

-- Example 3: Monthly transaction trend
-- Question: "Show me the monthly transaction trend"
SELECT DATE_TRUNC('month', transaction_date) AS month,
       COUNT(*) AS transactions, SUM(total_amount) AS revenue
FROM samples.bakehouse.sales_transactions
GROUP BY DATE_TRUNC('month', transaction_date)
ORDER BY month;

-- Example 4: Top customers by spend
-- Question: "Who are our top 10 customers by total spend?"
SELECT c.customer_name, SUM(t.total_amount) AS total_spend
FROM samples.bakehouse.sales_transactions t
JOIN samples.bakehouse.customers c ON t.customer_id = c.customer_id
GROUP BY c.customer_name
ORDER BY total_spend DESC
LIMIT 10;
