-- Category performance per week: demand (views), orders, revenue, and rank.
WITH views AS (
    SELECT week, category, COUNT(*) AS product_views
    FROM events
    WHERE event_type = 'product_view' AND category IS NOT NULL
    GROUP BY 1, 2
),
orders AS (
    SELECT
        week,
        category,
        COUNT(*)    AS orders,
        SUM(revenue) AS revenue
    FROM events
    WHERE event_type = 'purchase' AND category IS NOT NULL
    GROUP BY 1, 2
),
combined AS (
    SELECT
        COALESCE(v.week, o.week)         AS week,
        COALESCE(v.category, o.category) AS category,
        COALESCE(v.product_views, 0)     AS product_views,
        COALESCE(o.orders, 0)            AS orders,
        COALESCE(o.revenue, 0)           AS revenue
    FROM views v
    FULL OUTER JOIN orders o USING (week, category)
)
SELECT
    c.week,
    cal.week_start,
    c.category,
    c.product_views,
    c.orders,
    ROUND(c.revenue, 2)                                              AS revenue,
    ROUND(c.revenue / SUM(c.revenue) OVER (PARTITION BY c.week), 4)  AS revenue_share,
    RANK() OVER (PARTITION BY c.week ORDER BY c.revenue DESC)        AS revenue_rank,
    ROUND(c.revenue / NULLIF(c.orders, 0), 2)                        AS aov,
    ROUND(c.orders * 1000.0 / NULLIF(c.product_views, 0), 3)         AS orders_per_1k_views,
    ROUND(
        c.revenue / NULLIF(LAG(c.revenue) OVER (PARTITION BY c.category ORDER BY c.week), 0) - 1, 4)
                                                                     AS revenue_wow
FROM combined c
JOIN calendar cal USING (week)
ORDER BY c.week, revenue_rank;
