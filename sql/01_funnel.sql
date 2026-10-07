-- Weekly conversion funnel: session -> product view -> cart -> checkout -> purchase.
-- Grain: one row per calendar week of the simulation.
WITH session_facts AS (
    SELECT
        e.week,
        e.session_id,
        ANY_VALUE(e.user_id)                                       AS user_id,
        COUNT(*) FILTER (WHERE e.event_type = 'product_view')      AS product_views,
        BOOL_OR(e.event_type = 'add_to_cart')                      AS has_cart,
        BOOL_OR(e.event_type = 'checkout')                         AS has_checkout,
        BOOL_OR(e.event_type = 'purchase')                         AS has_purchase,
        COALESCE(SUM(e.revenue) FILTER (WHERE e.event_type = 'purchase'), 0) AS revenue
    FROM events e
    GROUP BY 1, 2
),
weekly AS (
    SELECT
        week,
        COUNT(*)                                    AS sessions,
        COUNT(DISTINCT user_id)                     AS active_users,
        SUM(product_views)                          AS product_views,
        SUM(has_cart::INT)                          AS carts,
        SUM(has_checkout::INT)                      AS checkouts,
        SUM(has_purchase::INT)                      AS purchases,
        SUM(revenue)                                AS revenue
    FROM session_facts
    GROUP BY week
)
SELECT
    w.week,
    c.week_start,
    w.sessions,
    w.active_users,
    w.product_views,
    w.carts,
    w.checkouts,
    w.purchases,
    ROUND(w.revenue, 2)                                                          AS revenue,
    ROUND(w.carts::DOUBLE      / NULLIF(w.sessions, 0), 6)                       AS cart_rate,
    ROUND(w.checkouts::DOUBLE  / NULLIF(w.carts, 0), 6)                          AS cart_to_checkout_rate,
    ROUND(w.purchases::DOUBLE  / NULLIF(w.checkouts, 0), 6)                      AS checkout_to_purchase_rate,
    ROUND(w.purchases::DOUBLE  / NULLIF(w.sessions, 0), 6)                       AS session_conversion_rate,
    ROUND(w.sessions::DOUBLE   / NULLIF(w.active_users, 0), 4)                   AS sessions_per_active_user,
    ROUND(w.revenue          / NULLIF(w.purchases, 0), 2)                        AS aov,
    ROUND(w.revenue          / NULLIF(w.sessions, 0), 4)                         AS revenue_per_session,
    ROUND(w.product_views::DOUBLE / NULLIF(w.sessions, 0), 3)                    AS views_per_session
FROM weekly w
JOIN calendar c USING (week)
ORDER BY w.week;
