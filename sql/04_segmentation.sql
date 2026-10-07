-- Weekly performance by segment, produced in one pass with GROUPING SETS.
-- Segments: device, acquisition channel, tenure (new vs returning), daypart.
WITH session_facts AS (
    SELECT
        e.week,
        e.session_id,
        ANY_VALUE(e.user_id)                                         AS user_id,
        ANY_VALUE(e.device)                                          AS device,
        MIN(e.event_time)                                            AS session_started_at,
        BOOL_OR(e.event_type = 'add_to_cart')                        AS has_cart,
        BOOL_OR(e.event_type = 'purchase')                           AS has_purchase,
        COALESCE(SUM(e.revenue) FILTER (WHERE e.event_type = 'purchase'), 0) AS revenue
    FROM events e
    GROUP BY 1, 2
),
session_dim AS (
    SELECT
        s.*,
        u.acquisition_channel,
        CASE WHEN u.signup_week > 0 AND u.signup_week >= s.week - 3 THEN 'new' ELSE 'returning' END AS tenure,
        CASE WHEN EXTRACT(ISODOW FROM s.session_started_at) >= 6 THEN 'weekend' ELSE 'weekday' END  AS daypart
    FROM session_facts s
    JOIN users u USING (user_id)
),
grouped AS (
    SELECT
        week,
        CASE
            WHEN GROUPING(device) = 0              THEN 'device'
            WHEN GROUPING(acquisition_channel) = 0 THEN 'channel'
            WHEN GROUPING(tenure) = 0              THEN 'tenure'
            ELSE 'daypart'
        END                                                          AS segment_type,
        COALESCE(device, acquisition_channel, tenure, daypart)        AS segment_value,
        COUNT(*)                                                     AS sessions,
        COUNT(DISTINCT user_id)                                      AS active_users,
        SUM(has_cart::INT)                                           AS carts,
        SUM(has_purchase::INT)                                       AS purchases,
        SUM(revenue)                                                 AS revenue
    FROM session_dim
    GROUP BY GROUPING SETS (
        (week, device),
        (week, acquisition_channel),
        (week, tenure),
        (week, daypart)
    )
)
SELECT
    g.week,
    c.week_start,
    g.segment_type,
    g.segment_value,
    g.sessions,
    g.active_users,
    g.carts,
    g.purchases,
    ROUND(g.revenue, 2)                                              AS revenue,
    ROUND(g.sessions::DOUBLE / SUM(g.sessions) OVER (PARTITION BY g.week, g.segment_type), 4)
                                                                     AS session_share,
    ROUND(g.carts::DOUBLE     / NULLIF(g.sessions, 0), 6)             AS cart_rate,
    ROUND(g.purchases::DOUBLE / NULLIF(g.sessions, 0), 6)             AS session_conversion_rate,
    ROUND(g.revenue / NULLIF(g.purchases, 0), 2)                      AS aov,
    ROUND(g.revenue / NULLIF(g.sessions, 0), 4)                       AS revenue_per_session
FROM grouped g
JOIN calendar c USING (week)
ORDER BY g.segment_type, g.segment_value, g.week;
