-- North Star metric tree.
--   North Star = weekly purchasing users
--   Revenue    = active users x sessions per active user x session conversion rate x AOV
-- The identity is reconstructed in SQL so any definitional drift shows up as a
-- non-zero decomposition_error rather than as a silent reporting difference.
WITH session_facts AS (
    SELECT
        e.week,
        e.session_id,
        ANY_VALUE(e.user_id)                                         AS user_id,
        BOOL_OR(e.event_type = 'purchase')                           AS has_purchase,
        COALESCE(SUM(e.revenue) FILTER (WHERE e.event_type = 'purchase'), 0) AS revenue
    FROM events e
    GROUP BY 1, 2
),
weekly AS (
    SELECT
        week,
        COUNT(DISTINCT user_id) FILTER (WHERE has_purchase) AS purchasing_users,
        COUNT(DISTINCT user_id)                             AS active_users,
        COUNT(*)                                            AS sessions,
        SUM(has_purchase::INT)                              AS purchases,
        SUM(revenue)                                        AS revenue
    FROM session_facts
    GROUP BY week
),
decomposed AS (
    SELECT
        w.week,
        c.week_start,
        w.purchasing_users                                          AS north_star_purchasing_users,
        w.active_users,
        w.sessions,
        w.purchases,
        ROUND(w.revenue, 2)                                         AS revenue,
        w.sessions::DOUBLE  / NULLIF(w.active_users, 0)             AS sessions_per_active_user,
        w.purchases::DOUBLE / NULLIF(w.sessions, 0)                 AS session_conversion_rate,
        w.revenue           / NULLIF(w.purchases, 0)                AS aov,
        w.revenue           / NULLIF(w.active_users, 0)             AS arpu
    FROM weekly w
    JOIN calendar c USING (week)
),
checked AS (
    SELECT
        d.*,
        d.active_users * d.sessions_per_active_user * d.session_conversion_rate * d.aov AS revenue_reconstructed
    FROM decomposed d
)
SELECT
    week,
    week_start,
    north_star_purchasing_users,
    active_users,
    sessions,
    purchases,
    revenue,
    ROUND(sessions_per_active_user, 4)  AS sessions_per_active_user,
    ROUND(session_conversion_rate, 6)   AS session_conversion_rate,
    ROUND(aov, 2)                       AS aov,
    ROUND(arpu, 2)                      AS arpu,
    ROUND(revenue_reconstructed - revenue, 6)                        AS decomposition_error,
    LAG(north_star_purchasing_users) OVER w                          AS prev_purchasing_users,
    ROUND(
        north_star_purchasing_users::DOUBLE
        / NULLIF(LAG(north_star_purchasing_users) OVER w, 0) - 1, 4) AS purchasing_users_wow,
    ROUND(revenue / NULLIF(LAG(revenue) OVER w, 0) - 1, 4)           AS revenue_wow,
    ROUND(
        session_conversion_rate / NULLIF(LAG(session_conversion_rate) OVER w, 0) - 1, 4)
                                                                     AS conversion_wow,
    ROUND(aov / NULLIF(LAG(aov) OVER w, 0) - 1, 4)                   AS aov_wow,
    ROUND(
        revenue / NULLIF(AVG(revenue) OVER (ORDER BY week ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING), 0) - 1, 4)
                                                                     AS revenue_vs_trailing_4w
FROM checked
WINDOW w AS (ORDER BY week)
ORDER BY week;
