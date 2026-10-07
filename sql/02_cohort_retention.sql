-- Cohort retention: users grouped by signup week, tracked across activity weeks.
-- cohort_week 0 is the pre-window base (users who existed before the simulation start).
WITH cohort AS (
    SELECT user_id, signup_week AS cohort_week
    FROM users
),
cohort_size AS (
    SELECT cohort_week, COUNT(*) AS cohort_size
    FROM cohort
    GROUP BY 1
),
user_week AS (
    SELECT
        e.user_id,
        e.week,
        COUNT(DISTINCT e.session_id)                                       AS sessions,
        BOOL_OR(e.event_type = 'purchase')                                 AS purchased,
        COALESCE(SUM(e.revenue) FILTER (WHERE e.event_type = 'purchase'), 0) AS revenue
    FROM events e
    GROUP BY 1, 2
)
SELECT
    c.cohort_week,
    s.cohort_size,
    uw.week                                                                AS activity_week,
    cal.week_start,
    uw.week - c.cohort_week                                                AS week_index,
    COUNT(DISTINCT uw.user_id)                                             AS active_users,
    ROUND(COUNT(DISTINCT uw.user_id)::DOUBLE / s.cohort_size, 6)           AS retention_rate,
    COUNT(DISTINCT uw.user_id) FILTER (WHERE uw.purchased)                 AS purchasing_users,
    ROUND(
        COUNT(DISTINCT uw.user_id) FILTER (WHERE uw.purchased)::DOUBLE
        / NULLIF(COUNT(DISTINCT uw.user_id), 0), 6)                        AS active_purchase_rate,
    SUM(uw.sessions)                                                       AS sessions,
    ROUND(SUM(uw.revenue), 2)                                              AS revenue,
    ROUND(SUM(uw.revenue) / NULLIF(COUNT(DISTINCT uw.user_id), 0), 2)      AS revenue_per_active_user
FROM user_week uw
JOIN cohort c       USING (user_id)
JOIN cohort_size s  USING (cohort_week)
JOIN calendar cal   ON cal.week = uw.week
GROUP BY 1, 2, 3, 4, 5
ORDER BY c.cohort_week, week_index;
