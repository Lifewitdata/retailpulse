-- Analysis-ready session table for a single experiment window.
-- Grain: one session, restricted to users present in the LOGGED assignment table.
-- Sessions belonging to users the assignment logger dropped simply do not join,
-- which is exactly how the EXP-003 defect surfaces downstream.
WITH assigned AS (
    SELECT
        user_id,
        variant,
        assigned_week,
        first_device,
        pre_period_sessions
    FROM assignments
    WHERE experiment_id = $exp_id
),
session_facts AS (
    SELECT
        e.week,
        e.session_id,
        ANY_VALUE(e.user_id)                                       AS user_id,
        ANY_VALUE(e.device)                                        AS device,
        MIN(e.event_time)                                          AS session_started_at,
        COUNT(*) FILTER (WHERE e.event_type = 'product_view')      AS product_views,
        BOOL_OR(e.event_type = 'add_to_cart')                      AS has_cart,
        BOOL_OR(e.event_type = 'checkout')                         AS has_checkout,
        BOOL_OR(e.event_type = 'purchase')                         AS has_purchase,
        COALESCE(SUM(e.revenue) FILTER (WHERE e.event_type = 'purchase'), 0) AS revenue
    FROM events e
    WHERE e.week BETWEEN $start_week AND $end_week
    GROUP BY 1, 2
)
SELECT
    s.week,
    s.session_id,
    s.user_id,
    s.device,
    s.session_started_at,
    CASE WHEN EXTRACT(ISODOW FROM s.session_started_at) >= 6 THEN 'weekend' ELSE 'weekday' END AS daypart,
    s.product_views,
    s.has_cart::INT      AS has_cart,
    s.has_checkout::INT  AS has_checkout,
    s.has_purchase::INT  AS has_purchase,
    s.revenue,
    a.variant,
    a.assigned_week,
    a.first_device,
    a.pre_period_sessions,
    u.acquisition_channel,
    u.signup_week,
    CASE WHEN u.signup_week > 0 AND u.signup_week >= s.week - 3 THEN 'new' ELSE 'returning' END AS tenure
FROM session_facts s
JOIN assigned a ON a.user_id = s.user_id
JOIN users u    ON u.user_id = s.user_id
ORDER BY s.week, s.session_id;
