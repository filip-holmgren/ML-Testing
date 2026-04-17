WITH cs_agg AS (
  SELECT
    cve_id,
    MAX(score) AS max_score,
    AVG(score) AS avg_score,
    MAX(
      CASE severity
        WHEN 'low' THEN 1
        WHEN 'medium' THEN 2
        WHEN 'high' THEN 3
        WHEN 'critical' THEN 4
        ELSE 0
      END
    ) AS severity_max
  FROM cve_scores
  GROUP BY cve_id
),
license_agg AS (
  SELECT
    pl.package_id,
    (ARRAY_AGG(l.id ORDER BY l.id))[1] AS license_id
  FROM package_license pl
  JOIN licenses l ON l.id = pl.license_id
  GROUP BY pl.package_id
),
historical_fp AS (
  SELECT DISTINCT
    p.purl,
    f.cve_id
  FROM fp_labels fpl
  JOIN findings f ON f.id = fpl.finding_id
  JOIN dependencies d ON d.id = f.dependency_id
  JOIN packages p ON p.id = d.package_id
  WHERE fpl.is_false_positive = true
)
SELECT
  s.tool,
  COALESCE(d.is_transitive::int, 0) as is_transitive,
  p.ecosystem,
  lic.license_id AS license_id,
  c.fix_available::int,
  COALESCE(c.score, 0) AS base_score,
  COALESCE(c.confidence, 0) as confidence,
  COALESCE(array_length(c.cwes, 1), 0) AS num_cwes,
  COALESCE((CURRENT_DATE - c.published_at::date), -1) AS cve_age_days,
  (c.published_at IS NULL)::int AS cve_age_missing,
  CASE WHEN c.status = 'affected' THEN 1 ELSE 0 END AS affected,
  COALESCE(cs.max_score, 0) AS max_score,
  COALESCE(cs.avg_score, 0) AS avg_score,
  COALESCE(cs.severity_max, 0) AS severity_max,
  -- safer (still imperfect) version signal
  CASE
    WHEN c.affects_version IS NOT NULL
         AND p.version = c.affects_version
    THEN 1
    WHEN c.affects_range IS NOT NULL
         AND p.version IS NOT NULL
         AND POSITION(p.version IN c.affects_range) > 0
    THEN 1
    ELSE 0
  END AS is_version_affected,
  -- strong signal
  fpl.is_false_positive::int AS label
FROM findings f
JOIN dependencies d ON f.dependency_id = d.id
JOIN scans s ON d.scan_id = s.id
JOIN packages p ON d.package_id = p.id
JOIN cves c ON f.cve_id = c.id
LEFT JOIN cs_agg cs ON cs.cve_id = c.id
LEFT JOIN license_agg lic ON lic.package_id = p.id
LEFT JOIN historical_fp hfp
  ON hfp.purl = p.purl
 AND hfp.cve_id = f.cve_id
LEFT JOIN fp_labels fpl ON fpl.finding_id = f.id
WHERE
  fpl.is_false_positive IS NOT NULL;