import sqlite3
import csv

"""
Compute immediate-upstream water quality metrics for REC2 river segments.

This version:
- removes all wetland-related columns and processing
- includes base median_flow from rec2_wetland_calcs_v2
- adds upstream_total_flow as the sum of median_flow across all immediate upstream segments

Main steps
----------
1. Build an immediate upstream network using riverlines:
   Each segment is linked to all segments whose NextDownID equals its HydroID.

2. Calculate upstream metrics:
   - TN and TP are converted to loads using:
       load = concentration * median_flow * 31536
     and summed across upstream segments.
   - E. coli metrics (median, Q95, G540) are flow-weighted averages using median_flow.
   - suspended_sed_load is summed across upstream segments.
   - upstream_total_flow is the sum of median_flow across all immediate upstream segments.

3. Handle headwater segments:
   If a segment has no upstream segments, its own values are used instead.

4. Record upstream segment IDs:
   A pipe-delimited list of immediate upstream nzsegments is generated.
   If no upstream segments exist, the segment lists itself.
"""

rec_gpkg = r"D:\Data\NIWA\REC2_geodata_version_5\REC2.gpkg"
riverlines_layer = "riverlines"
calc_layer = "rec2_wetland_calcs_v2"

output_csv = r"<PROJECT_DIRECTORY>\Wetlands\Intermediate\upstream_quality_nzsegment.csv"

conn = sqlite3.connect(rec_gpkg)
cur = conn.cursor()

query = f"""
WITH upstream_links AS (
    SELECT
        d.nzsegment AS downstream_nzsegment,
        u.nzsegment AS upstream_nzsegment
    FROM "{riverlines_layer}" d
    LEFT JOIN "{riverlines_layer}" u
        ON u.NextDownID = d.HydroID
),

upstream_values AS (
    SELECT
        l.downstream_nzsegment AS nzsegment,
        l.upstream_nzsegment,
        c.median_flow,
        c.TN,
        c.TP,
        c.Ecoli_median,
        c.Ecoli_Q95,
        c.Ecoli_G540,
        c.suspended_sed_load
    FROM upstream_links l
    LEFT JOIN "{calc_layer}" c
        ON l.upstream_nzsegment = c.nzsegment
),

upstream_lists AS (
    SELECT
        downstream_nzsegment AS nzsegment,
        GROUP_CONCAT(upstream_nzsegment, '|') AS upstream_nzsegments
    FROM (
        SELECT DISTINCT
            downstream_nzsegment,
            upstream_nzsegment
        FROM upstream_links
        WHERE upstream_nzsegment IS NOT NULL
        ORDER BY downstream_nzsegment, upstream_nzsegment
    )
    GROUP BY downstream_nzsegment
),

upstream_stats AS (
    SELECT
        nzsegment,

        SUM(CASE
            WHEN median_flow IS NOT NULL
            THEN median_flow
            ELSE 0
        END) AS upstream_total_flow,

        SUM(CASE
            WHEN TN IS NOT NULL AND median_flow IS NOT NULL
            THEN TN * median_flow * 31536
            ELSE 0
        END) AS upstream_TN_load,

        SUM(CASE
            WHEN TP IS NOT NULL AND median_flow IS NOT NULL
            THEN TP * median_flow * 31536
            ELSE 0
        END) AS upstream_TP_load,

        CASE
            WHEN SUM(CASE WHEN Ecoli_median IS NOT NULL AND median_flow IS NOT NULL THEN median_flow END) > 0
            THEN
                SUM(CASE WHEN Ecoli_median IS NOT NULL AND median_flow IS NOT NULL THEN Ecoli_median * median_flow END) * 1.0
                / SUM(CASE WHEN Ecoli_median IS NOT NULL AND median_flow IS NOT NULL THEN median_flow END)
            ELSE NULL
        END AS upstream_Ecoli_median_fw,

        CASE
            WHEN SUM(CASE WHEN Ecoli_Q95 IS NOT NULL AND median_flow IS NOT NULL THEN median_flow END) > 0
            THEN
                SUM(CASE WHEN Ecoli_Q95 IS NOT NULL AND median_flow IS NOT NULL THEN Ecoli_Q95 * median_flow END) * 1.0
                / SUM(CASE WHEN Ecoli_Q95 IS NOT NULL AND median_flow IS NOT NULL THEN median_flow END)
            ELSE NULL
        END AS upstream_Ecoli_Q95_fw,

        CASE
            WHEN SUM(CASE WHEN Ecoli_G540 IS NOT NULL AND median_flow IS NOT NULL THEN median_flow END) > 0
            THEN
                SUM(CASE WHEN Ecoli_G540 IS NOT NULL AND median_flow IS NOT NULL THEN Ecoli_G540 * median_flow END) * 1.0
                / SUM(CASE WHEN Ecoli_G540 IS NOT NULL AND median_flow IS NOT NULL THEN median_flow END)
            ELSE NULL
        END AS upstream_Ecoli_G540_fw,

        SUM(CASE WHEN suspended_sed_load IS NOT NULL THEN suspended_sed_load ELSE 0 END) AS upstream_suspended_sed_load_sum,

        COUNT(upstream_nzsegment) AS upstream_count

    FROM upstream_values
    GROUP BY nzsegment
)

SELECT
    base.nzsegment,
    base.median_flow,

    CASE
        WHEN base.TN IS NOT NULL AND base.median_flow IS NOT NULL
        THEN base.TN * base.median_flow * 31536
        ELSE NULL
    END AS TN_load,
    
    CASE
        WHEN base.TP IS NOT NULL AND base.median_flow IS NOT NULL
        THEN base.TP * base.median_flow * 31536
        ELSE NULL
    END AS TP_load,
    
    base.suspended_sed_load AS suspended_sed_load,

    COALESCE(l.upstream_nzsegments, CAST(base.nzsegment AS TEXT)) AS upstream_nzsegments,

    CASE
        WHEN COALESCE(s.upstream_count, 0) = 0 THEN base.median_flow
        ELSE s.upstream_total_flow
    END AS upstream_total_flow,

    CASE
        WHEN COALESCE(s.upstream_count, 0) = 0
            THEN CASE
                WHEN base.TN IS NOT NULL AND base.median_flow IS NOT NULL
                THEN base.TN * base.median_flow * 31536
                ELSE NULL
            END
        ELSE s.upstream_TN_load
    END AS upstream_TN_load,

    CASE
        WHEN COALESCE(s.upstream_count, 0) = 0
            THEN CASE
                WHEN base.TP IS NOT NULL AND base.median_flow IS NOT NULL
                THEN base.TP * base.median_flow * 31536
                ELSE NULL
            END
        ELSE s.upstream_TP_load
    END AS upstream_TP_load,

    CASE
        WHEN COALESCE(s.upstream_count, 0) = 0 THEN base.Ecoli_median
        ELSE s.upstream_Ecoli_median_fw
    END AS upstream_Ecoli_median_fw,

    CASE
        WHEN COALESCE(s.upstream_count, 0) = 0 THEN base.Ecoli_Q95
        ELSE s.upstream_Ecoli_Q95_fw
    END AS upstream_Ecoli_Q95_fw,

    CASE
        WHEN COALESCE(s.upstream_count, 0) = 0 THEN base.Ecoli_G540
        ELSE s.upstream_Ecoli_G540_fw
    END AS upstream_Ecoli_G540_fw,

    CASE
        WHEN COALESCE(s.upstream_count, 0) = 0 THEN base.suspended_sed_load
        ELSE s.upstream_suspended_sed_load_sum
    END AS upstream_suspended_sed_load_sum

FROM "{calc_layer}" base
LEFT JOIN upstream_lists l
    ON base.nzsegment = l.nzsegment
LEFT JOIN upstream_stats s
    ON base.nzsegment = s.nzsegment
ORDER BY base.nzsegment
"""

cur.execute(query)
rows = cur.fetchall()
conn.close()

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "nzsegment",
        "median_flow",
        "TN_load",
        "TP_load",
        "suspended_sed_load",
        "upstream_nzsegments",
        "upstream_total_flow",
        "upstream_TN_load",
        "upstream_TP_load",
        "upstream_Ecoli_median_fw",
        "upstream_Ecoli_Q95_fw",
        "upstream_Ecoli_G540_fw",
        "upstream_suspended_sed_load_sum"
    ])
    writer.writerows(rows)

print(f"Wrote {len(rows):,} rows to:")
print(output_csv)