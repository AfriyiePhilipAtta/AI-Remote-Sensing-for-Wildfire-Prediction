#!/usr/bin/env python3
"""
══════════════════════════════════════════════════════════════════════════════
  GEE DATA EXTRACTION — STANDALONE                                            
  Upper West Ghana · Sentinel-2 · ERA5 · MODIS MCD64A1                        
                                                                              
  This script ONLY extracts data from Google Earth Engine and saves a CSV.    
  Run this FIRST, then feed the CSV into the main pipeline script.            
                                                                              
  Prerequisites:                                                              
    pip install earthengine-api pandas                                        
    earthengine authenticate                                                  
                                                                              
══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import time
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
#  SETTINGS — EDIT THESE
# ──────────────────────────────────────────────────────────────────────────────
GEE_PROJECT  = "quiet-subset-447718-q0"   # ← YOUR GEE project ID
OUTPUT_CSV   = "upper_west_fire_spread.csv"

GEE_SEASONS = [
    ("2019-11-01", "2020-04-30"),
    ("2020-11-01", "2021-04-30"),
    ("2021-11-01", "2022-04-30"),
    ("2022-11-01", "2023-04-30"),
]

GEE_INTERVAL = 3        # days per composite
GEE_SCALE    = 500      # metres per pixel
GEE_PPC      = 500      # points per class per interval


# ──────────────────────────────────────────────────────────────────────────────
#  GEE EXTRACTION
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("═" * 70)
    print("  GEE DATA EXTRACTION — Upper West Ghana")
    print("═" * 70)

    # ── Import Earth Engine ───────────────────────────────────────────────────
    try:
        import ee
    except ImportError:
        print("\n  ERROR: earthengine-api not installed.")
        print("  Fix:   pip install earthengine-api")
        print("  Then:  earthengine authenticate")
        sys.exit(1)

    # ── Initialize ────────────────────────────────────────────────────────────
    try:
        ee.Initialize(project=GEE_PROJECT)
        print(f"\n  [GEE] Initialized — project: {GEE_PROJECT}")
    except Exception as exc:
        print(f"\n  ERROR: GEE initialization failed — {exc}")
        print("  Fix:")
        print("    1. Run: earthengine authenticate")
        print(f"    2. Make sure project '{GEE_PROJECT}' exists")
        print("    3. Enable Earth Engine API in Google Cloud Console")
        sys.exit(1)

    t_start = time.time()

    # ── Region of Interest ────────────────────────────────────────────────────
    ghana = ee.FeatureCollection("FAO/GAUL/2015/level1")
    roi   = ghana.filter(
        ee.Filter.And(
            ee.Filter.eq("ADM0_NAME", "Ghana"),
            ee.Filter.eq("ADM1_NAME", "Upper West")
        )
    ).geometry()
    print("  [GEE] ROI: Upper West, Ghana")

    # ── Static Layers ─────────────────────────────────────────────────────────
    dem       = ee.Image("USGS/SRTMGL1_003")
    slope     = ee.Terrain.slope(dem).rename("slope")
    aspect    = ee.Terrain.aspect(dem).rename("aspect")
    elevation = dem.rename("elevation")
    terrain   = slope.addBands([aspect, elevation])
    landcover = (ee.Image("ESA/WorldCover/v100/2020")
                 .select("Map").rename("landcover"))
    print("  [GEE] Static layers: DEM, slope, aspect, landcover")

    # ── MODIS Burned Area ─────────────────────────────────────────────────────
    def get_modis_fire(t0, t1):
        coll = (ee.ImageCollection("MODIS/061/MCD64A1")
                .filterBounds(roi)
                .filterDate(t0, t1)
                .select("BurnDate"))
        return ee.Image(
            ee.Algorithms.If(
                coll.size().gt(0),
                coll.max().gt(0).unmask(0),
                ee.Image.constant(0).clip(roi)
            )
        ).rename("fire").toByte()

    # ── Sentinel-2 Spectral Indices ───────────────────────────────────────────
    def get_s2(t0, t1):
        s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(roi)
              .filterDate(t0, t1)
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30)))

        def compute_indices(img):
            ndvi = img.normalizedDifference(["B8", "B4"]).rename("ndvi")
            nbr  = img.normalizedDifference(["B8", "B12"]).rename("nbr")
            ndwi = img.normalizedDifference(["B3", "B8"]).rename("ndwi")
            return ndvi.addBands([nbr, ndwi])

        fallback = (ee.Image.constant([0, 0, 0])
                    .rename(["ndvi", "nbr", "ndwi"]).unmask(0))

        return ee.Image(
            ee.Algorithms.If(
                s2.size().gt(0),
                s2.map(compute_indices).median().unmask(0),
                fallback
            )
        )

    # ── ERA5-Land Meteorology ─────────────────────────────────────────────────
    def get_era5(t0, t1):
        era = (ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
               .filterBounds(roi)
               .filterDate(t0, t1))

        A, B = 17.625, 243.04

        T_c  = era.select("temperature_2m").mean().subtract(273.15)
        Td_c = era.select("dewpoint_temperature_2m").mean().subtract(273.15)

        rh = (Td_c.multiply(A).divide(Td_c.add(B)).exp()
              .divide(T_c.multiply(A).divide(T_c.add(B)).exp())
              .multiply(100).min(100).max(0)
              .rename("relative_humidity"))

        u10 = era.select("u_component_of_wind_10m").mean()
        v10 = era.select("v_component_of_wind_10m").mean()
        ws  = u10.pow(2).add(v10.pow(2)).sqrt().rename("wind_speed")
        wd  = (v10.atan2(u10)
               .multiply(-180 / 3.14159265358979)
               .add(270).mod(360)
               .rename("wind_dir"))

        rain = era.select("total_precipitation_sum").sum().rename("rainfall")
        sm   = (era.select("volumetric_soil_water_layer_1")
                .mean().rename("soil_moisture"))

        return T_c.rename("temp").addBands([rh, ws, wd, rain, sm])

    # ── Build Feature Stack ───────────────────────────────────────────────────
    def build_stack(t0_ee, iv=GEE_INTERVAL):
        t1 = t0_ee.advance(iv, "day")
        t2 = t1.advance(iv, "day")

        s2  = get_s2(t0_ee, t1)
        met = get_era5(t0_ee, t1)
        ft  = get_modis_fire(t0_ee, t1).rename("fire_t")
        ft1 = get_modis_fire(t1, t2).rename("fire_t_plus1")
        sp  = ft1.And(ft.Not()).rename("spread").toByte()

        return (s2.addBands([met, terrain, landcover, ft, ft1, sp])
                .unmask(0).clip(roi))

    # ── Stratified Sampling ───────────────────────────────────────────────────
    def sample_iv(t0_ee, date_str):
        stack = build_stack(t0_ee)
        samp  = stack.stratifiedSample(
            numPoints=GEE_PPC,
            classBand="spread",
            classValues=[0, 1],
            classPoints=[GEE_PPC, GEE_PPC],
            region=roi,
            scale=GEE_SCALE,
            geometries=True,
            seed=42,
            dropNulls=True
        )

        def add_meta(f):
            c = f.geometry().coordinates()
            return f.set({
                "lon":  c.get(0),
                "lat":  c.get(1),
                "date": date_str
            })

        return samp.map(add_meta)

    # ── Loop Over All Seasons ─────────────────────────────────────────────────
    print("\n  [GEE] Building collection across all seasons …")
    all_cols = []

    for s_start, s_end in GEE_SEASONS:
        print(f"\n  [GEE] Season: {s_start} → {s_end}")

        n_days  = ee.Date(s_end).difference(ee.Date(s_start), "day")
        offsets = ee.List.sequence(
            0,
            n_days.subtract(GEE_INTERVAL * 2),
            GEE_INTERVAL
        )

        n_intervals = offsets.size().getInfo()
        print(f"         {n_intervals} intervals × {GEE_INTERVAL} days")

        def process(d):
            date = ee.Date(s_start).advance(d, "day")
            ds   = date.format("YYYY-MM-dd")
            return sample_iv(date, ds)

        col = ee.FeatureCollection(offsets.map(process)).flatten()
        all_cols.append(col)

    full_col = ee.FeatureCollection(all_cols).flatten()

    # ── Export to Google Drive ────────────────────────────────────────────────
    print("\n  [GEE] Submitting export task to Google Drive …")

    task = ee.batch.Export.table.toDrive(
        collection=full_col,
        description="upper_west_fire_spread",
        folder="GEE_exports",
        fileNamePrefix=OUTPUT_CSV.replace(".csv", ""),
        fileFormat="CSV",
        selectors=[
            "date", "lat", "lon",
            "ndvi", "nbr", "ndwi",
            "temp", "relative_humidity", "wind_speed", "wind_dir",
            "rainfall", "soil_moisture",
            "slope", "aspect", "elevation", "landcover",
            "fire_t", "fire_t_plus1", "spread"
        ]
    )
    task.start()

    elapsed = time.time() - t_start

    print(f"""
═══════════════════════════════════════════════════════════════════════════
  EXPORT TASK SUBMITTED  ({elapsed:.0f}s to build the collection)
═══════════════════════════════════════════════════════════════════════════
  Task name     : upper_west_fire_spread
  Drive folder  : GEE_exports
  Output file   : {OUTPUT_CSV.replace(".csv", "")}.csv  (may split into chunks)

  Monitor progress:
    https://code.earthengine.google.com/tasks

  Typical wait  : 10–30 minutes  (GEE runs this server-side in parallel)

  NEXT STEPS:
    1. Wait for the task to show  ✅ COMPLETED  at the link above
    2. Download the CSV from Google Drive  →  GEE_exports/
    3. Rename / move it to:  {OUTPUT_CSV}
    4. Then run:  python3 wildfire_pipeline.py
═══════════════════════════════════════════════════════════════════════════
    """)


if __name__ == "__main__":
    main()