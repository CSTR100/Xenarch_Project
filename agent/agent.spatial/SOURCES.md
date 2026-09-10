# Spatial agent — sources

Purpose: per flagged chip, compute **where it sits and how it relates** to what is
around it: containment in known craters/units, slope from DEMs, distance to known
hardware, shadow geometry, and point patterns across all anomalies in the scene. The
references here are mostly **datasets** the relations are computed against, plus a
few standard methods.

## 1. Crater catalogs (containment: floor / rim / ejecta)

| Body | Source | Link |
|---|---|---|
| Moon ≥1 km | Robbins 2019, JGR Planets | https://doi.org/10.1029/2018JE005592 |
| Moon ≥1 km, data | Robbins lunar DB (USGS Astropedia) | https://astrogeology.usgs.gov/search/map/Moon/Research/Craters/lunar_crater_database_robbins_2018 |
| Moon, LU78287GT (Salamunićcar) | https://doi.org/10.1016/j.pss.2012.07.019 | |
| Mars ≥1 km | Robbins & Hynek 2012 | https://doi.org/10.1029/2011JE003966 |
| Mars, data | https://craters.sjrdesign.net/ | |
| Mars, CDA-based (Lagain et al. 2021) | https://doi.org/10.1029/2020EA001598 | small craters |
| Sub-km craters | none global — use `scene_objects` CV detection (or DeepMoon later) | https://github.com/silburt/DeepMoon |

## 2. Geologic unit maps (unit, age, contact distance)

| Body | Source | Link |
|---|---|---|
| Moon | Unified Geologic Map of the Moon, GIS v2 | https://astrogeology.usgs.gov/search/map/Moon/Geology/Unified_Geologic_Map_of_the_Moon_GIS_v2 |
| Mars | SIM 3292 (Tanaka et al. 2014), GIS | https://pubs.usgs.gov/sim/3292/ |
| Both | USGS Astropedia search | https://astrogeology.usgs.gov/search |

## 3. Digital elevation models (slope, aspect, elevation)

| Body | Product | Link | Resolution |
|---|---|---|---|
| Moon | LOLA GDR | https://pds-geosciences.wustl.edu/missions/lro/lola.htm | 118 m–1 km |
| Moon | SLDEM2015 (LOLA + Kaguya TC) | https://astrogeology.usgs.gov/search/map/moon_lro_lola_selene_kaguya_tc_dem_merge_60n60s_59m | 60 m |
| Moon | LROC NAC DTMs | https://wms.lroc.asu.edu/lroc/rdr_product_select | 2–5 m, sparse |
| Mars | MOLA MEGDR | https://pds-geosciences.wustl.edu/missions/mgs/megdr.html | 463 m |
| Mars | HRSC + MOLA blended DEM | https://astrogeology.usgs.gov/search/map/mars_mgs_mola_mex_hrsc_blended_dem_global_200m | 200 m |
| Mars | HiRISE DTMs | https://www.uahirise.org/dtm/ | 1–2 m, sparse |
| Mars | Global CTX mosaic (Murray Lab; basemap for 10× context, not a DEM) | https://murray-lab.caltech.edu/CTX/ (mirror: https://astrogeology.usgs.gov/search/map/mars-mro-ctx-global-mosaic-murray-lab-v1) | 5 m/px |

## 4. Nomenclature and known objects

| Source | Link |
|---|---|
| Gazetteer of Planetary Nomenclature (bulk CSV/KML) | https://planetarynames.wr.usgs.gov/AdvancedSearch |
| Anthropogenic feature coordinates, Moon (Wagner et al. 2017) | https://doi.org/10.1016/j.icarus.2016.05.011 |
| Lunar impact/landing sites (NSSDCA) | https://nssdc.gsfc.nasa.gov/planetary/lunar/lunar_impact_sites.html |
| Mars landing sites (NSSDCA) | https://nssdc.gsfc.nasa.gov/planetary/mars_mission_landings.html |
| Rover traverse maps (MSL, Perseverance) | https://mars.nasa.gov/maps/location/ |

## 5. Illumination and viewing geometry

| Source | Link | Use |
|---|---|---|
| NAIF SPICE toolkit | https://naif.jpl.nasa.gov/naif/ | sun azimuth/elevation at any time and place |
| spiceypy | https://spiceypy.readthedocs.io/ | Python bindings |
| LRO SPICE kernels | https://naif.jpl.nasa.gov/pub/naif/pds/data/lro-l-spice-6-v1.0/ | |
| MRO SPICE kernels | https://naif.jpl.nasa.gov/pub/naif/pds/data/mro-m-spice-6-v1.0/ | |
| PDS3 label keywords (INCIDENCE_ANGLE, SUB_SOLAR_AZIMUTH …) | https://pds.nasa.gov/datastandards/pds3/standards/ | parse from the product label first |

## 6. Methods (few, standard)

| Method | Source | Link |
|---|---|---|
| Shadow-length heights (hardware) | Wagner et al. 2017 | https://doi.org/10.1016/j.icarus.2016.05.011 |
| Nearest-neighbour clustering (Clark–Evans) | Clark & Evans 1954, Ecology | https://doi.org/10.2307/1931034 |
| Ripley's K | Ripley 1977, J. R. Stat. Soc. B | https://doi.org/10.1111/j.2517-6161.1977.tb01615.x |
| Secondary-crater clustering and chains | McEwen & Bierhaus 2006 | https://doi.org/10.1146/annurev.earth.34.031405.125018 |
| Ejecta extent vs diameter | Melosh, Impact Cratering | https://ui.adsabs.harvard.edu/abs/1989icgp.book.....M |
| Rockfall / boulder trails and slope | Bickel et al. 2020 | https://doi.org/10.1038/s41467-020-16653-3 |
| Lunar slope statistics from LOLA | Rosenburg et al. 2011, JGR | https://doi.org/10.1029/2010JE003716 |
| Planetary coordinate reference systems (IAU) | Archinal et al. 2018, Celest. Mech. Dyn. Astron. | https://doi.org/10.1007/s10569-017-9805-5 |

## 7. Stack docs

| Tool | Link | Use |
|---|---|---|
| ISIS (USGS) | https://isis.astrogeology.usgs.gov/ | campt/caminfo for pixel → lat/lon, decode MOC .imq |
| GDAL / rasterio | https://rasterio.readthedocs.io/ | DEM sampling, reprojection |
| PROJ planetary CRS (IAU codes) | https://proj.org/ | e.g. `IAU_2015:30100` Moon, `IAU_2015:49900` Mars |
| geopandas / shapely | https://geopandas.org/ | point-in-polygon against unit maps |
| scipy.spatial | https://docs.scipy.org/doc/scipy/reference/spatial.html | KD-trees, nearest neighbour |
| OpenCV Hough circles/lines | https://docs.opencv.org/ | `scene_objects` |
| OpenPlanetary (community tools, basemaps) | https://www.openplanetary.org/ | |
