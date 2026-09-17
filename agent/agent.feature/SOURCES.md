# Feature agent — sources

Purpose: per flagged chip, decide **what this thing is locally** (class, size, relief,
freshness, or artifact) and whether the ContextBrief's expected landforms explain it.
Grounded by a labeled **exemplar library** (image retrieval), measurement tools, and
class definitions from the references below.

## 1. Exemplar image sources (build the labeled chip library from these)

| Source | Link | Notes |
|---|---|---|
| LROC featured images | https://www.lroc.asu.edu/images | captioned NAC/WAC crops, every lunar landform |
| LROC image search / QuickMap | https://quickmap.lroc.asu.edu/ | pull NAC at chosen lat/lon and scale |
| HiRISE image catalog | https://www.uahirise.org/catalog/ | captioned Mars images |
| HiRISE science themes | https://www.uahirise.org/science_themes/ | pre-sorted by landform class |
| HiRISE "HiPOD" | https://www.uahirise.org/hipod/ | one captioned crop per day |
| Moon Trek | https://trek.nasa.gov/moon/ | mosaics + layers |
| Mars Trek | https://trek.nasa.gov/mars/ | mosaics + layers |
| JMARS | https://jmars.asu.edu/ | GIS for locating exemplars |
| PDS Image Atlas | https://pds-imaging.jpl.nasa.gov/tools/atlas/ | raw products (already used by downloader) |

## 2. Labeled datasets (exemplars, priors, later a trained classifier)

| Dataset | Link | Content |
|---|---|---|
| **Mars-Bench** (Purohit et al. 2025) — 20 Mars datasets, cls/seg/det, HiRISE/CTX/HRSC + rover; **cloned as submodule `Mars-Bench/`, see `MARS_BENCH.md`** | https://github.com/kerner-lab/Mars-Bench | MIT code, CC-BY-4.0 data |
| Mars-Bench paper | https://arxiv.org/abs/2510.24010 | |
| Mars-Bench datasets on Hugging Face | https://huggingface.co/collections/Mirali33/mars-bench-models | `Mirali33/mb-*` |
| Mars-Bench on Zenodo | https://zenodo.org/communities/mars-bench/records | zip mirrors |
| DoMars16k (Wilhelm et al. 2020, CTX landforms) | https://doi.org/10.3390/rs12233981 | source of mb-domars16k |
| HiRISE landmark dataset v3.2 (Wagstaff et al.) | https://doi.org/10.5281/zenodo.4002935 | ~64k labeled HiRISE crops: crater, dark dune, slope streak, bright dune, impact ejecta, swiss cheese, spider, other |
| DeepMoon (Silburt et al. 2019) | https://github.com/silburt/DeepMoon | crater detection on LRO/Kaguya DEM; code + data |
| Silburt et al. 2019, Icarus | https://doi.org/10.1016/j.icarus.2018.06.022 | paper for the above |
| Robbins 2019 lunar crater database | https://doi.org/10.1029/2018JE005592 | 1.3M craters ≥1 km |
| Robbins & Hynek 2012 Mars crater database | https://doi.org/10.1029/2011JE003966 | 384k craters ≥1 km |
| Global lunar rockfall map (Bickel et al. 2020) | https://doi.org/10.1038/s41467-020-16653-3 | boulder + trail detections, NAC |
| Mars rockfalls / boulder tracks (Bickel et al. 2020, IEEE JSTARS) | https://doi.org/10.1109/JSTARS.2020.2991588 | rockfall on Mars |
| LROC pit atlas (Wagner & Robinson) | https://doi.org/10.1016/j.icarus.2014.04.002 | lunar pits with coordinates |
| Mars Global Digital Dune Database | https://doi.org/10.1029/2007JE002943 | dune fields |
| Apollo landing-site NAC images | https://www.lroc.asu.edu/images?query=apollo+landing+sites | hardware exemplars (positive class for eval) |
| Mars lander/rover hardware in HiRISE | https://www.uahirise.org/catalog/ (search by lander/rover name, e.g. Perseverance, Curiosity, Phoenix, Beagle 2) | hardware exemplars |

## 3. Class definitions and morphometry (system prompt + measurement tools)

| Topic | Source | Link |
|---|---|---|
| Landform definitions, all classes | Encyclopedia of Planetary Landforms | https://doi.org/10.1007/978-1-4614-3134-3 |
| Fresh crater depth/diameter, rim height | Pike 1977 | https://ui.adsabs.harvard.edu/abs/1977LPSC....8.3427P |
| Simple–complex transition, morphology | Melosh, Impact Cratering | https://ui.adsabs.harvard.edu/abs/1989icgp.book.....M |
| Crater degradation states | Fassett & Thomson 2014 | https://doi.org/10.1002/2014JE004698 |
| Small-crater morphometry from NAC DTMs | Stopar et al. 2017, Icarus | https://doi.org/10.1016/j.icarus.2017.05.022 |
| Boulder size–frequency | Krishna & Kumar 2016, Icarus | https://doi.org/10.1016/j.icarus.2015.10.033 |
| Irregular mare patches | Braden et al. 2014 | https://doi.org/10.1038/ngeo2252 |
| Lunar swirls | Denevi et al. 2016 | https://doi.org/10.1016/j.icarus.2016.01.017 |
| RSL | McEwen et al. 2011 | https://doi.org/10.1126/science.1204816 |
| Gullies | Malin & Edgett 2000 | https://doi.org/10.1126/science.288.5475.2330 |
| Dust-devil tracks | Reiss et al. 2016 | https://doi.org/10.1007/s11214-016-0308-6 |
| Brain terrain / periglacial | Levy et al. 2009 | https://doi.org/10.1029/2008JE003273 |
| Lunar Sourcebook (regolith, surface properties) | https://www.lpi.usra.edu/publications/books/lunar_sourcebook/ | |

## 4. Shadow-length height and photometry (tools)

| Topic | Source | Link |
|---|---|---|
| Hardware heights from NAC shadows | Wagner et al. 2017, Icarus | https://doi.org/10.1016/j.icarus.2016.05.011 |
| Boulder heights from shadows (method) | Bickel et al. 2020 (rockfall map, Methods) | https://doi.org/10.1038/s41467-020-16653-3 |
| Lunar photometry (Hapke) for albedo vs relief | Hapke 2012, Theory of Reflectance and Emittance Spectroscopy | https://doi.org/10.1017/CBO9781139025683 |
| Lunar opposition effect | Buratti et al. 1996, Icarus | https://doi.org/10.1006/icar.1996.0209 |

## 5. Instrument artifacts (artifact classes in the exemplar library)

| Instrument | Source | Link |
|---|---|---|
| LROC NAC calibration and known artifacts | Humm et al. 2016 | https://doi.org/10.1007/s11214-015-0201-8 |
| HiRISE instrument and CCD layout | McEwen et al. 2007 | https://doi.org/10.1029/2005JE002605 |
| HiRISE data processing / known issues | Delamere et al. 2010, Icarus | https://doi.org/10.1016/j.icarus.2009.03.012 |
| CTX | Malin et al. 2007 | https://doi.org/10.1029/2006JE002808 |
| TMC-2 | Chowdhury et al. 2020 | https://doi.org/10.18520/cs/v118/i4/566-572 |

## 6. Stack docs

| Tool | Link |
|---|---|
| fastembed image embeddings (CLIP ONNX) | https://qdrant.github.io/fastembed/examples/Supported_Models/ |
| OpenCV (blob/edge/Hough, profiles) | https://docs.opencv.org/ |
| scikit-image (measure, segmentation) | https://scikit-image.org/docs/stable/ |
| Anthropic vision input | https://platform.claude.com/docs/en/build-with-claude/vision |
