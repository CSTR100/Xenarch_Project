# Literature agent — sources

Purpose: write the per-scene **ContextBrief** (expected natural landforms, unit, rare
natural features nearby, known hardware, instrument artifacts) with citations. Never
sees the anomaly. Everything below is either ingested into the RAG store, loaded as a
lookup table, or used at runtime as an API.

## 1. Bibliographic APIs (runtime search + metadata)

| Source | Link | Use |
|---|---|---|
| NASA ADS | https://ui.adsabs.harvard.edu/ | primary planetary-science index; free API key |
| ADS API docs | https://ui.adsabs.harvard.edu/help/api/ | search, export BibTeX/abstracts |
| arXiv astro-ph.EP | https://arxiv.org/list/astro-ph.EP/recent | preprints; full text |
| arXiv API | https://info.arxiv.org/help/api/index.html | programmatic search |
| Semantic Scholar API | https://api.semanticscholar.org/ | citation graph, open-access PDFs |
| OpenAlex | https://openalex.org/ | open bibliographic metadata |
| Crossref | https://www.crossref.org/documentation/retrieve-metadata/rest-api/ | DOI → metadata |

## 2. Conference abstracts and grey literature (bulk ingest)

| Source | Link | Use |
|---|---|---|
| LPSC (Lunar and Planetary Science Conference) | https://www.lpi.usra.edu/meetings/ | 2-page abstracts, every feature class, yearly |
| LPI publications | https://www.lpi.usra.edu/publications/ | books, technical reports |
| USGS Astrogeology | https://www.usgs.gov/centers/astrogeology-science-center | maps, reports |
| USGS publications warehouse | https://pubs.usgs.gov/ | SIM maps, open-file reports |
| NASA NTRS | https://ntrs.nasa.gov/ | technical reports, mission docs |

## 3. Reference books (class definitions, size ranges)

| Source | Link |
|---|---|
| Encyclopedia of Planetary Landforms (Hargitai & Kereszturi) | https://doi.org/10.1007/978-1-4614-3134-3 |
| Lunar Sourcebook (Heiken, Vaniman, French) | https://www.lpi.usra.edu/publications/books/lunar_sourcebook/ |
| Impact Cratering: A Geologic Process (Melosh) | https://ui.adsabs.harvard.edu/abs/1989icgp.book.....M |
| The Geology of Mars (Chapman, ed.) | https://doi.org/10.1017/CBO9780511536014 |
| Planetary Geology (Melosh 2011, Planetary Surface Processes) | https://doi.org/10.1017/CBO9780511977848 |

## 4. Regional geology — unit descriptions

| Source | Link |
|---|---|
| Unified Geologic Map of the Moon (Fortezzo et al. 2020), GIS | https://astrogeology.usgs.gov/search/map/Moon/Geology/Unified_Geologic_Map_of_the_Moon_GIS_v2 |
| Fortezzo et al. 2020 LPSC abstract | https://www.hou.usra.edu/meetings/lpsc2020/pdf/2760.pdf |
| Geologic Map of Mars, SIM 3292 (Tanaka et al. 2014) | https://pubs.usgs.gov/sim/3292/ |
| USGS planetary geologic mapping program | https://planetarymapping.wr.usgs.gov/ |
| Mare basalt stratigraphy (Hiesinger et al. 2011, GSA SP 477) | https://doi.org/10.1130/2011.2477(01) |
| Apollo landing-site geology (LPI Apollo pages) | https://www.lpi.usra.edu/lunar/missions/apollo/ |

## 5. Landform reviews (one or two per taxonomy class)

| Class | Source | Link |
|---|---|---|
| Crater morphometry | Pike 1977, Size-dependence in shape of fresh impact craters | https://ui.adsabs.harvard.edu/abs/1977LPSC....8.3427P |
| Crater degradation | Fassett & Thomson 2014, JGR Planets | https://doi.org/10.1002/2014JE004698 |
| Boulders / rockfalls | Bickel et al. 2020, global lunar rockfall map, Nat. Commun. | https://doi.org/10.1038/s41467-020-16653-3 |
| Irregular mare patches | Braden et al. 2014, Nature Geoscience | https://doi.org/10.1038/ngeo2252 |
| Lunar swirls | Denevi et al. 2016, Icarus | https://doi.org/10.1016/j.icarus.2016.01.017 |
| Pits / skylights | Wagner & Robinson 2014, Icarus | https://doi.org/10.1016/j.icarus.2014.04.002 |
| Fresh impacts (temporal pairs) | Speyerer et al. 2016, Nature | https://doi.org/10.1038/nature19829 |
| Wrinkle ridges | Watters 1988, JGR | https://doi.org/10.1029/JB093iB09p10236 |
| Lunar scarps (lobate) | Watters et al. 2010, Science | https://doi.org/10.1126/science.1189590 |
| Rilles | Hurwitz et al. 2013, Planet. Space Sci. | https://doi.org/10.1016/j.pss.2012.10.019 |
| Mars gullies | Malin & Edgett 2000, Science | https://doi.org/10.1126/science.288.5475.2330 |
| Recurring slope lineae | McEwen et al. 2011, Science | https://doi.org/10.1126/science.1204816 |
| Mars dunes | Hayward et al. 2007, Mars Global Digital Dune Database | https://doi.org/10.1029/2007JE002943 |
| Dust-devil tracks | Reiss et al. 2016, Space Sci. Rev. | https://doi.org/10.1007/s11214-016-0308-6 |
| Chaos terrain | Rodriguez et al. 2005, Icarus | https://doi.org/10.1016/j.icarus.2004.11.021 |
| Polygonal ground | Levy et al. 2009, JGR | https://doi.org/10.1029/2008JE003273 |
| Secondary craters | McEwen & Bierhaus 2006, Ann. Rev. Earth Planet. Sci. | https://doi.org/10.1146/annurev.earth.34.031405.125018 |

## 6. Instrument descriptions and artifact sheets

| Instrument | Source | Link |
|---|---|---|
| LROC (NAC/WAC) | Robinson et al. 2010, Space Sci. Rev. | https://doi.org/10.1007/s11214-010-9634-2 |
| LROC NAC calibration | Humm et al. 2016, Space Sci. Rev. | https://doi.org/10.1007/s11214-015-0201-8 |
| LROC PDS archive + SIS | https://pds.lroc.asu.edu/ | |
| HiRISE | McEwen et al. 2007, JGR | https://doi.org/10.1029/2005JE002605 |
| HiRISE PDS archive (SIS in /document) | https://hirise-pds.lpl.arizona.edu/PDS/ | |
| CTX | Malin et al. 2007, JGR | https://doi.org/10.1029/2006JE002808 |
| CTX PDS archive | https://pds-imaging.jpl.nasa.gov/data/mro/ctx/ | |
| MOC | Malin & Edgett 2001, JGR | https://doi.org/10.1029/2000JE001455 |
| Clementine UVVIS | Nozette et al. 1994, Science | https://doi.org/10.1126/science.266.5192.1835 |
| Clementine PDS archive | https://pds-imaging.jpl.nasa.gov/data/clementine/ | |
| Chandrayaan-2 TMC-2 | Chowdhury et al. 2020, Current Science | https://doi.org/10.18520/cs/v118/i4/566-572 |
| Chandrayaan-2 data (PRADAN) | https://pradan.issdc.gov.in/ | |
| PDS Imaging Node | https://pds-imaging.jpl.nasa.gov/ | |
| PDS Geosciences Node | https://pds-geosciences.wustl.edu/ | |

## 7. Known human objects (as context, not target)

| Source | Link |
|---|---|
| Wagner et al. 2017, Coordinates of anthropogenic features on the Moon, Icarus | https://doi.org/10.1016/j.icarus.2016.05.011 |
| LROC Apollo landing sites | https://www.lroc.asu.edu/images?query=apollo+landing+sites (also https://www.lroc.asu.edu/featured_sites) |
| NASA lunar landing/impact site catalog (NSSDCA) | https://nssdc.gsfc.nasa.gov/planetary/lunar/lunar_impact_sites.html |
| Mars landing sites (NSSDCA) | https://nssdc.gsfc.nasa.gov/planetary/mars_mission_landings.html |

## 8. Nomenclature

| Source | Link |
|---|---|
| Gazetteer of Planetary Nomenclature | https://planetarynames.wr.usgs.gov/ |
| Gazetteer bulk download (CSV/KML) | https://planetarynames.wr.usgs.gov/AdvancedSearch |

## 9. Stack docs

| Tool | Link |
|---|---|
| Anthropic API (Messages, tool use, citations, structured outputs) | https://platform.claude.com/docs |
| Anthropic Python SDK | https://github.com/anthropics/anthropic-sdk-python |
| LanceDB (vector + full-text hybrid) | https://lancedb.github.io/lancedb/ |
| fastembed (ONNX text/image embeddings, no torch) | https://qdrant.github.io/fastembed/ |
| PyMuPDF | https://pymupdf.readthedocs.io/ |
| Voyage AI embeddings (hosted alternative) | https://docs.voyageai.com/ |
