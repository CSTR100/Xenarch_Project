# Xenarch context agents — plan

The VAE (`xenarch_core.py`) stays the detector and is never told what anomalies look
like. Three agents run **after** it and turn "this chip is un-geology-like" into "this is
probably X, here, because Y, see Z". The literature agent does not investigate the
anomaly. It supplies *what natural looks like here* — the same philosophy as the VAE's
training folder, but in words and citations — so the feature and spatial agents can
judge whether the flagged thing is explained by the expected geology. Whatever is not
explained is the residual anomaly.

| Agent | Question it answers | Grounded by |
|---|---|---|
| **Feature** | *What is this thing, locally?* Morphology class, size, relief, freshness, or "artifact". | Reference exemplar chips + landform reference books + measurement tools |
| **Spatial** | *Where does it sit and how does it relate to what's around it?* Containment, alignment, clustering, shadow-derived height, slope, distance to known objects. | Crater/boulder catalogs, geologic unit maps, DEMs, illumination geometry — computed, not recalled |
| **Literature (natural context)** | *What natural geology is expected here?* A per-scene brief on the terrain, its usual landforms and their scales, rare-but-natural features known nearby, and this instrument's artifacts. Runs **first** and feeds the other two. | Local RAG corpus of regional geology + landform references + structured tables |

Decision from the earlier plan still holds: RAG and tool-use, not fine-tuning
(no provenance, no training pairs, corpus changes). See §7 for where small trained
models do fit.

## 1. Orchestration — a code-driven pipeline, not an agent-of-agents

```
scene + metadata ─► Literature agent ─► ContextBrief (once per scene, cached)
                                             │
VAE flags chip ─► build CaseFile ─► Feature agent ∥ Spatial agent ─► Synthesis ─► Verdict
                     (brief attached)
```

- **Literature runs once per scene, before any chip is examined.** Its brief is
  attached to every CaseFile from that scene, so the feature and spatial agents start
  with "you are in Mare Tranquillitatis basalts, unit Im2, expect 10–200 m simple
  craters, wrinkle ridges, scattered boulders < 5 m; IMPs (Ina-type) occur in this
  region; NAC images at this sun angle show CCD-boundary offsets" — with sources.
- Feature and Spatial run in parallel on each flagged chip, with the brief in their
  prompt, and are asked one question: *is this explained by the expected natural
  context, and if not, what remains unexplained?*
- **Synthesis** is a short final step (code plus one Claude call, no tools) that
  reads brief + feature + spatial output and writes the verdict, reusing the brief's
  citations. It never retrieves on its own.
- Orchestration is plain Python: deterministic, testable, each agent evaluated alone.
  A top-level orchestrator agent is a later option, not the start.
- Each agent is a Claude tool-use loop (`client.beta.messages.tool_runner`, model
  `claude-opus-5`, adaptive thinking, structured output, cached system prompt,
  server-side refusal `fallbacks` on by default). They differ in system prompt, tools,
  and what they see: Literature sees metadata and text; Feature sees images; Spatial
  sees mostly numbers and a scene graph.

### ContextBrief (one per scene)

```json
{"scene_id": "M175124932R", "body": "MOON", "region": "Mare Tranquillitatis",
 "geologic_unit": {"code": "Im2", "description": "...", "age": "Imbrian"},
 "expected_landforms": [
   {"class": "simple_crater", "size_range_m": [5, 500], "notes": "...", "sources": ["doc:..."]},
   {"class": "wrinkle_ridge", "size_range_m": [200, 5000], "sources": ["doc:..."]},
   {"class": "boulder", "size_range_m": [1, 8], "notes": "rare on mature mare", "sources": ["doc:..."]}],
 "rare_natural_nearby": [{"class": "irregular_mare_patch", "example": "Ina", "distance_km": 320, "sources": [...]}],
 "known_human_objects": [{"name": "Apollo 11 LM descent stage", "distance_km": 0.2, "sources": [...]}],
 "instrument_artifacts": [{"name": "NAC CCD boundary offset", "signature": "...", "sources": [...]}],
 "illumination_notes": "sun elevation 24°: shadows ≈ 2.2× object height",
 "sources": [{"doc_id": "...", "title": "...", "quote": "..."}]}
```

### CaseFile (one per flagged chip)

```json
{"case_id": "...", "brief": "<ContextBrief, or its id>",
 "scene": {"path": "...", "mission": "LRO", "instrument": "NAC", "body": "MOON",
  "product_id": "M175124932R", "pixel_scale_m": 0.5, "center_latlon": [0.674, 23.473],
  "footprint": [[..],[..],[..],[..]], "sun_azimuth_deg": 90.4, "sun_elevation_deg": 24.1},
 "chip": {"chip_id": 17, "top_x": 3072, "top_y": 1536, "size": 256,
  "fine_bbox": [x1,y1,x2,y2], "metrics": {"mse": .., "edge": .., "contextual": ..,
  "latent": .., "gradient": ..}, "combined": .., "confidence": ..},
 "crops": {"chip_png": "...", "context_3x_png": "...", "context_10x_png": "..."},
 "feature": null, "spatial": null, "verdict": null}
```

Filled in by the downloader (scene block — needs the lat/lon + illumination
extension), by the core (chip block), by the literature agent (brief), then by each
per-chip agent in turn.

## 2. Feature agent

**Sees:** chip crop, 3× context, the five VAE metrics, pixel scale, sun geometry.

**Tools:**
- `measure(bbox)` → extent in metres, major/minor axis, orientation, brightness
  contrast, edge straightness score, symmetry score.
- `profile(line)` → brightness profile along a line (rim-floor-rim for craters).
- `shadow_height(bbox)` → segments the shadow on the anti-sun side, returns length
  and height = length × tan(sun elevation). A 4 m feature casting a 2 m-tall shadow
  is not a boulder-sized crater.
- `nearest_exemplars(k)` → CLIP-style image retrieval over a labeled exemplar
  library (fastembed image models, ONNX, no torch): "here are the 5 most similar
  reference chips and their labels". Returned as images so Claude can compare.
- `crater_morphometry(diameter_m, depth_m)` → depth/diameter against Pike-style
  fresh-crater relations; returns degradation estimate.

**Reference material (this is what you can pin down):**
- Exemplar library: 20–50 labeled chips per class at 2–3 pixel scales, drawn from
  the LROC/HiRISE featured-image archives and your own downloads. Classes mirror the
  taxonomy: simple crater (fresh/degraded), secondary chain, boulder, boulder trail,
  rille, scarp, wrinkle ridge, pit, swirl, IMP, dune, dust-devil track, RSL, gully,
  plus artifact classes (CCD seam, saturation bleed, dropout, jitter) and hardware
  (each landing site).
- Books/atlases for the class definitions in the system prompt and for the
  Literature corpus: *Encyclopedia of Planetary Landforms* (Hargitai & Kereszturi),
  *Lunar Sourcebook*, Pike 1977 (crater depth/diameter), Melosh *Impact Cratering*,
  USGS planetary mapping guides.

**Uses the brief:** the expected-landform list narrows the candidate classes and
gives size ranges, so "a 40 m bowl in a unit where 5–500 m simple craters are
expected" is closed as natural, while "a 4 m object 2 m tall in a unit where
boulders are rare" is flagged as unexplained.

**Output:** `{class, subclass, confidence, measurements: {...},
explained_by_context: bool, matched_expected_landform: "simple_crater" | null,
evidence: [exemplar ids, measurements], artifact_suspected: bool, notes}`.

## 3. Spatial agent — grounding is data, not papers

The spatial question is different: there's no paper that says "an object 40 m from a
fresh crater rim, collinear with two other bright spots, is X". Instead, the
relationships are **computed** against reference datasets, and the agent interprets
the computed scene graph. The references are catalogs, maps and DEMs:

| Reference | Moon | Mars | Gives |
|---|---|---|---|
| Crater catalog | Robbins 2019 lunar DB (≥1 km), LU1–LU5 small-crater DBs | Robbins & Hynek 2012 (≥1 km) | "is this inside / on the rim / in the ejecta of a known crater" |
| Geologic unit map | USGS Unified Geologic Map of the Moon (Fortezzo 2020) | Tanaka et al. 2014 global map | unit, age, contact distance |
| DEM | LOLA + SLDEM2015 (60 m), NAC DTMs where available | MOLA (463 m), HRSC, HiRISE DTMs | elevation, slope, aspect → rockfall/ mass-wasting plausibility |
| Nomenclature | USGS Gazetteer | same | named features nearby |
| Hardware sites | LROC hardware coordinate list | lander/rover track maps | distance to known human objects |
| Illumination | from the PDS label | same | expected shadow direction and length |

Methods worth citing in the system prompt (few, all standard):
shadow-length height estimation; nearest-neighbour / Ripley's K for clustering
(used for secondary craters and boulder fields); rockfall/boulder-trail studies
(Bickel et al.); ejecta-blanket extent ≈ 1 crater radius, ray systems; collinearity
and regular-spacing tests as technosignature cues.

**Tools:**
- `scene_objects()` → detected objects in the 10× context: craters (Hough/template),
  bright/dark blobs (LoG), linear features (Hough lines), shadow masks. Each with
  centre, size, orientation, and chip-relative position. Classical CV, no torch.
- `catalog_objects(radius_m)` → craters, named features, hardware, geologic unit
  from the tables above, by lat/lon.
- `terrain(latlon)` → elevation, slope, aspect from the DEM.
- `relations(object_a, object_b)` → distance, bearing, containment (floor / rim /
  continuous ejecta / outside), alignment with sun azimuth, alignment with each other.
- `point_pattern(objects)` → nearest-neighbour ratio, collinearity score, spacing
  regularity across all anomalies in the scene.

**What it reasons about:**
- Shadow consistency: bright-with-shadow-on-anti-sun-side ⇒ positive relief; the
  reverse ⇒ depression; no shadow at low sun ⇒ likely albedo or artifact.
- Context: on a crater floor, on a steep slope with a trail uphill (rockfall), at a
  mare/highland contact, inside a known hardware radius.
- Multi-anomaly patterns: three chips in a line with equal spacing, or a radial
  arrangement around one bright object (blast zone + descent stage + tracks).
- Instrument geometry: anomalies that line up with CCD boundaries or the
  push-broom direction are artifacts.

**Uses the brief:** the brief's known human objects, rare-natural features nearby,
and instrument artifact signatures become named objects the relations can be
computed against ("collinear with the NAC CCD boundary", "inside the Apollo 11
hardware radius").

**Output:** `{scene_graph: {nodes, edges}, context: {unit, nearest_crater, slope,
hardware_within_m}, relief: "positive|negative|none", pattern_flags: [...],
explained_by_context: bool, interpretation, confidence}`.

## 4. Literature agent — natural context, not anomaly hunting

**Runs once per scene**, from metadata alone (body, lat/lon, unit, instrument, sun
geometry), before any chip is looked at. It never sees the anomaly. Its job is to
write the ContextBrief: what the literature says is normal for this place and this
instrument, with sources.

**Tools:**
- `geologic_unit(latlon)` → unit code, description, age (from the USGS map tables).
- `nearby_features(latlon, radius_km)` → gazetteer names, catalog craters, rare
  natural features, hardware sites.
- `instrument_sheet(instrument)` → known artifacts and their visual signature.
- `search_literature(query, body, feature_class, k)` → hybrid retrieval.
- `read_document(doc_id, section)`.
- `web_search` restricted to ads, arxiv, lpi.usra.edu, nasa.gov, usgs.gov,
  uahirise.org, lroc.asu.edu — fallback only when the local corpus is thin.

**Corpus emphasis shifts to natural geology:**
1. Regional geology: unit descriptions from the USGS lunar and Mars maps, mare
   basalt / highland / polar terrain reviews, landing-site geology reports.
2. Landform references with size ranges and morphology: *Encyclopedia of Planetary
   Landforms*, *Lunar Sourcebook*, crater morphometry (Pike), boulder population
   studies, wrinkle ridge, rille, scarp, dune, RSL, gully reviews.
3. Rare-but-natural catalogs: IMPs, swirls, pits and skylights, fresh impacts,
   rockfalls.
4. Instrument SIS and calibration papers (artifact descriptions).
5. Hardware sites — kept, but framed as "known objects in the scene", not as the
   target.
6. LPSC abstracts and ADS records filtered by region and feature class.

Target v1 size: 200–500 documents. Coverage of the units and landforms in your
downloads matters more than volume. Stack as before: pymupdf → section-aware chunks
(captions tagged) → fastembed embeddings → LanceDB hybrid search; SQLite for the
tables. Chunks carry `body`, `instrument`, `feature_class`, `unit`, `doc_id`,
`section`, `year`, `is_caption`.

**Caching:** the brief is keyed by scene id and stored next to the metadata JSON, so
re-running the chip agents costs nothing on the literature side. Briefs for scenes
in the same unit share most of their content — the retrieval results can be cached
per (unit, instrument) as well.

**A second use of the brief:** it tells you what the VAE's training folder should
contain for this terrain. If the brief says "boulders and wrinkle ridges expected"
and the training folder has neither, the detector will flag natural geology; the
brief is an audit of the baseline corpus as much as context for the agents.

## 4b. Synthesis step

Code assembles brief + feature output + spatial output; one Claude call with no
tools writes the verdict, allowed to cite only sources already present in the brief.

```json
{"verdict": "explained_natural | probably_artifact | probably_hardware | unexplained",
 "explained_by": {"landform": "simple_crater", "sources": ["doc:..."]} ,
 "unexplained_residual": "straight 12 m edge with 2 m relief; no expected landform matches",
 "hypotheses": [{"label": "...", "confidence": "high|medium|low", "reasoning": "...",
                 "evidence": [{"doc_id": "...", "quote": "..."}], "would_confirm": "..."}],
 "vae_trigger_explained": "edge=0.91 dominates: straight shadow edge",
 "agents_agree": true}
```

Rules: cite or don't claim; at most three hypotheses; `unexplained` is a valid and
important outcome — it is the review queue; if Feature and Spatial disagree, say so.

## 5. Running it on the project

Setup (once):

```bash
.venv/bin/uv pip install anthropic lancedb fastembed pymupdf opencv-python-headless rasterio
export ANTHROPIC_API_KEY=...
python agents/ingest.py load-tables data/reference/{gazetteer,landing_sites,robbins_moon}.csv
python agents/ingest.py add-docs papers/*.pdf --body moon
python agents/ingest.py add-exemplars data/exemplars/   # labeled chips → image index
```

Per scene:

```bash
python agents/explain.py --image <scene.png> --meta <product>.json --top 5
```

which runs the core scorer (or loads a saved result), builds the ContextBrief for
the scene (cached as `<product>_brief.json`), builds CaseFiles for the top-N, runs
Feature ∥ Spatial with the brief attached, then Synthesis, and writes
`<scene>_explained.json`. `python agents/brief.py --meta <product>.json` produces
just the brief, which is useful on its own for auditing the training folder.

Web app: one hook in `run_analysis` (`xenarch_mk19_script.py`) after
`scored = out["chips"]`, gated by `config["explain"]`, attaching `explanation` to each
detection. Scoring never calls the API; only the explain step costs.

Layout:

```
agents/
  casefile.py      # schema + builders (from core result + meta JSON)
  feature.py       # Feature agent + tools
  spatial.py       # Spatial agent + CV/catalog/DEM tools
  literature.py    # Literature agent → ContextBrief
  synthesis.py     # final verdict from brief + feature + spatial
  brief.py         # CLI: brief only
  explain.py       # orchestration + CLI
  ingest.py        # tables, docs, exemplars
data/reference/    # CSVs, DEM tiles, geologic map shapefiles (git-ignored)
```

## 6. Evaluation — one small set per agent, then end-to-end

- Feature: 100 labeled chips (from the exemplar library, held out) → class accuracy,
  artifact recall.
- Spatial: 30 chips with hand-checked context (unit, nearest crater, relief sign,
  slope band) → field accuracy; plus synthetic tests (planted collinear blobs,
  known shadow lengths).
- Literature: 15–20 scenes across units you actually download (mare, highlands,
  Mars cratered plains, chaos terrain, polar) with a hand-written expected-landform
  list → recall of expected landforms, correct unit, citation validity (quoted text
  really appears in the cited chunk).
- End-to-end: 30–50 chips with known answers (Apollo sites in repo, Ina, Reiner
  Gamma, Marius Hills pit, Perseverance hardware, CCD-seam chips, ordinary craters
  and ridges → `explained_natural`) → verdict accuracy; the ordinary-terrain cases
  are the important ones, since the goal is to close natural hits with a citation.

## 7. Phases

| Phase | Work | Done when |
|---|---|---|
| 0 | CaseFile schema; downloader writes lat/lon, footprint, pixel scale, sun geometry; taxonomy frozen; download reference tables (gazetteer, Robbins, hardware sites), one DEM tile, one geologic map | `casefile.py` builds a CaseFile from an existing result + meta JSON |
| 1 | Spatial tools without any LLM: `scene_objects`, `catalog_objects`, `terrain`, `relations`, `point_pattern`; unit tests on Apollo 11 image | scene graph JSON looks right by eye |
| 2 | Literature ingest + Literature agent producing briefs (moved up: the other agents consume it) | a brief for a Tranquillitatis NAC scene lists the right unit and landforms with valid citations |
| 3 | Feature tools + exemplar library + image index; Feature agent; Spatial agent wrapper over phase-1 tools; both read the brief | class accuracy measured |
| 4 | Synthesis, `explain.py` orchestration, web hook, end-to-end eval, prompt/retrieval tuning | ordinary craters close as `explained_natural`; Apollo 11 ends `probably_hardware` with a valid citation |
| later | small trained models as priors: chip classifier on exemplars, learned crater detector (DeepMoon-style) once torch is installed; orchestrator agent if specialists plateau | — |

## Open questions

- DEM coverage: SLDEM/MOLA are global but coarse; NAC/HiRISE DTMs are sparse. Slope
  from a 60 m DEM is fine for context, useless for a 4 m object — say so in output.
- Catalog resolution: Robbins stops at ~1 km craters; sub-km craters come only from
  `scene_objects`, so those relations carry CV detection error.
- TMC2 footprint fields in the PDS4 XML the downloader already saves — verify.
