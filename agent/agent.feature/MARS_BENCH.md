# Mars-Bench in the feature agent

Submodule: `Mars-Bench/` → https://github.com/kerner-lab/Mars-Bench
(Purohit et al., NeurIPS 2025 Datasets & Benchmarks, https://arxiv.org/abs/2510.24010)

Update / first clone:

```bash
git submodule update --init agent/agent.feature/Mars-Bench
```

## Can we use it?

Yes.

| | |
|---|---|
| Code license | MIT (declared in `pyproject.toml`; no LICENSE file in the repo yet) |
| Dataset licenses | CC-BY-4.0 on every Hugging Face dataset checked (domars16k, landmark_cls, surface_cls, frost_cls, boulder_det, crater_binary_seg, dust_devil_det, change_cls_hirise) — attribution required, commercial use allowed |
| Coverage | **Mars only.** 20 datasets, orbital (HiRISE, CTX, HRSC) and rover (MER, MSL). No lunar data; lunar exemplars come from LROC (see SOURCES.md §1) |
| Dependencies of the training code | torch, torchvision, pytorch-lightning, hydra, transformers, effdet … — heavy, and the venv has no torch. We do **not** need any of it for the first three uses below |

## What we take from it, in order

### 1. Exemplar library (now, no torch)

The `nearest_exemplars` tool needs labeled reference chips. Mars-Bench gives tens of
thousands, already split into train/val/test, straight from Hugging Face via the
`datasets` package (pure Python + pyarrow):

| HF dataset | Task | What it contributes |
|---|---|---|
| `Mirali33/mb-domars16k` | 15-class landform classification, CTX | the core Mars landform exemplars: aeolian bedforms, cliff, ridge, channel, mounds, gullies, slope streaks, mass wasting, crater, crater field, mixed/rough/smooth/textured terrain |
| `Mirali33/mb-landmark_cls` | HiRISE landmarks (Wagstaff et al.) | crater, dark/bright dune, slope streak, impact ejecta, swiss cheese, spider, other |
| `Mirali33/mb-surface_cls` | surface classification | terrain types |
| `Mirali33/mb-frost_cls` | frost / no frost | seasonal-process exemplars (false-positive class for the VAE) |
| `Mirali33/mb-crater_binary_seg`, `mb-crater_multi_seg` | crater masks | crater exemplars with outlines → rim/floor measurements |
| `Mirali33/mb-boulder_det`, `mb-boulder_seg` | boulder boxes/masks | boulder exemplars with sizes |
| `Mirali33/mb-dust_devil_det` | dust-devil boxes | dust devil / track exemplars |
| `Mirali33/mb-conequest_det`, `mb-conequest_seg` | cone boxes/masks | volcanic cone exemplars |
| `Mirali33/mb-change_cls_hirise`, `mb-change_cls_ctx` | temporal-pair change classification | "new feature vs nothing changed" pairs — useful negatives |

Ingest: `agents/ingest.py add-exemplars --hf Mirali33/mb-domars16k --split train`
loads images + labels, embeds with fastembed (CLIP ONNX), and writes them to the
exemplar index with `source=mars-bench`, `dataset`, `label`, `body=MARS`,
`instrument` (CTX / HiRISE from the dataset card). Detection/segmentation sets are
cropped to their boxes/masks before embedding so each exemplar is one object.

### 2. Prompt and class definitions (now)

`Mars-Bench/marsbench/configs/prompts/classification/domars16k.yaml` is a ready-made
VLM system prompt with the 15 DoMars16k classes, abbreviations and one-line
definitions. Reuse the definitions verbatim in the feature agent's system prompt for
Mars scenes, mapped onto our taxonomy:

| DoMars16k | Xenarch feature class |
|---|---|
| ael, aec | dune / aeolian bedform |
| cli, rid | scarp / ridge |
| fsf | channel / rille-like |
| sfe | mound / cone |
| fsg | gully |
| fse | slope streak (RSL-adjacent) |
| fss | mass wasting |
| cra, sfx | simple crater / crater field |
| mix, rou, smo, tex | terrain background (no discrete feature) |

### 3. Evaluation of the feature agent (now, needs API key)

Their test splits are a labeled eval set for free. Run the Claude-backed feature
agent on `mb-domars16k` and `mb-landmark_cls` test images with the same prompt
protocol and compare to the paper's baselines (ResNet/ViT/Swin fine-tuned; Gemini 2.0
Flash and GPT-4o-mini zero-shot; metric: normalized F1). The repo has VLM configs for
Gemini and ChatGPT under `marsbench/configs/vlm/`; a Claude entry follows the same
shape (`agents/eval_marsbench.py`, planned). This is the number that says whether the
feature agent is worth trusting on Mars landforms before touching anomalies.

### 4. Trained prior (later, needs torch)

Once torch is installed, `python -m marsbench.main task=classification
model_name=vit data_name=domars16k load_from_hf=true repo_id=Mirali33/mb-domars16k`
trains a landform classifier in one command. Its softmax over the 15 classes becomes
a cheap prior handed to the feature agent alongside the exemplars — and the same
classifier can score every VAE chip offline as a first-pass "this is ordinary
terrain" filter.

## What it does not give us

- Nothing lunar. Lunar exemplars: LROC featured images, QuickMap crops, the Apollo
  NAC images already in the repo, the pit atlas, Bickel rockfall map.
- No hardware class. Human-made objects come from LROC Apollo sites and HiRISE lander
  images (SOURCES.md §2).
- No instrument-artifact class. Those exemplars are cut from our own downloads.

## Attribution

Cite Purohit et al. 2025 (BibTeX in `Mars-Bench/README.md`) and the original dataset
papers listed on each Hugging Face card; CC-BY-4.0 requires attribution when
exemplars or eval results are published.
