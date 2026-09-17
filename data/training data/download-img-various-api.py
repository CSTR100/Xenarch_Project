#!/usr/bin/env python3
"""
Multi-dataset planetary imagery downloader.

Covers the sources shared in chat:

  # HiRISE EDR (Mars Reconnaissance Orbiter)
  #   Browse UI : https://pds-imaging.jpl.nasa.gov/tools/atlas/search?gather.common.instrument=HIRISE&gather.common.product_type=EDR
  #   API used  : PDS Imaging Atlas Solr search  https://pds-imaging.jpl.nasa.gov/solr/pds_archives/search
  #   Downloads : raw .IMG files from https://hirise-pds.lpl.arizona.edu/PDS/<FILE_NAME_SPECIFICATION>
  #   Status    : verified working (see download_hirise_edr.py, the dedicated script for this one)

  # CTX (Context Camera, Mars Reconnaissance Orbiter) -- "low quality but ok"
  #   Browse UI : https://pds-imaging.jpl.nasa.gov/tools/atlas/search?gather.common.mission=mro&gather.common.instrument=CTX
  #   API used  : same PDS Imaging Atlas Solr search as HiRISE, filtered to instrument=ctx
  #   Downloads : constructed from ATLAS_VOLUME_URL + FILE_PATH + FILE_NAME_SPECIFICATION
  #   Status    : UNVERIFIED -- several sampled products 404'd through the pds-imaging.jpl.nasa.gov
  #               mirror at test time (2026-08-27). CTX EDRs may have moved to a different mirror
  #               (e.g. the PDS Geosciences Node or MSSS). Check ATLAS_DATA_URL / ATLAS_LABEL_URL
  #               in the returned doc and adjust build_url() below if downloads keep failing.

  # LRO (Lunar Reconnaissance Orbiter) -- "kinda" -- using the LROC camera instrument
  #   Browse UI : https://pds-imaging.jpl.nasa.gov/tools/atlas/search?gather.common.mission=lro
  #   API used  : same Solr search, filtered to instrument=lroc (the plain "mission=lro" filter
  #               also matches non-imagery instruments like LAMP, which is why LROC is pinned here)
  #   Downloads : ATLAS_VOLUME_URL + "/" + FILE_SPECIFICATION_NAME, following redirects
  #               (redirects through lroc.sese.asu.edu -> lroc.im-ldi.com -> pds.mcp.nasa.gov)
  #   Status    : verified working, but LROC NAC frames are large (~500 MB each) -- the disk
  #               guard below matters a lot more here than for the other datasets.

  # MOC (Mars Orbiter Camera, Mars Global Surveyor) -- "low res"
  #   Browse UI : https://pds-imaging.jpl.nasa.gov/tools/atlas/search?gather.common.spacecraft=mars_global_surveyor&gather.common.instrument=MOC
  #   API used  : same Solr search, filtered to spacecraft=mars global surveyor, instrument=moc
  #   Downloads : https://pds-imaging.jpl.nasa.gov/data/mgs/moc/ + FILE_PATH + FILE_NAME
  #               (compressed .imq images -- need MOC/ISIS tools to decompress into a raster)
  #   Status    : verified working

  # Chandrayaan-2 TMC2 (ISRO Pradan)
  #   Browse UI : https://pradan.issdc.gov.in/ch2/protected/browse.xhtml?id=tmc2
  #             : https://pradan.issdc.gov.in/ch2/protected/browse.xhtml
  #   API used  : NONE -- this is a JSF (JavaServer Faces) portal behind an ISSDC login.
  #               There is no public/documented REST API; the "protected" path in the URL
  #               means every request needs an authenticated session (cookies + a JSF
  #               viewstate/CSRF token generated per session). Scripting this requires
  #               logging in through a real browser first, then replaying the session
  #               cookie -- it cannot be done anonymously like the NASA PDS endpoints above.
  #   Status    : NOT IMPLEMENTED. See download_pradan_tmc2() below for what's needed.

All NASA/PDS datasets default to sampling ~1% of the archive (systematic sampling,
1 product every `stride` results) rather than downloading sequential blocks, and all
downloads stop automatically if free disk space drops below --min-free-mb.

This script only defines the downloader -- run it explicitly to actually fetch data:
    python download_all_datasets.py --dataset hirise --pct 1.0
    python download_all_datasets.py --dataset all --pct 1.0 --outdir data
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time
import urllib.request
import urllib.parse

SOLR_URL = "https://pds-imaging.jpl.nasa.gov/solr/pds_archives/search"
USER_AGENT = "planetary-dataset-sample-downloader/1.0 (contact: giulia.sironi.02@gmail.com)"


# --- per-dataset configuration ------------------------------------------------
#
# `fq` are the Solr filter-query clauses that select this instrument/mission in the
# PDS Imaging Atlas index. `build_url(doc)` turns one Solr result document into the
# actual file URL to download, using the fields observed in that dataset's documents.

def _hirise_url(doc):
    # hirise-pds.lpl.arizona.edu hosts self-labeled PDS3 .IMG files directly.
    return "https://hirise-pds.lpl.arizona.edu/PDS/" + doc["FILE_NAME_SPECIFICATION"]


def _ctx_url(doc):
    # The Atlas record's ATLAS_VOLUME_URL still points at the retired
    # .../data/mro/mars_reconnaissance_orbiter/ctx/ tree (404s since ~2026);
    # the volumes now live under .../data/mro/ctx/.
    volume = doc["ATLAS_VOLUME_URL"].rstrip("/").rsplit("/", 1)[-1]
    return ("https://pds-imaging.jpl.nasa.gov/data/mro/ctx/" + volume + "/"
            + doc["FILE_PATH"] + "/" + doc["FILE_NAME_SPECIFICATION"])


def _moc_url(doc):
    return "https://pds-imaging.jpl.nasa.gov/data/mgs/moc/" + doc["FILE_PATH"] + doc["FILE_NAME"]


def _lroc_url(doc):
    return doc["ATLAS_VOLUME_URL"] + "/" + doc["FILE_SPECIFICATION_NAME"]


DATASETS = {
    "hirise": {
        "label": "HiRISE EDR (Mars Reconnaissance Orbiter)",
        "fq": ["ATLAS_INSTRUMENT_NAME:hirise", "PRODUCT_TYPE:edr"],
        "build_url": _hirise_url,
        "filename_field": "FILE_NAME_SPECIFICATION",
    },
    "ctx": {
        "label": "CTX (Mars Reconnaissance Orbiter)",
        "fq": ["ATLAS_INSTRUMENT_NAME:ctx", 'ATLAS_MISSION_NAME:"Mars Reconnaissance Orbiter"'],
        "build_url": _ctx_url,
        "filename_field": "FILE_NAME_SPECIFICATION",
    },
    "moc": {
        "label": "MOC (Mars Global Surveyor)",
        "fq": ["ATLAS_INSTRUMENT_NAME:moc", 'ATLAS_SPACECRAFT_NAME:"Mars Global Surveyor"'],
        "build_url": _moc_url,
        "filename_field": "FILE_NAME",
    },
    "lroc": {
        "label": "LROC (Lunar Reconnaissance Orbiter)",
        "fq": ["ATLAS_INSTRUMENT_NAME:lroc"],
        "build_url": _lroc_url,
        "filename_field": "FILE_SPECIFICATION_NAME",
    },
    "clementine": {
        # Lunar global mapping mission (1994), ~600k UVVIS frames. The raw
        # EDRs use Clementine-JPEG compression (undecodable without ISIS), so
        # download the archive's browse JPEG instead -- UVVIS native resolution
        # is only 384x288, and the browse JPEG is that full size.
        "label": "Clementine UVVIS (Moon)",
        "fq": ["ATLAS_MISSION_NAME:clementine", "ATLAS_INSTRUMENT_NAME:uvvis"],
        "build_url": lambda doc: doc["ATLAS_BROWSE_URL"],
        "filename_field": "ATLAS_BROWSE_URL",
    },
    "lroc-wac": {
        # Same archive as lroc but restricted to the Wide Angle Camera:
        # WAC frames are a few MB instead of the ~500 MB NAC strips, which
        # makes sampling the Moon at scale practical.
        "label": "LROC WAC (Lunar Reconnaissance Orbiter, wide-angle)",
        "fq": ["ATLAS_INSTRUMENT_NAME:lroc", "FILE_SPECIFICATION_NAME:*WAC*"],
        "build_url": _lroc_url,
        "filename_field": "FILE_SPECIFICATION_NAME",
    },
}


# --- thumbnail pre-filter -------------------------------------------------------

def _fix_extras_url(u):
    # The Atlas index still points thumbnails/browse at retired hosts/paths.
    u = u.replace("https://pdsimg.jpl.nasa.gov//data/mro/mars_reconnaissance_orbiter/ctx/",
                  "https://pds-imaging.jpl.nasa.gov/data/mro/ctx/")
    return u


def _looks_blank(arr):
    """Shared blank test: True when an 8-bit grayscale array is a white-out,
    black frame, or mostly data gaps -- judged on the interior (nonzero) pixels."""
    interior = arr[arr > 0]
    return (interior.size < arr.size * 0.05 or interior.std() < 6
            or (interior > 230).mean() > 0.6 or interior.mean() < 25)


def thumbnail_is_blank(doc):
    """Fetch the catalog's tiny thumbnail (a few KB) and test it for blankness
    BEFORE spending a full-product download. Returns True (blank -- skip),
    False (looks fine), or None (no usable thumbnail -- download anyway)."""
    url = doc.get("ATLAS_THUMBNAIL_URL") or doc.get("ATLAS_BROWSE_URL")
    if not url:
        return None
    try:
        import io
        import numpy as np
        from PIL import Image
        req = urllib.request.Request(_fix_extras_url(url),
                                     headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        arr = np.asarray(Image.open(io.BytesIO(data)).convert("L"), dtype="float32")
        return bool(_looks_blank(arr))
    except Exception:
        return None


# --- Solr search helpers -------------------------------------------------------

def _solr_get(params, retries=4):
    """Catalog query with retry -- the Atlas Solr endpoint throws transient
    502/503s under load, which shouldn't kill a long sampling run."""
    qs = urllib.parse.urlencode(params, doseq=True)
    req = urllib.request.Request(f"{SOLR_URL}?{qs}", headers={"User-Agent": USER_AGENT})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except Exception as e:
            if attempt == retries:
                raise
            print(f"    catalog query attempt {attempt}/{retries} failed ({e}), retrying")
            time.sleep(5 * attempt)


def solr_query(fq, count, start=0):
    data = _solr_get({"q": "*:*", "fq": fq, "rows": str(count),
                      "start": str(start), "wt": "json"})
    return data["response"]["docs"]


def solr_num_found(fq):
    data = _solr_get({"q": "*:*", "fq": fq, "rows": "0", "wt": "json"})
    return data["response"]["numFound"]


def iter_sample_docs(fq, pct, stride=None, max_files=None):
    """Systematic sample spread evenly across the archive.

    By default samples pct% of the archive (1 product every total//target).
    Pass `stride` to instead take exactly 1 product every `stride` results,
    and `max_files` to cap how many products are yielded either way.
    """
    total = solr_num_found(fq)
    if stride:
        target = max(1, total // stride)
    else:
        target = max(1, int(total * pct / 100))
        stride = max(1, total // target)
    if max_files:
        target = min(target, max_files)
    print(f"    archive size: {total} products => ~{target} products "
          f"(1 every {stride}{f', capped at {max_files}' if max_files else ''})")
    offset = 0
    yielded = 0
    while offset < total and yielded < target:
        try:
            docs = solr_query(fq, 1, start=offset)
        except Exception as e:
            # Deep pagination (large start=) reliably 502s on the biggest
            # archives -- skip this offset rather than aborting the whole run.
            print(f"    catalog query at offset {offset} failed ({e}), skipping")
            offset += stride
            continue
        if docs:
            yield docs[0]
            yielded += 1
        offset += stride


# --- download helpers -----------------------------------------------------------

def free_mb(path):
    return shutil.disk_usage(path).free / (1024 * 1024)


def download_file(url, dest_path, retries=3):
    if os.path.exists(dest_path):
        print(f"    already present, skipping: {os.path.basename(dest_path)}")
        return True
    tmp_path = dest_path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=180) as resp, open(tmp_path, "wb") as f:
                total = resp.getheader("Content-Length")
                total = int(total) if total else None
                downloaded = 0
                chunk = 1024 * 256
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
                    downloaded += len(buf)
                    if total:
                        pct = downloaded * 100 // total
                        print(f"\r    {os.path.basename(dest_path)}: {pct}% "
                              f"({downloaded}/{total} bytes)", end="")
            print()
            os.replace(tmp_path, dest_path)
            return True
        except Exception as e:
            print(f"\n    attempt {attempt}/{retries} failed: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            time.sleep(2 * attempt)
    return False


def _read_pds3_image(path):
    """Read the IMAGE object of a PDS3 .IMG file (e.g. HiRISE EDR) into a
    float32 array, stripping per-line prefix/suffix bytes that GDAL's PDS
    driver mishandles and masking 0xFF gap/missing pixels as NaN."""
    import numpy as np
    import re

    with open(path, "rb") as f:
        label = f.read(65536).decode("latin-1", errors="replace")

    def _label_int(pattern, default=None):
        mm = re.search(pattern, label)
        if mm:
            return int(mm.group(1))
        if default is None:
            raise ValueError(f"label field not found: {pattern}")
        return default

    offset = _label_int(r"\^IMAGE\s*=\s*(\d+)\s*<BYTES>") - 1
    img_block = re.search(r"OBJECT\s*=\s*IMAGE\b(.*?)END_OBJECT\s*=\s*IMAGE\b",
                          label, re.S)
    if not img_block:
        raise ValueError("no IMAGE object in label")
    blk = img_block.group(1)

    def _blk_int(field, default=None):
        mm = re.search(rf"{field}\s*=\s*(\d+)", blk)
        if mm:
            return int(mm.group(1))
        if default is None:
            raise ValueError(f"IMAGE field not found: {field}")
        return default

    lines = _blk_int("LINES")
    samples = _blk_int("LINE_SAMPLES")
    bits = _blk_int("SAMPLE_BITS", 8)
    prefix = _blk_int("LINE_PREFIX_BYTES", 0)
    suffix = _blk_int("LINE_SUFFIX_BYTES", 0)

    bpp = bits // 8
    rec = prefix + samples * bpp + suffix
    raw = np.fromfile(path, dtype=np.uint8, offset=offset, count=lines * rec)
    lines = raw.size // rec  # tolerate truncated downloads
    raw = raw[: lines * rec].reshape(lines, rec)[:, prefix:prefix + samples * bpp]
    if bpp == 2:
        dtype = ">u2" if "MSB" in blk else "<u2"
        arr = raw.reshape(lines, samples, 2).copy().view(dtype)[:, :, 0].astype("float32")
        arr[arr == 0xFFFF] = np.nan
    else:
        arr = raw.astype("float32")
        arr[arr == 0xFF] = np.nan  # MISSING_CONSTANT / gap value
    return arr


def convert_and_resize(src_path, png_path, max_px):
    """Decode a downloaded product and save it as an 8-bit PNG terrain patch
    of at most `max_px` x `max_px`: the short side is scaled down to `max_px`
    and the long side is center-cropped. (Orbital products are often extremely
    elongated strips -- e.g. HiRISE EDRs at 1024 x ~25000 px -- so fitting the
    whole strip into a square thumbnail would destroy all detail.)
    Uses rasterio (GDAL) for PDS .IMG files, Pillow for everything else.
    Returns True on success."""
    import numpy as np
    from PIL import Image

    arr = None
    if src_path.upper().endswith(".IMG"):
        try:
            arr = _read_pds3_image(src_path)
        except Exception as e:
            print(f"    PDS3 parse failed ({e}), falling back to rasterio")
    if arr is None:
        try:
            import rasterio
            with rasterio.open(src_path) as ds:
                arr = ds.read(1).astype("float32")
        except Exception:
            try:
                arr = np.asarray(Image.open(src_path).convert("L"), dtype="float32")
            except Exception as e:
                print(f"    convert failed ({os.path.basename(src_path)}): {e}")
                return False

    # Percentile stretch to 8-bit -- raw planetary data is 10-16 bit and would
    # come out nearly black if divided by the full dtype range.
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        print(f"    convert failed ({os.path.basename(src_path)}): no valid pixels")
        return False
    # Estimate the stretch bounds ignoring saturated/nodata extremes (black
    # fill borders and pure-white gaps would otherwise flatten the contrast).
    interior = finite[(finite > 0) & (finite < finite.max())]
    if interior.size > finite.size * 0.01:
        finite = interior
    lo, hi = np.percentile(finite, (0.5, 99.5))
    if hi <= lo:
        lo, hi = finite.min(), max(finite.max(), finite.min() + 1)
    arr = np.nan_to_num(np.clip((arr - lo) / (hi - lo), 0, 1), nan=0.0)
    img = Image.fromarray((arr * 255).astype("uint8"), mode="L")

    if min(img.size) > max_px:
        scale = max_px / min(img.size)
        img = img.resize((max(1, round(img.width * scale)),
                          max(1, round(img.height * scale))), Image.LANCZOS)
    if max(img.size) > max_px:
        left = (img.width - min(img.width, max_px)) // 2
        top = (img.height - min(img.height, max_px)) // 2
        img = img.crop((left, top,
                        left + min(img.width, max_px),
                        top + min(img.height, max_px)))

    # Reject blank frames (saturated white-outs, black/night frames, gaps).
    if _looks_blank(np.asarray(img, dtype="float32")):
        print(f"    rejected as blank (white-out/black): {os.path.basename(src_path)}")
        return None

    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    img.save(png_path)
    print(f"    resized -> {png_path} ({img.width}x{img.height})")
    return True


def download_pradan_tmc2(*_args, **_kwargs):
    """Chandrayaan-2 TMC2 browse data (ISSDC Pradan) -- NOT IMPLEMENTED.

    Pradan (https://pradan.issdc.gov.in/ch2/protected/browse.xhtml?id=tmc2) sits behind
    a login-gated JSF portal, not a public API. To make this work you would need to:
      1. Log in through a real browser with an ISSDC account.
      2. Capture the session cookie(s) and JSF viewstate/CSRF token from a logged-in
         request (browser dev tools -> Network tab on a search/download request).
      3. Replay that session (cookies + viewstate) in this script's requests, refreshing
         it as it expires -- Pradan sessions are typically short-lived.
    There is no equivalent of the NASA PDS Solr search here, so query parameters and
    result parsing would have to be reverse-engineered from the portal's own requests.
    """
    raise NotImplementedError(
        "Pradan/Chandrayaan-2 requires an authenticated ISSDC session; see the "
        "docstring of download_pradan_tmc2() for what's needed before this can run."
    )


# --- main -------------------------------------------------------------------------

def run_dataset(name, pct, outdir, delay, min_free_mb,
                stride=None, max_files=None, resize=None, delete_raw=False,
                prefilter=False):
    cfg = DATASETS[name]
    print(f"\n=== {cfg['label']} ===")
    manifest_path = os.path.join(outdir, "manifest.csv")
    write_header = not os.path.exists(manifest_path)

    # With a quality gate active (prefilter or resize-reject), --max-files
    # counts KEPT images, so keep sampling the archive until the quota of
    # good ones is met rather than counting skipped/rejected frames.
    kept_quota = max_files if (prefilter or resize) else None
    iter_cap = None if kept_quota else max_files

    ok, failed, kept, skipped_blank, stopped_for_space = 0, 0, 0, 0, 0
    with open(manifest_path, "a", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        if write_header:
            writer.writerow(["dataset", "product_id", "filename", "download_url", "local_path", "status"])
        for doc in iter_sample_docs(cfg["fq"], pct, stride=stride, max_files=iter_cap):
            if kept_quota and kept >= kept_quota:
                break
            free = free_mb(outdir)
            if free < min_free_mb:
                print(f"    free space dropped to {free:.0f} MB (< {min_free_mb} MB), stopping.")
                stopped_for_space += 1
                break
            fname_field = cfg["filename_field"]
            if fname_field not in doc:
                continue
            try:
                url = cfg["build_url"](doc)
            except KeyError as e:
                print(f"    skipping product, missing field {e}")
                continue
            fname = os.path.basename(doc[fname_field])
            dest = os.path.join(outdir, name, fname)
            png = os.path.join(outdir, name + "_png",
                               os.path.splitext(fname)[0] + ".png")
            if resize and os.path.exists(png):
                kept += 1
                continue
            if prefilter and not os.path.exists(dest):
                if thumbnail_is_blank(doc):
                    print(f"  [{name}] {fname}: thumbnail looks blank, skipping download")
                    skipped_blank += 1
                    writer.writerow([name, doc.get("PRODUCT_ID", ""), fname, url,
                                      "", "SKIPPED_BLANK_THUMB"])
                    mf.flush()
                    continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            print(f"  [{name}] {fname} (free space: {free:.0f} MB)")
            success = download_file(url, dest)
            status = "OK" if success else "FAILED"
            if success:
                # Save the full PDS Atlas catalog record (coordinates, times,
                # mission info, ...) alongside the image -- it's already in hand
                # from the search query, so this costs no extra requests.
                meta_path = os.path.join(outdir, name + "_meta",
                                         os.path.splitext(fname)[0] + ".json")
                os.makedirs(os.path.dirname(meta_path), exist_ok=True)
                with open(meta_path, "w", encoding="utf-8") as jf:
                    json.dump(doc, jf, indent=2, sort_keys=True)
            if success and resize:
                result = convert_and_resize(dest, png, resize)
                if result:
                    status = "OK+PNG"
                    kept += 1
                    if delete_raw:
                        os.remove(dest)
                        dest = png
                elif result is None:
                    # blank frame: nothing worth keeping, drop the raw too
                    status = "REJECTED_BLANK"
                    if delete_raw:
                        os.remove(dest)
                        dest = ""
            ok += success
            failed += not success
            writer.writerow([name, doc.get("PRODUCT_ID", ""), fname, url,
                              dest if success else "", status])
            mf.flush()
            time.sleep(delay)

    print(f"  {name}: {ok} downloaded, {kept} kept, {skipped_blank} pre-filtered as blank, "
          f"{failed} failed"
          f"{', stopped for low disk space' if stopped_for_space else ''}.")


def main():
    ap = argparse.ArgumentParser(description="Download samples from multiple planetary imagery datasets")
    ap.add_argument("--dataset", choices=list(DATASETS) + ["all"], default="all",
                     help="which dataset to sample (default: all)")
    ap.add_argument("--pct", type=float, default=1.0,
                     help="percent of each dataset's archive to sample (default 1.0 = 1%%)")
    ap.add_argument("--outdir", default="data", help="destination folder")
    ap.add_argument("--delay", type=float, default=1.0, help="pause in seconds between downloads")
    ap.add_argument("--min-free-mb", type=float, default=300,
                     help="abort downloads if free disk space drops below this threshold (MB)")
    ap.add_argument("--stride", type=int, default=None,
                     help="take exactly 1 product every N archive results (overrides --pct); "
                          "combine with --max-files, archives hold millions of products")
    ap.add_argument("--max-files", type=int, default=None,
                     help="hard cap on downloads per dataset")
    ap.add_argument("--resize", type=int, default=None, metavar="PX",
                     help="also save each image as an 8-bit PNG with longest side PX "
                          "(into <outdir>/<dataset>_png/); needs rasterio+pillow")
    ap.add_argument("--delete-raw", action="store_true",
                     help="with --resize: delete the raw product after a successful conversion")
    ap.add_argument("--prefilter", action="store_true",
                     help="fetch the catalog thumbnail first and skip blank "
                          "(white/black) frames before downloading the full product")
    args = ap.parse_args()

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in names:
        run_dataset(name, args.pct, outdir, args.delay, args.min_free_mb,
                    stride=args.stride, max_files=args.max_files,
                    resize=args.resize, delete_raw=args.delete_raw,
                    prefilter=args.prefilter)


if __name__ == "__main__":
    main()
