#!/usr/bin/env python3
"""
Download a sample of HiRISE EDR products (Mars Reconnaissance Orbiter)
by querying the PDS Imaging Atlas Solr API and downloading the .IMG files
directly from the HiRISE PDS archive (University of Arizona).

Usage:
    python download_hirise_edr.py --count 20 --outdir data
    python download_hirise_edr.py --every 10 --outdir data
    python download_hirise_edr.py --count 50 --ccd RED

Search API: https://pds-imaging.jpl.nasa.gov/solr/pds_archives/search
Data archive: https://hirise-pds.lpl.arizona.edu/PDS/<FILE_NAME_SPECIFICATION>
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
HIRISE_BASE = "https://hirise-pds.lpl.arizona.edu/PDS/"
USER_AGENT = "hirise-edr-sample-downloader/1.0 (contact: giulia.sironi.02@gmail.com)"


def solr_query(count, start=0, ccd=None):
    fq = ["ATLAS_INSTRUMENT_NAME:hirise", "PRODUCT_TYPE:edr"]
    if ccd:
        fq.append(f"CCD_NAME:{ccd}*")
    params = {
        "q": "*:*",
        "fq": fq,
        "rows": str(count),
        "start": str(start),
        "wt": "json",
    }
    qs = urllib.parse.urlencode(params, doseq=True)
    url = f"{SOLR_URL}?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return data["response"]["docs"]


def solr_num_found(ccd=None):
    fq = ["ATLAS_INSTRUMENT_NAME:hirise", "PRODUCT_TYPE:edr"]
    if ccd:
        fq.append(f"CCD_NAME:{ccd}*")
    params = {"q": "*:*", "fq": fq, "rows": "0", "wt": "json"}
    qs = urllib.parse.urlencode(params, doseq=True)
    req = urllib.request.Request(f"{SOLR_URL}?{qs}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return data["response"]["numFound"]


def free_mb(path):
    return shutil.disk_usage(path).free / (1024 * 1024)


def download_file(url, dest_path, retries=3):
    if os.path.exists(dest_path):
        print(f"  already present, skipping: {os.path.basename(dest_path)}")
        return True
    tmp_path = dest_path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp_path, "wb") as f:
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
                        print(f"\r  {os.path.basename(dest_path)}: {pct}% ({downloaded}/{total} bytes)", end="")
            print()
            os.replace(tmp_path, dest_path)
            return True
        except Exception as e:
            print(f"\n  attempt {attempt}/{retries} failed: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            time.sleep(2 * attempt)
    return False


def iter_every_nth_docs(every, ccd=None):
    """Systematic sampling: take 1 product every `every` results across the whole archive."""
    total = solr_num_found(ccd=ccd)
    target = max(1, total // every)
    print(f"Total archive: {total} products. Taking 1 every {every} => ~{target} products.")
    offset = 0
    while offset < total:
        docs = solr_query(1, start=offset, ccd=ccd)
        if docs:
            yield docs[0]
        offset += every


def main():
    ap = argparse.ArgumentParser(description="Download a HiRISE EDR sample")
    ap.add_argument("--count", type=int, default=20, help="number of products to download as a sequential block (default 20)")
    ap.add_argument("--start", type=int, default=0, help="offset into the search results (block mode)")
    ap.add_argument("--every", type=int, default=None,
                     help="if set, systematically sample 1 product every N results across the whole "
                          "archive instead of a sequential block (e.g. --every 10)")
    ap.add_argument("--ccd", default=None, help="filter by CCD, e.g. RED, IR, BG")
    ap.add_argument("--outdir", default="data", help="destination folder")
    ap.add_argument("--delay", type=float, default=1.0, help="pause in seconds between downloads")
    ap.add_argument("--min-free-mb", type=float, default=300,
                     help="abort the download if free disk space drops below this threshold (MB)")
    args = ap.parse_args()

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    if args.every:
        print(f"Systematic sampling mode: 1 HiRISE EDR product every {args.every}...")
        docs_iter = iter_every_nth_docs(args.every, ccd=args.ccd)
    else:
        print(f"Querying the PDS Imaging Atlas API for {args.count} HiRISE EDR products...")
        docs = solr_query(args.count, start=args.start, ccd=args.ccd)
        if not docs:
            print("No products found.")
            sys.exit(1)
        print(f"Found {len(docs)} products. Starting download into: {outdir}")
        docs_iter = iter(docs)

    manifest_path = os.path.join(outdir, "..", "manifest.csv")
    manifest_path = os.path.abspath(manifest_path)
    write_header = not os.path.exists(manifest_path)

    ok, failed, stopped_for_space = 0, 0, 0
    with open(manifest_path, "a", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        if write_header:
            writer.writerow([
                "product_id", "ccd", "orbit", "observation_start_time",
                "target", "file_name_specification", "download_url", "local_path", "status"
            ])
        for i, doc in enumerate(docs_iter, 1):
            free = free_mb(outdir)
            if free < args.min_free_mb:
                print(f"\nFree space dropped to {free:.0f} MB (< threshold {args.min_free_mb} MB). "
                      f"Stopping download for safety.")
                stopped_for_space += 1
                break
            file_spec = doc.get("FILE_NAME_SPECIFICATION")
            if not file_spec:
                continue
            url = HIRISE_BASE + file_spec
            fname = os.path.basename(file_spec)
            dest = os.path.join(outdir, fname)
            print(f"[{i}] {fname} (free space: {free:.0f} MB)")
            success = download_file(url, dest)
            status = "OK" if success else "FAILED"
            ok += success
            failed += not success
            writer.writerow([
                doc.get("PRODUCT_ID", ""),
                doc.get("CCD_NAME", ""),
                doc.get("ORBIT_NUMBER", ""),
                doc.get("OBSERVATION_START_TIME", doc.get("START_TIME", "")),
                doc.get("TARGET_NAME", ""),
                file_spec,
                url,
                dest if success else "",
                status,
            ])
            mf.flush()
            time.sleep(args.delay)

    print(f"\nDone: {ok} downloaded, {failed} failed"
          f"{', stopped due to low disk space' if stopped_for_space else ''}.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
