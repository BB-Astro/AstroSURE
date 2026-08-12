"""Robust MAST downloader for the Arp SNAP targets (program 15446).

Downloads, per target, the two HAP (hst_*) FLC exposures plus the visit/filter
level combined HAP DRC into

    ~/DocM2max/Astro/ArpNNN_MAST/mastDownload/HST/<obs_id>/<filename>

Each file is streamed to a .part sidecar with HTTP Range resume and retried
until its size matches the MAST catalog size, so a broken connection never
leaves a truncated FITS behind.
"""
import json
import os
import sys
import time

import requests
from astroquery.mast import Observations

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser("~/DocM2max/Astro")
API = "https://mast.stsci.edu/api/v0.1/Download/file?uri="

HAP_SVM = {
    141: "hst_15446_23_acs_wfc_f606w_jdrz23",
    180: "hst_15446_40_acs_wfc_f606w_jdrz40",
    197: "hst_15446_46_acs_wfc_f606w_jdrz46",
    176: "hst_15446_38_acs_wfc_f606w_jdrz38",
    280: "hst_15446_85_acs_wfc_f606w_jdrz85",
    219: "hst_15446_59_acs_wfc_f606w_jdrz59",
    221: "hst_15446_61_acs_wfc_f606w_jdrz61",
    303: "hst_15446_0a_acs_wfc_f606w_jdrz0a",
    190: "hst_15446_43_acs_wfc_f606w_jdrz43",
    164: "hst_15446_33_acs_wfc_f606w_jdrz33",
    293: "hst_15446_94_acs_wfc_f606w_jdrz94",
}

MAX_TRIES = 40
BLOCK = 1 << 18  # small blocks keep the stall check responsive
# A stalled MAST connection never recovers on its own: drop it and resume with a
# fresh Range request as soon as the moving rate falls under MIN_RATE.
MIN_RATE = 400_000        # bytes/s
RATE_WINDOW = 25.0        # seconds before the rate is judged


def fetch(uri, dest, expected, log):
    """Stream `uri` to `dest`, resuming/retrying until size == expected."""
    if os.path.exists(dest) and os.path.getsize(dest) == expected:
        log("  cached  %s" % os.path.basename(dest))
        return True
    part = dest + ".part"
    url = API + requests.utils.quote(uri, safe="")
    for attempt in range(1, MAX_TRIES + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have > expected:
            os.remove(part)
            have = 0
        headers = {"Range": "bytes=%d-" % have} if have else {}
        try:
            r = requests.get(url, headers=headers, stream=True,
                             timeout=(20, 45))
            if r.status_code == 206:
                mode = "ab"
            elif r.status_code == 200:
                mode, have = "wb", 0
            else:
                r.close()
                raise RuntimeError("HTTP %d" % r.status_code)
            mark_t = time.time()
            mark_b = 0
            got_now = 0
            with open(part, mode) as fh:
                for block in r.iter_content(BLOCK):
                    if not block:
                        continue
                    fh.write(block)
                    got_now += len(block)
                    # sliding window: judge only the most recent RATE_WINDOW
                    # seconds, otherwise a fast start hides a later stall
                    dt = time.time() - mark_t
                    if dt >= RATE_WINDOW:
                        rate = (got_now - mark_b) / dt
                        if rate < MIN_RATE:
                            raise RuntimeError("stalled at %.0f kB/s" % (rate / 1e3))
                        mark_t, mark_b = time.time(), got_now
            r.close()
        except Exception as exc:  # noqa: BLE001
            try:
                r.close()
            except Exception:  # noqa: BLE001
                pass
            got = os.path.getsize(part) if os.path.exists(part) else 0
            log("  try %d/%d interrupted at %d/%d bytes: %s"
                % (attempt, MAX_TRIES, got, expected, exc))
            time.sleep(2)
            continue
        got = os.path.getsize(part)
        if got == expected:
            os.replace(part, dest)
            log("  OK      %s (%d bytes, %d try/tries)"
                % (os.path.basename(dest), got, attempt))
            return True
        log("  try %d/%d short: %d/%d bytes" % (attempt, MAX_TRIES, got, expected))
        time.sleep(2)
    log("  FAILED  %s" % os.path.basename(dest))
    return False


def select(prod, svm_id):
    keep = []
    for row in prod:
        fn = str(row["productFilename"])
        sub = str(row["productSubGroupDescription"])
        oid = str(row["obs_id"])
        if not fn.startswith("hst_"):
            continue  # pitfall 1: never mix in the short-rootname variant
        if sub == "FLC" and oid.startswith(svm_id):
            keep.append((oid, fn, str(row["dataURI"]), int(row["size"])))
        elif sub == "DRC" and oid == svm_id:
            keep.append((oid, fn, str(row["dataURI"]), int(row["size"])))
    return keep


def main():
    only = [int(a) for a in sys.argv[1:]] or sorted(HAP_SVM)
    tag = "-".join(str(a) for a in only)
    rep_path = os.path.join(HERE, "report_%s.json" % tag)
    report = {}

    def log(msg):
        print(msg, flush=True)

    for arp in only:
        svm_id = HAP_SVM[arp]
        tdir = os.path.join(ROOT, "Arp%03d_MAST" % arp)
        log("\n===== ARP %d  %s  ->  %s" % (arp, svm_id, tdir))
        obs = Observations.query_criteria(obs_id=svm_id)
        prod = Observations.get_product_list(obs)
        items = select(prod, svm_id)
        log("  %d products, %.1f MB expected"
            % (len(items), sum(i[3] for i in items) / 1e6))
        files = []
        for oid, fn, uri, size in items:
            ddir = os.path.join(tdir, "mastDownload", "HST", oid)
            os.makedirs(ddir, exist_ok=True)
            dest = os.path.join(ddir, fn)
            ok = fetch(uri, dest, size, log)
            files.append({"file": fn, "obs_id": oid, "path": dest,
                          "size": size, "ok": ok})
        report[arp] = {"svm_id": svm_id, "dir": tdir, "files": files}
        with open(rep_path, "w") as fh:
            json.dump(report, fh, indent=1)
        log("  target done")
    log("\nALL_TARGETS_DONE")


if __name__ == "__main__":
    main()
