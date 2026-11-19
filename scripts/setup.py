"""cranker-llm setup: pick a model for this GPU, download it (and llama.cpp on Windows) with checksums, write the
config.

    py scripts/setup.py                      list the profiles that fit this GPU and the recommended one
    py scripts/setup.py --profile NAME       download what it needs (asks first) and write the config
    py scripts/setup.py --profile NAME --dry-run
    py scripts/setup.py --profile NAME --model-path D:/models/x.gguf    use a file you already have (checked)
    py scripts/setup.py --profile NAME --llama-server /usr/local/bin/llama-server

Everything goes to CRANKER_HOME (~/.cranker-llm by default): bin/, models/, templates/, logs/, config.json.
llama.cpp: on Windows with an NVIDIA GPU the pinned CUDA build is downloaded from ggml-org's GitHub releases; on
Linux and macOS, install it yourself (a package or a CUDA/Metal build) and pass --llama-server, or have
llama-server on PATH. Standard library only.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOME = Path(os.environ.get("CRANKER_HOME") or Path.home() / ".cranker-llm")
WINDOWS = os.name == "nt"
UA = {"User-Agent": "cranker-llm-setup"}

LLAMA_TAG = "b12421"  # measured with every profile here (15-11-2026); Bonsai needs b11556 or newer
LLAMA_RELEASE_API = f"https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{LLAMA_TAG}"


# ------------------------------------------------------------------------------------------------- downloading

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def final_url(url):
    opener = urllib.request.build_opener(NoRedirect)
    for _ in range(8):
        try:
            opener.open(urllib.request.Request(url, method="HEAD", headers=UA), timeout=30)
            return url
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308):
                url = urllib.parse.urljoin(url, e.headers["Location"])
                continue
            raise
    return url


RETRY_WAIT = 3  # s before a piece's next attempt, times the attempt


def _piece(url, path, start, end):
    pos = start  # where this piece goes on from, after a connection that was cut or raised
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={pos}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "r+b") as f:
                if r.status != 206:  # the whole file from its start: written at pos it would be the wrong bytes
                    raise RuntimeError(f"the server ignored the range (HTTP {r.status})")
                f.seek(pos)
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
                    pos += len(b)
            if pos == end + 1:
                return
        except Exception:
            time.sleep(RETRY_WAIT * (attempt + 1))
    raise RuntimeError(f"piece {start}-{end} failed")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def download(url, out, size, sha256, parts=32, jobs=8):
    """In ranges over several connections (Hugging Face throttles some files per connection: 0.4 MB/s on one,
    24 MB/s on eight), then the SHA-256; the file gets its name only when the checksum matches. A download that
    was cut goes on from the pieces it has (NAME.part.json lists them): 17 GB cut at 90 % started from zero."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    url = final_url(url)
    part = out.with_name(out.name + ".part")
    state = out.with_name(out.name + ".part.json")
    step = -(-size // parts)
    ranges = [(s, min(s + step, size) - 1) for s in range(0, size, step)]
    this = {"size": size, "sha256": sha256, "step": step}
    have = set()
    try:  # the pieces a run before this one finished, when it was after the same file
        was = json.loads(state.read_text(encoding="utf-8"))
        if {k: was.get(k) for k in this} == this and part.stat().st_size == size:
            have = {i for i in was["done"] if 0 <= i < len(ranges)}
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if have:
        print(f"  going on from a download that was cut: {len(have)} of {len(ranges)} pieces are here")
    else:
        with open(part, "wb") as f:
            f.truncate(size)
    lock = threading.Lock()

    def one(i):
        _piece(url, part, *ranges[i])
        with lock:
            have.add(i)
            state.write_text(json.dumps({**this, "done": sorted(have)}), encoding="utf-8")

    todo = [i for i in range(len(ranges)) if i not in have]
    t0 = time.time()
    ex = concurrent.futures.ThreadPoolExecutor(jobs)
    try:
        pending = {ex.submit(one, i) for i in todo}
        while pending:  # a wait with a timeout: on Windows one without it does not see Ctrl+C until a piece ends
            done, pending = concurrent.futures.wait(pending, timeout=0.5,
                                                    return_when=concurrent.futures.FIRST_EXCEPTION)
            for f in done:
                f.result()
    except KeyboardInterrupt:
        # the pool's threads are not daemons: left to themselves they would download every piece still waiting
        ex.shutdown(wait=False, cancel_futures=True)
        print(f"\ninterrupted with {len(have)} of {len(ranges)} pieces: the same command goes on from them",
              flush=True)
        os._exit(130)
    finally:
        ex.shutdown(wait=True, cancel_futures=True)  # after a piece that failed: the ones running finish and count
    got_bytes = sum(ranges[i][1] - ranges[i][0] + 1 for i in todo)
    dt = max(time.time() - t0, 0.1)
    print(f"  {got_bytes:,} bytes in {dt:.0f} s ({got_bytes / dt / 1e6:.1f} MB/s); checking the SHA-256")
    got = sha256_of(part)
    if got != sha256:
        state.unlink(missing_ok=True)  # no piece of it can be trusted: the next run starts over
        raise RuntimeError(f"checksum mismatch for {out.name}: {got}, expected {sha256} (left at {part}; the next "
                           "run downloads it again)")
    part.replace(out)
    state.unlink(missing_ok=True)
    out.with_name(out.name + ".sha256").write_text(f"{sha256}  {out.name}\n", encoding="utf-8")


def have_file(path, size, sha256):
    """True when path is the expected file: its .sha256 next to it and its size match, or else its hash does"""
    path = Path(path)
    if not path.exists() or path.stat().st_size != size:
        return False
    side = path.with_name(path.name + ".sha256")
    if side.exists() and side.read_text(encoding="utf-8").split()[0] == sha256:
        return True
    print(f"  hashing {path} ({size / 1e9:.1f} GB)...")
    if sha256_of(path) != sha256:
        return False
    side.write_text(f"{sha256}  {path.name}\n", encoding="utf-8")
    return True


# ------------------------------------------------------------------------------------------------ this machine

def gpu():
    """(name, total MiB, CUDA version the driver supports) of the first NVIDIA GPU, or None"""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        name, mib = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        head = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10).stdout
        m = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", head)  # "CUDA UMD Version" in the 600-series drivers
        return name, int(float(mib)), (tuple(int(x) for x in m.group(1).split(".")) if m else None)
    except Exception:
        return None


def llama_assets(cuda):
    """The pinned release's Windows CUDA build and its runtime: the newest CUDA the driver supports"""
    with urllib.request.urlopen(urllib.request.Request(LLAMA_RELEASE_API, headers=UA), timeout=30) as r:
        rel = json.loads(r.read().decode("utf-8"))
    builds = {}
    for a in rel["assets"]:
        m = re.match(r"(llama-%s|cudart-llama)-bin-win-cuda-([\d.]+)-x64\.zip$" % LLAMA_TAG, a["name"])
        if m:
            builds.setdefault(m.group(2), {})[m.group(1).split("-")[0]] = a
    ok = sorted((v for v in builds if cuda is None or tuple(int(x) for x in v.split(".")) <= cuda),
                key=lambda v: tuple(int(x) for x in v.split(".")))
    if not ok:
        raise RuntimeError(f"no CUDA build of {LLAMA_TAG} for a driver with CUDA {cuda}: update the NVIDIA driver")
    pick = builds[ok[-1]]
    return [(a["name"], a["browser_download_url"], a["size"], a["digest"].split(":", 1)[1])
            for a in (pick["llama"], pick["cudart"])]


# ------------------------------------------------------------------------------------------------------- main

def ask_yes(text, assume):
    if assume:
        return True
    try:
        return input(text + " [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def main():
    ap = argparse.ArgumentParser(description="cranker-llm setup")
    ap.add_argument("--profile")
    ap.add_argument("--model-path", help="use this .gguf instead of downloading (its checksum is checked)")
    ap.add_argument("--llama-server", help="use this llama-server instead of downloading llama.cpp")
    ap.add_argument("--dry-run", action="store_true", help="say what it would do, change nothing")
    ap.add_argument("--yes", action="store_true", help="do not ask before downloading")
    a = ap.parse_args()
    profiles = json.loads((ROOT / "profiles" / "profiles.json").read_text(encoding="utf-8"))
    g = gpu()
    vram = g[1] / 1024 if g else 0
    print(f"home: {HOME}")
    print(f"GPU: {g[0]}, {vram:.0f} GB, driver CUDA {'.'.join(map(str, g[2])) if g[2] else '?'}" if g
          else "GPU: no NVIDIA GPU found (nvidia-smi); on a Mac use a Metal build of llama.cpp and pick by memory")

    if not a.profile:
        fits = [p for p in profiles if not g or p["min_vram_gb"] <= vram + 0.5]
        print("\nprofiles" + (" that fit this GPU" if g else "") + ":")
        for p in fits:
            print(f"  {p['name']:<22} {p['title']}\n  {'':<22} [{p['status']}] {p['measured']}\n")
        rec = next((p for p in fits if p["status"] == "measured"), fits[0] if fits else None)
        if rec:
            print(f"recommended: py {Path(__file__)} --profile {rec['name']}")
        else:
            print("none fits: see the README for smaller models")
        return 0

    prof = next((p for p in profiles if p["name"] == a.profile), None)
    if not prof:
        print(f"no profile {a.profile}: {', '.join(p['name'] for p in profiles)}")
        return 1
    if g and prof["min_vram_gb"] > vram + 0.5:
        print(f"warning: {prof['name']} wants {prof['min_vram_gb']} GB and this GPU has {vram:.0f} GB")

    m = prof["model"]
    model_path = Path(a.model_path) if a.model_path else HOME / "models" / m["file"]
    plan = []
    if a.model_path:
        if not have_file(model_path, m["size"], m["sha256"]):
            print(f"{model_path} is not {m['file']} (size or SHA-256 differ)")
            return 1
        print(f"model: {model_path} (checked)")
    elif have_file(model_path, m["size"], m["sha256"]):
        print(f"model: {model_path} (already there, checked)")
    else:
        url = f"https://huggingface.co/{m['repo']}/resolve/main/{m['file']}"
        plan.append((m["file"], url, m["size"], m["sha256"], model_path))

    server = None
    if a.llama_server:
        server = Path(a.llama_server)
    elif shutil.which("llama-server"):
        server = Path(shutil.which("llama-server"))
    elif (HOME / "bin" / ("llama-server.exe" if WINDOWS else "llama-server")).exists():
        server = HOME / "bin" / ("llama-server.exe" if WINDOWS else "llama-server")
    elif WINDOWS and g:
        for name, url, size, sha in llama_assets(g[2]):
            plan.append((name, url, size, sha, HOME / "downloads" / name))
        server = HOME / "bin" / "llama-server.exe"
    else:
        print(f"llama-server not found: install llama.cpp {LLAMA_TAG} or newer (a CUDA, Metal or Vulkan build) and "
              f"pass --llama-server PATH")
        return 1
    print(f"llama-server: {server}" + ("" if server.exists() else " (to be unpacked)"))

    if plan:
        total = sum(p[2] for p in plan)
        print(f"\nto download ({total / 1e9:.2f} GB):")
        for name, url, size, sha, out in plan:
            print(f"  {name}  {size / 1e9:.2f} GB  from {urllib.parse.urlsplit(url).netloc}  sha256 {sha[:16]}...")
        print(f"model license: {m['license']}")
        if a.dry_run:
            print("\n(dry run: nothing downloaded or written)")
            return 0
        if not ask_yes("download them?", a.yes):
            print("nothing downloaded")
            return 1
        for name, url, size, sha, out in plan:
            print(f"{name}:")
            download(url, out, size, sha)
            if name.endswith(".zip"):
                with zipfile.ZipFile(out) as z:
                    z.extractall(HOME / "bin")
    elif a.dry_run:
        print("\n(dry run: nothing to download; the config would be written)")
        return 0

    cfg_file = HOME / "config.json"
    cfg = json.loads(cfg_file.read_text(encoding="utf-8-sig")) if cfg_file.exists() else {}
    cfg.update(prof["config"])
    cfg.update({"model": str(model_path), "llama_server": str(server), "template": "auto", "profile": prof["name"]})
    HOME.mkdir(parents=True, exist_ok=True)
    cfg_file.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {cfg_file}\nnext: py {ROOT / 'scripts' / 'worker.py'} start")
    return 0


if __name__ == "__main__":
    sys.exit(main())
