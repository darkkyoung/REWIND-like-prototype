"""Read pinned source/tools and a single candidate member from existing mounted artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

PINS = json.loads(Path(__file__).with_name("pins.json").read_text())


def verified_write(path, stream, pin):
    h = hashlib.sha256()
    # Reuse a previously extracted verified member.
    if path.exists():
        with path.open("rb") as f:
            if hashlib.file_digest(f,"sha256").hexdigest() != pin:
                raise ValueError(f"cached member changed: {path}")
        return
    with path.open("xb") as out:
        while block := stream.read(1024*1024):
            h.update(block)
            out.write(block)
    if h.hexdigest() != pin:
        path.unlink()
        raise ValueError(f"SHA mismatch: {path}")


def prepare(bundle, anchored, dest):
    dest.mkdir(parents=True,exist_ok=False)
    (dest/"tools").mkdir()
    # For a mounted ZIP, ZipFile seeks to the required members.
    def copy_members(read):
        for name,pin in PINS["tools"].items():
            with read("tools/"+name) as stream:
                verified_write(dest/"tools"/name,stream,pin)
        with read("source/v2_gpu_training_128_v1.zip") as stream:
            verified_write(dest/"source.zip",stream,PINS["source"])
    if bundle.is_dir():
        copy_members(lambda name:(bundle/name).open("rb"))
    else:
        with zipfile.ZipFile(bundle) as z:
            copy_members(z.open)
    with zipfile.ZipFile(dest/"source.zip") as z:
        z.extractall(dest/"source")
    if anchored.suffix == ".zip":
        with zipfile.ZipFile(anchored) as z, z.open("results/anchored.pt") as stream:
            verified_write(dest/"anchored.pt",stream,PINS["anchored"])
    else:
        with anchored.open("rb") as stream:
            verified_write(dest/"anchored.pt",stream,PINS["anchored"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ("bundle","anchored","dest"):
        parser.add_argument("--"+key,type=Path,required=True)
    prepare(**vars(parser.parse_args()))
