"""Recheck the archived T4 evidence without loading weights or executing archived code."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from .adapter import PINS
from .decision import decide

FAILED_ZIP = "f849916e5b9c052fa9cf60451d4471183aafe6709e4848c320cbf4e05891dd52"


def inspect(path, index):
    with path.open("rb") as stream:
        if hashlib.file_digest(stream,"sha256").hexdigest() != FAILED_ZIP:
            raise ValueError("T4 archive SHA")
    with zipfile.ZipFile(path) as archive:
        def doc(name):
            return json.loads(archive.read(name))
        manifest = doc("SHA256_MANIFEST.json")
        if any(hashlib.sha256(archive.read(k)).hexdigest() != v for k,v in manifest.items()):
            raise ValueError("T4 archive member hash")
        inputs = doc("logs/selected_input_sha256.json")
        members = json.loads(index.read_bytes())["members"]
        if any(members[k]["sha256"] != v for k,v in inputs.items()) or doc("pins.json") != PINS:
            raise ValueError("selected input identity")
        parity = doc("result/reference_only/parity.json")
        strict = decide("strict",parity["clips"])
        return dict(zip_sha256=FAILED_ZIP,manifest_entries=len(manifest),verified_input_pins=len(inputs),
                    execution=doc("execution.json"),published_environment=doc("logs/published_environment.json"),
                    prior_strict=strict,observed=doc("logs/recalculated_parity.json"),
                    deltas={k:r["published_parity"] for k,r in parity["clips"].items()},
                    binding_diff=doc("logs/binding_diff.json"),candidate_forward="not run")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("archive",type=Path)
    p.add_argument("index",type=Path)
    a = p.parse_args()
    print(json.dumps(inspect(a.archive,a.index),indent=2))
