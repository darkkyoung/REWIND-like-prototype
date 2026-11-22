"""Read the pinned official manifest and check fixed validation membership and byte duplicates."""
import json
from pathlib import Path
import zipfile

import rewind_v1
from .adapter import PINS
from .run import split_audit


def inspect(doc):
    selected = PINS["clips"].items()
    if doc["content_sha256"] != "51657741aedb25ddafaf6fbffef706f4465729f787a3fa0fa004d8dcaa65dd9c":
        raise ValueError("A manifest identity")
    result = split_audit(doc["records"],selected)
    selection = json.loads((Path(__file__).resolve().parents[1]/"v12_anchored/selection.json").read_text())
    keys = {(seq,i) for seq,start in selection["train"] for i in range(start,start+50)}
    train = [r for r in doc["records"] if (r["sequence_id"],r["frame_index"]) in keys]
    valid_keys = {(s,f) for s,start in PINS["clips"].values() for f in range(start,start+50)}
    valid = [r for r in doc["records"] if (r["sequence_id"],r["frame_index"]) in valid_keys]
    pins = {pin["sha256"] for r in train for pin in r["files"].values()}
    result["root_train300_manifest_rows"] = len(train)
    result["root_train300_exact_overlap"] = sum(pin["sha256"] in pins for r in valid for pin in r["files"].values())
    if len(train) != 300 or result["root_train300_exact_overlap"]:
        raise ValueError("RootAnchored train overlap or missing files")
    return result


if __name__ == "__main__":
    path = Path(rewind_v1.__file__).parents[1]/"reference/v2_snapshot_manifest.zip"
    with zipfile.ZipFile(path) as z, z.open("audited_manifest.json") as f:
        print(json.dumps(inspect(json.load(f)),indent=2))
