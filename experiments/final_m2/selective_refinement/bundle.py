"""Read only the expected result members, including nested handoff ZIPs."""
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile, is_zipfile

NAMES = {'control_800.pt','improved_800.pt'} | {
    f'{split}_{arm}_predictions.npz' for split in ('opening130','gunplay47','unused_motion_preselected')
    for arm in ('control','improved')}


def unpack_results(archive, out):
    files, manifests = {}, []
    def scan(source, depth):
        with ZipFile(source) as z:
            for item in z.infolist():
                name = Path(item.filename).name
                if item.file_size>200*1024**2:
                    continue
                if name in NAMES:
                    raw=z.read(item)
                    if name in files and files[name]!=raw:
                        raise ValueError('ambiguous '+name)
                    files[name]=raw
                elif name=='checksums.json':
                    manifests.append(json.loads(z.read(item)))
                elif name.endswith('.zip') and depth<3:
                    nested=io.BytesIO(z.read(item))
                    if is_zipfile(nested):
                        scan(nested,depth+1)
    scan(archive,0)
    actual={key:hashlib.sha256(blob).hexdigest() for key,blob in files.items()}
    if files.keys()!=NAMES or not any(all(doc.get(k)==v for k,v in actual.items()) for doc in manifests):
        raise ValueError('required frozen artifacts missing or checksum mismatch')
    out.mkdir(parents=True,exist_ok=False)
    for key,blob in files.items():
        (out/key).write_bytes(blob)
    (out/'checksums.json').write_text(json.dumps(actual,indent=2))
