"""Verify the ALREADY GENERATED Colab ZIP; no training or inference."""
import argparse,hashlib,json,zipfile
from pathlib import Path
EXPECTED_BYTES=12373875
EXPECTED_SHA='4ee135ed64654aa563c975fdca748eba102615ccda633b050166b9f93793c024'
def recover(archive,output):
    archive,output=Path(archive),Path(output)
    assert archive.stat().st_size==EXPECTED_BYTES, 'incomplete or different archive'
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==EXPECTED_SHA, 'original full ZIP SHA mismatch'
    assert not output.exists(), 'use a new output directory'
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        assert len(names)==len(set(names))
        assert all(Path(n).name==n and n not in ('','.','..') for n in names), 'flat files only'
        assert z.testzip() is None
        assert sum(i.file_size for i in z.infolist())<50000000
        checks=json.loads(z.read('checksums.json'))
        assert 'console.log' in names
        for n,h in checks.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
        output.mkdir()
        for n in names:(output/n).write_bytes(z.read(n))
    print(json.dumps({'full_archive_sha256':EXPECTED_SHA,'crc':'PASS','checksummed_entries':len(checks),'output':str(output)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--output',required=True);a=p.parse_args();recover(a.archive,a.output)
