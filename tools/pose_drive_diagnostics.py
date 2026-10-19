"""Opt-in bounded Drive probe. Whitelisted telemetry, never exception strings/URLs/headers."""
import json
from pathlib import Path
import tempfile
import time
from urllib.parse import urlsplit, parse_qs


def error_category(exc):
    # Only predefined labels leave this function. No class repr or message.
    import requests
    from google.auth.exceptions import RefreshError, TransportError
    kinds=[(requests.exceptions.SSLError,'tls_error'),(requests.exceptions.ConnectTimeout,'connect_timeout'),
        (requests.exceptions.ReadTimeout,'read_timeout'),(requests.exceptions.Timeout,'timeout'),
        (requests.exceptions.ConnectionError,'connection_error'),(RefreshError,'credential_refresh_error'),
        (TransportError,'auth_transport_error'),(requests.exceptions.RequestException,'request_error')]
    for _ in range(4):
        for cls,name in kinds:
            if isinstance(exc,cls):return name
        exc=getattr(exc,'__cause__',None) or getattr(exc,'__context__',None)
        if exc is None:break
    return 'other_exception'


class Audit:
    def __init__(self,path):self.path=Path(path);self.events=[];self.stage='initial_refresh'
    def add(self,**row):
        if len(self.events)>=512:return
        self.events.append({'event':len(self.events)+1,'stage':self.stage,**row})
        self.path.write_text(json.dumps(self.events,indent=2),encoding='utf-8')


def audited_session(credentials,audit):
    from google.auth.transport.requests import AuthorizedSession
    from requests.adapters import HTTPAdapter
    from rewind_archive.sources import session_opener
    class RawProxy:
        def __init__(self,raw):self.raw=raw
        def __getattr__(self,name):return getattr(self.raw,name)
        def read1(self,*args,**kwargs):
            try:return self.raw.read1(*args,**kwargs)
            except Exception as exc:
                audit.add(phase='body_read',outcome='exception',category=error_category(exc));raise
    class Session(AuthorizedSession):
        def request(self,method,url,**kwargs):
            start=time.perf_counter()
            is_media=parse_qs(urlsplit(url).query).get('alt')==['media']
            state={'phase':'request','endpoint':'media_range' if is_media else 'metadata',
                'credential_valid_before':bool(self.credentials.valid),'credential_expired_before':bool(self.credentials.expired),
                'timeout_seconds':kwargs.get('timeout') if isinstance(kwargs.get('timeout'),(int,float)) else None}
            try:
                response=super().request(method,url,**kwargs)
                code=response.status_code
                audit.add(**state,outcome='response',http_status=code if type(code)is int and 100<=code<=599 else None,
                    elapsed_seconds=time.perf_counter()-start,credential_valid_after=bool(self.credentials.valid))
                if callable(getattr(response.raw,'read1',None)):
                    response.raw=RawProxy(response.raw)
                return response
            except Exception as exc:
                audit.add(**state,outcome='exception',category=error_category(exc),elapsed_seconds=time.perf_counter()-start)
                raise
    session=Session(credentials,max_refresh_attempts=0)
    session.mount('https://',HTTPAdapter(max_retries=0))
    session_opener(session)
    audit.add(phase='session_policy',outcome='passed',max_refresh_attempts=0,https_retries=0)
    return session


def probe_once(root='/content'):
    """Fresh isolated attempt, at most 1 Pose file; never trains on probe output."""
    from google.colab import auth
    import google.auth
    from google.auth.transport.requests import Request
    from requests import Session
    from requests.adapters import HTTPAdapter
    from rewind_archive.contracts import Limits,PreparationError
    from rewind_archive.drive_metadata import MetadataLimits,build_drive_manifest,save_metadata_bundle
    from rewind_archive.pilot import run_metadata_pilot
    from tools.pose_pilot_colab import PART_SELECTION
    work=Path(tempfile.mkdtemp(prefix='rewind_drive_probe_',dir=root));audit=Audit(work/'safe_http_audit.json')
    session=None;status={'status':'running','training_dependency':False,'automatic_retries':0}
    try:
        auth.authenticate_user()
        credentials,_=google.auth.default(scopes=['https://www.googleapis.com/auth/drive.readonly'])
        with Session() as transport:
            transport.mount('https://',HTTPAdapter(max_retries=0))
            class BoundedRequest:
                def __call__(self,*args,**kwargs):
                    kwargs['timeout']=5
                    return Request(session=transport)(*args,**kwargs)
            credentials.refresh(BoundedRequest())
        audit.add(phase='initial_refresh',outcome='passed')
        session=audited_session(credentials,audit)
        audit.stage='metadata'
        result=build_drive_manifest(PART_SELECTION,session,MetadataLimits())
        save_metadata_bundle(result,work/'metadata')
        if result['manifest'] is None:raise PreparationError(result['report']['stop_reason'])
        audit.stage='extraction'
        limits=Limits(max_transfer_bytes=1024**2,max_decompressed_bytes=16*1024**2,max_disk_bytes=8*1024**2,
            max_files=1,max_members=1000,max_ranges=32,max_elapsed_seconds=45)
        report=run_metadata_pilot(result,work/'pilot',['*/json/frame_*.json'],limits,session)
        status.update(status='passed' if report['inspection_candidates_usable'] and report['complete_files']>=1 and report['verification']['status']=='passed' else 'failed',
            stop_reason=report['stop_reason'],complete_files=report['complete_files'],verification=report.get('verification'))
    except Exception as exc:
        status.update(status='failed',stage=audit.stage,reason=exc.reason if isinstance(exc,PreparationError) else error_category(exc))
        audit.add(phase='probe',outcome='exception',category=error_category(exc))
    finally:
        if session is not None:session.close()
        (work/'probe_status.json').write_text(json.dumps(status,indent=2))
    print('Drive probe:',status,'reports:',work)
    return work,status
