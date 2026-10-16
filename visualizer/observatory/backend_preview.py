"""Bounded, capability-authenticated sessions for the backend GPU renderer."""
import importlib.util
import multiprocessing
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Annotated, Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .gpu_renderer import worker_main


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


Matrix = Annotated[list[Annotated[float, Field(ge=-1e12,le=1e12)]], Field(min_length=16,max_length=16)]


class BackendView(StrictModel):
    name: Literal['main', 'mini']
    width: int = Field(ge=1,le=2048)
    height: int = Field(ge=1,le=2048)
    inverse_projection: Matrix
    camera_world: Matrix


class BackendControls(StrictModel):
    steps: Literal[96,192,384] = 192
    dust: bool = True
    dust_scale: float = Field(default=1,ge=0,le=4)
    stars: bool = True
    young: bool = True
    exposure: float = Field(default=.02,ge=0,le=10)
    psf: float = Field(default=0,ge=0,le=8)
    structure: float = Field(default=0,ge=0,le=4)
    stretch: Literal[0,1,2] = 0


class BackendFrame(StrictModel):
    session_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    volume_key: str = Field(pattern=r'^[0-9a-f]{64}$')
    sequence: int = Field(ge=0,le=2**53-1)
    views: list[BackendView] = Field(min_length=1,max_length=2)
    controls: BackendControls = Field(default_factory=BackendControls)

    @model_validator(mode='after')
    def bounded_views(self):
        if len({view.name for view in self.views}) != len(self.views):
            raise ValueError('Duplicate view names')
        if sum(view.width*view.height for view in self.views) > 4_194_304:
            raise ValueError('Backend frame exceeds the combined four-megapixel limit')
        return self


class BackendPreview:
    def __init__(self, storage, volume_root):
        self.root = Path(volume_root)
        self.device = int(os.environ.get('SKIRT_VIEWER_PREVIEW_GPU','1'))
        self.budget = int(os.environ.get('SKIRT_VIEWER_PREVIEW_CACHE_MIB','6144'))*1024**2
        self.ttl = 120
        self.lock, self.session_lock = threading.Lock(), threading.Lock()
        self.sessions = {}
        self.process = self.connection = None
        self.info = None
        self.key_path = Path(storage)/'backend-preview.key'
        self.key_path.parent.mkdir(parents=True,exist_ok=True)
        try:
            fd = os.open(self.key_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd,'w') as stream:
                stream.write(secrets.token_urlsafe(32)+'\n')
        self.access_key = self.key_path.read_text().strip()
        if len(self.access_key) < 32:
            raise ValueError('Backend preview access key is too short')

    @staticmethod
    def local(request):
        return (request.client is not None and request.client.host in ('127.0.0.1','::1')
                and request.url.hostname in ('127.0.0.1','::1','localhost'))

    @staticmethod
    def bearer(request):
        value = request.headers.get('authorization','')
        return value[7:] if value.startswith('Bearer ') else ''

    def status(self, request):
        return dict(available=importlib.util.find_spec('moderngl') is not None,
                    running=self.process is not None and self.process.is_alive(),gpu=self.info,
                    device=self.device,texture_budget_mib=self.budget//1024**2,
                    auth_required=not self.local(request),session_ttl_seconds=self.ttl)

    def start(self):
        if self.process is not None and self.process.is_alive():
            return
        self.stop_worker()
        context = multiprocessing.get_context('spawn')
        parent, child = context.Pipe()
        self.connection = parent
        self.process = context.Process(target=worker_main,args=(child,str(self.root),self.device,self.budget,self.ttl),
                                       name='observatory-preview-gpu',daemon=True)
        self.process.start()
        child.close()
        if not parent.poll(30):
            self.stop_worker()
            raise HTTPException(503,'GPU worker did not start within 30 seconds')
        status, value = parent.recv()
        if status != 'ok':
            self.stop_worker()
            raise HTTPException(503,f'GPU worker unavailable: {value}')
        self.info = value

    def call(self, command=None, session=None, payload=None):
        # At most one pending GPU call: slow clients cannot build an unbounded queue.
        if not self.lock.acquire(blocking=False):
            raise HTTPException(429,'GPU worker busy; retry the latest frame',headers={'Retry-After':'1'})
        try:
            self.start()
            if command is None:
                return self.info
            self.connection.send((command,session,payload))
            if not self.connection.poll(60):
                self.stop_worker()
                raise HTTPException(504,'GPU frame timed out; worker released. Retry with a smaller image or grid.')
            status, value = self.connection.recv()
            if status != 'ok':
                raise HTTPException(422,value)
            return value
        except (OSError,EOFError) as error:
            self.stop_worker()
            raise HTTPException(503,'GPU worker stopped; retry the preview') from error
        finally:
            self.lock.release()

    def create_session(self, request):
        if not self.local(request) and not secrets.compare_digest(self.bearer(request),self.access_key):
            raise HTTPException(401,'Enter the backend GPU access key from var/backend-preview.key',
                                headers={'WWW-Authenticate':'Bearer'})
        with self.session_lock:
            now = time.monotonic()
            self.sessions = {key:value for key,value in self.sessions.items() if now-value['last'] < self.ttl}
            if len(self.sessions) >= 4:
                raise HTTPException(429,'Four backend sessions are active; close a preview or wait two minutes')
            gpu = self.call()
            identifier, token = secrets.token_hex(16), secrets.token_urlsafe(32)
            self.sessions[identifier] = dict(token=token,last=now,sequence=-1)
            return dict(session_id=identifier,token=token,gpu=gpu)

    def authorize(self, request, identifier):
        session = self.sessions.get(identifier)
        if session is None or time.monotonic()-session['last'] >= self.ttl:
            self.sessions.pop(identifier,None)
            raise HTTPException(401,'Backend session expired; click Build / update to reconnect')
        if not secrets.compare_digest(self.bearer(request),session['token']):
            raise HTTPException(401,'Invalid backend session token')
        session['last'] = time.monotonic()
        return session

    def frame(self, request, payload):
        with self.session_lock:
            session = self.authorize(request,payload.session_id)
            if payload.sequence <= session['sequence']:
                raise HTTPException(409,'Obsolete frame request')
            session['sequence'] = payload.sequence
        return self.call('render',payload.session_id,payload.model_dump())

    def release(self, request, identifier):
        with self.session_lock:
            self.authorize(request,identifier)
            self.sessions.pop(identifier)
        try:
            if self.process is not None and self.process.is_alive():
                self.call('release',identifier)
        except HTTPException:
            pass  # Idle expiry also frees GL state if the worker is busy.

    def stop_worker(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=2)
            self.process.close()
        if self.connection is not None:
            self.connection.close()
        self.process = self.connection = None
        self.info = None

    def close(self):
        with self.lock:
            self.stop_worker()
