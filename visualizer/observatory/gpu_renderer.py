"""Headless NVIDIA renderer. All GL objects live in one dedicated worker process."""
import io
import ctypes
import json
import mmap
import os
import re
import signal
import struct
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
VERTEX = '''#version 330
in vec2 position;
out vec2 vUv;
void main() { vUv=position*.5+.5; gl_Position=vec4(position,0.,1.); }
'''
DEPTH_OFFSETS = [.5, .125, .875, .375, .625, .0625, .9375, .5]


def egl_device_index(cuda_device):
    """EGL enumeration is not CUDA/nvidia-smi order; match EGL_CUDA_DEVICE_NV."""
    lib=ctypes.CDLL('libEGL.so.1')
    lib.eglGetProcAddress.argtypes=[ctypes.c_char_p]
    lib.eglGetProcAddress.restype=ctypes.c_void_p
    query_address=lib.eglGetProcAddress(b'eglQueryDevicesEXT')
    attribute_address=lib.eglGetProcAddress(b'eglQueryDeviceAttribEXT')
    if not query_address or not attribute_address:
        raise ValueError('EGL cannot identify NVIDIA GPU devices; refusing an ambiguous GPU selection')
    query=ctypes.CFUNCTYPE(ctypes.c_uint,ctypes.c_int,ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_int))(query_address)
    attribute=ctypes.CFUNCTYPE(ctypes.c_uint,ctypes.c_void_p,ctypes.c_int,ctypes.POINTER(ctypes.c_ssize_t))(attribute_address)
    devices=(ctypes.c_void_p*32)();count=ctypes.c_int()
    if not query(32,devices,ctypes.byref(count)):
        raise ValueError('Could not enumerate EGL GPU devices')
    for index in range(count.value):
        ordinal=ctypes.c_ssize_t(-1)
        if attribute(devices[index],0x323A,ctypes.byref(ordinal)) and ordinal.value==cuda_device:
            return index
    raise ValueError(f'NVIDIA GPU {cuda_device} is not available through EGL')


def shader(name, filename):
    # Read the same source as the browser, so physics/display edits cannot drift.
    source = re.search(r'\bconst '+name+r'\s*=\s*`([\s\S]*?)`;', (ROOT/'frontend'/filename).read_text())
    if source is None:
        raise ValueError(f'Missing shared shader {name}')
    source = source[1].replace('varying vec2 vUv;', 'in vec2 vUv;').replace('texture2D(', 'texture(')
    if 'gl_FragColor' in source:
        source = 'out vec4 outColor;\n'+source.replace('gl_FragColor', 'outColor')
    if '#include <colorspace_fragment>' in source:
        # Same sRGB OETF as Three.js colorspace_pars_fragment (display only).
        source = source.replace('#include <colorspace_fragment>', '''
        outColor.rgb=mix(pow(outColor.rgb,vec3(.41666))*1.055-vec3(.055),
                        outColor.rgb*12.92,lessThanEqual(outColor.rgb,vec3(.0031308)));''')
    return '#version 330\n'+source


def halton(index, base):
    value, fraction = 0., 1.
    while index:
        fraction /= base
        value += fraction*(index % base)
        index //= base
    return value


class GPURenderer:
    def __init__(self, root, device=1, cache_bytes=6*1024**3, ttl=120):
        import moderngl
        self.gl = moderngl
        egl_index=egl_device_index(device)
        override=os.environ.get('GLCONTEXT_DEVICE_INDEX')
        if override is not None and override!=str(egl_index):
            raise ValueError('GLCONTEXT_DEVICE_INDEX conflicts with SKIRT_VIEWER_PREVIEW_GPU; refusing to use another GPU')
        self.ctx = moderngl.create_context(standalone=True, backend='egl', device_index=egl_index, require=330)
        self.info = dict(renderer=self.ctx.info['GL_RENDERER'], vendor=self.ctx.info['GL_VENDOR'],
                         version=self.ctx.info['GL_VERSION'], device=device, egl_device=egl_index)
        if 'NVIDIA' not in self.info['vendor'] or 'llvmpipe' in self.info['renderer'].lower():
            self.ctx.release()
            raise ValueError('Backend preview requires NVIDIA hardware; software rendering is not accepted')
        self.root, self.budget, self.ttl = Path(root), cache_bytes, ttl
        self.volumes, self.states = OrderedDict(), {}
        self.buffer = self.ctx.buffer(np.array([-1,-1, 3,-1, -1,3], dtype='f4').tobytes())
        self.programs, self.quads = {}, {}
        for name, filename in [('volumeFragmentShader','volume.js'), ('compositeFragmentShader','viewer.js'),
                               ('blurFragmentShader','viewer.js')]:
            program = self.ctx.program(vertex_shader=VERTEX, fragment_shader=shader(name, filename))
            self.programs[name] = program
            self.quads[name] = self.ctx.vertex_array(program, [(self.buffer, '2f', 'position')])
        self.parking_objects=[]
        self.parking=self.target(1,1,'f1',self.parking_objects)
        self.parking.use()

    def evict(self):
        now = time.monotonic()
        for key, state in list(self.states.items()):
            if now-state['last'] > self.ttl:
                self.release_state(key)
        for key, volume in list(self.volumes.items()):
            if now-volume['last'] > self.ttl:
                self.release_volume(key)

    def release_state(self, key):
        # ModernGL tracks the bound FBO. Never leave that pointer referring to a
        # deleted view framebuffer: subsequent clears restore the tracked FBO.
        self.parking.use()
        state = self.states.pop(key)
        for obj in state['objects']:
            obj.release()

    def release_session(self, session):
        for key in list(self.states):
            if key[0] == session:
                self.release_state(key)

    def release_volume(self, key):
        for texture in self.volumes.pop(key)['textures']:
            texture.release()
        for state_key, state in list(self.states.items()):
            if state['volume'] == key:
                self.release_state(state_key)

    def volume(self, key):
        if not re.fullmatch('[0-9a-f]{64}', key):
            raise ValueError('Invalid volume key')
        if key in self.volumes:
            self.volumes.move_to_end(key)
            self.volumes[key]['last'] = time.monotonic()
            return self.volumes[key]
        folder = self.root/key
        meta = json.loads((folder/'volume.json').read_text())
        n = int(meta['size'])
        if not 1 <= n <= min(512, self.ctx.info['GL_MAX_3D_TEXTURE_SIZE']):
            raise ValueError('Unsupported backend volume dimensions')
        size = 2*n**3*4*4
        if size > self.budget:
            raise ValueError('Volume exceeds the configured backend texture budget; choose a smaller grid')
        while self.volumes and sum(v['bytes'] for v in self.volumes.values())+size > self.budget:
            self.release_volume(next(iter(self.volumes)))
        textures = []
        try:
            for name in ('stars-dust.bin', 'young.bin'):
                path = folder/name
                if path.stat().st_size != size//2:
                    raise ValueError('Volume file length does not match metadata')
                texture = self.ctx.texture3d((n,n,n), 4, dtype='f4', alignment=1)
                textures.append(texture)
                # Avoid a single 2-GiB upload for 512³ RGBA32F. Bounded z slabs
                # also limit transient driver staging memory; mmap avoids copies.
                with path.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
                    with memoryview(data) as buffer:
                        for z in range(0,n,16):
                            depth=min(16,n-z)
                            with buffer[z*n*n*16:(z+depth)*n*n*16] as slab:
                                texture.write(slab,viewport=(0,0,z,n,n,depth),alignment=1)
                texture.filter = (self.gl.LINEAR, self.gl.LINEAR)
                texture.repeat_x = texture.repeat_y = texture.repeat_z = False
            self.ctx.finish()
            error=self.ctx.error
            if error != 'GL_NO_ERROR':
                raise ValueError(f'GPU could not upload the volume ({error}); choose a smaller grid')
        except Exception:
            for texture in textures:
                texture.release()
            raise
        volume = dict(meta=meta, textures=textures, bytes=size, last=time.monotonic())
        self.volumes[key] = volume
        return volume

    def target(self, width, height, dtype, objects):
        texture = self.ctx.texture((width,height), 4, dtype=dtype)
        texture.filter = (self.gl.LINEAR, self.gl.LINEAR)
        texture.repeat_x = texture.repeat_y = False
        framebuffer = self.ctx.framebuffer(color_attachments=[texture])
        framebuffer.clear()
        objects.extend([framebuffer, texture])
        return framebuffer

    def render_view(self, session, request, view, linear=False):
        volume = self.volume(request['volume_key'])
        signature = json.dumps([request['volume_key'], view, request['controls']], sort_keys=True)
        key = (session, view['name'])
        if key in self.states and self.states[key]['signature'] != signature:
            self.release_state(key)
        width, height = view['width'], view['height']
        if key not in self.states:
            objects = []
            try:
                history = [self.target(width,height,'f4',objects) for _ in range(2)]
                blur = [self.target(width,height,'f2',objects) for _ in range(2)]
                display = self.target(width,height,'f1',objects)
            except Exception:
                for obj in objects:
                    obj.release()
                raise
            self.states[key] = dict(signature=signature, volume=request['volume_key'], samples=0,
                                    history=history, blur=blur, display=display, objects=objects)
        state = self.states[key]
        state['last'] = time.monotonic()
        controls = request['controls']
        # Each FBO owns its viewport. Setting ctx.viewport here would mutate the
        # PREVIOUS view's bound FBO and corrupt alternating main/mini dimensions.
        self.ctx.disable(self.gl.BLEND | self.gl.DEPTH_TEST | self.gl.CULL_FACE)
        index = state['samples']
        if index < 8:
            program = self.programs['volumeFragmentShader']
            values = dict(starsDust=0,youngSources=1,history=2,boxMin=volume['meta']['bounds_kpc'][0],
                          boxMax=volume['meta']['bounds_kpc'][1],extinction=volume['meta']['extinction_per_kpc'],
                          resolution=(width,height),sampleIndex=index,depthJitter=DEPTH_OFFSETS[index],
                          pixelJitter=(halton(index,2)-.5,halton(index,3)-.5) if index else (0,0),
                          steps=controls['steps'],dustScale=controls['dust_scale'] if controls['dust'] else 0.,
                          showStars=float(controls['stars']),showYoung=float(controls['young']))
            for name, value in values.items():
                program[name].value = value
            for name, field in [('inverseProjection','inverse_projection'),('cameraWorld','camera_world')]:
                program[name].write(np.asarray(view[field],dtype='f4').tobytes())  # Three.js column-major
            for slot, texture in enumerate(volume['textures']):
                texture.use(slot)
            state['history'][(index+1)%2].color_attachments[0].use(2)
            state['history'][index%2].use()
            self.quads['volumeFragmentShader'].render(vertices=3)
            state['samples'] += 1
        radiance = state['history'][(state['samples']-1)%2]
        if linear:
            return np.frombuffer(radiance.read(components=3,dtype='f4'),dtype='f4').reshape(height,width,3)
        blurred = radiance
        if controls['psf'] > 0 or controls['structure'] > 0:
            program = self.programs['blurFragmentShader']
            program['radianceTexture'].value = 0
            program['resolution'].value = (width,height)
            program['sigma'].value = controls['psf'] or 1.
            for framebuffer, direction in zip(state['blur'], [(1,0),(0,1)]):
                program['direction'].value = direction
                blurred.color_attachments[0].use(0)
                framebuffer.use()
                self.quads['blurFragmentShader'].render(vertices=3)
                blurred = framebuffer
        program = self.programs['compositeFragmentShader']
        for name, value in dict(radianceTexture=0,blurTexture=1,exposure=controls['exposure'],
                                psfMix=float(controls['psf']>0),structureStrength=controls['structure'],
                                stretchMode=controls['stretch']).items():
            program[name].value = value
        radiance.color_attachments[0].use(0)
        blurred.color_attachments[0].use(1)
        state['display'].use()
        self.quads['compositeFragmentShader'].render(vertices=3)
        pixels = state['display'].read(components=3, alignment=1)
        image = Image.frombytes('RGB',(width,height),pixels).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        buffer = io.BytesIO()
        image.save(buffer, format='PNG', compress_level=1)
        return buffer.getvalue(), state['samples']

    def render(self, session, request):
        start = time.perf_counter()
        self.volume(request['volume_key'])
        load_ms = (time.perf_counter()-start)*1000
        images, views = [], []
        start = time.perf_counter()
        for view in request['views']:
            png, samples = self.render_view(session, request, view)
            images.append(png)
            views.append(dict(name=view['name'],bytes=len(png),samples=samples,width=view['width'],height=view['height']))
        meta = dict(sequence=request['sequence'],views=views,gpu=self.info,load_ms=load_ms,
                    render_encode_ms=(time.perf_counter()-start)*1000,
                    resident_bytes=sum(v['bytes'] for v in self.volumes.values()))
        header = json.dumps(meta).encode()
        return struct.pack('<I',len(header))+header+b''.join(images)

    def close(self):
        for key in list(self.states):
            self.release_state(key)
        for key in list(self.volumes):
            self.release_volume(key)
        for obj in [*self.quads.values(),*self.programs.values(),self.buffer]:
            obj.release()
        for obj in self.parking_objects:
            obj.release()
        self.ctx.release()


def worker_main(connection, root, device, budget, ttl):
    # Let the API finish in-flight work on Ctrl-C; its shutdown owns this worker.
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    renderer = None
    try:
        renderer = GPURenderer(root, device, budget, ttl)
        connection.send(('ok', renderer.info))
        while True:
            if not connection.poll(1):
                renderer.evict()
                continue
            command, session, payload = connection.recv()
            if command == 'close':
                break
            try:
                if command == 'release':
                    renderer.release_session(session)
                    result = None
                else:
                    renderer.evict()
                    result = renderer.render(session, payload)
                connection.send(('ok', result))
            except Exception as error:
                renderer.release_session(session)
                connection.send(('error', str(error)))
    except (EOFError, BrokenPipeError):
        pass
    except Exception as error:
        try:
            connection.send(('error', str(error)))
        except (OSError, EOFError):
            pass
    finally:
        if renderer is not None:
            renderer.close()
        connection.close()
