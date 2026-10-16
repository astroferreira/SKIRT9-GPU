"""Single GPU worker with persistent job records, bounded queue and cancellation."""

import fcntl
import hashlib
import json
import os
import queue
import re
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path

from .simulation import QUALITY, RenderRequest, prepare, render_fits, render_rgb_fits
from .spectra import process_sed
from .catalog import RawDataset
from .raw import materialize
from .kinematics import availability as velocity_availability, input_identity as kinematics_identity, materialize as materialize_kinematics


ACTIVE = {"queued", "preparing", "running", "processing"}
ARTIFACTS = {"preview.png", "observation_view_total.fits", "rgb.png", "observation_rgb_total.fits",
             "observation.ski", "manifest.json", "console.log", "observation_log.txt",
             "observation_spectrum_sed.dat", "spectrum.png", "spectrum.json", "spectrum.csv", "spectrum.fits"}


def execution_limit(base_timeout, quality):
    return base_timeout * QUALITY[quality].get("timeout_multiplier", 1)


def requested_products(request):
    if isinstance(request, RenderRequest):
        return request.selected_instruments()
    return tuple(request.get("instruments") or [request.get("output", "image")])


def primary_product(request):
    selected = requested_products(request)
    preferred = request.primary_product() if isinstance(request, RenderRequest) else request.get("output", "image")
    if preferred in selected:
        return preferred
    return next((product for product in ("rgb", "image", "sed") if product in selected), selected[0])


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def tail(path, size=18000):
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - size))
        return stream.read().decode("utf-8", errors="replace")


def execution_log(folder):
    scientific = folder / "observation_log.txt"
    return scientific if scientific.is_file() else folder / "console.log"


def execution_evidence(folder):
    path = execution_log(folder)
    text = re.sub(r"\x1b\[[0-9;]*m", "", path.read_text(errors="replace")) if path.is_file() else ""
    warnings = list(dict.fromkeys(line.strip() for line in text.splitlines() if " ! " in line))
    gpu_lines = [line.strip() for line in text.splitlines() if "GPU acceleration" in line or "CUDA devices" in line]
    confirmed = any("enabled on" in line for line in gpu_lines)
    if not confirmed:
        warnings.append("GPU activation was not confirmed in the execution log; inspect the backend log.")
    return {"warnings": warnings, "gpu_log": gpu_lines, "gpu_confirmed": confirmed, "execution_log": path.name}


def progress(text):
    phase, percent = "Preparing model", None
    for line in text.splitlines():
        if "Starting setup" in line:
            phase, percent = "Building galaxy model", None
        elif "Starting primary emission" in line:
            phase, percent = "Primary emission", 0
        elif "Starting secondary emission" in line:
            phase, percent = "Dust / secondary emission", None
        elif "Starting final output" in line or ("wrote" in line and ".fits" in line):
            phase, percent = "Writing observation", None
        if "launched" in line.lower() or "photon packages" in line.lower():
            match = re.search(r"([\d.]+)%", line)
            if match:
                percent = min(100., float(match.group(1)))
    return phase, percent


class Jobs:
    def __init__(self, catalog, root, executable, threads=16, timeout=7200, dust_batch_cells=None):
        # None lets SKIRTGPU size stochastic-dust batches from the number of emitting cells
        if dust_batch_cells is not None and (not isinstance(dust_batch_cells, int) or isinstance(dust_batch_cells, bool)
                                             or not 1 <= dust_batch_cells <= 1048576):
            raise ValueError("Dust GPU batch size must be an integer between 1 and 1048576")
        self.catalog, self.root, self.executable = catalog, root, executable
        self.threads, self.timeout = threads, timeout
        self.dust_batch_cells = dust_batch_cells
        root.mkdir(parents=True, exist_ok=True)
        self.process_lock = (root / ".worker.lock").open("w")
        try:
            fcntl.flock(self.process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.process_lock.close()
            raise RuntimeError("A render worker already owns this directory; run one server process")
        self.lock = threading.RLock()
        self.items = {}
        self.process = None
        self.process_id = None
        self.pending = queue.Queue()
        self.stopping = threading.Event()
        for path in root.glob("*/job.json"):
            try:
                item = json.loads(path.read_text())
                if item["status"] in ACTIVE:
                    item.update(status="interrupted", error="Server stopped before this render completed", finished_at=time.time())
                    write_json(path, item)
                elif item["status"] == "completed" and item.get("log_metadata_version") != 2:
                    # Reuse completed FITS/images after improvements to log parsing.
                    evidence = execution_evidence(path.parent)
                    manifest_path = path.parent / "manifest.json"
                    if manifest_path.is_file():
                        manifest = json.loads(manifest_path.read_text())
                        if manifest.get("image", {}).get("blank"):
                            evidence["warnings"].append("No signal landed in this FOV/filter at this packet count; try a wider FOV or more packets.")
                        manifest.update(evidence)
                        write_json(manifest_path, manifest)
                    item.update(gpu_confirmed=evidence["gpu_confirmed"], warnings=evidence["warnings"], log_metadata_version=2)
                    write_json(path, item)
                self.items[item["id"]] = item
            except (ValueError, KeyError):
                continue
        self.worker = threading.Thread(target=self.run, daemon=True, name="skirtgpu-render-worker")
        self.worker.start()

    def update(self, identifier, **changes):
        with self.lock:
            self.items[identifier].update(changes)
            write_json(self.root / identifier / "job.json", self.items[identifier])

    def submit(self, request):
        dataset = self.catalog.datasets[request.dataset_id]
        if not dataset.info()["available"]:
            raise ValueError("Dataset inputs are unavailable")
        if dataset.entry.get("render_available") is False:
            raise ValueError(dataset.entry.get("render_unavailable_reason", "This dataset is available for viewing only"))
        if isinstance(dataset,RawDataset) and {'image', 'rgb'} & set(request.selected_instruments()) and (request.model_fov_kpc or dataset.entry['default_model_fov_kpc']) < request.camera.fov_kpc:
            raise ValueError('Physical model region must be at least as large as the camera FOV')
        if not self.executable.is_file() or not os.access(self.executable, os.X_OK):
            raise ValueError("SKIRTGPU executable is unavailable")
        digest, _ = dataset.fingerprint()
        if request.source_velocities:
            available = velocity_availability(dataset)
            if not available['available']:
                raise ValueError(available['reason'])
            digest = hashlib.sha256(json.dumps({'inputs': digest, 'kinematics': kinematics_identity(dataset)},
                                               sort_keys=True).encode()).hexdigest()
        binary_stat = self.executable.stat()
        signature = hashlib.sha256(json.dumps({"request": request.render_identity(), "inputs": digest,
                                               "binary": [str(self.executable), binary_stat.st_size, binary_stat.st_mtime_ns],
                                               "pipeline_version": 2}, sort_keys=True).encode()).hexdigest()
        with self.lock:
            if self.stopping.is_set():
                raise ValueError("Render service is shutting down")
            for item in self.items.values():
                if item["signature"] == signature and (item["status"] in ACTIVE or (
                    item["status"] == "completed" and self.artifacts_complete(item))):
                    return self.get(item["id"]), True
            if sum(item["status"] in ACTIVE for item in self.items.values()) >= 5:
                raise ValueError("The render queue is full (5 jobs); wait or cancel an observation")
            identifier = uuid.uuid4().hex
            folder = self.root / identifier
            folder.mkdir()
            self.items[identifier] = {"id": identifier, "signature": signature, "status": "queued", "created_at": time.time(),
                                      "request": request.model_dump(mode="json"), "dataset_name": dataset.entry["name"],
                                      "cancel_requested": False, "phase": "Waiting for GPU worker", "phase_percent": None}
            self.update(identifier)
            self.pending.put(identifier)
            return self.get(identifier), False

    def artifacts_complete(self, item):
        names = []
        selected = requested_products(item["request"])
        if "image" in selected:
            names.extend(("preview.png", "observation_view_total.fits"))
        if "rgb" in selected:
            names.extend(("rgb.png", "observation_rgb_total.fits"))
        if "sed" in selected:
            names.extend(("spectrum.png", "spectrum.json", "spectrum.csv", "spectrum.fits", "observation_spectrum_sed.dat"))
        return all((self.root / item["id"] / name).is_file() for name in names)

    def get(self, identifier):
        with self.lock:
            item = dict(self.items[identifier])
        folder = self.root / identifier
        if item["status"] == "running":
            item["phase"], item["phase_percent"] = progress(tail(execution_log(folder)))
        if item["status"] == "completed":
            base = f"/api/jobs/{identifier}/files"
            products = {}
            if "image" in requested_products(item["request"]):
                products["image"] = {"preview_url": f"{base}/preview.png", "image_url": f"{base}/preview.png",
                                     "fits_url": f"{base}/observation_view_total.fits", "metadata": item.get("image")}
            if "rgb" in requested_products(item["request"]):
                products["rgb"] = {"preview_url": f"{base}/rgb.png", "image_url": f"{base}/rgb.png",
                                   "fits_url": f"{base}/observation_rgb_total.fits", "metadata": item.get("rgb")}
            if "sed" in requested_products(item["request"]):
                products["sed"] = {"preview_url": f"{base}/spectrum.png", "spectrum_url": f"/api/jobs/{identifier}/spectrum",
                                   "fits_url": f"{base}/spectrum.fits", "csv_url": f"{base}/spectrum.csv",
                                   "raw_url": f"{base}/observation_spectrum_sed.dat", "metadata": item.get("spectrum")}
            primary = primary_product(item["request"])
            item.update(products=products, primary_product=primary, preview_url=products[primary]["preview_url"],
                        fits_url=products[primary]["fits_url"])
            if primary in ("image", "rgb"):
                item["image_url"] = products[primary]["image_url"]
            else:
                item.update(spectrum_url=products["sed"]["spectrum_url"], csv_url=products["sed"]["csv_url"],
                            raw_url=products["sed"]["raw_url"])
        item["elapsed_seconds"] = round(max(0, item.get("finished_at", time.time()) - item.get("started_at", item["created_at"])), 1)
        return item

    def list(self):
        with self.lock:
            identifiers = sorted(self.items, key=lambda i: self.items[i]["created_at"], reverse=True)[:60]
        return [self.get(i) for i in identifiers]

    def cancel(self, identifier):
        with self.lock:
            item = self.items[identifier]
            if item["status"] not in ACTIVE:
                return self.get(identifier)
            self.update(identifier, cancel_requested=True)
            if item["status"] == "queued":
                self.update(identifier, status="cancelled", finished_at=time.time(), phase="Cancelled")
            if self.process_id == identifier and self.process is not None and self.process.poll() is None:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        return self.get(identifier)

    def run(self):
        while not self.stopping.is_set():
            try:
                identifier = self.pending.get(timeout=.25)
            except queue.Empty:
                continue
            try:
                self.execute(identifier)
            except Exception as error:
                with self.lock:
                    cancelled = self.items[identifier]["cancel_requested"]
                self.update(identifier, status="cancelled" if cancelled else "failed", error=str(error),
                            finished_at=time.time(), phase="Cancelled" if cancelled else "Render failed")
            finally:
                # An exception after Popen (e.g. an output write failure) must not
                # orphan a live GPU process and allow the next job to overlap it.
                proc = self.process
                if proc is not None and proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                    except ProcessLookupError:
                        proc.wait()
                with self.lock:
                    self.process = self.process_id = None
                self.pending.task_done()

    def execute(self, identifier):
        with self.lock:
            if self.items[identifier]["cancel_requested"]:
                return
            self.update(identifier, status="preparing", started_at=time.time(), phase="Preparing observation")
            request = RenderRequest.model_validate(self.items[identifier]["request"])
        dataset = self.catalog.datasets[request.dataset_id]
        folder = self.root / identifier
        raw_export=None;run_request=request
        if isinstance(dataset,RawDataset):
            self.update(identifier,phase='Exporting selected physical model region')
            dataset,run_request,raw_export=materialize(dataset,request,self.catalog.cache)
        kinematics = None
        if request.source_velocities:
            self.update(identifier, phase='Importing resolved source velocities')
            metadata = (Path(raw_export['cache_directory'])/f'{dataset.id}_physics.json') if raw_export else None
            dataset, kinematics = materialize_kinematics(dataset, self.catalog.cache, metadata)
        manifest = prepare(dataset, run_request, folder, self.catalog.cache)
        manifest['request']=request.model_dump(mode='json')
        if raw_export:manifest['on_demand_export']=raw_export
        if kinematics:
            manifest['kinematics'] = kinematics
            manifest['limitations'].append('Source bulk velocities only; diffuse dust is stationary and no unresolved dispersion is added.')
        command = [str(self.executable), "-g", "-t", str(self.threads), "-o", str(folder), str(folder / "observation.ski")]
        # GPU source launch uses SKIRTGPU's windowed source-spectrum cache, so million-particle BPASS
        # catalogues stay within VRAM; keep all configured host workers feeding the resident transport.
        overrides = {"SKIRTGPU_PRECOMPUTE_SOURCE_SPECTRA": "0",
                     "SKIRTGPU_BATCH_FIRST_FORCED_CHUNK": "2097152",
                     "SKIRTGPU_PRIMARY_LIFECYCLE_THREADS": str(self.threads)}
        if self.dust_batch_cells is not None:
            overrides["SKIRTGPU_STOCHASTIC_DUST_BATCH_CELLS"] = str(self.dust_batch_cells)
        env = {k: v for k, v in os.environ.items() if not k.startswith("SKIRTGPU")}
        env.update(overrides)
        binary_stat = self.executable.stat()
        limit = execution_limit(self.timeout, request.quality)
        manifest.update(command=command, environment=overrides, executable=str(self.executable),
                        executable_identity={"bytes": binary_stat.st_size, "mtime_ns": binary_stat.st_mtime_ns},
                        job_id=identifier, created_at=self.items[identifier]["created_at"], execution_limit_seconds=limit)
        write_json(folder / "manifest.json", manifest)
        with (folder / "console.log").open("w") as log:
            with self.lock:
                if self.items[identifier]["cancel_requested"] or self.stopping.is_set():
                    raise RuntimeError("Cancelled before launch")
                self.process = subprocess.Popen(command, cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                stdin=subprocess.DEVNULL, start_new_session=True)
                self.process_id = identifier
                proc = self.process
                self.update(identifier, status="running", phase="Building galaxy model")
            deadline = time.monotonic() + limit
            terminate_at = None
            timed_out = False
            while proc.poll() is None:
                cancelled = self.items[identifier]["cancel_requested"] or self.stopping.is_set()
                timed_out = time.monotonic() >= deadline
                if cancelled or timed_out:
                    try:
                        if terminate_at is None:
                            os.killpg(proc.pid, signal.SIGTERM)
                            terminate_at = time.monotonic()
                        elif time.monotonic() - terminate_at > 3:
                            os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                time.sleep(.2)
            code = proc.wait()
        if self.items[identifier]["cancel_requested"] or self.stopping.is_set():
            raise RuntimeError("Observation cancelled")
        if timed_out:
            raise RuntimeError(f"Render exceeded the {limit:g}s execution limit")
        if code:
            raise RuntimeError(f"SKIRTGPU exited with code {code}. " + tail(folder / "console.log", 1600))
        selected = request.selected_instruments()
        product = {}
        blank_products = []
        if "image" in selected:
            self.update(identifier, status="processing", phase="Developing broadband image", phase_percent=None)
            image_info = render_fits(folder / "observation_view_total.fits", folder / "preview.png", request.pixels)
            product["image"] = image_info
            if image_info["blank"]:
                blank_products.append("broadband image")
        if "rgb" in selected:
            self.update(identifier, status="processing", phase="Composing RGB image", phase_percent=None)
            rgb_info = render_rgb_fits(folder / "observation_rgb_total.fits", folder / "rgb.png", request.pixels, request.rgb_bands)
            product["rgb"] = rgb_info
            if rgb_info["blank"]:
                blank_products.append("RGB image")
        if "sed" in selected:
            self.update(identifier, status="processing", phase="Preparing spectral plot and tables", phase_percent=None)
            spectrum_info = process_sed(folder / "observation_spectrum_sed.dat", folder, request.spectral_mode,
                                        dataset.entry["name"], manifest["spectral_observation"]["frame"],
                                        resolving_power=request.spectral_resolution)
            product["spectrum"] = spectrum_info
            if spectrum_info["blank"]:
                blank_products.append("integrated SED")
        evidence = execution_evidence(folder)
        warnings = evidence["warnings"]
        if blank_products:
            warnings.append(f"No signal was recorded for the {', '.join(blank_products)} at this photon budget.")
        if "sed" in selected:
            warnings.extend(manifest["limitations"][-2:])
        manifest.update(**product, **evidence, completed_at=time.time(), returncode=code)
        write_json(folder / "manifest.json", manifest)
        with self.lock:
            if self.items[identifier]["cancel_requested"]:
                raise RuntimeError("Observation cancelled")
            self.update(identifier, status="completed", phase="Observation ready", phase_percent=100,
                        finished_at=time.time(), **product, warnings=warnings,
                        gpu_confirmed=evidence["gpu_confirmed"], log_metadata_version=2)

    def close(self):
        self.stopping.set()
        with self.lock:
            for identifier in list(self.items):
                if self.items[identifier]["status"] in ACTIVE:
                    self.cancel(identifier)
        self.worker.join(timeout=8)
        # Hold the inter-process lock until the worker has exited. Process teardown
        # releases it even when a long input conversion is still completing.
        if not self.worker.is_alive():
            self.process_lock.close()
