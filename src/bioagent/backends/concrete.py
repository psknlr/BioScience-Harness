"""Concrete execution backends.

Each backend reports honestly when it cannot run: `ContainerBackend.available()`
is False on this machine (no docker/podman/nerdctl), so container components
resolve to UNAVAILABLE with a reason rather than silently appearing usable.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from ..runtime.component import ComponentManifest
from ..runtime.registry import Loader
from ..status import ExecutionStatus
from .base import Backend
from .http import HTTPBackend, HTTPRequest  # noqa: F401 - re-exported


class PythonBackend(Backend):
    """Calls an in-process python entrypoint resolved by the Loader."""

    backend = "python"

    def __init__(self, loader: Loader) -> None:
        self.loader = loader

    def rebind(self, loader: Loader) -> "PythonBackend":
        return PythonBackend(loader)

    def invoke(self, manifest: ComponentManifest, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        fn = self.loader.load(manifest.id)
        if fn is None:
            return self._result(manifest, ExecutionStatus.UNAVAILABLE, t0,
                                error=manifest.blocking_reason or "entrypoint could not be loaded")
        try:
            value = fn(**kwargs)
            return self._result(manifest, ExecutionStatus.SUCCEEDED, t0, value=value)
        except TypeError as exc:
            # wrong signature is a contract failure, not an environment problem
            return self._result(manifest, ExecutionStatus.FAILED, t0,
                                error=f"signature mismatch: {exc}")
        except Exception as exc:  # noqa: BLE001
            return self._result(manifest, ExecutionStatus.FAILED, t0,
                                error=f"{type(exc).__name__}: {exc}")


class MCPBackend(Backend):
    """Routes to a platform-native MCP connector via an injected dispatcher.

    With no dispatcher bound the result is RESOLVED, never SUCCEEDED — this is the
    exact v1 false-success path, now unrepresentable.
    """

    backend = "mcp"

    def __init__(self, dispatcher: Callable[..., Any] | None = None) -> None:
        self._dispatcher = dispatcher

    def available(self) -> bool:
        return self._dispatcher is not None

    def unavailable_reason(self) -> str:
        return "" if self._dispatcher else "no MCP dispatcher bound to this runtime"

    def invoke(self, manifest: ComponentManifest, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        server = manifest.runtime.server or (manifest.native_connectors[0]
                                             if manifest.native_connectors else "")
        if not server:
            return self._result(manifest, ExecutionStatus.UNAVAILABLE, t0,
                                error="component declares no MCP server")
        if self._dispatcher is None:
            return self._result(
                manifest, ExecutionStatus.RESOLVED, t0,
                value={"resolved_connector": server, "tool": manifest.runtime.entrypoint,
                       "arguments": kwargs, "dispatched": False,
                       "note": "routing resolved; no dispatcher bound so nothing executed"},
                metadata={"connector": server})
        try:
            value = self._dispatcher(server, manifest.runtime.entrypoint or manifest.name, **kwargs)
            return self._result(manifest, ExecutionStatus.SUCCEEDED, t0, value=value,
                                metadata={"connector": server})
        except Exception as exc:  # noqa: BLE001
            return self._result(manifest, ExecutionStatus.FAILED, t0,
                                error=f"{type(exc).__name__}: {exc}",
                                metadata={"connector": server})


class DatasetBackend(Backend):
    """Streams a bounded slice of a local dataset."""

    backend = "dataset"

    def __init__(self, lake_dir: Path | str) -> None:
        self.lake_dir = Path(lake_dir)

    def available(self) -> bool:
        return self.lake_dir.is_dir()

    def unavailable_reason(self) -> str:
        return "" if self.available() else f"data lake not present at {self.lake_dir}"

    def has_dataset(self, name: str) -> bool:
        """Whether this backend can actually read `name` right now.

        The resolver used to answer this with a default `lambda _cid: False`, so
        a dataset sitting in the lake resolved as missing and the component was
        permanently UNAVAILABLE — and auto-fetch dead-ended, because the
        re-resolution after a successful download consulted the same always-False
        probe. Exposing the check here lets the runtime wire the two together.
        """
        n = str(name).strip()
        if not n or not self.lake_dir.is_dir():
            return False
        candidate = (self.lake_dir / n)
        try:
            # confine the lookup to the lake: a dataset name is never a path out
            candidate = candidate.resolve()
            if self.lake_dir.resolve() not in candidate.parents:
                return False
        except (OSError, RuntimeError):
            return False
        return candidate.exists()

    def invoke(self, manifest: ComponentManifest, *, nrows: int | None = 5,
               columns: list[str] | None = None, **_: Any) -> Any:
        from ..adapters.datalake import DataLakeAdapter

        t0 = time.perf_counter()
        if not self.available():
            return self._result(manifest, ExecutionStatus.UNAVAILABLE, t0,
                                error=self.unavailable_reason())
        adapter = DataLakeAdapter(self.lake_dir)

        class _Cap:
            kind = "dataset"
            name = manifest.name

        res = adapter.invoke(_Cap(), nrows=nrows, columns=columns)
        res.capability = manifest.id
        res.adapter = f"{self.backend}-backend"
        return res


class SubprocessBackend(Backend):
    """Runs an upstream project in its own interpreter — never imports its code."""

    backend = "subprocess"

    def __init__(self, project_roots: dict[str, Path] | None = None,
                 timeout_s: float = 120.0) -> None:
        self.project_roots = {k: Path(v) for k, v in (project_roots or {}).items()}
        self.timeout_s = timeout_s

    @staticmethod
    def _code_for(manifest: ComponentManifest, arguments: dict) -> str:
        return _SubprocessCodeBuilder.build(manifest, arguments)

    def invoke(self, manifest: ComponentManifest, *, code: str | None = None,
               **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        root = self.project_roots.get(manifest.provider.project)
        if root is None or not root.is_dir():
            return self._result(
                manifest, ExecutionStatus.UNAVAILABLE, t0,
                error=(f"upstream project {manifest.provider.project!r} is not installed; "
                       "this backend invokes upstream code in place and never copies it"))
        if not code:
            # A planner never supplies `code=`, so requiring it made every
            # subprocess capability permanently unexecutable through the normal
            # path. When the manifest names a "module:function" entrypoint the
            # call can be constructed from it, with the step's own arguments.
            code = self._code_for(manifest, kwargs)
            if not code:
                return self._result(
                    manifest, ExecutionStatus.UNAVAILABLE, t0,
                    error=("no invocation code supplied and the manifest declares no "
                           "'module:function' entrypoint to build one from"))
            kwargs = {}
        try:
            proc = subprocess.run(  # noqa: S603
                [sys.executable, "-I", "-c", code], cwd=str(root),
                capture_output=True, text=True, timeout=self.timeout_s, check=False)
        except subprocess.TimeoutExpired:
            return self._result(manifest, ExecutionStatus.TIMEOUT, t0,
                                error=f"timed out after {self.timeout_s}s")
        if proc.returncode != 0:
            return self._result(manifest, ExecutionStatus.FAILED, t0,
                                error=proc.stderr.strip()[:1500])
        out = proc.stdout.strip()
        try:
            value = json.loads(out) if out else None
        except json.JSONDecodeError:
            value = {"stdout": out[:4000]}
        return self._result(manifest, ExecutionStatus.SUCCEEDED, t0, value=value)


class _SubprocessCodeBuilder:
    """Renders a manifest entrypoint into a self-contained invocation script."""

    @staticmethod
    def build(manifest: ComponentManifest, arguments: dict) -> str:
        target = (manifest.runtime.entrypoint or "").strip()
        if ":" not in target:
            return ""
        module, _, func = target.partition(":")
        if not module or not func:
            return ""
        return (
            "import json, sys\n"
            f"import {module} as _m\n"
            f"_fn = getattr(_m, {func!r})\n"
            f"_args = json.loads({json.dumps(json.dumps(arguments, default=str))!r})\n"
            "_out = _fn(**_args)\n"
            "try:\n"
            "    sys.stdout.write(json.dumps(_out, default=str))\n"
            "except (TypeError, ValueError):\n"
            "    sys.stdout.write(json.dumps({'repr': repr(_out)}))\n"
        )


class ContainerBackend(Backend):
    """Container execution — unavailable here, and says so rather than pretending."""

    backend = "container"
    RUNTIMES = ("docker", "podman", "nerdctl")

    def __init__(self) -> None:
        self.runtime_bin = next((r for r in self.RUNTIMES if shutil.which(r)), None)

    def available(self) -> bool:
        return self.runtime_bin is not None

    def unavailable_reason(self) -> str:
        if self.available():
            return ""
        return (f"no container runtime found (looked for {', '.join(self.RUNTIMES)}); "
                "container-isolated execution is not possible on this machine")

    def invoke(self, manifest: ComponentManifest, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        if not self.available():
            return self._result(manifest, ExecutionStatus.UNAVAILABLE, t0,
                                error=self.unavailable_reason())
        image = manifest.runtime.image
        entrypoint = (manifest.runtime.entrypoint or "").strip()
        if not entrypoint:
            # Running the image's default command executes whatever the image
            # does, not what this component declares. Reporting that as this
            # component's SUCCEEDED result — and advancing it to READY — meant
            # every component sharing an image produced the same "successful"
            # run. Without an entrypoint there is nothing specific to execute.
            return self._result(
                manifest, ExecutionStatus.UNAVAILABLE, t0,
                error=("component declares no runtime.entrypoint, so the container would run "
                       f"the default command of {image!r} rather than this component; declare "
                       "an entrypoint to make the invocation specific"))
        timeout_s = kwargs.pop("timeout_s", 300)
        payload = json.dumps({"entrypoint": entrypoint, "arguments": kwargs}, default=str)
        cmd = [self.runtime_bin, "run", "--rm", "--network", "none",
               "--env", f"BIOAGENT_INVOCATION={payload}", image, *self._argv(entrypoint, kwargs)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,  # noqa: S603
                                  timeout=timeout_s, check=False)
        except subprocess.TimeoutExpired:
            return self._result(manifest, ExecutionStatus.TIMEOUT, t0, error="container timed out")
        status = ExecutionStatus.SUCCEEDED if proc.returncode == 0 else ExecutionStatus.FAILED
        out = proc.stdout.strip()
        try:
            value = json.loads(out) if out else None
        except json.JSONDecodeError:
            value = {"stdout": out[:4000]}
        return self._result(manifest, status, t0, value=value,
                            error=(None if proc.returncode == 0 else proc.stderr[:1500]))

    @staticmethod
    def _argv(entrypoint: str, kwargs: dict) -> list[str]:
        """Command the container runs: the entrypoint, then `--key value` pairs."""
        argv = [entrypoint]
        for key, val in kwargs.items():
            argv += [f"--{key}", str(val)]
        return argv


class NoneBackend(Backend):
    """For declarative components (skills, benchmarks, roles) with no entrypoint."""

    backend = "none"

    def available(self) -> bool:
        return False

    def unavailable_reason(self) -> str:
        return ("component is declarative (catalogue/specification metadata) and exposes "
                "no invocable entrypoint")

    def invoke(self, manifest: ComponentManifest, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        return self._result(manifest, ExecutionStatus.UNAVAILABLE, t0,
                            error=self.unavailable_reason(),
                            value={"kind": manifest.kind, "specification_only": True})
