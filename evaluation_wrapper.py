"""Participant model wrapper for the DNDX benchmark.

This wrapper preserves the public ``VLMModel.generate_with_metrics`` contract.
Three backends are supported and selected via ``backend=...``:

* ``"dummy"``       – smoke-test only; deterministic hash-based answers.
* ``"transformers"`` – Hugging Face Transformers, single-GPU eager generate.
* ``"sglang"``      – SGLang OpenAI-compatible HTTP server.

The sglang backend can either reuse a running server (if
``SGLANG_SERVER_URL`` is set, or ``localhost:8999`` already answers
``/health_generate``) or auto-launch one in a subprocess. The auto-launch
path is the default and is what the competition grader will exercise, since
the wrapper must be self-contained.
"""

from __future__ import annotations

import atexit
import base64
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class GenerationConfig:
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass
class GenerationResult:
    text: str
    token_count: int
    ttft_seconds: float
    elapsed_seconds: float
    meta: dict[str, Any]


# ---------------------------------------------------------------------------
# sglang backend helpers
# ---------------------------------------------------------------------------


_DEFAULT_SGLANG_PORT = 8999
_SGLANG_HEALTH_PATH = "/health_generate"
_SGLANG_CHAT_PATH = "/v1/chat/completions"
_SGLANG_LAUNCH_TIMEOUT_S = 900  # model load + CUDA graph capture can be slow
_SGLANG_HEALTH_POLL_S = 2.0


def _server_alive(base_url: str, timeout: float = 3.0) -> bool:
    try:
        r = requests.get(f"{base_url}{_SGLANG_HEALTH_PATH}", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _pick_sglang_port() -> int:
    env_port = os.environ.get("SGLANG_PORT")
    if env_port:
        try:
            return int(env_port)
        except ValueError:
            pass
    return _DEFAULT_SGLANG_PORT


class _SGLangServer:
    """Manages a local sglang HTTP server (launch, health, warmup, shutdown)."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

        explicit = os.environ.get("SGLANG_SERVER_URL")
        if explicit:
            self.base_url = explicit.rstrip("/")
            self.proc: subprocess.Popen | None = None
            self.owns_server = False
        else:
            port = _pick_sglang_port()
            self.base_url = f"http://127.0.0.1:{port}"
            if _server_alive(self.base_url):
                # Reuse an already-running server (e.g. left over from earlier tests).
                self.proc = None
                self.owns_server = False
            else:
                self.proc = self._launch(port=port)
                self.owns_server = True
                atexit.register(self.shutdown)

        # The sglang server registers its model under the model_path by default.
        self.served_model = os.environ.get("SGLANG_SERVED_MODEL", model_path)

        self._wait_ready()
        self._warmup()

    # -- launch ---------------------------------------------------------------

    def _launch(self, *, port: int) -> subprocess.Popen:
        log_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "logs"
        )
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(
            log_dir,
            f"sglang_server_{port}_{int(time.time())}.log",
        )
        log_handle = open(log_path, "ab")

        # Default command tuned for Qwen3.5-2B on PPU 810E (see EXPERIMENT.md
        # §17 — "对照组 G"): fa3 attention + PPU SAIL FLA CUDA kernel + radix
        # cache off + page-size 64. This G-group configuration beats the
        # triton baseline on TTFT (-85 %) and throughput (+30 %).
        # 现在额外叠加 piecewise CUDA Graph(参 SGLang PR #15320 思想):
        #   - ViT blocks+merger 段压成 graph(配 SGLANG_VIT_ENABLE_CUDA_GRAPH=1)
        #   - decode 主体走 piecewise CUDA Graph(配 wrapper GEMV 已预热)
        # 不开 --enable-torch-compile:PPU 上 Inductor 对 causal_conv1d triton
        # 内核 codegen 有 bug(launcher() missing _grid_2),piecewise CUDA Graph
        # 自带的 eager 编译模式不需要显式 torch.compile。
        cmd = [
            sys.executable,
            "-u",
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.model_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--trust-remote-code",
            "--tp-size",
            str(int(os.environ.get("SGLANG_TP_SIZE", "1"))),
            "--attention-backend",
            os.environ.get("SGLANG_ATTENTION_BACKEND", "fa3"),
            "--enable-multimodal",
            "--dtype",
            os.environ.get("SGLANG_DTYPE", "bfloat16"),
            "--mem-fraction-static",
            os.environ.get("SGLANG_MEM_FRACTION", "0.85"),
            "--page-size",
            os.environ.get("SGLANG_PAGE_SIZE", "64"),
            "--log-level",
            os.environ.get("SGLANG_LOG_LEVEL", "info"),
            # === INT8 GEMV / GEMM 走我们 patches 注入的 w8a8_int8.py ===
            # 不设这个会 fallback 到 compressed_tensors_w8a8_int8,该路径会调
            # sgl_kernel.int8_scaled_mm(PPU sgl_kernel 里没这个 op,必崩)
            "--quantization", os.environ.get("SGLANG_QUANTIZATION", "w8a8_int8"),
        ]

        # Radix cache is on by default in sglang; for short, mostly-unique
        # eval-style prompts it adds lookup overhead without real reuse
        # benefit. Set SGLANG_FORCE_RADIX_CACHE=1 to re-enable.
        if os.environ.get("SGLANG_FORCE_RADIX_CACHE", "0") != "1":
            cmd.append("--disable-radix-cache")

        # === piecewise CUDA Graph + ViT 优化(ViT PR #15320 思路) ===
        # 默认开;用户可用 SGLANG_DISABLE_CUDA_GRAPH=1 关掉(debug / 对照实验)。
        # 注:--linear-attn-backend 没有 fla 选项(PPU FLA 是 env var 控制,见下)
        # PPU 实测:
        #   - piecewise CUDA Graph + eager compiler: ✅ 工作(SGLANG_VIT_ENABLE_CUDA_GRAPH=0 时)
        #   - SGLANG_VIT_ENABLE_CUDA_GRAPH=1 + 多模态请求: ❌ 触发 Inductor codegen
        #     "Cannot copy between CPU and CUDA tensors during CUDA graph capture
        #     unless the CPU tensor is pinned"(PR #15320 的 PPU 兼容性问题)
        #   - 决策:默认开 piecewise CUDA Graph + 普通 CUDA Graph,关 ViT CG;
        #     用 SGLANG_VIT_ENABLE_CUDA_GRAPH_OVERRIDE=1 可重新打开(用户测试用)
        if os.environ.get("SGLANG_DISABLE_CUDA_GRAPH", "0") != "1":
            cmd.extend([
                "--enable-piecewise-cuda-graph",
                "--piecewise-cuda-graph-max-tokens",
                os.environ.get("SGLANG_PIECEWISE_CUDA_GRAPH_MAX_TOKENS", "4096"),
                "--piecewise-cuda-graph-compiler", "eager",
                "--mm-attention-backend", "fa3",
                # 普通 CUDA Graph max BS — baseline 用 256,我们之前误设 32 太小
                # 单样本 TTFT 影响小,但 batched 推理会更稳
                "--cuda-graph-max-bs", os.environ.get("SGLANG_CUDA_GRAPH_MAX_BS", "256"),
            ])

        extra_args = os.environ.get("SGLANG_EXTRA_ARGS", "").strip()
        if extra_args:
            cmd.extend(extra_args.split())

        env = os.environ.copy()
        # Force the PPU-optimised FLA CUDA kernel for hybrid linear/GDN
        # attention layers. Without this, sglang falls back to the community
        # Triton kernel which leaves a ~300 ms prefill plateau on Qwen3.5.
        env["SGLANG_SAIL_FLA_CUDA"] = "1"

        # === ViT Piecewise CUDA Graph(PR #15320)===
        # PPU 实测关闭(见上方注释);用户可显式开启重试
        env["SGLANG_VIT_ENABLE_CUDA_GRAPH"] = (
            "1" if os.environ.get("SGLANG_VIT_ENABLE_CUDA_GRAPH_OVERRIDE", "0") == "1" else "0"
        )
        env.setdefault("SGLANG_MM_FEATURE_CACHE_MB", "4096")
        env.setdefault("SGLANG_VLM_CACHE_SIZE_MB", "0")

        # === GEMV / 量化分派 + 自动 warmup(本仓库 patches 注入)===
        # 本路线目标:GEMV 真正进入 CUDA Graph,不被 ACEXT 互斥绕过。
        # GEMV 默认开启 + warmup 默认开启;ACEXT 默认关闭。
        env["SGLANG_SAIL_DEEPGEMM_DENSE"] = "1"
        env.setdefault("SGLANG_SAIL_USE_GEMV", "1")
        env.setdefault("SGLANG_SAIL_GEMV_M_MAX", "8")
        env.setdefault("SGLANG_SAIL_GEMV_WARMUP_ON_LOAD", "1")
        env["SGLANG_SAIL_PREFILL_BF16"] = "1"

        # === ACEXT(与 GEMV 互斥接管 dense linear)===
        # §37 实测 ACEXT 比 GEMV 又快 +2.4% (423 vs 418 tok/s),且来源是
        # PPU sgl_kernel 内置(`sgl_kernel.int8_gemm`),无 patch 成本。
        # 默认开启;关 ACEXT 回退到 GEMV: SGLANG_SAIL_USE_ACEXT_CUDA=0
        use_acext = os.environ.get("SGLANG_SAIL_USE_ACEXT_CUDA", "1") == "1"
        env["SGLANG_SAIL_USE_ACEXT_CUDA"] = "1" if use_acext else "0"

        # === DEBUG:GEMV 是否真在 CG capture/replay 中被调用 ===
        # 设 1 时 wrapper __call__ 会输出每帧调用 + capturing 状态。
        # 期望看到:capture 阶段 bs=1..32 各自调一次;replay 阶段不再调 Python。
        if os.environ.get("SGLANG_DEBUG_GEMV", "0") == "1":
            env["SGLANG_DEBUG_GEMV"] = "1"

        # custom all-reduce 还不支持 CUDA Graph,必须关 + 用 symm-mem 替代
        # (注意:这些是命令行参数,这里再设 env 是为了将来 launch_server 的 fallback 路径)
        env.setdefault("SGLANG_DISABLE_CUSTOM_ALL_REDU", "1")
        env.setdefault("SGLANG_ENABLE_TORCH_SYMM_MEM", "1")

        return subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid if os.name == "posix" else None,
        )

    # -- readiness ------------------------------------------------------------

    def _wait_ready(self) -> None:
        deadline = time.time() + _SGLANG_LAUNCH_TIMEOUT_S
        last_err: Exception | None = None
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(
                    f"sglang server exited early (code={self.proc.returncode}). "
                    "Check logs/sglang_server_*.log for details."
                )
            try:
                r = requests.get(
                    f"{self.base_url}{_SGLANG_HEALTH_PATH}", timeout=3
                )
                if r.status_code == 200:
                    return
            except Exception as exc:  # noqa: BLE001
                last_err = exc
            time.sleep(_SGLANG_HEALTH_POLL_S)
        raise RuntimeError(
            f"sglang server did not become ready at {self.base_url} within "
            f"{_SGLANG_LAUNCH_TIMEOUT_S}s (last error: {last_err})"
        )

    def _warmup(self) -> None:
        # A tiny text-only chat completion forces sglang to compile/warm any
        # remaining lazy kernels before the first measured request.
        payload = {
            "model": self.served_model,
            "messages": [{"role": "user", "content": "ok"}],
            "max_tokens": 4,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": False,
        }
        try:
            requests.post(
                f"{self.base_url}{_SGLANG_CHAT_PATH}",
                json=payload,
                timeout=120,
            )
        except Exception:
            # Warmup is best-effort; the benchmark loop has its own warmup pass.
            pass

    # -- shutdown -------------------------------------------------------------

    def shutdown(self) -> None:
        if not self.proc or self.proc.poll() is not None:
            return
        try:
            if os.name == "posix":
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                except Exception:
                    self.proc.terminate()
            else:
                self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# VLMModel
# ---------------------------------------------------------------------------


class VLMModel:
    """
    Default participant wrapper.

    `backend="dummy"` is for demo-only smoke tests.
    `backend="transformers"` uses a local Hugging Face model directory.
    `backend="sglang"` launches (or reuses) a local SGLang server and talks to
    it via the OpenAI-compatible HTTP API.

    Participants can replace the internals while preserving `generate_with_metrics`.
    """

    def __init__(
        self,
        model_path: str,
        *,
        backend: str = "auto",
        device: str = "auto",
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.backend = backend
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._backend_name = "dummy"
        self._sglang: _SGLangServer | None = None

        if backend == "sglang":
            self._load_sglang_backend()
        elif backend in {"auto", "transformers"}:
            try:
                self._load_transformers_backend()
                self._backend_name = "transformers"
            except Exception as exc:
                if backend == "transformers":
                    raise
                self._load_dummy_backend(str(exc))
        else:
            self._load_dummy_backend(f"backend={backend}")

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def generate_with_metrics(
        self,
        *,
        image,
        prompt: str,
        choices: dict[str, str],
        generation_config: GenerationConfig,
        sample_id: str,
    ) -> GenerationResult:
        if self._backend_name == "sglang":
            return self._generate_with_sglang(
                image=image,
                prompt=prompt,
                generation_config=generation_config,
            )
        if self._backend_name == "transformers":
            return self._generate_with_transformers(
                image=image,
                prompt=prompt,
                generation_config=generation_config,
            )
        return self._generate_with_dummy(
            prompt=prompt,
            choices=choices,
            generation_config=generation_config,
            sample_id=sample_id,
        )

    # -- sglang backend -------------------------------------------------------

    def _load_sglang_backend(self) -> None:
        self._sglang = _SGLangServer(self.model_path)
        self._backend_name = "sglang"

    def _encode_image_data_url(self, image) -> str:
        if image.mode != "RGB":
            image = image.convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _generate_with_sglang(
        self,
        *,
        image,
        prompt: str,
        generation_config: GenerationConfig,
    ) -> GenerationResult:
        assert self._sglang is not None
        data_url = self._encode_image_data_url(image)

        payload: dict[str, Any] = {
            "model": self._sglang.served_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": int(generation_config.max_new_tokens),
            "temperature": float(generation_config.temperature),
            "top_p": float(generation_config.top_p),
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        t0 = time.perf_counter()
        first_chunk_at: float | None = None
        text_parts: list[str] = []
        completion_tokens = 0
        prompt_tokens = 0

        with requests.post(
            f"{self._sglang.base_url}{_SGLANG_CHAT_PATH}",
            json=payload,
            stream=True,
            timeout=600,
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", "replace") if isinstance(raw_line, bytes) else raw_line
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    obj = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    chunk = delta.get("content")
                    if chunk:
                        if first_chunk_at is None:
                            first_chunk_at = time.perf_counter()
                        text_parts.append(chunk)
                usage = obj.get("usage")
                if usage:
                    completion_tokens = usage.get("completion_tokens", completion_tokens) or completion_tokens
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens) or prompt_tokens
        t1 = time.perf_counter()

        text = "".join(text_parts).strip()
        ttft = (
            (first_chunk_at - t0)
            if first_chunk_at is not None
            else (t1 - t0)
        )
        token_count = int(completion_tokens) if completion_tokens > 0 else max(1, len(text_parts))
        return GenerationResult(
            text=text,
            token_count=token_count,
            ttft_seconds=ttft,
            elapsed_seconds=t1 - t0,
            meta={
                "backend": "sglang",
                "base_url": self._sglang.base_url,
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "streamed_chunks": len(text_parts),
            },
        )

    # -- transformers backend (preserved) -------------------------------------

    def _load_transformers_backend(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        ).eval()
        self._tokenizer = getattr(self._processor, "tokenizer", None)

    def _generate_with_transformers(
        self,
        *,
        image,
        prompt: str,
        generation_config: GenerationConfig,
    ) -> GenerationResult:
        import torch
        from transformers import TextIteratorStreamer

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        input_len = inputs.input_ids.shape[1]
        streamer = TextIteratorStreamer(
            self._processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_kwargs = {
            **inputs,
            "max_new_tokens": generation_config.max_new_tokens,
            "temperature": generation_config.temperature,
            "top_p": generation_config.top_p,
            "do_sample": generation_config.temperature > 0,
            "use_cache": True,
            "streamer": streamer,
        }

        output_holder: dict[str, Any] = {}

        def _run_generate() -> None:
            with torch.no_grad():
                output_holder["output_ids"] = self._model.generate(**generation_kwargs)

        worker = threading.Thread(target=_run_generate, daemon=True)
        start = time.perf_counter()
        worker.start()

        first_chunk_at = None
        chunks: list[str] = []
        for chunk in streamer:
            now = time.perf_counter()
            if first_chunk_at is None and chunk:
                first_chunk_at = now
            chunks.append(chunk)
        worker.join()
        end = time.perf_counter()

        output_ids = output_holder["output_ids"]
        generated_ids = output_ids[0][input_len:]
        text = "".join(chunks).strip()
        if not text:
            text = self._processor.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

        ttft = (first_chunk_at - start) if first_chunk_at is not None else (end - start)
        return GenerationResult(
            text=text,
            token_count=int(generated_ids.shape[0]),
            ttft_seconds=ttft,
            elapsed_seconds=end - start,
            meta={"backend": "transformers"},
        )

    # -- dummy backend (preserved) --------------------------------------------

    def _load_dummy_backend(self, reason: str) -> None:
        self._dummy_reason = reason

    def _generate_with_dummy(
        self,
        *,
        prompt: str,
        choices: dict[str, str],
        generation_config: GenerationConfig,
        sample_id: str,
    ) -> GenerationResult:
        start = time.perf_counter()
        usable_choices = [key for key, value in choices.items() if (value or "").strip()]
        picked = usable_choices[hash(sample_id) % len(usable_choices)] if usable_choices else "A"
        text = (
            f"Answer: {picked}\n"
            f"Explanation: dummy backend selected a deterministic option for smoke testing."
        )
        token_count = max(1, min(generation_config.max_new_tokens, len(text.split())))
        end = time.perf_counter()
        return GenerationResult(
            text=text,
            token_count=token_count,
            ttft_seconds=max(end - start, 1e-4),
            elapsed_seconds=max(end - start, 2e-4),
            meta={"backend": "dummy", "reason": getattr(self, "_dummy_reason", "n/a"), "prompt_chars": len(prompt)},
        )