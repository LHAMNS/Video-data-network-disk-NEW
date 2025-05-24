# GPU Upgrade & AVI Migration Guide

> **Scope** – This guide describes the precise steps required to migrate the Web‑UI conversion server from the legacy CPU/MP4 pipeline to the new GPU‑accelerated Raptor pipeline that outputs **AVI** containers. Follow the sequence exactly; each pre‑condition is mandatory.

---

## 1  Overview of Functional Changes

| Area             | Old Behaviour                                      | New Behaviour                                              |
| ---------------- | -------------------------------------------------- | ---------------------------------------------------------- |
| Error‑correction | `ReedSolomonEncoder` (CPU)                         | `get_optimal_error_corrector()` → Raptor LDPC (GPU)        |
| Frame generation | `FrameGenerator` / `OptimizedFrameGenerator` (CPU) | `GPUFrameGenerator` with CUDA fallback to CPU              |
| Video encoding   | `StreamingVideoEncoder` (MP4/H.264)                | `StreamingDirectAVIEncoder` (AVI/MJPEG)                    |
| Metadata         | None                                               | `VideoRaptorEncoder` adds metadata/calibration/sync frames |
| File extension   | `.mp4`                                             | `.avi`                                                     |
| MIME type        | `video/mp4`                                        | `video/x‑msvideo`                                          |
| Checksum         | not present                                        | 16‑digit SHA‑256 prefix stored in metadata                 |

All application‑level REST and Socket.IO semantics remain unchanged.

---

## 2  Prerequisites

1. **CUDA‑capable GPU** with driver ≥ 525 and CUDA toolkit ≥ 12.0.
2. **FFmpeg ≥ 6.1** compiled with `--enable-libmjpeg` and support for AVI muxing (default in official builds).
3. **Python ≥ 3.10** with the packages listed in `requirements.txt` **plus**:

   * `cupy` (GPU arrays)
   * `numba[cuda]` (kernel JIT)
   * `pyldpc` (Raptor inner codes)
   * `opencv-python` (MJPEG validation)
4. Linux kernel 5.15+ or Windows 10 22H2. macOS is **not supported** for GPU mode.

> **Tip:** If the system lacks a compatible GPU, the code automatically falls back to CPU paths; performance will, however, be substantially lower.

---

## 3  Directory Additions

```
converter/
 ├── gpu_error_correction.py        # new – Raptor LDPC front‑end
 ├── gpu_frame_generator.py         # new – CUDA frame synthesis
 └── encoder/
     └── StreamingDirectAVIEncoder.py  # new – MJPEG‑in‑AVI writer
```

Ensure these files are committed **before** applying the patch.

---

## 4  Applying the Source Patch

The patch only touches `server.py`. Place `server.patch` at the project root and execute:

```bash
cd <project‑root>
patch -p1 < server.patch
```

The following will occur automatically:

* New imports (`hashlib`, GPU modules) added.
* Default output extension switched to `.avi`.
* Default MIME switched to `video/x‑msvideo`.
* GPU‑aware replacement for `ConversionTask._conversion_worker`.

> **Validation:** `git diff` must show **no remaining hunks** after the patch.

---

## 5  Dependencies Update

Append the following to `requirements.txt` (exact versions tested in CI):

```
cupy-cuda12x==13.2.0
numba==0.59.1
pyldpc==0.5.2
opencv-python>=4.9.0.0
```

Then create/refresh the virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate.bat
pip install -r requirements.txt
```

---

## 6  Database & Cache Compatibility

* The cache schema is **unchanged**; no migration steps required.
* Old `.mp4` outputs already present in `output/` remain readable and downloadable. Download endpoints still infer MIME via `mimetypes.guess_type`; hence legacy files continue to serve correctly.

---

## 7  Functional Test Procedure

1. **Unit tests** (run in venv):

   ```bash
   pytest tests/test_gpu_pipeline.py -q
   ```

2. **Manual end‑to‑end smoke test** – Start the server:

   ```bash
   python server.py --debug
   ```

   * Upload a ≤ 10 MB sample file via Web‑UI.
   * Observe log – it must print:

     * `Initializing Raptor error correction`
     * `Using GPU-accelerated frame generator`
     * `Encoding complete [TASK_ID]: { ... }`
   * Confirm a `.avi` file appears in `output/`.
   * Download via UI and play with VLC; video must open and show metadata frame for ≈ 1 s.

3. **Fallback test** – Run the same procedure on a machine **without** a CUDA‑enabled GPU. Logs should print `Using CPU frame generator`; the process must still succeed, albeit slower.

---

## 8  Deployment Checklist

* [ ] All new Python modules committed.
* [ ] Patch applied cleanly.
* [ ] `requirements.txt` updated and dependencies installed.
* [ ] FFmpeg ≥ 6.1 present in `$PATH`.
* [ ] GPU validation test (Section 7) passes.
* [ ] Monitoring/cleanup background threads verified (no regressions).

---

## 9  Rollback Strategy

1. `git revert <merge‑commit>` or `git checkout` the previous commit on `server.py`.
2. Delete new GPU modules if not needed.
3. Remove added packages from the virtual environment.
4. Restart the Flask server.

No data migration is required; the system will resume MP4 output immediately.

---

## 10  Troubleshooting Matrix

| Symptom                               | Likely Cause                         | Resolution                                                 |
| ------------------------------------- | ------------------------------------ | ---------------------------------------------------------- |
| `RuntimeError: CUDA driver not found` | NVIDIA driver absent or incompatible | Install correct driver; verify `nvidia-smi` works          |
| `Error correction failed: ...`        | Raptor parameters outside spec       | Check `error_correction_ratio` (0 < r ≤ 0.5)               |
| Output verification fails             | FFmpeg build lacks MJPEG decoder     | Update FFmpeg to ≥ 6.1 official build                      |
| Download shows 0 bytes                | Reverse‑proxy strips chunked data    | Disable compression or use `proxy_buffering off;` in Nginx |

---

## 11  Maintainers

| Name     | Role      | Contact                 |
| -------- | --------- | ----------------------- |
| L. Hamns | Tech Lead | \<internal‑slack>@hamns |
| Y. Chen  | DevOps    | \<internal‑slack>@ychen |

---

*Document version: 1.0 – 24 May 2025*

---

## Appendix A – Complete `ConversionTask._conversion_worker` Implementation

> Copy‑and‑paste if you prefer a manual edit rather than applying `server.patch`.

```python
# --------------- BEGIN NEW _conversion_worker ----------------
    def _conversion_worker(self):
        """GPU-aware conversion worker thread"""
        try:
            self._update_task_status("initializing")

            # ---------- 1.  GPU error-correction ----------
            if self.error_correction_enabled:
                logger.info(f"Initializing Raptor error correction [{self.task_id}]")
                self.error_correction = get_optimal_error_corrector(self.error_correction_ratio)

            # ---------- 2.  Frame generator ----------
            try:
                self.frame_generator = GPUFrameGenerator(
                    resolution=self.resolution,
                    fps=self.fps,
                    color_count=self.color_count,
                    nine_to_one=self.nine_to_one
                )
                logger.info(f"Using GPU-accelerated frame generator [{self.task_id}]")
            except RuntimeError:
                generator_class = OptimizedFrameGenerator if self.use_optimized_generator else FrameGenerator
                self.frame_generator = generator_class(
                    resolution=self.resolution,
                    fps=self.fps,
                    color_count=self.color_count,
                    nine_to_one=self.nine_to_one
                )
                logger.info(f"Using CPU frame generator [{self.task_id}]")

            # ---------- 3.  Encoder initialisation ----------
            physical_width  = self.video_params["physical_width"]
            physical_height = self.video_params["physical_height"]

            # Metadata encoder (Raptor) for calibration & sync
            self.raptor_encoder = VideoRaptorEncoder(physical_width, physical_height, self.fps)

            from converter.encoder import StreamingDirectAVIEncoder   # lazy import
            self.video_encoder = StreamingDirectAVIEncoder(
                width=physical_width,
                height=physical_height,
                fps=self.fps,
                output_path=self.output_path
            )
            self.video_encoder.start()

            # ---------- 4.  Metadata / calibration frames ----------
            if self.params.get("metadata_frames", True):
                logger.info(f"Adding metadata frames [{self.task_id}]")

                file_info = {
                    'filename': self.original_filename,
                    'size': self.file_size,
                    'checksum': hashlib.sha256(str(self.file_id).encode()).hexdigest()[:16],
                    'total_symbols': 0
                }

                self.video_encoder.add_frame(self.raptor_encoder.create_metadata_frame(file_info))
                self.video_encoder.add_frame(self.raptor_encoder.create_calibration_frame())

                sync_frame = np.zeros((physical_height, physical_width, 3), dtype=np.uint8)
                self.raptor_encoder._add_sync_pattern(sync_frame)
                self.video_encoder.add_frame(sync_frame)

                self.processed_frames = 3

            self._update_task_status("processing")

            # ---------- 5.  Read source data ----------
            data_generator = cache_manager.read_cached_file(self.file_id)
            all_data = bytearray()
            for chunk in data_generator:
                if not self.running:
                    logger.info(f"Data collection interrupted [{self.task_id}]")
                    self._update_task_status("stopped")
                    return
                all_data.extend(chunk)

            # ---------- 6.  Error-correction ----------
            if self.error_correction_enabled and self.error_correction:
                logger.info(f"Applying Raptor error correction [{self.task_id}]…")
                self._update_task_status("error_correction")
                try:
                    encoded_data, stats = self.error_correction.process_file_data(bytes(all_data))
                    logger.info(f"Raptor encoding complete [{self.task_id}]: "
                                f"{stats['throughput_mbps']:.1f} MB/s, "
                                f"redundancy: {stats['redundancy_ratio']:.2%}")
                    data_source = encoded_data
                except Exception as e:
                    error_msg = f"Error correction failed: {e}"
                    logger.error(f"{error_msg} [{self.task_id}]", exc_info=True)
                    self._update_task_status("error", error_msg)
                    socketio.emit('conversion_error', {"error": error_msg, "task_id": self.task_id})
                    return
            else:
                data_source = bytes(all_data)

            self._update_task_status("converting")

            # ---------- 7.  Frame generation & encoding ----------
            for frame in self.frame_generator.generate_frames_from_data(
                    data_source, callback=self._frame_generated_callback):
                if not self.running:
                    logger.info(f"Frame generation interrupted [{self.task_id}]")
                    self._update_task_status("stopped")
                    return
                if not self.video_encoder.add_frame(frame):
                    error_msg = "Failed to add frame to video encoder"
                    logger.error(f"{error_msg} [{self.task_id}]")
                    self._update_task_status("error", error_msg)
                    socketio.emit('conversion_error', {"error": error_msg, "task_id": self.task_id})
                    return

            # ---------- 8.  Finalise ----------
            if self.running:
                logger.info(f"All frames processed [{self.task_id}], finalizing…")
                self._update_task_status("finalizing")
                try:
                    stats = self.video_encoder.stop()
                    logger.info(f"Encoding complete [{self.task_id}]: {stats}")

                    is_valid, err = self._verify_output_video()
                    self.output_file_verified = is_valid
                    if not is_valid:
                        logger.error(f"Output verification failed [{self.task_id}]: {err}")
                        self._update_task_status("error", f"Output verification failed: {err}")
                        socketio.emit('conversion_error',
                                      {"error": f"Output verification failed: {err}",
                                       "task_id": self.task_id})
                        return

                    self._update_task_status("completed", output_path=str(self.output_path))
                    socketio.emit('conversion_complete',
                                  {"output_file": str(self.output_path),
                                   "filename": self.output_path.name,
                                   "duration": time.time() - self.start_time,
                                   "frames": self.processed_frames,
                                   "task_id": self.task_id})
                except Exception as e:
                    error_msg = f"Encoding completion error: {e}"
                    logger.error(f"{error_msg} [{self.task_id}]", exc_info=True)
                    self._update_task_status("error", error_msg)
                    socketio.emit('conversion_error',
                                  {"error": error_msg, "task_id": self.task_id})

        except Exception as e:
            error_msg = f"Conversion error: {e}"
            logger.error(f"{error_msg} [{self.task_id}]", exc_info=True)
            self._update_task_status("error", error_msg)
            socketio.emit('conversion_error', {"error": error_msg, "task_id": self.task_id})

        finally:
            self.running = False
            if hasattr(self, 'video_encoder') and self.video_encoder:
                try:
                    if getattr(self.video_encoder, 'running', False):
                        self.video_encoder.stop()
                except Exception as e:
                    logger.error(f"Error stopping video encoder [{self.task_id}]: {e}")
            self.event.set()
# --------------- END NEW _conversion_worker ----------------
```

*Updated: 24 May 2025*
