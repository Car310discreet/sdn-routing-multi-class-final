"""
multimedia_evaluator.py - Module 4: video transfer over the selected path and QoS/QoE measurement.

For one routing paradigm, `VideoEvaluator.run()`:
  1. streams the H.264 transport stream (assets/surreal.ts, 66 s, 4,007 frames) Sender -> Receiver with
     FFmpeg (`-re -c copy` over UDP) while the background traffic runs;
  2. logs one row per second to timeseries.csv: t_s, rtt_ms, jitter_ms, throughput_mbps, packet_loss_pct
       RTT          one ICMP echo per second, Receiver -> Sender (ping -D -O), so it follows the route
       jitter, loss a 200 kbit/s connectionless UDP probe Sender -> Receiver on the same route
                    (udp_probe.py): per-second loss from sequence numbers, RFC 3550 jitter from
                    shared-clock transit times; it keeps working where iperf3's handshake would fail
       throughput   video bytes delivered to the Receiver per second
     summary.json also holds the exact whole-stream video packet loss.
     Both come from the OpenFlow counters of the Sender -> Receiver route entries on the first and last
     switch, minus the UDP probe's datagrams. Access-port counters cannot be used: the background
     generator also sources and sinks iperf3 flows at the Sender's and Receiver's hosts;
  3. scores the received video frame by frame (frames.csv: frame, psnr_db, ssim) and overall.

Video scoring needs FFmpeg's native H.264 decoder, which conceals errors: Fedora's ffmpeg-free only has
OpenH264, which decodes nothing from a badly damaged stream. The system ffmpeg is used when it has the
native decoder (Ubuntu's does), otherwise the static build from the imageio-ffmpeg package (which crashes
decoding on some kernels, e.g. Ubuntu 24.04 with Linux 7.0). `python routing/multimedia_evaluator.py
--rescore RUN_DIR` re-scores the saved received.ts files of a run whose scoring failed. Frames
identical to the original have infinite PSNR; per-frame means cap them at PSNR_CAP_DB.
`frames_decoded` counts the frames actually recovered from the received file.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import udp_probe  # noqa: E402

log = logging.getLogger(__name__)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_VIDEO = os.path.join(PROJECT_DIR, "assets", "surreal.ts")
VIDEO_PORT = 1234
PROBE_PORT = 5201
PROBE_PPS = 25            # 25 x 1000-byte datagrams = 200 kbit/s, ~4 % of the video bitrate
PROBE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "udp_probe.py")
PSNR_CAP_DB = 60.0


def has_native_h264(binary: str) -> bool:
    try:
        proc = subprocess.run([binary, "-hide_banner", "-decoders"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and re.search(r"^\s*V\S*\s+h264\s", proc.stdout, re.MULTILINE) is not None


def ffmpeg_binary() -> str:
    if has_native_h264("ffmpeg"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        log.warning("No ffmpeg with the native H.264 decoder; scoring with the system ffmpeg (may fail on damaged video)")
        return "ffmpeg"


@dataclass
class StreamSummary:
    paradigm: str
    duration_s: float
    mean_rtt_ms: float | None
    mean_jitter_ms: float | None
    mean_throughput_mbps: float
    packets_sent: int
    packets_received: int
    packet_loss_pct: float
    frames_reference: int
    frames_decoded: int
    psnr_db: float | None              # FFmpeg overall average (MSE-averaged, as in the papers)
    ssim: float | None
    psnr_frame_mean_db: float | None   # mean of per-frame values, identical frames capped
    ssim_frame_mean: float | None


PROBE_FRAME_BYTES = 1000 + 8 + 20 + 14     # probe payload + UDP + IPv4 + Ethernet, as OVS counts it


def parse_flow_counters(dump: str) -> tuple[int, int]:
    """Summed (n_packets, n_bytes) of every flow in an `ovs-ofctl dump-flows` output."""
    packets = sum(int(v) for v in re.findall(r"n_packets=(\d+)", dump))
    octets = sum(int(v) for v in re.findall(r"n_bytes=(\d+)", dump))
    return packets, octets


def route_counters(switch: str, src_ip: str, dst_ip: str) -> tuple[int, int]:
    """(packets, bytes) that `switch` has forwarded from src_ip to dst_ip (root only)."""
    proc = subprocess.run(["ovs-ofctl", "-O", "OpenFlow13", "dump-flows", switch,
                           f"ip,nw_src={src_ip},nw_dst={dst_ip}"], capture_output=True, text=True, check=True)
    return parse_flow_counters(proc.stdout)


class _PingSampler(threading.Thread):
    """`ping -D -O -i 1` in the Receiver's namespace; keeps (timestamp, rtt or None) samples."""

    LINE = re.compile(r"^\[(?P<ts>[\d.]+)\].*?(?:time=(?P<rtt>[\d.]+) ms|no answer yet)")

    def __init__(self, client_node, server_ip: str):
        super().__init__(daemon=True, name="ping-sampler")
        self.proc = client_node.popen(["ping", "-D", "-O", "-i", "1", "-W", "2", server_ip],
                                      stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.samples: list[tuple[float, float | None]] = []
        self._lock = threading.Lock()

    def run(self):
        for line in self.proc.stdout:
            match = self.LINE.search(line)
            if match:
                rtt = float(match.group("rtt")) if match.group("rtt") else None
                with self._lock:
                    self.samples.append((float(match.group("ts")), rtt))

    def samples_between(self, t0: float, t1: float):
        with self._lock:
            return [(ts, rtt) for ts, rtt in self.samples if t0 <= ts < t1]

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class VideoEvaluator:
    ffmpeg = ffmpeg_binary()

    def __init__(self, net, spec, server_host: str, client_host: str, video_path: str = DEFAULT_VIDEO,
                 log_seconds: int = 60):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        self.net, self.spec = net, spec
        self.server_host, self.client_host = server_host, client_host
        self.video_path, self.log_seconds = video_path, log_seconds

    def run(self, paradigm: str, out_dir: str) -> StreamSummary:
        os.makedirs(out_dir, exist_ok=True)
        received_path = os.path.join(out_dir, "received.ts")
        server, client = self.net.get(self.server_host), self.net.get(self.client_host)
        server_ip, client_ip = self.spec.hosts[self.server_host].ip, self.spec.hosts[self.client_host].ip
        tx_switch = self.spec.hosts[self.server_host].switch
        rx_switch = self.spec.hosts[self.client_host].switch
        url = f"udp://{client_ip}:{VIDEO_PORT}"
        probe_sent, probe_recv = os.path.join(out_dir, "probe_sent.csv"), os.path.join(out_dir, "probe_recv.csv")

        with open(os.path.join(out_dir, "receiver.log"), "w") as rx_log, \
                open(os.path.join(out_dir, "sender.log"), "w") as tx_log:
            receiver = client.popen(["ffmpeg", "-hide_banner", "-y", "-i", url, "-c", "copy", received_path],
                                    stdout=rx_log, stderr=subprocess.STDOUT)
            pinger = _PingSampler(client, server_ip)
            pinger.start()
            probe_receiver = client.popen([sys.executable, PROBE_SCRIPT, "recv", "--port", str(PROBE_PORT),
                                           "--duration", str(self.log_seconds + 15), "--out", probe_recv])
            time.sleep(1.0)
            stream_start = time.time()
            probe = server.popen([sys.executable, PROBE_SCRIPT, "send", "--dst", client_ip, "--port",
                                  str(PROBE_PORT), "--rate", str(PROBE_PPS), "--size", "1000",
                                  "--duration", str(self.log_seconds), "--out", probe_sent])
            sender = server.popen(["ffmpeg", "-hide_banner", "-re", "-i", self.video_path, "-c", "copy",
                                   "-f", "mpegts", url], stdout=tx_log, stderr=subprocess.STDOUT)
            try:
                rows, totals = self._sample(sender, pinger, tx_switch, rx_switch, server_ip, client_ip)
            finally:
                if sender.poll() is None:
                    sender.terminate()
                for proc in (probe, probe_receiver):
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                time.sleep(2)                    # let in-flight packets arrive
                receiver.terminate()             # ffmpeg finalises the file on SIGTERM
                try:
                    receiver.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    receiver.kill()
                pinger.stop()

        self._merge_probe(rows, totals, probe_sent, probe_recv, stream_start)
        with open(os.path.join(out_dir, "timeseries.csv"), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["t_s", "rtt_ms", "jitter_ms", "throughput_mbps",
                                                        "packet_loss_pct"])
            writer.writeheader()
            writer.writerows(rows)
        decoded, frames, overall = self.frame_quality(received_path, os.path.join(out_dir, "frames.csv"))
        summary = self._summarize(paradigm, rows, totals, decoded, frames, overall)
        with open(os.path.join(out_dir, "summary.json"), "w") as handle:
            json.dump(asdict(summary), handle, indent=2)
        log.info("[%s] PSNR %s dB, SSIM %s, video loss %.2f %%, RTT %s ms, jitter %s ms, %.2f Mbit/s", paradigm,
                 summary.psnr_db, summary.ssim, summary.packet_loss_pct, summary.mean_rtt_ms,
                 summary.mean_jitter_ms, summary.mean_throughput_mbps)
        return summary

    def _sample(self, sender, pinger, tx_switch, rx_switch, server_ip, client_ip, step=0.25, drain_s=2.5):
        """Route-entry counters every `step` s while the video plays: per-second throughput and RTT,
        whole-stream packet totals (still including the UDP probe, removed by _merge_probe)."""
        def read():
            rx_pkts, rx_bytes = route_counters(rx_switch, server_ip, client_ip)
            tx_pkts, _ = route_counters(tx_switch, server_ip, client_ip)
            return time.time(), rx_bytes, rx_pkts, tx_pkts

        history = [read()]
        start, sender_end = history[0][0], None
        while True:
            time.sleep(step)
            history.append(read())
            if sender_end is None and sender.poll() is not None:
                sender_end = history[-1][0]
            if sender_end is not None and history[-1][0] - sender_end >= drain_s:
                break
        times, rx_bytes, rx_pkts, tx_pkts = (np.array(column, dtype=float) for column in zip(*history))
        at = lambda series, t: float(np.interp(t, times, series))  # noqa: E731
        rows = []
        for second in range(1, min(int(sender_end - start), self.log_seconds) + 1):
            t0, t1 = start + second - 1, start + second
            rtts = [r for _, r in pinger.samples_between(t0, t1) if r is not None]
            rows.append({"t_s": second, "rtt_ms": rtts[-1] if rtts else None, "jitter_ms": None,
                         "throughput_mbps": round((at(rx_bytes, t1) - at(rx_bytes, t0)) * 8 / 1e6, 4),
                         "packet_loss_pct": None})
        totals = {"sent": int(tx_pkts[-1] - tx_pkts[0]), "received": int(rx_pkts[-1] - rx_pkts[0]),
                  "duration": sender_end - start}
        return rows, totals

    @staticmethod
    def _merge_probe(rows, totals, sent_path, recv_path, start):
        """Per-second probe loss and jitter into `rows`; the probe's own datagrams out of the
        route counters, so throughput and whole-stream loss are the video's alone."""
        try:
            with open(sent_path) as sent, open(recv_path) as received:
                sent_rows, recv_rows = list(csv.DictReader(sent)), list(csv.DictReader(received))
            stats = udp_probe.per_second(sent_rows, recv_rows, start, len(rows))
        except (OSError, ValueError, KeyError) as exc:
            log.warning("UDP probe logs unusable (%s); per-second loss and jitter left empty, "
                        "throughput includes the probe", exc)
            return
        totals["sent"] = max(0, totals["sent"] - len(sent_rows))
        totals["received"] = max(0, totals["received"] - len(recv_rows))
        probe_bytes = {}
        for row in recv_rows:
            second = int(float(row["recv_ts"]) - start) + 1
            probe_bytes[second] = probe_bytes.get(second, 0) + PROBE_FRAME_BYTES
        for row in rows:
            row["packet_loss_pct"] = stats[row["t_s"]]["loss_pct"]
            row["jitter_ms"] = stats[row["t_s"]]["jitter_ms"]
            row["throughput_mbps"] = round(max(0.0, row["throughput_mbps"] - probe_bytes.get(row["t_s"], 0) * 8 / 1e6),
                                           4)

    # -- video quality ------------------------------------------------------------------------
    def decoded_frames(self, path: str) -> int:
        proc = subprocess.run([self.ffmpeg, "-hide_banner", "-i", path, "-map", "0:v:0", "-f", "null", "-"],
                              capture_output=True, text=True)
        counts = re.findall(r"frame=\s*(\d+)", proc.stderr)
        return int(counts[-1]) if counts else 0

    def frame_quality(self, received_path: str, csv_path: str):
        """(frames decoded, per-frame rows, {psnr, ssim}) - an unusable file is reported, not raised."""
        if not os.path.exists(received_path) or os.path.getsize(received_path) == 0:
            log.warning("No video received")
            return 0, [], {}
        decoded = self.decoded_frames(received_path)
        if decoded == 0:
            log.warning("No frame of the received video could be decoded")
            return 0, [], {}
        out_dir = os.path.dirname(csv_path)
        psnr_log, ssim_log = os.path.join(out_dir, "psnr.log"), os.path.join(out_dir, "ssim.log")
        graph = (f"[0:v]split[r1][r2];[1:v]split[d1][d2];"
                 f"[r1][d1]psnr=stats_file={psnr_log};[r2][d2]ssim=stats_file={ssim_log}")
        proc = subprocess.run([self.ffmpeg, "-hide_banner", "-i", self.video_path, "-i", received_path,
                               "-lavfi", graph, "-f", "null", "-"], capture_output=True, text=True)
        psnr = re.search(r"PSNR .*average:(\S+)", proc.stderr)
        ssim = re.search(r"SSIM .*All:(\S+)", proc.stderr)
        if proc.returncode != 0 or not psnr or not ssim:
            log.warning("Video quality analysis failed (%d frames decoded): %s", decoded, proc.stderr[-300:])
            return decoded, [], {}
        frames = {}
        for path, key, field_name in ((psnr_log, "psnr_db", "psnr_avg"), (ssim_log, "ssim", "All")):
            with open(path) as handle:
                for line in handle:
                    fields = dict(item.split(":", 1) for item in line.split() if ":" in item)
                    frames.setdefault(int(fields["n"]), {})[key] = float(fields[field_name])
            os.remove(path)
        rows = [{"frame": n, **values} for n, values in sorted(frames.items())]
        with open(csv_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["frame", "psnr_db", "ssim"])
            writer.writeheader()
            writer.writerows(rows)
        return decoded, rows, {"psnr": float(psnr.group(1)), "ssim": float(ssim.group(1))}

    def _summarize(self, paradigm, rows, totals, decoded, frames, overall) -> StreamSummary:
        def mean(values):
            values = [v for v in values if v is not None]
            return round(sum(values) / len(values), 3) if values else None

        capped = [min(f["psnr_db"], PSNR_CAP_DB) for f in frames if "psnr_db" in f and not math.isnan(f["psnr_db"])]
        loss = (1 - totals["received"] / totals["sent"]) * 100 if totals["sent"] else 0.0
        return StreamSummary(
            paradigm=paradigm, duration_s=round(totals["duration"], 2),
            mean_rtt_ms=mean(r["rtt_ms"] for r in rows), mean_jitter_ms=mean(r["jitter_ms"] for r in rows),
            mean_throughput_mbps=mean(r["throughput_mbps"] for r in rows) or 0.0,
            packets_sent=totals["sent"], packets_received=totals["received"],
            packet_loss_pct=round(max(0.0, loss), 3), frames_reference=self.decoded_frames(self.video_path),
            frames_decoded=decoded, psnr_db=overall.get("psnr"), ssim=overall.get("ssim"),
            psnr_frame_mean_db=round(sum(capped) / len(capped), 3) if capped else None,
            ssim_frame_mean=mean(f.get("ssim") for f in frames))


# -- re-scoring a finished run ---------------------------------------------------------------
SCORE_FIELDS = ("frames_reference", "frames_decoded", "psnr_db", "ssim", "psnr_frame_mean_db", "ssim_frame_mean")


def rescore_dir(directory: str, video_path: str = DEFAULT_VIDEO) -> dict | None:
    """Re-score <paradigm dir>/received.ts; update frames.csv and summary.json. Returns the new score fields."""
    received, summary_path = os.path.join(directory, "received.ts"), os.path.join(directory, "summary.json")
    if not (os.path.exists(received) and os.path.exists(summary_path)):
        return None
    evaluator = VideoEvaluator.__new__(VideoEvaluator)          # no network needed for scoring
    evaluator.video_path = video_path
    decoded, frames, overall = evaluator.frame_quality(received, os.path.join(directory, "frames.csv"))
    capped = [min(f["psnr_db"], PSNR_CAP_DB) for f in frames if "psnr_db" in f and not math.isnan(f["psnr_db"])]
    ssims = [f["ssim"] for f in frames if "ssim" in f]
    scores = {"frames_reference": evaluator.decoded_frames(video_path), "frames_decoded": decoded,
              "psnr_db": overall.get("psnr"), "ssim": overall.get("ssim"),
              "psnr_frame_mean_db": round(sum(capped) / len(capped), 3) if capped else None,
              "ssim_frame_mean": round(sum(ssims) / len(ssims), 3) if ssims else None}
    with open(summary_path) as handle:
        summary = json.load(handle)
    summary.update(scores)
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    return scores


def needs_rescore(directory: str) -> bool:
    try:
        with open(os.path.join(directory, "summary.json")) as handle:
            summary = json.load(handle)
    except (OSError, ValueError):
        return False
    return summary.get("psnr_db") is None and os.path.exists(os.path.join(directory, "received.ts")) \
        and os.path.getsize(os.path.join(directory, "received.ts")) > 0


def rescore_run(run_dir: str, only_missing: bool = True, workers: int = 4) -> int:
    """Re-score every paradigm of a run (by default only those without a PSNR); also updates run.json."""
    from concurrent.futures import ProcessPoolExecutor
    import glob
    directories = sorted(d for d in glob.glob(os.path.join(run_dir, "scenario_*", "*"))
                         if os.path.isdir(d) and (not only_missing or needs_rescore(d)))
    if not directories:
        return 0
    log.info("Re-scoring %d videos in %s with %s", len(directories), run_dir, VideoEvaluator.ffmpeg)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = dict(zip(directories, pool.map(rescore_dir, directories)))
    run_json = os.path.join(run_dir, "run.json")
    if os.path.exists(run_json):
        with open(run_json) as handle:
            meta = json.load(handle)
        for directory, scores in results.items():
            k, paradigm = os.path.basename(os.path.dirname(directory)).split("_", 1)[1], os.path.basename(directory)
            entry = meta.get("scenarios", {}).get(k, {}).get("paradigms", {}).get(paradigm)
            if entry and scores:
                entry.setdefault("summary", {}).update(scores)
        with open(run_json, "w") as handle:
            json.dump(meta, handle, indent=2, default=str)
    for directory, scores in results.items():
        log.info("  %s: PSNR %s dB, SSIM %s, %s frames", os.path.relpath(directory, run_dir),
                 (scores or {}).get("psnr_db"), (scores or {}).get("ssim"), (scores or {}).get("frames_decoded"))
    return len(directories)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Re-score the received videos of finished runs")
    parser.add_argument("--rescore", nargs="+", required=True, metavar="RUN_DIR")
    parser.add_argument("--all", action="store_true", help="re-score every video, not only those without a PSNR")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for run_dir in args.rescore:
        rescore_run(run_dir, not args.all, args.workers)
