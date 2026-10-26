"""
multimedia_evaluator.py - the AIRSDN measurements of a routing approach.

As in the paper (Section 4.2) and the paper's original experiment scripts, on the route currently in use:
  1. RTT and throughput are monitored for 60 s each (Figs. 16-17):
       ping -c 60 client -> server              -> per-second RTT
       iperf3 -R -t 60 (server sends to client) -> per-second throughput
  2. the 66 s H.264 video (assets/surreal.ts) is streamed server -> client with FFmpeg
     (-re -c copy over UDP) and compared with the original:
       PSNR and SSIM averages (FFmpeg psnr/ssim filters) and per-frame values
Results: timeseries.csv (t_s, rtt_ms, throughput_mbps), frames.csv, summary.json.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass

log = logging.getLogger(__name__)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_VIDEO = os.path.join(PROJECT_DIR, "assets", "surreal.ts")
VIDEO_PORT = 1234
IPERF_PORT = 5202


@dataclass
class Summary:
    approach: str
    mean_rtt_ms: float | None
    ping_loss_pct: float | None
    mean_throughput_mbps: float | None
    psnr_db: float | None
    ssim: float | None
    frames_received: int


class Evaluator:
    def __init__(self, net, spec, server_host: str, client_host: str, video_path: str = DEFAULT_VIDEO,
                 monitor_seconds: int = 60):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        self.net, self.spec = net, spec
        self.server_host, self.client_host = server_host, client_host
        self.video_path, self.monitor_seconds = video_path, monitor_seconds

    def run(self, approach: str, out_dir: str) -> Summary:
        os.makedirs(out_dir, exist_ok=True)
        rtts, ping_loss = self.monitor_rtt()
        throughput = self.monitor_throughput()
        with open(os.path.join(out_dir, "timeseries.csv"), "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["t_s", "rtt_ms", "throughput_mbps"])
            for second in range(1, self.monitor_seconds + 1):
                writer.writerow([second, rtts.get(second), throughput.get(second)])

        received = os.path.join(out_dir, "received.ts")
        self.stream_video(received, out_dir)
        frames, psnr, ssim = self.video_quality(received, os.path.join(out_dir, "frames.csv"))

        mean = lambda values: round(sum(values) / len(values), 3) if values else None
        summary = Summary(approach, mean(list(rtts.values())), ping_loss, mean(list(throughput.values())),
                          psnr, ssim, frames)
        with open(os.path.join(out_dir, "summary.json"), "w") as handle:
            json.dump(asdict(summary), handle, indent=2)
        log.info("[%s] RTT %s ms, throughput %s Mbps, PSNR %s dB, SSIM %s", approach, summary.mean_rtt_ms,
                 summary.mean_throughput_mbps, psnr, ssim)
        return summary

    # -- RTT and throughput (Figs. 16-17) -------------------------------------------------
    def monitor_rtt(self) -> tuple[dict[int, float], float | None]:
        client = self.net.get(self.client_host)
        output = client.cmd(f"ping {self.spec.hosts[self.server_host].ip} -c {self.monitor_seconds}")
        rtts = {int(m.group(1)): float(m.group(2)) for m in re.finditer(r"icmp_seq=(\d+) .*?time=([\d.]+) ms", output)}
        match = re.search(r"([\d.]+)% packet loss", output)
        return rtts, float(match.group(1)) if match else None

    def monitor_throughput(self) -> dict[int, float]:
        server, client = self.net.get(self.server_host), self.net.get(self.client_host)
        iperf_server = server.popen(["iperf3", "-s", "-1", "-p", str(IPERF_PORT)],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1)
        iperf = client.popen(["iperf3", "-c", self.spec.hosts[self.server_host].ip, "-p", str(IPERF_PORT), "-b", "0",
                              "-R", "-t", str(self.monitor_seconds), "-J"],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        try:
            stdout, _ = iperf.communicate(timeout=self.monitor_seconds + 120)
        except subprocess.TimeoutExpired:
            iperf.kill()
            stdout = ""
        if iperf_server.poll() is None:
            iperf_server.terminate()
        try:
            intervals = json.loads(stdout)["intervals"]
        except (ValueError, KeyError, TypeError):
            log.warning("iperf3 produced no result; throughput left empty")
            return {}
        return {round(i["sum"]["end"]): round(i["sum"]["bits_per_second"] / 1e6, 4) for i in intervals}

    # -- video ------------------------------------------------------------------------------
    def stream_video(self, received_path: str, out_dir: str) -> None:
        server, client = self.net.get(self.server_host), self.net.get(self.client_host)
        url = f"udp://{self.spec.hosts[self.client_host].ip}:{VIDEO_PORT}"
        with open(os.path.join(out_dir, "receiver.log"), "w") as rx_log, \
                open(os.path.join(out_dir, "sender.log"), "w") as tx_log:
            receiver = client.popen(["ffmpeg", "-hide_banner", "-y", "-i", url, "-c", "copy", received_path],
                                    stdout=rx_log, stderr=subprocess.STDOUT)
            time.sleep(1)
            sender = server.popen(["ffmpeg", "-hide_banner", "-re", "-i", self.video_path, "-c", "copy",
                                   "-f", "mpegts", url], stdout=tx_log, stderr=subprocess.STDOUT)
            sender.wait()
            time.sleep(2)                    # let in-flight packets arrive
            receiver.terminate()             # ffmpeg finalizes the file on SIGTERM
            try:
                receiver.wait(timeout=15)
            except subprocess.TimeoutExpired:
                receiver.kill()

    def video_quality(self, received_path: str, csv_path: str) -> tuple[int, float | None, float | None]:
        """Average and per-frame PSNR/SSIM of the received video (one FFmpeg decode pass)."""
        if not os.path.exists(received_path) or os.path.getsize(received_path) == 0:
            log.warning("No video received")
            return 0, None, None
        out_dir = os.path.dirname(csv_path)
        psnr_log, ssim_log = os.path.join(out_dir, "psnr.log"), os.path.join(out_dir, "ssim.log")
        graph = (f"[0:v]split[r1][r2];[1:v]split[d1][d2];"
                 f"[r1][d1]psnr=stats_file={psnr_log};[r2][d2]ssim=stats_file={ssim_log}")
        proc = subprocess.run(["ffmpeg", "-hide_banner", "-i", self.video_path, "-i", received_path,
                               "-lavfi", graph, "-f", "null", "-"], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg quality analysis failed: {proc.stderr[-500:]}")
        psnr = re.search(r"PSNR .*average:(\S+)", proc.stderr)
        ssim = re.search(r"SSIM .*All:(\S+)", proc.stderr)
        frames = {}
        for path, key, field_name in ((psnr_log, "psnr_db", "psnr_avg"), (ssim_log, "ssim", "All")):
            for line in open(path):
                fields = dict(item.split(":", 1) for item in line.split() if ":" in item)
                frames.setdefault(int(fields["n"]), {})[key] = float(fields[field_name])
            os.remove(path)
        with open(csv_path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["frame", "psnr_db", "ssim"])
            writer.writeheader()
            writer.writerows({"frame": n, **values} for n, values in sorted(frames.items()))
        return len(frames), float(psnr.group(1)) if psnr else None, float(ssim.group(1)) if ssim else None
