"""
udp_probe.py - connectionless UDP probe for per-second loss and jitter.

The sender emits fixed-rate datagrams carrying a sequence number and a send timestamp; the
receiver logs every arrival. Mininet hosts share one clock, so one-way transit times are exact.
Unlike iperf3 there is no TCP control connection to set up, so the probe also works on paths
with seconds of queueing delay and heavy loss (where iperf3's handshake fails).

Run inside a Mininet host namespace (multimedia_evaluator starts both ends):
    python udp_probe.py recv --port 5201 --duration 75 --out probe_recv.csv
    python udp_probe.py send --dst 10.0.0.9 --port 5201 --rate 25 --size 1000 --duration 62 --out probe_sent.csv
"""
from __future__ import annotations

import argparse
import csv
import socket
import struct
import time

HEADER = struct.Struct("!Id")      # sequence number, send time (unix seconds)


def send(dst: str, port: int, rate: float, size: int, duration: float, out: str) -> None:
    if rate <= 0 or size < HEADER.size:
        raise ValueError("rate must be > 0 and size >= header size")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    padding = b"\0" * (size - HEADER.size)
    start = time.time()
    with open(out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seq", "send_ts"])
        seq = 0
        while True:
            due = start + seq / rate
            if due - start >= duration:
                break
            delay = due - time.time()
            if delay > 0:
                time.sleep(delay)
            now = time.time()
            try:
                sock.sendto(HEADER.pack(seq, now) + padding, (dst, port))
            except OSError:
                pass                     # a send error is just a lost probe packet
            writer.writerow([seq, f"{now:.6f}"])
            seq += 1


def receive(port: int, duration: float, out: str) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(0.5)
    deadline = time.time() + duration
    with open(out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seq", "send_ts", "recv_ts"])
        while time.time() < deadline:
            try:
                data, _ = sock.recvfrom(65535)
            except socket.timeout:
                continue
            now = time.time()
            if len(data) >= HEADER.size:
                seq, sent = HEADER.unpack_from(data)
                writer.writerow([seq, f"{sent:.6f}", f"{now:.6f}"])
            handle.flush()


def per_second(sent_rows, recv_rows, start: float, seconds: int) -> dict[int, dict]:
    """
    Per-second loss (packets sent in second t that never arrived) and RFC 3550 jitter
    (smoothed |transit difference| of consecutive arrivals, value at the end of second t).
    """
    received = {int(r["seq"]): (float(r["send_ts"]), float(r["recv_ts"])) for r in recv_rows}
    sent_by_second: dict[int, list[int]] = {}
    for row in sent_rows:
        second = int(float(row["send_ts"]) - start) + 1
        if 1 <= second <= seconds:
            sent_by_second.setdefault(second, []).append(int(row["seq"]))

    jitter_at: dict[int, float] = {}
    jitter, previous = 0.0, None
    for seq, (sent, arrived) in sorted(received.items(), key=lambda kv: kv[1][1]):
        transit = arrived - sent
        if previous is not None:
            jitter += (abs(transit - previous) * 1000 - jitter) / 16
        previous = transit
        second = int(sent - start) + 1
        if 1 <= second <= seconds:
            jitter_at[second] = jitter

    result, last_jitter = {}, None
    for second in range(1, seconds + 1):
        seqs = sent_by_second.get(second, [])
        lost = sum(1 for s in seqs if s not in received)
        last_jitter = jitter_at.get(second, last_jitter)
        result[second] = {"loss_pct": round(100 * lost / len(seqs), 3) if seqs else None,
                          "jitter_ms": round(last_jitter, 3) if last_jitter is not None else None}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("send")
    s.add_argument("--dst", required=True)
    s.add_argument("--port", type=int, default=5201)
    s.add_argument("--rate", type=float, default=25, help="packets per second")
    s.add_argument("--size", type=int, default=1000, help="datagram payload bytes")
    s.add_argument("--duration", type=float, required=True)
    s.add_argument("--out", required=True)
    r = sub.add_parser("recv")
    r.add_argument("--port", type=int, default=5201)
    r.add_argument("--duration", type=float, required=True)
    r.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.mode == "send":
        send(args.dst, args.port, args.rate, args.size, args.duration, args.out)
    else:
        receive(args.port, args.duration, args.out)
