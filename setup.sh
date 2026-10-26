#!/usr/bin/env bash
# setup.sh - one-time setup on Ubuntu 24.04: emulator and tools, the Floodlight controller, the Python environment.
# Run as your normal user; the steps that need root use sudo.
set -euo pipefail
cd "$(dirname "$0")"

# Mininet (with mnexec), Open vSwitch, iperf3, FFmpeg (native H.264 decoder), JDK 8 + Maven for Floodlight
sudo apt-get update
sudo apt-get install -y mininet openvswitch-switch iperf3 ffmpeg iproute2 git curl python3-venv openjdk-8-jdk maven
sudo systemctl enable --now openvswitch-switch
sudo modprobe -a openvswitch sch_htb sch_netem

# Floodlight (master: /wm/routing/paths, /wm/staticentrypusher); builds only with JDK 8
FLOODLIGHT="$HOME/floodlight"
if [ ! -f "$FLOODLIGHT/target/floodlight.jar" ]; then
    [ -d "$FLOODLIGHT" ] || git clone https://github.com/floodlight/floodlight.git "$FLOODLIGHT"
    (
        cd "$FLOODLIGHT"
        git submodule update --init
        export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64 PATH="/usr/lib/jvm/java-8-openjdk-amd64/bin:$PATH"
        # openflowj 3.3.0-SNAPSHOT is no longer published; install the jar bundled with Floodlight
        mvn -q -B install:install-file -Dfile=lib/openflowj-3.3.0-SNAPSHOT.jar \
            -DgroupId=org.projectfloodlight -DartifactId=openflowj -Dversion=3.3.0-SNAPSHOT -Dpackaging=jar
        mvn -q -B package -Dmaven.test.skip=true
    )
fi

# Python environment (exact versions of the reported results)
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install --no-deps -r requirements.txt
.venv/bin/python -m pytest -q tests
echo "Setup complete. Start Floodlight with ./start_floodlight.sh, then run ./run_all.sh"
