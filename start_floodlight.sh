#!/bin/bash
# start_floodlight.sh - the Floodlight controller (OpenFlow on 6653, REST API on 8080), built by setup.sh in
# ~/floodlight; Floodlight runs only on Java 8.
cd "$HOME/floodlight" && exec /usr/lib/jvm/java-8-openjdk-amd64/bin/java -jar target/floodlight.jar
