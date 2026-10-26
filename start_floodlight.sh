#!/bin/bash
# Start the Floodlight controller (OpenFlow on 6653, REST API on 8080)
# Java 8: the JDK setup.sh installs (Fedora), else Ubuntu's openjdk-8 package
JAVA="$HOME/opt/jdk8/bin/java"
[ -x "$JAVA" ] || JAVA=/usr/lib/jvm/java-8-openjdk-amd64/bin/java
cd "$HOME/floodlight" && exec "$JAVA" -jar target/floodlight.jar
