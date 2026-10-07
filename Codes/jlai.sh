#!/usr/bin/env bash
set -e

n="${1:?usage: jlai.sh <part-number>}"
base=/home/isetbz/jlai

evince "$base/PDF-Files/Demystifying AI Sorcery (Part-$n).pdf" &
evince "$base/PDF-Files/Lab-AI (Part-$n).pdf" &

docker ps -aq | xargs -r docker rm -f

cd "$base/Docker"
docker compose -f "jlai$n.yml" down
docker compose -f "jlai$n.yml" up -d

sleep 3
firefox --private-window localhost:2468
