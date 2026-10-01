#!/usr/bin/bash

evince /home/isetbz/jlai/PDF-Files/'Demystifying AI Sorcery (Part-1).pdf' &
evince /home/isetbz/jlai/PDF-Files/'Lab-AI (Part-1).pdf' & 
docker ps -aq | xargs -r docker stop | xargs -r docker rm &&
cd /home/isetbz/jlai/Docker && 
docker-compose -f jlai1.yml down && 
docker-compose -f jlai1.yml up -d && 
cd .. && 
firefox --private-window localhost:2468

