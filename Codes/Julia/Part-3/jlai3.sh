#!/usr/bin/bash

evince /home/isetbz/jlai/PDF-Files/'Demystifying AI Sorcery (Part-3).pdf' &
evince /home/isetbz/jlai/PDF-Files/'Lab-AI (Part-3).pdf' & 
docker ps -aq | xargs -r docker stop | xargs -r docker rm &&
cd /home/isetbz/jlai/Docker &&
docker-compose -f jlai3.yml down && 
docker-compose -f jlai3.yml up -d && 
cd .. &&
firefox --private-window localhost:2468 # 1234

