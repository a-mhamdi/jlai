cd %USERPROFILE%\jlai

REM The `docker-compose.yml` file runs both a `Jupyter Lab` and a `Pluto` services, and allows to run `Julia` from within the `Jupyter Notebook`.

start "" "Lab-AI (Part-1)".pdf
start "" "Lab-AI (Part-2)".pdf
start "" "Lab-AI (Part-3)".pdf

REM LAUNCH CHROME
REM Run `pluto` on port `1234`
:: start chrome /incognito 192.168.99.100:1234
REM Run `Jupyter Lab` on port `2468`
start chrome /incognito 192.168.99.100:2468
REM START DOCKER DEAMON
docker-machine start
REM Destroy the containers to liberate ports.
docker-compose down
REM Start new containers
docker-compose up -d

echo Please, type `docker-compose down` before leaving.

REM RUN A CONTAINER "raia" FROM IMAGE "abmhamdi/jlai"
:: `docker run --rm --name raia -d -p 2468:2468 abmhamdi/jlai`

REM EXIT UPON COMPLETION
:: exit

