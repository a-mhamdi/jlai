::The `docker-compose.yml` file runs both a `Jupyter notebook` and a `Julia` service, and allows to run `Julia` from within the `Jupyter Notebook`.

start "" c:\jlai\"Lab-AI (Part-1)".pdf
start "" c:\jlai\"Lab-AI (Part-2)".pdf
start "" c:\jlai\"Lab-AI (Part-3)".pdf

::LAUNCH CHROME
start chrome /incognito 192.168.99.100:2468?token=jlai
::START DOCKER DEAMON
docker-machine start
::Destroy the container to liberate port.
docker-compose down
::Start a new container
docker-compose up
::RUN A CONTAINER "raia" FROM IMAGE "jlai"
:: `docker run --rm --name raia -d -p 2468:2468 jlai`
::EXIT UPON COMPLETION
exit
