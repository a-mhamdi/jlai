FROM julia:latest

RUN apt update && apt upgrade -y

RUN apt install apt-utils gcc g++ build-essential\
	rsync wget vim tmux 

RUN apt clean

