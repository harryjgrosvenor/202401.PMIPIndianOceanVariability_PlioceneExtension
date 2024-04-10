# Jupyter & Climate Docker Image

[Docker](https://www.docker.com/) image to load [JupyterHub](https://jupyter.org/hub) via [Conda](https://docs.conda.io/en/latest/), with a few [additional packages](environment.yml) useful for climate/data analysis.


## Installation/use

* [Install Docker desktop](https://www.docker.com/get-started)
* Ensure Docker desktop is running
* Navigate to your project directory (i.e. with Jupyter notebook files)
* Download published image:

```
docker pull chrisbrierley/pmip4_iovar:latest
```

* Run published image:

```
docker run -i --rm --volume=${PWD}:/home/docker/jupyter-climate -w /home/docker/jupyter-climate -p 8888:8888 -t chrisbrierley/pmip4_iovar:latest
```

Help:

* [Docker run help](https://docs.docker.com/engine/reference/commandline/run/)

Docker run options used:

* -i = interactive
* -t = allow pseudo tty
* -p 8888:8888 = publish container ports to host
* --rm = remove container on exit
* -w PATH = working directory inside the container


## Building & running image from scratch

* Clone repo & navigate inside:

```
git clone git@github.com:hannahwoodward/docker-jupyter-climate.git && cd docker-jupyter-climate
```

* Build image from Dockerfile (takes ~15 minutes):

```
docker build -t chrisbrierley/pmip4_iovar:latest .
```

* Navigate to your project directory (i.e. with Jupyter notebook files)
* Run locally built image:

```
docker run -i --rm --volume=${PWD}:/home/docker/jupyter-climate -w /home/docker/jupyter-climate -p 8888:8888 -t jupyter-climate
```

Help:

* [Docker build help](https://docs.docker.com/engine/reference/commandline/build/)

Docker build options used:

* -t = name/tag the image, format `name:tag`


## Publishing image

```
docker login
docker tag jupyter-climate chrisbrierley/pmip4_iovar:latest
docker push chrisbrierley/pmip4_iovar:latest
```


## Troubleshooting

* Exit code 137 - need to increase Docker memory e.g. to 4GB
* No space left on device - `docker system prune`
