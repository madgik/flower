## Rabbitmq with automatic configuration

### Build

In order to change the initial rabbitmq configuration, go to the `init.sh`.

To build a new image you must be on the project root `flower/`, then

```
docker build -t <USERNAME>/flower_rabbitmq:<IMAGETAG> -f rabbitmq/Dockerfile .
```

## Run

Then run with

```
docker run -d -p 5672:5672 --name <CONTAINERNAME> <USERNAME>/flower_rabbitmq:<IMAGETAG>
```
