# Cat vs. Dog image classification API
## Running
Python 3.9 required
### Local
Install requirements using `pip install -r requirements.txt`

Run with `gunicorn -k uvicorn.workers.UvicornWorker -b <host>:<port> server.main:app`
### Using Docker
Build docker image:

`docker build -f Dockerfile.local -t <image_name> .`

Run container:

`docker run -p <host_machine_port>:8080 <image_name>`

## Usage
SwaggerUI available at `http://<host>:<port>/docs`

Environment variables used by application:
* **MODEL_ARCH** (Default: `resnet18`) - Enum for selecting model architecture. Possible values:
    * `resnet18`
    * `mobilenetv3`
* **REQUEST_MAX_ITEMS** (Default: `10`) - maximum number of images to process in one request