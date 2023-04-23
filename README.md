# Object Detection Model using YOLOV5 and Rabbitmq server
## The fundamental project aims to provide a summary of object detection using a pre-trained YOLO model from image and YouTube video links.

## Features

- The rabbitmq consumer server loads a pre-trained model, which is used to detect objects from image or video data.
- To run the rabbitmq server locally, a docker image is used. The pre-trained model is loaded into the consumer server, which can then detect objects from image or video data.
- By pushing payload as queue message to the consumer, the image or video data can be processed using a pre-trained model that is loaded into the rabbitmq consumer server. The rabbitmq server is run locally using a docker image.

## Technologies:

- Python3
- Rabbitmq
- Yolov5 for object detection

## Installation

This project requires [Python](https://www.python.org/) v3.11 to run.

Steps to run the project.

```sh
1. Clone the repositiory and open in your favourite terminal
2. run this commnad to install dependencies : pip3 install -r requirements.txt
3. run this commnad to start rabbitmq server locally : docker-compose up
4. Open new terminal and run the command to start consumer server : python3 app.py
5. Open rabbitmq GUI server in browser: http://localhost:15672/
6. Open Queue tab from the rabbitmq server and publish message as shown in screenshot below
   message format: {
                      "type": "image",
                      "url": "https://github.com/ultralytics/yolov5/raw/master/data/images/zidane.jpg",
                      "output_location": "results"
                    }
7. See the saved image/video with the overlayed detection in the output location
```
[![image.png](https://i.postimg.cc/5t4NvFW8/image.png)](https://postimg.cc/3kqTQNRw)
