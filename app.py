import json
import sys

import pika

from model import YOLOv5


def consumer(queue_name: str):
    """

    :param queue_name: Name of the Queue
    :return: consumer server
    """
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host='localhost', port=5672))
    channel = connection.channel()

    channel.queue_declare(queue=queue_name)

    def callback(ch, method, properties, body):
        message = json.loads(body)
        print("Received message:", message)
        print("Model initialized !!!")
        yolo = YOLOv5()

        if message['type'] == 'image':
            image_url = message['url']
            output_location = message['output_location']
            yolo.set_object(image_url, message["type"], output_location)
            yolo.detect_and_save()
            print("Saved output !!!")

        elif message['type'] == 'video':
            video_url = message['url']
            output_location = message['output_location']
            yolo.set_object(video_url, message["type"], output_location)
            yolo.detect_and_save()
            print("Saved output !!!")

    channel.basic_consume(queue=queue_name, on_message_callback=callback, auto_ack=True)

    print('Waiting for messages. To exit, press CTRL+C')
    channel.start_consuming()


if __name__ == '__main__':
    try:
        consumer("object-detection")
    except KeyboardInterrupt:
        print('Interrupted')
        sys.exit(0)
