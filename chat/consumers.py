# chat/consumers.py
from channels.generic.websocket import WebsocketConsumer
import json
from core import chat
from django.conf import settings
from .models import Chat


# override OUT_DIR
if settings.OUT_DIR is not None:
    chat.FLAGS.out_dir = settings.OUT_DIR

chatbot = chat.ChatBot()
default_hparams = chat.create_hparams(chat.FLAGS)
chatbot.nmt_main(chat.FLAGS, default_hparams)


class ChatConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()

    def disconnect(self, close_code):
        pass

    def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json['message']

        if (len(message) != 0):
            # self.send(text_data=json.dumps({
            #     'message': message
            # }))
            answer = chatbot._do_reply(message)
            self.send(text_data=json.dumps({
                'message': answer
            }))
            # saving chat message to database
            c = Chat(question=message, answer=answer, verify=Chat.VERIFY_NOT)
            c.save()
