# chat/consumers.py
from channels.generic.websocket import WebsocketConsumer
import json
from core import chat

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
            self.send(text_data=json.dumps({
                'message': '나  : ' +  message
            }))
            self.send(text_data=json.dumps({
                'message': '코에 : ' + chatbot._do_reply(message)
            }))
