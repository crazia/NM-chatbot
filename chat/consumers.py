# chat/consumers.py
from channels.generic.websocket import WebsocketConsumer
import json
from core import chat

class ChatConsumer(WebsocketConsumer, chat.ChatBot):

    def __init__(self):
        print('init baby')
    
    def connect(self):
        self.accept()

    def disconnect(self, close_code):
        pass

    def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json['message']
        
        self.send(text_data=json.dumps({
            'message': message
        }))
