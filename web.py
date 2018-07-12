#! /usr/bin/env python
# -*- coding:utf-8 -*-


import time
import json
from websocket import create_connection
# import websocket-client

# pip install websocket-client
def takeAction(action, data):
    if action == "__bet":
        #time.sleep(2)
        ws.send(json.dumps({
            "eventName": "__action",
            "data": {
                "action": "bet",
                "playerName": "ppp",
                "amount": 100
            }
        }))
    elif action == "__action":
        #time.sleep(2)
        ws.send(json.dumps({
            "eventName": "__action",
            "data": {
                "action": "allin",
                "playerName": "ppp"
            }
        }))


def doListen():
    try:
        global ws
        ws = create_connection("ws://poker-training.vtr.trendnet.org:3001")
        ws.send(json.dumps({
            "eventName": "__join",
            "data": {
                "playerName": "ppp"
            }
        }))
        while 1:
            result = ws.recv()
            msg = json.loads(result)
            event_name = msg["eventName"]
            data = msg["data"]
            print(event_name)
            print(data)
            takeAction(event_name, data)
    except Exception as e:
        print(e)
        doListen()


if __name__ == '__main__':
    doListen()
