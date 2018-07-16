#! /usr/bin/env python
# -*- coding:utf-8 -*-


import time
import json
from websocket import create_connection
import random
# import websocket-client

# pip install websocket-client
def takeAction(action, data):
    if action == "__bet":
        #time.sleep(2)
        ws.send(json.dumps({
            "eventName": "__action",
            "data": {
                "action": "bet",
                "playerName": "294da5cf6f00402d8549b9eba8e242ca",
                "amount": 100
            }
        }))
    elif action == "__action":
        #time.sleep(2)
        # rand_num = random.random()
        # if rand_num > 0.7:
        #     take_action = "allin"
        # else:
        #     take_action = "call"
        ws.send(json.dumps({
            "eventName": "__action",
            "data": {
                "action": "allin",
                "playerName": "294da5cf6f00402d8549b9eba8e242ca"
            }
        }))

    elif action == "__game_over":
        ws.send(json.dumps({
            "eventName": "__join",
            "data": {
                "playerName": "294da5cf6f00402d8549b9eba8e242ca"
            }
        }))
# http://poker-battle.vtr.trendnet.org:3001/#
# 

def doListen():
    try:
        global ws
        ws = create_connection("ws://poker-battle.vtr.trendnet.org:3001")
        ws.send(json.dumps({
            "eventName": "__join",
            "data": {
                "playerName": "294da5cf6f00402d8549b9eba8e242ca"
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
