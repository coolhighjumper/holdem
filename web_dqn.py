#! /usr/bin/env python
# -*- coding:utf-8 -*-


import time
import json
from websocket import create_connection
import math
import holdem.utils
import numpy as np
import holdem
from treys import Card, Evaluator
import holdem.DQN as DQN
import tensorflow as tf
# import websocket-client

community_card = []
class action_table:
    CHECK = 0
    CALL = 1
    RAISE = 2
    FOLD = 3

def transferCard(cards):
    try:
        result = []
        for i in cards:
            rank = Card.get_rank_int(i)
            suit = Card.get_suit_int(i)
            suit = int(math.log2(suit))
            result.append(suit * 13 + rank)
        one_hot_encoding = np.zeros(52)
        one_hot_encoding[result] += 1
    except Exception as e:
        print(cards)
        one_hot_encoding = np.zeros(52)
        
    return one_hot_encoding

def get_observation(data):
    stack = data['self']['chips']
    print('stack= ', stack)
    hand_cards = [Card.new(x[0]+x[1].lower()) for x in data['self']['cards']]
    print('community_card= ',community_card)
    print('hand_cards= ', hand_cards)
    cards = transferCard(hand_cards * 2 + community_card)
    print('cards= ',cards)
    to_call = data['self']['minBet']
    print('to_call= ',to_call)
    if len(community_card)==0:
        handrank = -1
    else:
        handrank = Evaluator().evaluate(community_card, hand_cards)
    print('handrank= ',handrank)
    betting = data['self']['bet']
    print('betting= ',betting)
    totalpot = data['self']['roundBet']
    print('totalpot= ',totalpot)
    return np.concatenate(([totalpot, to_call, stack - 3000, handrank, betting], cards))

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
        # print('get_observation')
        observation = get_observation(data)
        # print('get_action')
        print(observation.shape)
        action = RL2.choose_action(observation)
        print(action)
        amount = 0
        if action == 0:
            action = 'check'
        elif action == 1:
            action = 'call'
        elif action == 2:
            action = 'raise'
        elif action == 3:
            action == 'fold'
        ws.send(json.dumps({
            "eventName": "__action",
            "data": {
                "action": action,
                "playerName": "ppp"
            }
        }))
    elif action == "__deal":
        community_card = [Card.new(x[0]+x[1].lower()) for x in data['table']['board']]
        # print(community_card)

    # elif action == "__new_peer":
    #     stack_result = []
    #     for i in range(len(data)):
    #         stack_result.append([])

    # elif action == "__start_reload" and stack == 0:
    #     ws.send(json.dumps({
    #         "eventName": "__reload",
    #     }))

    elif action == "__show_action":
        pass


    elif action == "__round_end":
        pass
        # for i, player in enumerate(data['players']):
        #     stack_result[i].append(player['chips'])



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
            # print(data)
            takeAction(event_name, data)
    except Exception as e:
        print('hello')
        doListen()


if __name__ == '__main__':
    community_card = []
    tf.reset_default_graph()
    RL2 = DQN.DeepQNetwork(4, 57,
                      learning_rate=0.00001,
                      reward_decay=0.999,
                      e_greedy=0.9,
                      replace_target_iter=10000,
                      memory_size=50,
                      output_graph=True, nickname='2'
                      )
    RL2.load_model()
    doListen()
