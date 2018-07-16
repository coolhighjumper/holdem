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
import pandas as pd
# import websocket-client
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
        print('cards = ', result)
        one_hot_encoding = np.zeros(52)
        one_hot_encoding[result] += 1
    except Exception as e:
        print(cards)
        one_hot_encoding = np.zeros(52)
        
    return one_hot_encoding

def get_community_card(data):
    
    community_card = [Card.new(x[0]+x[1].lower()) for x in data['table']['board']]
    return community_card

def get_observation(data):
    global community_card
    # print(data)
    stack = data['self']['chips']
    # print('stack= ', stack)
    hand_cards = [Card.new(x[0]+x[1].lower()) for x in data['self']['cards']]
    # print('hand_cards= ', hand_cards)
    # print('community_card= ',community_card)
    
    cards = transferCard(hand_cards + hand_cards + community_card)
    # print('cards= ',cards)
    to_call = data['self']['minBet']
    # print('to_call= ',to_call)
    if len(community_card)==0:
        handrank = -1
    else:
        handrank = Evaluator().evaluate(community_card, hand_cards)
    # print('handrank= ',handrank)
    betting = data['self']['bet']
    # print('betting= ',betting)
    totalpot = betting
    for player in data['game']['players']:
        totalpot += player['bet']
    # print('totalpot= ',totalpot)
    return np.concatenate(([totalpot, to_call, stack - 3000, handrank, betting], cards))

# pip install websocket-client
def takeAction(action, data):
    global stack_result
    global next_action
    global observation
    global my_action
    global step
    global init_chip
    try:
        if action == "__bet":
            # #time.sleep(2)
            # step += 1
            # # print(step)
            # observation = get_observation(data)
            # # print('get_action')
            # # print(observation.shape)
            # my_action = RL2.choose_action(observation)
            # # print(action)
            # amount = 0
            # if my_action == 0:
            #     my_action = 'check'
            # elif my_action == 1:
            #     my_action = 'call'
            # elif my_action == 2:
            #     my_action = 'bet'
            #     amount = observation[1] + 10
            # elif my_action == 3:
            #     my_action == 'fold'
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
            # print('get_observation')
            step+=1
            next_action = True
            observation = get_observation(data)
            # print('get_action')
            # print(observation.shape)
            my_action = RL2.choose_action(observation)
            # print(action)
            amount = 0
            if my_action == 0:
                my_action = 'check'
                ws.send(json.dumps({
                    "eventName": "__action",
                    "data": {
                        "action": 'check',
                        "playerName": "294da5cf6f00402d8549b9eba8e242ca"
                    }
                }))
            elif my_action == 1:
                my_action = 'call'
                ws.send(json.dumps({
                    "eventName": "__action",
                    "data": {
                        "action": 'call',
                        "playerName": "294da5cf6f00402d8549b9eba8e242ca"
                    }
                }))
            elif my_action == 2:
                my_action = 'bet'
                amount = observation[1] + 10
                ws.send(json.dumps({
                    "eventName": "__action",
                    "data": {
                        "action": 'bet',
                        "amount": amount,
                        "playerName": "294da5cf6f00402d8549b9eba8e242ca"
                    }
                }))
                
            elif my_action == 3:
                my_action == 'fold'
                ws.send(json.dumps({
                    "eventName": "__action",
                    "data": {
                        "action": 'fold',
                        "playerName": "294da5cf6f00402d8549b9eba8e242ca"
                    }
                }))
            print(my_action)
            print(amount)
        elif action == "__deal":
            global community_card
            community_card = get_community_card(data)
            
            # print('community_card= ',community_card)
            # print(community_card)

        # elif action == "__new_peer":
        #     stack_result = []
        #     for i in range(len(data)):
        #         stack_result.append([])

        elif action == "__start_reload":
            ws.send(json.dumps({
                "eventName": "__reload",
            }))

        elif action == "__show_action" and next_action:
            pass
            # print('[info] store to memory')
            # next_action = False
            # for i, player in enumerate(data['players']):
            #     if player['playerName'] == 'f27f6f1c7c5cbf4e3e192e0a47b85300':
            #         player_data = player
            # observation_ = observation
            # # totalpot, to_call, stack - 3000, handrank, betting
            # observation_[0] = data['table']['totalBet']
            # observation_[1] = 0
            # observation_[2] = player_data['chips'] - 3000
            # observation_[4] = player_data['bet']
            # # print('observation_= ', observation)
            # RL2.store_transition(observation, my_action, [0,0,0,0], observation_, 0)



        elif action == "__round_end":
            print(data['players'])
            pass
            # for i, player in enumerate(data['players']):
            #     if player['playerName'] == 'f27f6f1c7c5cbf4e3e192e0a47b85300':
            #         chips = player['chips']
            #         print('chips= ',chips)
            #         stack_result.append(player['chips'])
            # print('[info] replace memory')
            # print(step)
            # if step == 0:
            #     pass
            # else:
            #     RL2.replace_transition(chips - init_chip, step-1, 0)
            # init_chip = chips
            # if step>10:
            #     print('[info] start to learn')
            #     RL2.learn()
            #     RL2.save_model()

        elif action == "__game_over":
            df = pd.DataFrame(stack_result)
            df.to_csv('./stacl_result/stack_'+step+'.csv')
            ws.send(json.dumps({
            "eventName": "__join",
            "data": {
                "playerName": "294da5cf6f00402d8549b9eba8e242ca"
            }
            }))
    except Exception as e:
        raise e
        


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
            # print(data)
            takeAction(event_name, data)
    except Exception as e:
        print('hello')
        doListen()


if __name__ == '__main__':
    community_card = []
    stack_result = []
    next_action = False
    observation = []
    my_action = 0
    step = 0
    init_chip = 3000
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
