class safePlayer():
	def __init__(self):
		self.memory = []
	def choose_action(self, observation):
		to_call = observation[1]
		handRank = observation[3]
		action = 0
		if (to_call > 0 and to_call < 500) or handRank < 500:
			action = 1
		elif to_call>=500:
			action = 3
		elif handRank >= 500:
			action = 3
		# if handRank > 5000:
		# 	action = 3

		return action
	def store_transition(self, s, a, r, s_, current_player):
		pass
	def replace_transition(self, r, how_many_round_, how_many_round):
		pass
	def learn(self):
		pass