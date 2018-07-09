class safePlayer():
	def choose_action(self, observation):
		current_player = observation[0]
		to_call = observation[2]
		action = 0
		if to_call > 0:
			action = 1
		return action
	def store_transition(self, s, a, r, s_, current_player):
		pass
	def replace_transition(self, r, how_many_round_, how_many_round):
		pass
	def learn(self):
		pass