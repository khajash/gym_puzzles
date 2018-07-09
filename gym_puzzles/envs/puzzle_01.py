import sys, math
import numpy as np

import Box2D
from Box2D.b2 import (polygonShape, circleShape, staticBody, dynamicBody, vec2, fixtureDef, contactListener)

import gym
from gym import spaces
from gym.utils import colorize, seeding

import my_gym
import pyglet
from pyglet import gl
# import pygame

# This is an environment in which an octagon robot assembles a simple puzzle
#
# Reward: There is a living penalty of -0.05. 
#
# State:
#
# To solve the game you need to get ?? points in ??? time steps.
#
# Created by Kate Hajash.

box_space = False

FPS    = 50
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

# INITIAL_RANDOM = 5

VIEWPORT_W, VIEWPORT_H = 640, 480
BORDER 	= 10			# border around screen to avoid placing blocks

# ROBOT SETTINGS
FR 		= 0.999 	# friction (between bodies)
RES 	= 0			# restitution 
DAMP 	= 5.0		# damping
DENSE 	= 5.0		# density of blocks
SPEED 	= 30/SCALE	# speed of robot agent

# PRECISION FOR BLOCKS
EPSILON = 0.5
ANG_EPSILON = 0.1

# REWARD STRUCTURE
LIVING_PENALTY = -0.01
BLOCK_REWARD = 10
FINAL_REWARD = 1000

S = 2 # scale of agents and blocks
AGENT_POLY = [
	(1/S,0), (2/S,0), (3/S,1/S), (3/S,2/S), 
	(2/S,3/S), (1/S,3/S), (0,2/S), (0,1/S)
	]

block_color 	= (0.5, 0.5, 0.5)
agent_color 	= (1., 1., 1.)
cp_color 		= (1., 1., 1.)
final_color 	= (58./255, 153./255, 1)

COLORS = {
	'agent'		: agent_color,
	't_block'	: block_color,
	'l_block'	: block_color,
	'i_block'	: block_color,
	'cp'		: cp_color,
	'final_pt'	: final_color,
	'wall'		: (0.2, 0.2, 0.2)
}

PUZZLE_REL_LOCATION = {
	't_block' : ((0., 0.75), 0.5*np.pi),
	# 'l_block' : ((-2./3., -2./3.), 0.5*np.pi),
	# 'i_block' : ((1., -0.5), 0.),
}

# DISCRETE ACTIONS
# ACTION_DICT = {
# 	0 : ((0, 1), "North"),
# 	1 : ((1, 1), "NorthEast"),
# 	2 : ((1, 0), "East"),
# 	3 : ((1, -1), "SouthEast"),
# 	4 : ((0, -1), "South"),
# 	5 : ((-1, -1), "SouthWest"),
# 	6 : ((-1, 0), "West"),
# 	7 : ((-1, 1), "NorthWest"),
# }

ACTION_DICT = {
	0 : ((0, 1), "North"),
	1 : ((1, 0), "East"),
	2 : ((0, -1), "South"),
	3 : ((-1, 0), "West"),
	# 4 : (-1, "Turn Right"),
	# 5 : (1, "Turn Left"), 
}

labels = False # make true to see overlay

# class ContactDetector(contactListener):
	# def __init__(self, env):
	#     contactListener.__init__(self)
	#     self.env = env
	# def BeginContact(self, contact):
	#     if self.env.hull==contact.fixtureA.body or self.env.hull==contact.fixtureB.body:
	#         self.env.game_over = True
	#     for leg in [self.env.legs[1], self.env.legs[3]]:
	#         if leg in [contact.fixtureA.body, contact.fixtureB.body]:
	#             leg.ground_contact = True
	# def EndContact(self, contact):
	#     for leg in [self.env.legs[1], self.env.legs[3]]:
	#         if leg in [contact.fixtureA.body, contact.fixtureB.body]:
	#             leg.ground_contact = False

###### HELPER FUNCTIONS #######################################################################

def set_final_loc(screen_wd, screen_ht, block_rel_dict):
	'''
	adjust location of blocks to center of screen
	'''
	screen_x = screen_wd//2 # Calculate centerpoint
	screen_y = screen_ht//2
	puzzle_abs_loc = {}

	for key, rel_loc in block_rel_dict.items(): ## CHECK PPM?
		puzzle_abs_loc[key] = (
			screen_x + rel_loc[0][0]*SCALE, # adjust x, y to screen
			screen_y + rel_loc[0][1]*SCALE, 
			rel_loc[1])

	return puzzle_abs_loc

def distance(pt1, pt2):
    x, y = [(a-b)**2 for a, b in zip(pt1, pt2)]
    return (x+y)**0.5


###### ENV CLASS #######################################################################

class Puzzle(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
		'video.frames_per_second' : FPS
	}

	def __init__(self):
		self._seed()
		self.viewer = None

		self.world = Box2D.b2World(gravity=(0,0), doSleep=False)

		self.agent = None
		self.block_names = ['t_block'] # 'l_block', 'i_block']
		self.blocks = None
		self.blks_in_place = 0
		self.prev_blks_in_place = 0
		self.boundary = None
		# self.reward = 0

		self.block_final_pos = set_final_loc(VIEWPORT_W, VIEWPORT_H, PUZZLE_REL_LOCATION)
		self.block_distance = {}

		# print(self.block_final_pos)
		# Define Observation Boundaries
		self.theta_threshold = 2*np.pi

		high = np.array([
			np.inf, np.inf, self.theta_threshold, # Block 1
			# np.inf, np.inf, self.theta_threshold, # Block 2
			# np.inf, np.inf, self.theta_threshold, # Block 3
			# np.inf, np.inf, self.theta_threshold, # Agent
			]) 

		print("initialize...")
		if box_space:
			self.action_space = spaces.Box(np.array([-1,-1,-1]), np.array([+1,+1,+1]))
		else:
			self.action_space = spaces.Discrete(6) # 45 degree angles of direction
		self.observation_space = spaces.Box(-high, high)
		
		# self.prev_shaping = None
		self._reset()

	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def _destroy(self):
		if not self.blocks: return # if NONE then skip reset
		for block in self.blocks:
			self.world.DestroyBody(block)
		self.blocks = []
		for bound in self.boundary:
			self.world.DestroyBody(bound)
		self.boundary = []
		# self.world.DestroyBody(self.agent)
		# self.agent = None

	def _generate_boundary(self):
		self.boundary = []
		# pass
		borders = [(0, 1/2), (1, 1/2), (1/2, 0), (1/2, 1)]
		for i, border in enumerate(borders):
			if i < 2:
				box_shape = [1, VIEWPORT_H/SCALE]
			else:
				box_shape = [VIEWPORT_W/SCALE, 1]

			wall = self.world.CreateStaticBody(
				position=(VIEWPORT_W/SCALE*border[0], VIEWPORT_H/SCALE*border[1]),
				fixtures = fixtureDef(
					shape=polygonShape(box=(box_shape)),
					),
				userData = 'wall'
				)
			self.boundary.append(wall)

		# agent_body = world.CreateStaticBody(position=(10, 5), shapes=agent_hex)


	def _calculate_distance(self):

		for block in self.blocks:
			# print("block: %s, final_pos: %s, current_loc: %s" % (
				# block.userData, self.block_final_pos[block.userData][:2], block.worldCenter*SCALE))
			self.block_distance[block.userData] = distance(
				block.worldCenter*SCALE, 
				self.block_final_pos[block.userData][:2])
		# print("calculated distance:", self.block_distance)


	def _generate_blocks(self):
		
		self.blocks = []

		for i, block in enumerate(self.block_names):
			# x = np.random.uniform(BORDER, VIEWPORT_W/SCALE-BORDER)
			# y = np.random.uniform(BORDER, VIEWPORT_H/SCALE-BORDER)
			block = self.world.CreateDynamicBody(
				position=(VIEWPORT_W/SCALE - 10, 
					VIEWPORT_H/SCALE - 10), 
				angle=0.5*np.pi, 
				# angle=np.random.uniform(0, 2*np.pi), 
				# angle=-np.pi*2, # 0.5*pi = 90 degree CCW (polar)
				linearDamping=DAMP, 
				angularDamping = DAMP,
				userData=block
				)
			print("block starting position: %s" % block.position)

			if i == 0: # t_block
				t_box = block.CreatePolygonFixture(
					box=(1/S, 1/S, (-1/S, 0.),0), 
					density=DENSE, 
					friction=FR, 
					restitution=RES)
				t_box2 = block.CreatePolygonFixture(
					box=(1/S, 3/S, (1/S, 0.),0), 
					density=DENSE, 
					friction=FR, 
					restitution=RES)
			
			elif i == 1: # l_block
				l_box = block.CreatePolygonFixture(
					box=(1/S, 1/S, (1/S, 0.5/S), 0), 
					density=DENSE,
					friction=FR, 
					restitution=RES)
				l_box2 = block.CreatePolygonFixture(
					box=(1/S, 2/S, (-1/S, -0.5/S), 0), 
					density=DENSE, 
					friction=FR, 
					restitution=RES)
			
			else: # i_block
				i_box = block.CreatePolygonFixture(
					box=(1/S, 2/S), 
					density=DENSE, 
					friction=FR, 
					restitution=RES)

			self.blocks.append(block)
		self._calculate_distance()

	def is_in_place(self, x, y, angle, block):
		
		f_x, f_y, f_angle = self.block_final_pos[block.userData]
		# f_x /= SCALE
		# f_y /= SCALE
		# print('is_in_place:')
		# print("final_position:", f_x, f_y, f_angle)
		# print("current_loc:", x, y, angle)

		if abs(f_x/SCALE - x) > EPSILON:
			return False
		if abs(f_y/SCALE - y) > EPSILON:
			return False
		if block.userData == 'i_block':
			diff = abs(f_angle-angle) % np.pi
			if  diff > ANG_EPSILON:
				return False
		elif abs(f_angle-angle) > ANG_EPSILON:
			return False

		return True

	def _reset(self):
		self._destroy()
		# self.world.contactListener_bug_workaround = ContactDetector(self)
		# self.world.contactListener = self.world.contactListener_bug_workaround
		self.game_over = False
		# self.prev_shaping = None
		# self.scroll = 0.0
		# self.lidar_render = 0

		W = VIEWPORT_W/SCALE
		H = VIEWPORT_H/SCALE

		init_x = W/2
		init_y = H/2

		# print("Destroyed blocks")

		self._generate_blocks()

		# print("generated blocks")
		# print(self.blocks)

		# self.agent = self.world.CreateDynamicBody(
		# 	position = (init_x, init_y),
		# 	fixtures = fixtureDef(
		# 		shape=polygonShape(vertices=[(x,y) for x,y in AGENT_POLY])),
		# 	angle=0, 
		# 	linearDamping=DAMP, 
		# 	angularDamping=DAMP,
		# 	userData='agent',
		# 	)

		self._generate_boundary()

		self.drawlist = self.boundary + self.blocks # + [self.agent]

		# print(self.action_space)
		if box_space:
			return self._step((0.5, 0, 0))[0]
		else:
			return self._step(0)[0]

	def _step(self, action):
		
		# print(action)
		# TAKE Action 
		if box_space:
			x, y, turn= action[0], action[1], action[2] # CONTINUOUS BOX
		
		else:
			assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
			if action < 4:
				x, y = ACTION_DICT[action][0] # DISCRETE
				turn = 0
			else:
				x, y = 0, 0
				turn = ACTION_DICT[action][0]
		# print(ACTION_DICT[action][1])		


		# print(action)
		# print(self.action_space)
		for block in self.blocks:
			block.linearVelocity = x * SPEED, y * SPEED
			block.angularVelocity = turn 
		

		# PROGRESS one timestep
		self.world.Step(1.0/FPS, 6*30, 2*30)

		# RETRIEVE Block locations + SET STATE
		state = []
		in_place = []
		# state = [self.agent.worldCenter[0], 
		# 	self.agent.worldCenter[1], 
		# 	self.agent.angle % (2*np.pi)]

		for block in self.blocks:
			x, y = block.worldCenter # is actual world center - unscaled
			angle = block.angle % (2*np.pi)
			state.extend([x, y, angle])
			in_place.append(self.is_in_place(x, y, angle, block))

		assert len(state)==3

		# self.scroll = pos.x - VIEWPORT_W/SCALE/5

		# shaping  = 130*pos[0]/SCALE   # moving forward is a way to receive reward (normalized to get 300 on completion)
		# shaping -= 5.0*abs(state[0])  # keep head straight, other than that and falling, any behavior is unpunished

		# CALCULATE rewards
		reward = 0
		
		prev_distance = self.block_distance.copy()
		# print(prev_distance)

		self._calculate_distance()
		
		# DISTANCE PENALTY
		for block, dist in self.block_distance.items():
			# print(prev_distance[block])
			# print(dist)
			if dist > prev_distance[block]:
				reward -= 1.
				# print('block further away')
			elif dist < prev_distance[block]:
				reward += 1.
				# print('block closer!!!!')
			else:
				reward -= 0.5
				# print('block didn\'t move')
			# reward += LIVING_PENALTY * dist/SCALE
			# print(dist/SCALE)

		# agent_penalty = LIVING_PENALTY * distance(self.agent.worldCenter, (VIEWPORT_W/SCALE/2, VIEWPORT_H/SCALE/2))
		# print("agent penalty: ", agent_penalty)
		# reward += agent_penalty


		# CHECK if DONE
		done = False

		# CALCULATE new blocks in place
		self.prev_blks_in_place = self.blks_in_place 
		self.blks_in_place = 0
		for complete in in_place:
			if complete == True:
				self.blks_in_place += 1

		# ASSIGN new reward
		reward += (self.blks_in_place-self.prev_blks_in_place) * BLOCK_REWARD 
			# -10 for moving a block out of place
			# +10 for moving a block in place
			# intended to avoid robot moving a block in and out of place to take get continuous reward

		if self.blks_in_place == 1:
			done = True
			reward += FINAL_REWARD

		return np.array(state), reward, done, {}

	def _render(self, mode='human', close=False):
		if close:
			if self.viewer is not None:
				self.viewer.close()
				self.viewer = None
			return

		from gym.envs.classic_control import rendering
		
		if self.viewer is None:
			self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
			if labels:
				for block in self.blocks:
					if block.userData == 't_block':
						self.T_block_label = pyglet.text.Label('T_block', font_size=10,
			                x=block.worldCenter[0]*SCALE, y=block.worldCenter[1]*SCALE, anchor_x='center', anchor_y='center',
			                # color=(0, 0, 0, 0))
			                color=(255,255,255,255))
					elif block.userData == 'l_block':
						self.L_block_label = pyglet.text.Label('L_block', font_size=10,
			                x=block.worldCenter[0]*SCALE, y=block.worldCenter[1]*SCALE, anchor_x='center', anchor_y='center',
			                # color=(0, 0, 0, 0))
			                color=(255,255,255,255))

					else:
						self.I_block_label = pyglet.text.Label('I_block', font_size=10,
			                x=block.worldCenter[0]*SCALE, y=block.worldCenter[1]*SCALE, anchor_x='center', anchor_y='center',
			                # color=(0, 0, 0, 0))
			                color=(255,255,255,255))
		self.viewer.set_bounds(0, VIEWPORT_W/SCALE, 0, VIEWPORT_H/SCALE)

		# DRAW BACKGROUND
		self.viewer.draw_polygon( [
			(0,					0),
			(VIEWPORT_W/SCALE,	0),
			(VIEWPORT_W/SCALE,	VIEWPORT_H/SCALE),
			(0, 				VIEWPORT_H/SCALE),
			], color=(0., 0., 0.) )

		# DRAW OBJECTS
		for obj in self.drawlist:
			for f in obj.fixtures:
				trans = f.body.transform
				path = [trans*v for v in f.shape.vertices]
				self.viewer.draw_polygon(path, color=COLORS[obj.userData])
			
			# DRAW CP
			if 'block' in obj.userData:
				x, y = obj.worldCenter
				t = rendering.Transform(translation=(x, y))
				self.viewer.draw_circle(0.05, 30, color=cp_color).add_attr(t)

		# DRAW FINAL POINTS
		for f_loc in self.block_final_pos.values():
			t = rendering.Transform(translation=(f_loc[0]/SCALE, f_loc[1]/SCALE))
			self.viewer.draw_circle(0.05, 30, color=final_color).add_attr(t)

		# DRAW LABELS >> NEED TO FIGURE OUT LABELLING!!!
		if labels:
			self.T_block_label.draw()
			self.L_block_label.draw()
			self.I_block_label.draw()

			self.viewer.window.flip() # flips between screen with labels and real screen -- not overlayed

		return self.viewer.render(return_rgb_array = mode=='rgb_array')



def run_random_actions():
	env.reset()
	print("completed reset")
	reward_sum = 0
	num_games = 10
	num_game = 0
	while num_game < num_games:
		env.render()
		observation, reward, done, _ = env.step(env.action_space.sample())
		reward_sum += reward
		# print(reward_sum)
		if done:
			print("Reward for this episode was: {}".format(reward_sum))
			reward_sum = 0
			num_game += 1
			env.reset()
		if escape: break

	env.render(close=True)


if __name__=="__main__":
	
	from pyglet.window import key
	
	escape = False
	def key_press(k, mod):
		global escape
		if k==key.ESCAPE: 
			escape = True
			print("escape")
		if k==key.LEFT: 
			escape = True
			print("Left")

	# env = gym.make("RobotPuzzle-v0")
	env = RobotPuzzle()
	# env.render()
	env.reset()
	env.render()
	env.viewer.window.on_key_press = key_press
	
	print("completed reset")
	reward_sum = 0
	num_games = 10
	num_game = 0
	while num_game < num_games:
		env.render()
		observation, reward, done, _ = env.step(env.action_space.sample())
		reward_sum += reward
		# print(reward_sum)
		if done:
			print("Reward for this episode was: {}".format(reward_sum))
			reward_sum = 0
			num_game += 1
			env.reset()
		if escape: break


	env.render(close=True)
	
