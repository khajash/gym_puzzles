import sys, math
import numpy as np
from scipy import misc

import Box2D
from Box2D.b2 import (polygonShape, circleShape, staticBody, dynamicBody, vec2, fixtureDef, contactListener)

import gym
from gym import spaces
from gym.utils import colorize, seeding

import my_gym
import pyglet
from pyglet import gl

'''
This is an environment in which an octagon robot assembles a simple puzzle

square screen
TODO: it scaled??

STATE:
only relative information


REWARD:
uses soft forces
give reward for change in distance


To solve the game you need to get ?? points in ??? time steps.
Created by Kate Hajash.
'''

FPS    = 50
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well
DS = 4. # downsample

# (240, 320, 3)
VIEWPORT_W, VIEWPORT_H = int(640/DS), int(640/DS)
BORDER 	= 1		# border around screen to avoid placing blocks

# ROBOT SETTINGS
FR 		= 0.999 	# friction (between bodies)
RES 	= 0			# restitution 
DAMP 	= 5.0		# damping
DENSE 	= 5.0		# density of blocks
SPEED 	= 10/SCALE	# speed of robot agent

# PRECISION FOR BLOCKS IN PLACE
# EPSILON = 5.0/DS
EPSILON = 10.0/DS

ANG_EPSILON = 0.1

# REWARD STRUCTURE
BLOCK_REWARD = 10
FINAL_REWARD = 1000

# AGENT
S = 2*DS # scale of agents and blocks

AGENT_POLY = [
	(-0.5/S,-1.5/S), (0.5/S,-1.5/S), (1.5/S,-0.5/S), (1.5/S,0.5/S), 
	(0.5/S,1.5/S), (-0.5/S,1.5/S), (-1.5/S,0.5/S), (-1.5/S,-0.5/S)
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
	't_block' : ((0., 0.75/DS), 0.),
	# 'l_block' : ((-2./3., -2./3.), 0.5*np.pi),
	# 'i_block' : ((1., -0.5), 0.),
}


class ContactDetector(contactListener):
	def __init__(self, env):
		contactListener.__init__(self)
		self.env = env
	def BeginContact(self, contact):
		# if block and agent in touch 
		for block in self.env.blocks:
			# print(block.userData)
			if block in [contact.fixtureA.body, contact.fixtureB.body]:
				if self.env.agent in [contact.fixtureA.body, contact.fixtureB.body]:
					# print([contact.fixtureA.body.userData, contact.fixtureB.body.userData])
					block.agent_contact = True
					# print("block and agent in contact!")
		if self.env.agent in [contact.fixtureA.body, contact.fixtureB.body]:
			# print("env agent in contact")
			# print([contact.fixtureA.body.userData, contact.fixtureB.body.userData])
			if 'wall' in [contact.fixtureA.body.userData, contact.fixtureB.body.userData]:
				# print("env in contact with wall")
				self.env.wall_contact = True
	def EndContact(self, contact):
		for block in self.env.blocks:
			if block in [contact.fixtureA.body, contact.fixtureB.body]:
				if self.env.agent in [contact.fixtureA.body, contact.fixtureB.body]:
					block.agent_contact = False
		if self.env.agent in [contact.fixtureA.body, contact.fixtureB.body]:
			if 'wall' in [contact.fixtureA.body.userData, contact.fixtureB.body.userData]:
				self.env.wall_contact = False

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

def unitVector(bodyA, bodyB):
	Ax, Ay = bodyA.worldCenter
	Bx, By = bodyB.worldCenter
	denom = max(abs(Bx-Ax), abs(By-Ay))
	return ((Bx-Ax)/denom, (By-Ay)/denom)


###### ENV CLASS #######################################################################

class RobotPuzzle3(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array', 'state_pixels'],
		'video.frames_per_second' : FPS
	}

	downsample = 2 # for paperspace w/ lowres screen
	obs_type = 'low-dim'
	unitize = False

	def __init__(self, act_type='box', obs_depth=3, frameskip=4):
		"""Action type: 'box' or 'discrete'
		Observation type: 'image' or 'low-dim'
		observation depth: integer indicating how many images should be part of observation
		"""
		self._seed()
		self.viewer = None
		self._act_type = act_type
		self._obs_depth = obs_depth
		if self.obs_type == 'image': self.frameskip = frameskip
		else: self.frameskip = 1
		

		self.world = Box2D.b2World(gravity=(0,0), doSleep=False)

		self.agent = None
		self.block_names = ['t_block'] # 'l_block', 'i_block']
		self.blocks = None
		self.blks_vertices = {}
		self.blks_in_place = 0
		self.prev_blks_in_place = 0
		self.boundary = None
		# self.reward = 0

		self.block_final_pos = set_final_loc(VIEWPORT_W, VIEWPORT_H, PUZZLE_REL_LOCATION)
		self.agent_dist = {}
		self.block_distance = {}
		self.block_angle = {}
		self.wall_contact = False

		# print(self.block_final_pos)
		# Define Observation Boundaries
		self.theta_threshold = 2*np.pi

		a_obs = [np.inf, np.inf, self.theta_threshold, np.inf, np.inf] # np.inf, np.inf, np.inf]
		# a_obs = [np.inf, np.inf] # np.inf, np.inf, np.inf]

		vert_obs = [np.inf]*16
		# blk_obs =[np.inf, np.inf, np.inf, np.inf, np.inf, self.theta_threshold, np.inf] # Block 1 (rel location, rel theta, distance)
		blk_obs =[np.inf, np.inf, self.theta_threshold, np.inf, np.inf] # Block 1 (rel location, rel theta, distance)

		# high = np.array(a_obs + blk_obs + [np.inf])
		high = np.array(a_obs + vert_obs + blk_obs + [np.inf])
			# np.inf, np.inf, self.theta_threshold, # Block 2
			# np.inf, np.inf, self.theta_threshold, # Block 3
			 # contact boolean
			

		print("initialize...")

		self.action_space = spaces.Box(np.array([-1,-1, -1]), np.array([+1,+1, +1]), dtype=np.float32)
	
		self.observation_space = spaces.Box(-high, high, dtype=np.float32)
		self.state = []

		self._reset()


	def _seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def _destroy(self):
		if not self.blocks: return # if NONE then skip reset
		self.world.contactListener = None
		for block in self.blocks:
			self.world.DestroyBody(block)
		self.blocks = []
		for bound in self.boundary:
			self.world.DestroyBody(bound)
		self.boundary = []
		self.world.DestroyBody(self.agent)
		self.agent = None

	def _generate_boundary(self):
		self.boundary = []
		# pass
		borders = [(0, 1/2), (1, 1/2), (1/2, 0), (1/2, 1)]
		for i, border in enumerate(borders):
			if i < 2:
				box_shape = [1/DS, VIEWPORT_H/SCALE]
			else:
				box_shape = [VIEWPORT_W/SCALE, 1/DS]

			wall = self.world.CreateStaticBody(
				position=(VIEWPORT_W/SCALE*border[0], VIEWPORT_H/SCALE*border[1]),
				fixtures = fixtureDef(
					shape=polygonShape(box=(box_shape)),
					),
				userData = 'wall'
				)
			self.boundary.append(wall)


	def _calculate_distance(self):

		for block in self.blocks:
			# print("block: %s, final_pos: %s, current_loc: %s" % (
				# block.userData, self.block_final_pos[block.userData][:2], block.worldCenter*SCALE))
			self.block_distance[block.userData] = distance(
				block.worldCenter*SCALE, 
				self.block_final_pos[block.userData][:2])
			
			fangle = self.block_final_pos[block.userData][2]
			self.block_angle[block.userData] = abs(fangle %(2*np.pi) - abs(block.angle) % (2*np.pi))

			self.agent_dist[block.userData] = distance(
				self.agent.worldCenter*SCALE,
				block.worldCenter*SCALE)
		# print("calculated distance:", self.block_distance)

			
	def _generate_blocks(self):
		
		self.blocks = []

		for i, block in enumerate(self.block_names):
			x = np.random.uniform(BORDER, VIEWPORT_W/SCALE-2)
			y = np.random.uniform(BORDER, VIEWPORT_H/SCALE-2)
			# x = VIEWPORT_W/SCALE/2 
			# y = VIEWPORT_H/SCALE/2 - 1
			block = self.world.CreateDynamicBody(
				position = (x, y),
				angle=0, 
				# angle=np.random.uniform(0, 2*np.pi), 
				linearDamping=DAMP, 
				angularDamping = DAMP,
				userData=block
				)
			# print("block starting position: %s" % block.position)
			block.agent_contact = False # store for contact

			if i == 0: # t_block
				t_box = block.CreatePolygonFixture(
					box=(1/S, 1/S, (0., -1/S),0), 
					density=DENSE, 
					friction=FR, 
					restitution=RES)
				t_box2 = block.CreatePolygonFixture(
					box=(3/S, 1/S, (0., 1/S),0), 
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
			# print("block contact: ", block.agent_contact)
			self.blocks.append(block)

			# print("from _generate_blocks")
			for fix in block.fixtures:
				if block.userData in self.blks_vertices.keys():
					extend_v = [v for v in fix.shape.vertices if v not in self.blks_vertices[block.userData]]
					self.blks_vertices[block.userData].extend(extend_v)
				else:
					self.blks_vertices[block.userData] = fix.shape.vertices
			# print(self.blks_vertices[block.userData])

	def is_in_place(self, x, y, angle, block):
		
		f_x, f_y, f_angle = self.block_final_pos[block.userData]
		# f_x /= SCALE
		# f_y /= SCALE
		# print('is_in_place:')
		# print("final_position:", f_x, f_y, f_angle)
		# print("current_loc:", x, y, angle)

		if abs(f_x - x) > EPSILON:
			return False
		if abs(f_y - y) > EPSILON:
			return False
		# IGNORE ANGLE
		# if block.userData == 'i_block':
		# 	diff = abs(f_angle-angle) % np.pi
		# 	if  diff > ANG_EPSILON:
		# 		return False
		# elif abs(f_angle-angle) > ANG_EPSILON:
		# 	return False

		return True

	def _reset(self):
		self._destroy()
		self.world.contactListener_bug_workaround = ContactDetector(self)
		self.world.contactListener = self.world.contactListener_bug_workaround
		self.game_over = False

		init_x = np.random.uniform(BORDER, VIEWPORT_W/SCALE-BORDER)
		init_y = np.random.uniform(BORDER, VIEWPORT_H/SCALE-BORDER)
		# init_x = VIEWPORT_W/SCALE/2 
		# init_y = VIEWPORT_H/SCALE/2 - 2

		self._generate_boundary()
		self._generate_blocks()

		self.agent = self.world.CreateDynamicBody(
			position = (init_x, init_y),
			fixtures = fixtureDef(
				shape=polygonShape(vertices=[(x,y) for x,y in AGENT_POLY])),
			angle=0, 
			linearDamping=DAMP, 
			angularDamping=DAMP,
			userData='agent',
			)

		self._calculate_distance()

		self.drawlist = self.boundary + self.blocks + [self.agent]

		return self._step(self.action_space.sample())[0]
		# print("action sample: ", self.action_space.sample())
		# print(self.action_space)


	def _step(self, action):
		
		x, y, turn= action[0], action[1], action[2] # CONTINUOUS BOX

		# TAKE Action
		self.agent.linearVelocity = x * SPEED, y * SPEED
		self.agent.angularVelocity = float(turn) # won't take numpy.float32 - needs to be float
		# print(self.agent.linearVelocity)
		# SOFT FORCE from agent
		for block in self.blocks:
			force = 1.1**(-self.agent_dist[block.userData]) # CHANGE STRENGTH of soft force over time
			soft_vect = unitVector(self.agent, block)
			soft_force = (force*soft_vect[0], force*soft_vect[1])
			block.ApplyForce(soft_force, block.worldCenter, True)

		self.world.Step(1.0/FPS, 6*30, 2*30)

		# RETRIEVE Block locations + SET STATE
		in_place = []
		in_contact = False

		# STATE = [agent_x, agent_y, agent_theta, 
				# block_vertices(global), block_relative_loc, 
				# block_distance, contact_boolean]

		if self.unitize: unit = VIEWPORT_H
		else: unit = 1

		# STATE add agent	
		self.state = [
			self.agent.worldCenter[0]*SCALE/unit, 
			self.agent.worldCenter[1]*SCALE/unit, 
			self.agent.angle % (2*np.pi),
			# self.agent.linearVelocity[0],
			# self.agent.linearVelocity[1],
			# self.agent.angularVelocity]
			]

		for block in self.blocks:

			# CALCULATE relative location 
			x, y = block.worldCenter # is actual world center - unscaled
			angle = block.angle % (2*np.pi)
			fx, fy, fangle = self.block_final_pos[block.userData]
			x *= SCALE
			y *= SCALE
			# print(x-fx, y-fy)
			a_diff = fangle %(2*np.pi) - angle

			in_place.append(self.is_in_place(x, y, angle, block))
			
			# STATE add relative agent block location
			# RELATIVE AGENT location
			self.state.extend([
				(self.agent.worldCenter[0]*SCALE-x)/unit, 
				(self.agent.worldCenter[1]*SCALE-y)/unit,
				])

			# STATE add world vertices location
			for v in self.blks_vertices[block.userData]:
				x, y = block.GetWorldPoint(v)
				self.state.extend([x*SCALE/unit, y*SCALE/unit])
			
			# RELATIVE BLOCK location
			self.state.extend([
				# block.linearVelocity[0],
				# block.linearVelocity[1],
				# block.angularVelocity,
				(x-fx)/unit, 
				(y-fy)/unit, 
				a_diff])
				

			# print("block.agent_contact: ", block.agent_contact)
			if block.agent_contact:
				in_contact = True

		# CALCULATE rewards
		reward = 0

		prev_agent_dist = self.agent_dist.copy()
		prev_distance = self.block_distance.copy()
		prev_angle = self.block_angle.copy()

		self._calculate_distance()

		# DISTANCE PENALTY
		for block, dist in self.block_distance.items():
			# print(prev_distance[block])
			# print("block distance:", dist)
			# print("agent_distance", self.agent_dist[block])

			if not self.obs_type == 'image':
				self.state.append(dist/unit)
				self.state.append(self.agent_dist[block]/unit)
			# print(dist*0.01)

			# Apply reward based on blocks distance to final
			deltaDist = prev_distance[block] - dist
			reward += deltaDist*20
			reward -= 0.025 * dist

			deltaAgent = prev_agent_dist[block] - self.agent_dist[block]
			# print(deltaDist*20) # <0.05
			reward += deltaAgent*10
			reward -= 0.1 * self.agent_dist[block]
			# print("agent distance penalty: ", -0.1 * self.agent_dist[block])
			
			# print("previous angle:", prev_angle)
			# print("current angle:", self.block_angle)
			# if prev_angle[block] > self.block_angle[block]:
			# 	# print("rotating towards final pos!!!")
			# 	reward -= 0.01
			# elif prev_angle[block] < self.block_angle[block]:
			# 	# print("rotating block away")
			# 	reward -= 5.

		# if in_contact:
		# 	# print("in contact with BLOCK!")
		# 	reward += 1.

		# if self.wall_contact:
		# 	# print("in contact with wall!! MINUS FIVE POINTS")
		# 	reward -= 5.

		if self.obs_type != 'image':
			self.state.append(1.0 if in_contact else 0.0)



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
			print("puzzle complete!!!!!!!!!!!!!!!!!")

		# print("state: ", self.state)
		# print("final rewa`d: ", reward)
		return np.array(self.state), reward, done, {}

	def _render(self, mode='human', close=False):
		
		if close:
			if self.viewer is not None:
				self.viewer.close()
				self.viewer = None
			return

		from gym.envs.classic_control import rendering

		if self.viewer is None:
			self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
		self.viewer.set_bounds(0, VIEWPORT_W/SCALE, 0, VIEWPORT_H/SCALE)

		# DRAW BACKGROUND
		self.viewer.draw_polygon( [
			(0,					0),
			(VIEWPORT_W/SCALE,	0),
			(VIEWPORT_W/SCALE,	VIEWPORT_H/SCALE),
			(0, 				VIEWPORT_H/SCALE),
			], color=(0., 0., 0.) )

		# DRAW OBJECTS
		# print(self.drawlist)
		for obj in self.drawlist:
			# print(obj.userData)
			for f in obj.fixtures:
				trans = f.body.transform
				path = [trans*v for v in f.shape.vertices]
				self.viewer.draw_polygon(path, color=COLORS[obj.userData])
			
			
			# DRAW CP
			if 'agent' in obj.userData:
				x, y = obj.position
				# print('position: ', obj.position)
				# print('position: ', obj.worldCenter)
				t = rendering.Transform(translation=(x, y))
				self.viewer.draw_circle(0.04, 30, color=block_color).add_attr(t)

			if 'block' in obj.userData:
				x, y = obj.worldCenter
				# print("world center: ", x, y)
				# print("local center: ", obj.localCenter)
				t = rendering.Transform(translation=(x, y))
				self.viewer.draw_circle(0.04, 30, color=cp_color).add_attr(t)
				for v in self.blks_vertices[obj.userData]:
					x, y = obj.GetWorldPoint(v)
					t = rendering.Transform(translation=(x, y))
					self.viewer.draw_circle(0.02, 30, color=cp_color).add_attr(t)

		# DRAW FINAL POINTS
		for f_loc in self.block_final_pos.values():
			t = rendering.Transform(translation=(f_loc[0]/SCALE, f_loc[1]/SCALE))
			self.viewer.draw_circle(0.04, 30, color=final_color).add_attr(t)

		return self.viewer.render(return_rgb_array = mode=='rgb_array')


####################################################################################################################


# class RobotPuzzle5Unitize(RobotPuzzle5):
# 	downsample = 2 # for paperspace w/ lowres screen
# 	obs_type = 'low-dim'
# 	unitize = True


# class RobotPuzzleHighRes(RobotPuzzle):
# 	downsample = 2 # for paperspace w/ lowres screen
# 	obs_type = 'low-dim'

# class RobotPuzzleHighResPixels(RobotPuzzle):
# 	downsample = 2 # for paperspace w/ lowres screen
# 	obs_type = 'image'

####################################################################################################################



####################################################################################################################
# Testing Environements
####################################################################################################################

def run_random_actions():
	env.reset()
	print("completed reset")
	reward_sum = 0
	num_games = 1
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

	env = gym.make("RobotPuzzle-v0")
	# env = SimplePuzzle()
	# env.render()
	env.reset()
	# env.render(mode="rgb_array")
	# env.viewer.window.on_key_press = key_press
	
	print("completed reset")
	reward_sum = 0
	num_games = 10
	num_game = 0
	# while num_game < num_games:
	env.render()
	# print(env.render(mode="rgb_array").shape)
	observation, reward, done, _ = env.step(env.action_space.sample())
	print(observation.shape)
	reward_sum += reward
	# print(reward_sum)
	# if done:
	# 	print("Reward for this episode was: {}".format(reward_sum))
	# 	reward_sum = 0
	# 	num_game += 1
	# 	env.reset()
	# if escape: break


	# env.render(close=True)
	
