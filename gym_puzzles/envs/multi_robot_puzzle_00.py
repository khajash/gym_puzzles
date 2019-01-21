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

# This is a 2-D environment in which two octagonal robots (holonomic control) moves blocks 
# to a specified location demarcated by a blue circle. 
#
# Actions: Linear velocity in x and y-dimensions and angular velocity
# 
# Reward: 
#
# State: 
# 	For each agent: relative location to block, distance to block, contact with block
# 	For each block: relative location to goal, distance to goal, global position of block's vertices 
#
# Created by Kate Hajash.

DS 	= 1. 		# downsample
FPS     = 50 		# smaller number is faster, larger number is slower
SCALE   = 30.0   	# affects how fast-paced the game is, forces should be adjusted as well

VIEWPORT_W, VIEWPORT_H = int(640/DS), int(480/DS)
BORDER 	= 1		# border around screen to avoid placing blocks

# ROBOT SETTINGS
FR 	= 0.999 	# friction (between bodies)
RES 	= 0		# restitution 
DAMP 	= 5.0		# damping
DENSE 	= 5.0		# density of blocks
SPEED 	= 10/SCALE*4	# speed of robot agent

# PRECISION FOR BLOCKS IN PLACE
EPSILON = 25.0/DS
ANG_EPSILON = 0.1

# AGENT
S = 2*DS 		# scale of agents and blocks

AGENT_POLY = [
	(-0.5/S,-1.5/S), (0.5/S,-1.5/S), (1.5/S,-0.5/S), (1.5/S,0.5/S), 
	(0.5/S,1.5/S), (-0.5/S,1.5/S), (-1.5/S,0.5/S), (-1.5/S,-0.5/S)
	]

grey 	= (0.5, 0.5, 0.5)
white 	= (1., 1., 1.)
blue 	= (58./255, 153./255, 1.)

COLORS = {
	'agent'		: white,
	't_block'	: grey,
	'l_block'	: grey,
	'i_block'	: grey,
	'cp'		: white,
	'final_pt'	: blue,
	'wall'		: (0.2, 0.2, 0.2)
}

PUZZLE_REL_LOCATION = {
	't_block' : ((0., 0.75/DS), 0.),
	# future will incorporate more blocks
	# 'l_block' : ((-2./3., -2./3.), 0.5*np.pi),
	# 'i_block' : ((1., -0.5), 0.),
}

###### CONTACT DETECTOR #######################################################################

class ContactDetector(contactListener):
	def __init__(self, env):
		contactListener.__init__(self)
		self.env = env
	def BeginContact(self, contact):
		for agent in self.env.agents:
			if agent in [contact.fixtureA.body, contact.fixtureB.body]:
				# contact with block
				if self.env.goal_block in [contact.fixtureA.body, contact.fixtureB.body]:
					agent.goal_contact = True
				# contact with wall
				if 'wall' in [contact.fixtureA.body.userData, contact.fixtureB.body.userData]:
					self.env.wall_contact = True
	def EndContact(self, contact):
		for agent in self.env.agents:
			if agent in [contact.fixtureA.body, contact.fixtureB.body]:
				if self.env.goal_block in [contact.fixtureA.body, contact.fixtureB.body]:
					agent.goal_contact = False
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

	for key, rel_loc in block_rel_dict.items(): ## CHECK PPM
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

class MultiRobotPuzzle(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array', 'state_pixels'],
		'video.frames_per_second' : FPS
	}

	downsample = 2 # for paperspace w/ lowres screen
	obs_type = 'low-dim'
	heavy = False
	number_agents = 2

	def __init__(self, obs_depth=3, frameskip=4):
		"""
		Action type: 'box'
		Observation type: 'image' or 'low-dim'
		observation depth: integer indicating how many images should be part of observation
		"""
		self._seed()
		self.viewer = None
		self._obs_depth = obs_depth
		if self.obs_type == 'image': self.frameskip = frameskip
		else: self.frameskip = 1

		self.world = Box2D.b2World(gravity=(0,0), doSleep=False)

		self.num_agents = self.number_agents
		self.agents = None
		self.block_names = ['t_block'] 
		self.blocks = None
		self.block_queue = self.block_names.copy()
		self.goal_block = None
		self.blks_vertices = {}
		self.blks_in_place = 0
		self.prev_blks_in_place = 0
		self.boundary = None

		self.block_final_pos = set_final_loc(VIEWPORT_W, VIEWPORT_H, PUZZLE_REL_LOCATION)
		self.agent_dist = {}
		self.block_distance = {}
		self.block_angle = {}
		self.wall_contact = False

		self.set_reward_params()

		# DEFINE Observation Boundaries
		self.theta_threshold = 2*np.pi
		# agent obs - relative location to block (x,y), distance to block, contact with block
		a_obs = [np.inf, np.inf, np.inf, np.inf] * self.num_agents
		# block obs - relative position (x, y) and rotation (theta) to goal, distance to goal
		blk_obs =[np.inf, np.inf, self.theta_threshold, np.inf] 
		# vertices obs - global position of block's vertices 
		vert_obs = [np.inf]*16
		high = np.array(a_obs + blk_obs + vert_obs)
	
		print("initialize...")

		if self.obs_type == 'image':
			self.observation_space = spaces.Box(low=0, high=255, 
				shape=(VIEWPORT_H * self._obs_depth, VIEWPORT_W, 3), dtype=np.uint8)
			self.state = np.zeros(shape=(self.observation_space.shape))
		else:
			self.observation_space = spaces.Box(-high, high, dtype=np.float32)
			self.state = []

		# DEFINE Action space
		action_high = np.array([1] * (3 * self.num_agents))
		self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)

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
		for agent in self.agents:
			self.world.DestroyBody(agent)
		self.agents = []

	def set_reward_params(self, agentDelta=10, agentDistance=0.1, blockDelta=50, blockDistance=0.025,
		puzzleComp=10000, outOfBounds=1000, blkOutOfBounds=100):
		self.weight_deltaAgent 			= agentDelta
		self.weight_agent_dist 			= agentDistance
		self.weight_deltaBlock 			= blockDelta
		self.weight_blk_dist 			= blockDistance
		self.puzzle_complete_reward 		= puzzleComp
		self.out_of_bounds_penalty		= outOfBounds
		self.blk_out_of_bounds_penalty 		= blkOutOfBounds

	def update_params(self, timestep, decay):
		self.shaped_blk_bounds_penalty = self.blk_out_of_bounds_penalty*decay**(-timestep)
		self.shaped_puzzle_reward = self.puzzle_complete_reward*decay**(-timestep)

	def update_goal(self, epoch, nb_epochs):
		self.scaled_epsilon = EPSILON*(2 - epoch/nb_epochs)

	def get_deltaAgent(self):
		return self.weight_deltaAgent

	def get_agentDist(self):
		return self.weight_agent_dist

	def get_deltaBlk(self):
		return self.weight_deltaBlock

	def get_blkDist(self):
		return self.weight_blk_dist

	def _generate_boundary(self):
		self.boundary = []
		borders = [(0, 1/2), (1, 1/2), (1/2, 0), (1/2, 1)]
		for i, border in enumerate(borders):
			if i < 2:
				box_shape = [1/DS, VIEWPORT_H/SCALE]
			else:
				box_shape = [VIEWPORT_W/SCALE, 1/DS]
			wall = self.world.CreateStaticBody(
				position = (VIEWPORT_W/SCALE*border[0], VIEWPORT_H/SCALE*border[1]),
				fixtures = fixtureDef(
					shape=polygonShape(box=(box_shape)),
					),
				userData = 'wall'
				)
			self.boundary.append(wall)

	def _calculate_distance(self):
		for block in self.blocks:
			self.block_distance[block.userData] = distance(
				block.worldCenter*SCALE, 
				self.block_final_pos[block.userData][:2])
			fangle = self.block_final_pos[block.userData][2]
			self.block_angle[block.userData] = abs(fangle %(2*np.pi) - abs(block.angle)%(2*np.pi))

	def _calculate_agent_distance(self):
		for block in self.blocks:
			if block.userData == self.goal_block.userData:
				for agent in self.agents:
					self.agent_dist[agent.userData] = distance(
						agent.worldCenter*SCALE, 
						block.worldCenter*SCALE)

	def _set_next_goal_block(self):
		for block in self.blocks:
			if block.userData == self.block_queue[0]:
				self.block_queue.pop(0)
				self.goal_block = block

	def _generate_blocks(self):
		global S
		global DENSE

		if self.heavy:
			scaled = S/2
			blk_dense = DENSE * 2
		self.blocks = []
		for i, block in enumerate(self.block_names):
			x = np.random.uniform(BORDER, VIEWPORT_W/SCALE-BORDER)
			y = np.random.uniform(BORDER, VIEWPORT_H/SCALE-BORDER)
			block = self.world.CreateDynamicBody(
				position = (x, y),
				angle=np.random.uniform(0, 2*np.pi), 
				linearDamping=DAMP, 
				angularDamping = DAMP,
				userData=block
				)
			block.agent_contact = False # store for contact listener

			if i == 0: # t_block
				t_box = block.CreatePolygonFixture(
					box=(1/scaled, 1/scaled, (0., -1/scaled),0), 
					density=blk_dense, 
					friction=FR, 
					restitution=RES)
				t_box2 = block.CreatePolygonFixture(
					box=(3/scaled, 1/scaled, (0., 1/scaled),0), 
					density=blk_dense, 
					friction=FR, 
					restitution=RES)
			
			elif i == 1: # l_block
				l_box = block.CreatePolygonFixture(
					box=(1/scaled, 1/scaled, (1/scaled, 0.5/scaled), 0), 
					density=blk_dense,
					friction=FR, 
					restitution=RES)
				l_box2 = block.CreatePolygonFixture(
					box=(1/scaled, 2/scaled, (-1/scaled, -0.5/scaled), 0), 
					density=blk_dense, 
					friction=FR, 
					restitution=RES)
			
			else: # i_block
				i_box = block.CreatePolygonFixture(
					box=(1/scaled, 2/scaled), 
					density=blk_dense, 
					friction=FR, 
					restitution=RES)
			
			self.blocks.append(block)

			# SAVE vertices data
			for fix in block.fixtures:
				if block.userData in self.blks_vertices.keys():
					extend_v = [v for v in fix.shape.vertices if v not in self.blks_vertices[block.userData]]
					self.blks_vertices[block.userData].extend(extend_v)
				else:
					self.blks_vertices[block.userData] = fix.shape.vertices

	def _generate_agents(self):
		self.agents = []
		for i in range(self.num_agents):
			x = np.random.uniform(BORDER, VIEWPORT_W/SCALE-BORDER)
			y = np.random.uniform(BORDER, VIEWPORT_H/SCALE-BORDER)
			agent = self.world.CreateDynamicBody(
				position = (x, y),
				fixtures = fixtureDef(
					shape=polygonShape(vertices=[(x,y) for x,y in AGENT_POLY])),
				angle=0, 
				linearDamping=DAMP, 
				angularDamping=DAMP,
				userData='agent_%s'%i,
				)
			agent.goal_contact = False # track contact with goal block
			self.agents.append(agent)

	def is_in_place(self, x, y, angle, block):
		f_x, f_y, f_angle = self.block_final_pos[block.userData]
		if abs(f_x - x) > EPSILON:
			return False
		if abs(f_y - y) > EPSILON:
			return False
		return True

	def _reset(self):
		self._destroy()
		self.world.contactListener_bug_workaround = ContactDetector(self)
		self.world.contactListener = self.world.contactListener_bug_workaround
		self.game_over = False

		self._generate_blocks()
		self._generate_agents()
		self._generate_boundary()
		
		# RESET goal block
		self.block_queue = self.block_names.copy()
		self._set_next_goal_block()

		self._calculate_distance()
		self._calculate_agent_distance()

		self.drawlist = self.boundary + self.blocks + self.agents

		return self._step(self.action_space.sample())[0]

	def _step(self, action):
		# CHOOSE Action 
		for i, agent in enumerate(self.agents):
			x, y, turn = action[0 + i*3], action[1 + i*3], action[2 + i*3]

			# TAKE Action
			agent.linearVelocity = x * SPEED, y * SPEED
			agent.angularVelocity = float(turn) # won't take numpy.float32 - needs to be float
			force = 1.1**(-self.agent_dist[agent.userData]) # CHANGE STRENGTH of soft force over time
			soft_vect = unitVector(agent, self.goal_block)
			soft_force = (force*soft_vect[0], force*soft_vect[1])
			self.goal_block.ApplyForce(soft_force, self.goal_block.worldCenter, True)
		
		# PROGRESS multiple timesteps:
		for _ in range(self.frameskip):
			self.world.Step(1.0/FPS, 6*30, 2*30)

		# CACLUATE distances
		prev_agent_dist = self.agent_dist.copy()
		prev_distance = self.block_distance.copy()
		prev_angle = self.block_angle.copy()
		
		self._calculate_distance()
		self._calculate_agent_distance()

		in_place = []
		in_contact = False

		# BUILD state
		self.state = []

		for agent in self.agents:
			# ADD location relative to goal block 
			x, y = self.goal_block.worldCenter
			self.state.extend([
				agent.worldCenter[0]*SCALE - x*SCALE, 
				agent.worldCenter[1]*SCALE - y*SCALE,
				])
			self.state.append(self.agent_dist[agent.userData])
			# ADD in contact 
			self.state.append(1.0 if agent.goal_contact else 0.0)

		for block in self.blocks:
			# CALCULATE relative location 
			x, y = block.worldCenter # actual world center - scale below
			angle = block.angle % (2*np.pi)
			fx, fy, fangle = self.block_final_pos[block.userData]
			x *= SCALE
			y *= SCALE
			a_diff = fangle %(2*np.pi)- angle

			in_place.append(self.is_in_place(x, y, angle, block))
			
			# STATE add relative block location
			self.state.extend([x-fx, y-fy, a_diff])
			self.state.append(distance((x,y), (fx, fy)))
			# STATE add world vertices location
			for v in self.blks_vertices[block.userData]:
				x, y = block.GetWorldPoint(v)
				self.state.extend([x*SCALE, y*SCALE])

		# CALCULATE rewards
		reward = 0

		# DISTANCE PENALTY - block to goal 
		# delta distance between prev timestep and cur timestep to encourage larger positive movements
		deltaDist = prev_distance[self.goal_block.userData] - self.block_distance[self.goal_block.userData]
		reward += deltaDist * self.weight_deltaBlock * DS/4.
		
		# negative reward based on distance between block and goal
		reward -= self.weight_blk_dist * self.block_distance[self.goal_block.userData] * DS/4.
		
		# DISTANCE PENALTY - agents to block
		for agent in self.agents:
			# delta distance between prev timestep and cur timestep to encourage larger positive movements
			deltaAgent = prev_agent_dist[agent.userData] - self.agent_dist[agent.userData]
			reward += deltaAgent * self.weight_deltaAgent * DS/4.
			
			# negative reward based on agent's distance to block
			reward -= self.weight_agent_dist * self.agent_dist[agent.userData] * DS/4.

			if agent.goal_contact:
				reward += 0.25

		# CHECK if done
		done = False
		self.done_status = None

		# CALCULATE new blocks in place
		self.prev_blks_in_place = self.blks_in_place 
		self.blks_in_place = 0
		for complete in in_place:
			if complete == True:
				self.blks_in_place += 1

		# ADD reward for each block
		# -10 for moving a block out of place
		# +10 for moving a block in place
		# intended to avoid robot moving a block in and out of place to take get continuous reward
		reward += (self.blks_in_place-self.prev_blks_in_place) * BLOCK_REWARD 
		
		# ADD reward for puzzle completion
		if self.blks_in_place == 1:
			done = True
			reward += FINAL_REWARD
			self.done_status = "puzzle complete!!"
			print("puzzle complete!!!!!!!!!!!!!!!!!")

		return np.array(self.state), reward, done, {}

	def _return_status(self):
		if self.done_status: return self.done_status
		else: return "Stayed in bounds"

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
			(0,				0),
			(VIEWPORT_W/SCALE,		0),
			(VIEWPORT_W/SCALE,		VIEWPORT_H/SCALE),
			(0, 				VIEWPORT_H/SCALE),],
			color=(0., 0., 0.) )

		# DRAW BOUNDARY
		self.viewer.draw_polyline( [
			(BORDER,			BORDER),
			(VIEWPORT_W/SCALE - BORDER,	BORDER),
			(VIEWPORT_W/SCALE - BORDER,	VIEWPORT_H/SCALE - BORDER),
			(BORDER,			VIEWPORT_H/SCALE - BORDER),
			(BORDER,			BORDER),],
			color=(0.2, 0.2, 0.2),
			linewidth=3)

		# diameter of circles (small and large)		
		sm_d = 0.02*4/DS
		lg_d = 0.04*4/DS

		# DRAW OBJECTS
		for obj in self.drawlist:
			for f in obj.fixtures:
				trans = f.body.transform
				path = [trans*v for v in f.shape.vertices]
				if 'agent' in obj.userData:
					self.viewer.draw_polygon(path, color=COLORS['agent'])
				else:
					self.viewer.draw_polygon(path, color=COLORS[obj.userData])
			
			# DRAW CP
			if 'agent' in obj.userData:
				x, y = obj.position
				t = rendering.Transform(translation=(x, y))
				self.viewer.draw_circle(lg_d, 30, color=block_color).add_attr(t)

			# DRAW BLOCK + VERTICES
			if 'block' in obj.userData:
				x, y = obj.worldCenter
				t = rendering.Transform(translation=(x, y))
				self.viewer.draw_circle(lg_d, 30, color=cp_color).add_attr(t)
				for v in self.blks_vertices[obj.userData]:
					x, y = obj.GetWorldPoint(v)
					t = rendering.Transform(translation=(x, y))
					self.viewer.draw_circle(sm_d, 30, color=cp_color).add_attr(t)

		# DRAW FINAL POINTS
		for f_loc in self.block_final_pos.values():
			t = rendering.Transform(translation=(f_loc[0]/SCALE, f_loc[1]/SCALE))
			self.viewer.draw_circle(EPSILON/SCALE, 30, color=final_color).add_attr(t)

		return self.viewer.render(return_rgb_array = mode=='rgb_array')

	def _get_image(self, save_img=False, count=0, downsample=1):
		image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
		arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
		arr = arr.reshape(image_data.height, image_data.width, 4)
		arr = arr[::-1,:,0:3]
		arr = arr[::downsample,::downsample,:]

		return arr


####################################################################################################################

class MultiRobotPuzzleHeavy(MultiRobotPuzzle):
	heavy = True
	number_agents = 5
	def __init__(self):
		super(MultiRobotPuzzleHeavy, self).__init__()


####################################################################################################################
# Testing Environements
####################################################################################################################

if __name__=="__main__":
	
	from pyglet.window import key
	escape = False

	a = np.array( [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] )
	def key_press(k, mod):
		global escape
		if k==key.ESCAPE: 
			escape = True
			print("escape")
		if k==key.LEFT 	and a[0] > -1.0:	a[0] -= 0.1
		if k==key.RIGHT	and a[0] < +1.0:	a[0] += 0.1
		if k==key.UP    and a[1] < +1.0:	a[1] += 0.1
		if k==key.DOWN	and a[1] > -1.0:	a[1] -= 0.1
		if k==key.SPACE: 			a[0], a[1] = 0, 0 

	env = MultiRobotPuzzle()
	env._render()
	env._reset()
	env.viewer.window.on_key_press = key_press
	
	reward_sum = 0
	num_games = 10
	num_game = 0
	while num_game < num_games:
		env._render()
		# observation, reward, done, _ = env.step(env.action_space.sample()) # random control
		observation, reward, done, _ = env._step(a) # keyboard control
		reward_sum += reward
		if done:
			print("Reward for this episode was: {}".format(reward_sum))
			reward_sum = 0
			num_game += 1
			env._reset()
		if escape: break

	env._render(close=True)
