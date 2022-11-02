import numpy as np

import Box2D
from Box2D.b2 import (polygonShape, fixtureDef, contactListener)

import gym
from gym import spaces
from gym.utils import seeding

import pyglet
from pyglet import gl

from gym_puzzles.envs.robot import Robot
from gym_puzzles.envs.blocks import Block

FPS 	= 50
SCALE   = 30.0   	# affects how fast-paced the game is, forces should be adjusted as well

BORDER 	= 1		# border around screen to avoid placing blocks

# ROBOT SETTINGS
FR 	    = 0.999 # friction (between bodies)
RES 	= 0		# restitution 
DAMP 	= 5.0		# damping
DENSE 	= 5.0		# density of blocks
SPEED 	= 40/SCALE	# speed of robot agent


# PRECISION FOR BLOCKS IN PLACE
EPSILON 	= 25.0
ANG_EPSILON = 0.1

# COLORS
GREY 	= (0.5, 0.5, 0.5)
LT_GREY = (0.2, 0.2, 0.2)
WHITE 	= (1., 1., 1.)
BLUE 	= (58./255, 153./255, 1.)


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

def distance(pt1, pt2):
	sqr_dist = (pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2 
	return sqr_dist**0.5

def unitVector(bodyA, bodyB):
	Ax, Ay = bodyA.worldCenter
	Bx, By = bodyB.worldCenter
	denom = max(abs(Bx-Ax), abs(By-Ay))
	return ((Bx-Ax)/denom, (By-Ay)/denom)

###### ENV CLASS #######################################################################

class RobotPuzzleBase(gym.Env):
	
	metadata = {
		'render.modes': ['human', 'rgb_array', 'agent'],
		'video.frames_per_second' : FPS,
	}

    # TODO: add more env config 

	def __init__(
		self, 
		num_agents: int = 2, 
		goal_velocity: float = 1.5, 
		heavy: bool = False,
		hardmode: bool = False
	):

		self.seed()
		self.viewer = None
		self.screen_width = 640
		self.screen_height = 480

		self.goal_velocity = goal_velocity
		self.heavy = heavy
		self.hardmode = hardmode

		self.world = Box2D.b2World(gravity=(0,0), doSleep=False)

		self.num_agents = num_agents
		self.agents = None
		self.goal_block = None
		self.blks_vertices = []
		self.boundary = []

		# REWARD
		self.agent_dist = {}
		self.block_distance = None
		self.block_angle = None
		self.wall_contact = False

		self.set_reward_params()

		# DEFINE Observation Boundaries
		theta_threshold = 2*np.pi
		# agent obs - relative location to block (x,y), rotation, contact with block
		a_obs_high = [ 2.5,  2.5,  theta_threshold, 1.0] * self.num_agents
		a_obs_low  = [-2.5, -2.5, -theta_threshold, 0.0] * self.num_agents
		# block obs - relative position (x, y) and rotation (theta) to goal
		blk_obs_high = [ 2.5,  2.5,  theta_threshold]
		blk_obs_low  = [-2.5, -2.5, -theta_threshold]
		# vertices obs - global position of T-block's vertices
		vert_obs_high = [1.5]*16
		vert_obs_low = [-1.5]*16
		high = np.array(a_obs_high + blk_obs_high + vert_obs_high)
		low = np.array(a_obs_low + blk_obs_low + vert_obs_low)
	

		self.observation_space = spaces.Box(low, high, dtype=np.float32)
		action_high = np.array([1] * (3 * self.num_agents))
		self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)

		print("initialize...")

		self.reset()


	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]		


	def set_reward_params(self, agentDelta=10, agentDistance=0.1, blockDelta=50, blockDistance=0.025,
		puzzleComp=100):
		self.weight_deltaAgent 			= agentDelta
		self.weight_agent_dist 			= agentDistance
		self.weight_deltaBlock 			= blockDelta
		self.weight_blk_dist 			= blockDistance
		self.puzzle_complete_reward 	= puzzleComp


	def update_params(self, timestep, decay):
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


	# def is_in_place(self, x, y, angle, block):
	# 	f_x, f_y, f_angle = self.goal_block_pos
	# 	if abs(f_x - x) > EPSILON:
	# 		return False
	# 	if abs(f_y - y) > EPSILON:
	# 		return False
	# 	return True
	

	def _generate_boundary(self, thickness):
		self.boundary = []
		borders = [(0, 1/2), (1, 1/2), (1/2, 0), (1/2, 1)]
		for i, border in enumerate(borders):
			if i < 2:
				box_shape = [thickness, self.screen_height/SCALE]
			else:
				box_shape = [self.screen_width/SCALE, thickness]
			wall = self.world.CreateStaticBody(
				position = (self.screen_width/SCALE*border[0], self.screen_height/SCALE*border[1]),
				fixtures = fixtureDef(
					shape=polygonShape(box=(box_shape)),
					),
				userData = 'wall'
				)
			self.boundary.append(wall)


	def _generate_blocks(self):
		if self.heavy:
			scale = 1
			blk_dense = DENSE * 2
		else:
			scale = 0.5
			blk_dense = DENSE
		
		x = np.random.uniform(self.screen_width/SCALE/3 + 2*BORDER, self.screen_width/SCALE*2/3-2*BORDER)
		y = np.random.uniform(3*BORDER, self.screen_height/SCALE-3*BORDER)
		# print(3*BORDER, self.screen_height/SCALE-3*BORDER)
		rot = np.random.uniform(0, 2*np.pi)

		self.goal_block = Block(
			world=self.world, 
			init_angle=rot,
			init_x=x, 
			init_y=y,
			scale=scale,
			density=blk_dense,
			shape="T"
		)


	def _generate_agents(self):
		self.agents = []
		for i in range(self.num_agents):
			x = np.random.uniform(BORDER, self.screen_width/SCALE/3-2*BORDER)
			y = np.random.uniform(BORDER, self.screen_height/SCALE-BORDER)

			self.agents.append(Robot(
				world=self.world,
				init_angle=0,
				init_x=x,
				init_y=y,
				name=f"agent_{i}",
				max_speed=5.0,
				scale=8.0,
				density=5.0,
			))


	def _destroy(self):
		if not self.goal_block: return # if NONE then skip reset
		self.world.contactListener = None

		# Destroy boundary
		for bound in self.boundary:
			self.world.DestroyBody(bound)
		self.boundary = []

		# Destroy blocks
		self.goal_block.destroy()
		self.goal_block = None

		# Destroy agents
		for agent in self.agents:
			agent.destroy()
		self.agents = []


	def reset(self):
		self._destroy()
		self.world.contactListener_bug_workaround = ContactDetector(self)
		self.world.contactListener = self.world.contactListener_bug_workaround
		self.game_over = False
		
		# Generate objects
		self._generate_blocks()
		self._generate_agents()
		self._generate_boundary(BORDER)
		
		# Set goal block - TODO make random version
		self.goal_block_pos = [
			# self.screen_width//2,
			5/6*self.screen_width - 4/3*BORDER,
			self.screen_height//2,
			0]

		# print(self.goal_block_pos)

		self._get_obs()

		return self.step(self.action_space.sample())[0]

	def _get_norm_pose(self, x, y, rot):
		width_scale  = self.screen_width / SCALE / 2
		height_scale = self.screen_height / SCALE / 2
		x = (x - width_scale) / width_scale
		y = (y - height_scale) / width_scale
		rot %= (2*np.pi)
		return x, y, rot

	def _get_obs(self):

		obs = []

		# Normalize all positions (-1,1) in width and height
		width_scale  = self.screen_width / SCALE / 2
		height_scale = self.screen_height / SCALE / 2
		
		# Block position
		# bx, by = self.goal_block.worldCenter
		bx, by, brot = self._get_norm_pose(*self.goal_block.worldCenter, self.goal_block.angle)
		# bx = (bx - width_scale) / width_scale
		# by = (by - height_scale) / height_scale
		# brot = self.goal_block.angle % (2*np.pi)

		# Agent Obs
		for agent in self.agents:
			# ax, ay = agent.worldCenter
			ax, ay, arot = self._get_norm_pose(*agent.worldCenter, agent.angle)
			
			# print("agent rotations: ", agent.angle % (2*np.pi), agent.angle)
			
			# ADD agent to block relative location 
			obs.extend([bx - ax, by - ay])
			obs.append(arot) # TODO: verify this range is correct
			# TODO: add velocity
			
			# Add distance between agent and block to dict - not necessary in obs
			self.agent_dist[agent.userData] = distance((ax, ay), (bx, by))
			
			# ADD in contact 
			obs.append(1.0 if agent.goal_contact else 0.0)

		# Block obs 
		# print(self.goal_block_pos)
		gx, gy = self.goal_block_pos[:2]
		# print(gx, gy)
		gx = (gx - self.screen_width / 2) / (self.screen_width / 2)
		gy = (gy - self.screen_height / 2) / (self.screen_width / 2)
		grot = self.goal_block_pos[2] % (2*np.pi)

		# print("scaled goal: ", gx, gy)
		
		# relative position of block to goal
		obs.extend([gx - bx, gy - by, grot - brot])
		
		self.block_distance = distance((bx, by), (gx, gy))
		self.block_angle = grot - brot

		# Obs add world vertices location
		obs.extend(self.goal_block.get_vertices(
			norm_fn=lambda  x, y : [(x-width_scale)/width_scale, (y-height_scale)/width_scale]))
		
		return obs


	def step(self, action):
		# APPLY Action 
		for i, agent in enumerate(self.agents):
			
			x, y, rot = action[0 + i*3], action[1 + i*3], action[2 + i*3]
			agent.step(x,y, rot)

			# TODO: move this to class
			force = 1.1**(-self.agent_dist[agent.userData]) # CHANGE STRENGTH of soft force over time
			soft_vect = unitVector(agent, self.goal_block)
			soft_force = (force*soft_vect[0], force*soft_vect[1])
			self.goal_block.apply_soft_force(soft_force)
		
		# PROGRESS multiple timesteps
		self.world.Step(1.0/FPS, 6*30, 2*30)

		# CACLUATE distances
		prev_agent_dist = self.agent_dist.copy()
		prev_distance = self.block_distance
		prev_angle = self.block_angle
		
		obs = self._get_obs()
		# TODO: FIX epsilon with scaling - not matching currently with scaling
		in_place = self.block_distance <= EPSILON / self.screen_width * 2
		# print("\nblock distance ", self.block_distance)
		# print(obs)

		# CALCULATE rewards
		reward = 0

		# DISTANCE PENALTY - block to goal 
		# delta distance between prev timestep and cur timestep to encourage larger positive movements
		deltaDist = prev_distance - self.block_distance
		reward += deltaDist * self.weight_deltaBlock 
		
		# negative reward based on distance between block and goal
		reward -= self.weight_blk_dist * self.block_distance
		
		# DISTANCE PENALTY - agents to block
		for agent in self.agents:
			# delta distance between prev timestep and cur timestep to encourage larger positive movements
			deltaAgent = prev_agent_dist[agent.userData] - self.agent_dist[agent.userData]
			reward += deltaAgent * self.weight_deltaAgent/4.
			
			# negative reward based on agent's distance to block
			reward -= self.weight_agent_dist * self.agent_dist[agent.userData]/4.

			if agent.goal_contact:
				reward += 0.25

		# CHECK if done
		done = False
		self.done_status = None
		
		# ADD reward for puzzle completion
		if in_place:
			done = True
			reward += self.puzzle_complete_reward
			self.done_status = "puzzle complete!!"
			print("puzzle complete!!!!!!!!!!!!!!!!!")

		return np.array(obs), reward, done, {}

	def _return_status(self):
		if self.done_status: return self.done_status
		else: return "Stayed in bounds"

	
	def render(self, mode='human', close=False):
		if close:
			if self.viewer is not None:
				self.viewer.close()
				self.viewer = None
			return
		
		from gym.envs.classic_control import rendering
		if self.viewer is None:
			self.viewer = rendering.Viewer(self.screen_width, self.screen_height)
		self.viewer.set_bounds(0, self.screen_width/SCALE, 0, self.screen_height/SCALE)

		# Background
		self.viewer.draw_polygon( [
			(0,							0),
			(self.screen_width/SCALE,	0),
			(self.screen_width/SCALE,	self.screen_height/SCALE),
			(0, 						self.screen_height/SCALE),],
			color=(0., 0., 0.) )
		
		# Boundary
		for obj in self.boundary:
			for f in obj.fixtures:
				trans = f.body.transform
				path = [trans*v for v in f.shape.vertices]
				self.viewer.draw_polygon(path, color=LT_GREY)

		# Block
		self.goal_block.draw(self.viewer, mode=mode)

		# Robot agents
		for agent in self.agents:
			agent.draw(self.viewer)

		# Goal position
		t = rendering.Transform(translation=(self.goal_block_pos[0]/SCALE, self.goal_block_pos[1]/SCALE))
		self.viewer.draw_circle(EPSILON/SCALE, 30, color=BLUE).add_attr(t)

		return self.viewer.render(return_rgb_array = mode=='rgb_array')


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

	env = RobotPuzzleBase(heavy=False)
	# print("finished init")
	env.render()
	env.reset()
	env.viewer.window.on_key_press = key_press
	
	reward_sum = 0
	num_games = 10
	num_game = 0
	t = 0
	while num_game < num_games:
		env.render()
		# observation, reward, done, _ = env.step(env.action_space.sample()) # random control
		observation, reward, done, _ = env.step(a) # keyboard control
		reward_sum += reward
		t += 1
		if done or t > 30:
			print("Reward for this episode was: {}. Timesteps: {}".format(reward_sum, t))
			reward_sum = 0
			t = 0
			num_game += 1
			env.reset()
		if escape: break

	env.render(close=True)
