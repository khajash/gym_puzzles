import sys, math
import numpy as np
from scipy import misc

import Box2D
from Box2D.b2 import (polygonShape, circleShape, staticBody, dynamicBody, vec2, fixtureDef, contactListener, dot)

import gym
from gym import spaces
from gym.utils import colorize, seeding

# import my_gym
import pyglet
from pyglet import gl

# This is an environment in which an octagon robot assembles a simple puzzle
#
# Reward: There is a living penalty of -0.05. 
#
# State:
#
# To solve the game you need to get ?? points in ??? time steps.
#
# Created by Kate Hajash.

FPS    = 50
SCALE  = 30.0   # affects how fast-paced the game is, forces should be adjusted as well
DS = 1. # downsample

# (240, 320, 3) (1440, 810)
VIEWPORT_W, VIEWPORT_H = int(1440/DS), int(810/DS) # divide by 2 for mac screen
BORDER 	= 4		# border around screen to avoid placing blocks
BOUNDS 	= 2

# ROBOT SETTINGS
FR 		= 0.999 	# friction (between bodies)
RES 	= 0			# restitution 
DAMP 	= 5.0		# damping
# DENSE 	= 5.0		# density of blocks
BLK_DENSE 	= 1.56	
AGT_DENSE 	= 17.3

SPEED 	= 1500	# speed of robot agent

# PRECISION FOR BLOCKS IN PLACE
EPSILON = 25.0*2
ANG_EPSILON = 0.1

# REWARD STRUCTURE
# LIVING_PENALTY = -0.01
BLOCK_REWARD = 10
FINAL_REWARD = 10000
OUT_OF_BOUNDS = 100000

# AGENT
# S = 2*DS # scale of agents and blocks
S = 1/1.25

AGENT_POLY = [
	(-0.62/S,-1.5/S), (0.62/S,-1.5/S), (1.5/S,-0.62/S), (1.5/S,0.62/S), 
	(0.62/S,1.5/S), (-0.62/S,1.5/S), (-1.5/S,0.62/S), (-1.5/S,-0.62/S)
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
		for agent in self.env.agents:
			if agent in [contact.fixtureA.body, contact.fixtureB.body]:
				if self.env.goal_block in [contact.fixtureA.body, contact.fixtureB.body]:
					agent.goal_contact = True
					# print("block and agent in contact!")
				# print([contact.fixtureA.body.userData, contact.fixtureB.body.userData])
				if 'wall' in [contact.fixtureA.body.userData, contact.fixtureB.body.userData]:
					# print("env in contact with wall")
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


def getLateralVelocity(body):
	currentRightNormal = body.GetWorldVector(localVector=(1.0, 0.0))
	return dot(currentRightNormal, body.linearVelocity) * currentRightNormal

def updateFriction(body):
	impulse = body.mass * -getLateralVelocity(body)	
	body.ApplyLinearImpulse(impulse, body.worldCenter, True)

###### ENV CLASS #######################################################################

class MultiRobotPuzzle1(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array', 'state_pixels'],
		'video.frames_per_second' : FPS
	}

	downsample = 2 # for paperspace w/ lowres screen
	obs_type = 'low-dim'

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

		self.num_agents = 2
		self.agents = None
		self.block_names = ['t_block'] # 'l_block', 'i_block']
		self.blocks = None
		self.block_queue = self.block_names.copy()
		self.goal_block = None
		self.blks_vertices = {}
		self.blks_in_place = 0
		self.prev_blks_in_place = 0
		self.boundary = None
		# self.reward = 0

		# self.block_final_pos = set_final_loc(VIEWPORT_W, VIEWPORT_H, PUZZLE_REL_LOCATION)
		self.agent_dist = {}
		self.block_distance = {}
		self.block_angle = {}
		self.wall_contact = False

		# DEFINE Observation Boundaries
		self.theta_threshold = 2*np.pi
		a_obs = [np.inf, np.inf, self.theta_threshold, np.inf, np.inf, np.inf] * self.num_agents
		# a_obs = [np.inf, np.inf, np.inf, np.inf] * self.num_agents
		# Global location and rotation (3), relative location (2), distance to block, contact 
		blk_obs =[np.inf, np.inf, self.theta_threshold, np.inf] # Block 1 (rel location, rel theta, distance)
		vert_obs = [np.inf]*16
		high = np.array(a_obs + blk_obs + vert_obs)
	
		print("initialize...")

		self.observation_space = spaces.Box(-high, high, dtype=np.float32)
		self.state = []

		# DEFINE Action space
		action_high = np.array([1] * (2 * self.num_agents))
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
		# for bound in self.boundary:
		# 	self.world.DestroyBody(bound)
		# self.boundary = []
		for agent in self.agents:
			self.world.DestroyBody(agent)
		self.agents = []



	def _generate_boundary(self):
		self.boundary = []
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
			self.block_angle[block.userData] = abs(fangle %(2*np.pi) - abs(block.angle)%(2*np.pi))

			# self.agent_dist[block.userData] = distance(
			# 	self.agent.worldCenter*SCALE,
			# 	block.worldCenter*SCALE)

	def _calculate_agent_distance(self):

		for block in self.blocks:
			if block.userData == self.goal_block.userData:
				for agent in self.agents:
					self.agent_dist[agent.userData] = distance(
						agent.worldCenter*SCALE, 
						block.worldCenter*SCALE)
					# print(agent.userData, agent.worldCenter*SCALE)
					# print(distance(
					# 	agent.worldCenter*SCALE, 
					# 	block.worldCenter*SCALE))

	def _out_of_bounds(self):
		
		for obj in self.blocks + self.agents:
			x, y = obj.worldCenter
			if x < BOUNDS or x > (VIEWPORT_W/SCALE - BOUNDS):
				return True
			elif y < BOUNDS or y > (VIEWPORT_H/SCALE - BOUNDS):
				return True
		return False
			

	def _set_next_goal_block(self):
		for block in self.blocks:
			if block.userData == self.block_queue[0]:
				self.block_queue.pop(0)
				self.goal_block = block
				# print("goal block: %s" % self.goal_block.userData)

	def _set_random_goal(self):
		goal_dict = {}
		x = np.random.uniform(VIEWPORT_W/SCALE*2/3, VIEWPORT_W/SCALE-BORDER)
		y = np.random.uniform(BORDER, VIEWPORT_H/SCALE-BORDER)
		goal_dict[self.goal_block.userData] = (x, y, 0)
		return goal_dict


	def _generate_blocks(self):
		
		self.blocks = []

		for i, block in enumerate(self.block_names):
			x = np.random.uniform(VIEWPORT_W/SCALE/3, VIEWPORT_W/SCALE*2/3)
			y = np.random.uniform(BORDER, VIEWPORT_H/SCALE-BORDER)
			# x = VIEWPORT_W/SCALE/2
			# y = VIEWPORT_H/SCALE/2
			block = self.world.CreateDynamicBody(
				position = (x, y),
				# angle=0, 
				angle=np.random.uniform(0, 2*np.pi), 
				linearDamping=DAMP, 
				angularDamping = DAMP,
				userData=block
				)
			# print("block starting position: %s" % block.position)
			block.agent_contact = False # store for contact

			BS = 1/1.9

			if i == 0: # t_block
				t_box = block.CreatePolygonFixture(
					box=(1/BS, 1/BS, (0., -1/BS),0), 
					density=BLK_DENSE, 
					friction=FR, 
					restitution=RES)
				t_box2 = block.CreatePolygonFixture(
					box=(3/BS, 1/BS, (0., 1/BS),0), 
					density=BLK_DENSE, 
					friction=FR, 
					restitution=RES)
			
			elif i == 1: # l_block
				l_box = block.CreatePolygonFixture(
					box=(1/BS, 1/BS, (1/BS, 0.5/BS), 0), 
					density=BLK_DENSE,
					friction=FR, 
					restitution=RES)
				l_box2 = block.CreatePolygonFixture(
					box=(1/BS, 2/BS, (-1/BS, -0.5/BS), 0), 
					density=BLK_DENSE, 
					friction=FR, 
					restitution=RES)
			
			else: # i_block
				i_box = block.CreatePolygonFixture(
					box=(1/BS, 2/BS), 
					density=BLK_DENSE, 
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
			x = np.random.uniform(BORDER, VIEWPORT_W/SCALE/3)
			y = np.random.uniform(BORDER, VIEWPORT_H/SCALE-BORDER)

			agent = self.world.CreateDynamicBody(
				position = (x, y),
				fixtures = fixtureDef(
					shape=polygonShape(vertices=[(x,y) for x,y in AGENT_POLY]),
					density=AGT_DENSE,
					friction=FR,
					restitution=RES,
					userData='body'),
				angle=np.random.uniform(0, 2*np.pi), 
				linearDamping=DAMP, 
				angularDamping=DAMP,
				userData='agent_%s'%i,
				)

			wheel1 = agent.CreatePolygonFixture(
					box=(0.2/S, 0.75/S, (1/S, 0.),0), 
					density=0, 
					friction=FR, 
					restitution=RES,
					userData='wheel1')

			wheel2 = agent.CreatePolygonFixture(
					box=(0.2/S, 0.75/S, (-1/S, 0.),0), 
					density=0, 
					friction=FR, 
					restitution=RES,
					userData='wheel2')

			agent.goal_contact = False # Track contact with goal block
			self.agents.append(agent)


	def is_in_place(self, x, y, angle, block):
		
		f_x, f_y, f_angle = self.block_final_pos[block.userData]
		# f_x /= SCALE
		# f_y /= SCALE
		# print('is_in_place:')
		# print("final_position:", f_x, f_y, f_angle)
		# print("current_loc:", x, y, angle)

		if abs(f_x*SCALE - x) > EPSILON:
			return False
		if abs(f_y*SCALE - y) > EPSILON:
			return False
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

		self._generate_blocks()
		self._generate_agents()
		# self._generate_boundary()
		
		# RESET goal block
		self.block_queue = self.block_names.copy()
		self._set_next_goal_block()

		self.block_final_pos = self._set_random_goal()

		self._calculate_distance()
		self._calculate_agent_distance()

		self.drawlist = self.blocks + self.agents

		return self._step(self.action_space.sample())[0]

	def _step(self, action):
		# print(action)
		# CHOOSE Action 
		for i, agent in enumerate(self.agents):
			turn, vel = action[0 + i*2], action[1 + i*2]

			# print("%s: turn: %s vel: %s" % (agent.userData, turn, vel))

			f = agent.GetWorldVector(localVector=(0.0, 1.0))
			p = agent.GetWorldPoint(localPoint=(0.0, 2.0))
			
			f = (f[0]*vel*SPEED, f[1]*vel*SPEED)

			# print(t)
			agent.ApplyForce(f, p, True)
			updateFriction(agent)
			agent.ApplyAngularImpulse( 0.1 * agent.inertia * agent.angularVelocity, True )
			# agent.ApplyTorque(turn, True)
			# print(agent)

			# for wheel in [f for f in agent.fixtures if 'wheel' in f.userData]:
			# 	# print(wheel.massData.center)
			# 	cp = wheel.massData.center

			# 	f = agent.GetWorldVector(localVector=(0.0, 1.0))
			# 	p = agent.GetWorldPoint(localPoint=(cp))
			# 	# print(p)

			# 	if '1' in wheel.userData:
			# 		agent.ApplyForce( (f[0]*left*5, f[1]*left*5), p, True )

			# 	else:
			# 		agent.ApplyForce( (f[0]*right*5, f[1]*right*5), p, True )
				# wheel.ApplyForceToCenter( (
                # p_force*side[0] + f_force*forw[0],
                # p_force*side[1] + f_force*forw[1]), True )

			max_torque = 20
			# # print(turn)
			torque = int(abs(turn)*max_torque)

			if abs(vel) < 0.1: turn = 0

			if turn < 0:
				agent.ApplyTorque(torque, True)
				# print(torque, "apply left torque")
			elif turn > 0:
				agent.ApplyTorque(-torque, True)
				# print(torque, "apply right torque")

			else:
				agent.ApplyTorque(0, True)
				# print("apply ZERO torque")

			


			# TAKE Action
			# agent.linearVelocity = x * SPEED, y * SPEED
			# agent.angularVelocity = float(turn) # won't take numpy.float32 - needs to be float

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
		# for k, v in self.agent_dist.items():
		# 	print(k, v)

		# for k, v in prev_agent_dist.items():
		# 	print("prev location")
		# 	print(k, v)


		# RETRIEVE Block locations + SET STATE
		in_place = []
		in_contact = False

		# BUILD state
		self.state = []

		for agent in self.agents:
			# ADD global location	
			self.state.extend([
				agent.worldCenter[0]*SCALE, 
				agent.worldCenter[1]*SCALE, 
				agent.angle % (2*np.pi)
				])
			# ADD location relative to goal block 
			# print(agent.userData, agent.worldCenter[0], agent.worldCenter[1])
			# print(agent.)
			x, y = self.goal_block.worldCenter
			self.state.extend([
				agent.worldCenter[0]*SCALE - x*SCALE, 
				agent.worldCenter[1]*SCALE - y*SCALE,
				])
			self.state.append(self.agent_dist[agent.userData])
			# ADD in contact 
			# self.state.append(1.0 if agent.goal_contact else 0.0)

		for block in self.blocks:
			# CALCULATE relative location 
			# TODO: LOOK INTO ANGLE
			x, y = block.worldCenter # is actual world center - unscaled
			angle = block.angle % (2*np.pi)
			fx, fy, fangle = self.block_final_pos[block.userData]
			# print("final location: ", fx, fy)
			x *= SCALE
			y *= SCALE
			# print(x-fx, y-fy)
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

		# DISTANCE of BLOCK 
		deltaDist = prev_distance[self.goal_block.userData] - self.block_distance[self.goal_block.userData]
		reward += deltaDist*20
		reward -= 0.025 * self.block_distance[self.goal_block.userData]

		# if self.block_distance[self.goal_block.userData] > prev_distance[self.goal_block.userData]:
		# 	reward -= 5.
		# 	# print('block further away')
		# elif self.block_distance[self.goal_block.userData] < prev_distance[self.goal_block.userData]:
		# 	reward += 1.
		# 	# print('block closer!!!!')
		# else:
		# 	reward -= 3.


		# ROTATING REWARD
		# if prev_angle[self.goal_block.userData] > self.block_angle[self.goal_block.userData]:
		# 	# print("rotating towards final pos!!!")
		# 	reward += 0.01
		# elif prev_angle[self.goal_block.userData] < self.block_angle[self.goal_block.userData]:
		# 	# print("rotating block away")
		# 	reward -= 0.05
		
		# DISTANCE PENALTY
		for agent in self.agents:
			# print("%s's distance: %s" % (agent.userData, self.agent_dist[agent.userData]))
			# print("prev %s's distance: %s" % (agent.userData, prev_agent_dist[agent.userData]))


			# DISTANCE OF AGENT
			deltaAgent = prev_agent_dist[agent.userData] - self.agent_dist[agent.userData]
			# print("%s's distance: %s" % (agent.userData, deltaAgent*10)) # <0.05
			reward += deltaAgent*10
			reward -= 0.1 * self.agent_dist[agent.userData]

		# CHECK if DONE
		done = False

		if self._out_of_bounds():
			done = True
			reward -= OUT_OF_BOUNDS
			return np.array(self.state), reward, done, {}

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
		# print("total reward: ", reward)
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

		# DRAW BOUNDARY LINES
		self.viewer.draw_polyline( [
			(BOUNDS,					BOUNDS),
			(VIEWPORT_W/SCALE - BOUNDS,	BOUNDS),
			(VIEWPORT_W/SCALE - BOUNDS,	VIEWPORT_H/SCALE - BOUNDS),
			(BOUNDS,					VIEWPORT_H/SCALE - BOUNDS),
			(BOUNDS,					BOUNDS),
			], color=(.75, 0., 0.,),
			linewidth=5)

		bound_color = (0.2, 0.2, 0.2)
		self.viewer.draw_polyline( [
			(BORDER,					BORDER),
			(VIEWPORT_W/SCALE - BORDER,	BORDER),
			(VIEWPORT_W/SCALE - BORDER,	VIEWPORT_H/SCALE - BORDER),
			(BORDER,					VIEWPORT_H/SCALE - BORDER),
			(BORDER,					BORDER),
			], color=bound_color,
			linewidth=3)

		self.viewer.draw_polyline( [ 
			(VIEWPORT_W/SCALE/3, BORDER),
			(VIEWPORT_W/SCALE/3, VIEWPORT_H/SCALE - BORDER),
			], color=bound_color,
			linewidth=3)

		self.viewer.draw_polyline( [ 
			(VIEWPORT_W/SCALE*2/3, BORDER),
			(VIEWPORT_W/SCALE*2/3, VIEWPORT_H/SCALE - BORDER),
			], color=bound_color,
			linewidth=3)


		# DRAW OBJECTS
		for obj in self.drawlist:
			
			# print(obj.userData)
			for f in obj.fixtures:
				trans = f.body.transform
				path = [trans*v for v in f.shape.vertices]

				if 'agent' in obj.userData:
					if 'wheel' in f.userData:
						self.viewer.draw_polygon(path, color=block_color)
					else:
						self.viewer.draw_polygon(path, color=COLORS['agent'])
				else:
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
			t = rendering.Transform(translation=(f_loc[0], f_loc[1]))
			self.viewer.draw_circle(EPSILON/SCALE, 30, color=final_color).add_attr(t)

		return self.viewer.render(return_rgb_array = mode=='rgb_array')

	def _get_image(self, save_img=False, count=0, downsample=1):
		image_data = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
		# print("width: %s, height: %s, format: %s" % (image_data.width, image_data.height, image_data.format))
		arr = np.fromstring(image_data.data, dtype=np.uint8, sep='')
		arr = arr.reshape(image_data.height, image_data.width, 4)
		arr = arr[::-1,:,0:3]
		arr = arr[::downsample,::downsample,:]

		return arr


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

	a = np.array( [0.0, 0.0, 0.0, 0.0] )
	def key_press(k, mod):
		global escape
		if k==key.ESCAPE: 
			escape = True
			print("escape")
		if k==key.LEFT 	and a[0] > -1.0:	a[0] -= 0.1
		if k==key.RIGHT	and a[0] < +1.0:	a[0] += 0.1
		if k==key.UP   and a[1] < +1.0:		a[1] += 0.1
		if k==key.DOWN	and a[1] > -1.0:	a[1] -= 0.1
		if k==key.SPACE: 					a[0], a[1] = 0, 0 

	# def key_release(k, mod):
	#     if k==key.LEFT  and a[0]==-1.0: a[0] = 0
	#     if k==key.RIGHT and a[0]==+1.0: a[0] = 0
	#     if k==key.UP:   and a[1]==+1.0: a[1] = 0
	#     if k==key.DOWN: and a[1]==-1.0: a[1] = 0


	# env = gym.make("MultiRobotPuzzle-v0")
	env = MultiRobotPuzzle1()
	print(type(env))
	# env.render()
	env._reset()
	env._render()

	env.viewer.window.on_key_press = key_press
	# env.viewer.window.on_key_release = key_release

	
	print("completed reset")
	reward_sum = 0
	num_games = 10
	num_game = 0
	while num_game < num_games:
		env._render()
		# print(env.render(mode="rgb_array").shape)
		observation, reward, done, _ = env._step(a)
		# print(observation.shape)
		reward_sum += reward
		# print(reward_sum)
		if done:
			print("Reward for this episode was: {}".format(reward_sum))
			reward_sum = 0
			num_game += 1
			env._reset()
		if escape: break


	env.render(close=True)
	
