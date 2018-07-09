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
SCALE  = 140.0*4   # affects how fast-paced the game is, forces should be adjusted as well
DS = 1. # downsample

# (240, 320, 3) (1440, 810)

# VIEWPORT_W, VIEWPORT_H = int(1440/4/DS), int(810/4/DS) # divide by 2 for mac screen
VIEWPORT_W, VIEWPORT_H = int(1440/DS), int(810/DS)
BORDER 	= 0.3		# border around screen to avoid placing blocks
BOUNDS 	= 0.1

# ROBOT SETTINGS
FR 			= 0.01 	# friction (between bodies)
RES 		= 0.		# restitution (maxes it bounce)
LINEAR_DAMP = 5.0
ANG_DAMP	= 5.0		# damping
BLK_DENSE 	= 1.56	
# BLK_DENSE 	= 20.0	
AGT_DENSE 	= 17.3		# density of blocks
FORCE 		= 0.75		# speed of robot agent

# PRECISION FOR BLOCKS IN PLACE
RATIO = SCALE/VIEWPORT_W
# EPSILON = 25.0/2
# EPSILON = 0.05
EPSILON 	= 0.1
ANG_EPSILON = 0.1
# SIMPLE 		= False
# ANYWHERE 	= True

SIMPLE 		= True
ANYWHERE 	= False

# REWARD STRUCTURE
# BLOCK_REWARD = 10
# FINAL_REWARD = 1000
# OUT_OF_BOUNDS = 10000

# AGENT
AGENT_POLY = [
	(-0.039,-0.095), (0.039,-0.095), (0.095,-0.039), (0.095,0.039), 
	(0.039,0.095), (-0.039,0.095), (-0.095,0.039), (-0.095,-0.039)
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

class MultiRobotPuzzle2(gym.Env):
	metadata = {
		'render.modes': ['human', 'rgb_array', 'state_pixels'],
		'video.frames_per_second' : FPS
	}

	unitize 		= True
	contact_weight 	= True # adds agents_in_contact/num_agents weight to puzzle completion
						   # also adds epsilon number to state info

	human_vision 	= False
	heavy			= False

	def __init__(self, frameskip=1, num_agents=2):
		"""
		Action type: 'box'
		Observation type: 'image' or 'low-dim'
		observation depth: integer indicating how many images should be part of observation
		"""
		self._seed()
		self.viewer = None
		self.frameskip = frameskip

		self.world = Box2D.b2World(gravity=(0,0), doSleep=False)

		self.num_agents = num_agents
		self.agents = None
		self.block_names = ['t_block'] # 'l_block', 'i_block']
		self.blocks = None
		self.block_queue = self.block_names.copy()
		self.goal_block = None
		self.blks_vertices = {}
		self.blks_in_place = 0
		self.prev_blks_in_place = 0
		self.boundary = None
		self.scaled_epsilon = EPSILON
		if self.heavy: 
			self.block_density = 20.0
		else:
			self.block_density = BLK_DENSE

		self.agent_dist = {}
		self.block_distance = {}
		self.block_angle = {}
		self.wall_contact = False

		self.set_reward_params()

		# DEFINE Observation Boundaries
		self.theta_threshold = 2*np.pi

		a_obs = [np.inf, np.inf, self.theta_threshold, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf] * self.num_agents
		# a_obs = [np.inf, np.inf, self.theta_threshold, np.inf, np.inf, np.inf] * self.num_agents

		# a_obs = [np.inf, np.inf, np.inf, np.inf] * self.num_agents
		# Global location and rotation (3), relative location (2), distance to block, contact 
		blk_obs =[np.inf, np.inf, self.theta_threshold, np.inf] # Block 1 (rel location, rel theta, distance)
		vert_obs = [np.inf]*16
		if self.contact_weight:
			high = np.array(a_obs + blk_obs + vert_obs + [np.inf])
		else:
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
		for bound in self.boundary:
			self.world.DestroyBody(bound)
		self.boundary = []
		for agent in self.agents:
			self.world.DestroyBody(agent)
		self.agents = []

	def set_reward_params(self, agentDelta=10, agentDistance=0.25, blockDelta=25, blockDistance=0.1,
		puzzleComp=10000, outOfBounds=1000, blkOutOfBounds=100):

		self.weight_deltaAgent 			= agentDelta
		self.weight_agent_dist 			= agentDistance
		self.weight_deltaBlock 			= blockDelta
		self.weight_blk_dist 			= blockDistance
		self.puzzle_complete_reward 	= puzzleComp
		self.out_of_bounds_penalty		= outOfBounds
		self.blk_out_of_bounds_penalty 	= blkOutOfBounds

	def update_params(self, timestep, decay):
		self.shaped_bounds_penalty = self.out_of_bounds_penalty*decay**(-timestep)
		self.shaped_blk_bounds_penalty = self.blk_out_of_bounds_penalty*decay**(-timestep)
		self.shaped_puzzle_reward = self.puzzle_complete_reward*decay**(-timestep)
		# print(self.shaped_bounds_penalty, self.shaped_blk_bounds_penalty)

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

	def norm_units(self, pt):
		ratio = SCALE/VIEWPORT_W
		return pt[0]*ratio, pt[1]*ratio

	def scale_units(self, pt):
		ratio = VIEWPORT_W/SCALE
		return pt[0]*ratio, pt[1]*ratio

	def norm_angle(self, a):
		theta = a % (2*np.pi)
		if theta <= np.pi: 
			norm_theta = -theta/np.pi
		else:
			norm_theta = (2*np.pi - theta)/np.pi
		return norm_theta

	def _calculate_distance(self):

		for block in self.blocks:
			# print("block: %s, final_pos: %s, current_loc: %s" % (
			# 	block.userData, self.block_final_pos[block.userData][:2], self.norm_units(block.worldCenter)))
			self.block_distance[block.userData] = distance(
				self.norm_units(block.worldCenter), 
				self.block_final_pos[block.userData][:2])
			
			fangle = self.block_final_pos[block.userData][2]
			self.block_angle[block.userData] = abs(fangle %(2*np.pi) - abs(block.angle)%(2*np.pi))

	def _calculate_agent_distance(self):

		for block in self.blocks:
			if block.userData == self.goal_block.userData:
				for agent in self.agents:
					self.agent_dist[agent.userData] = distance(
						self.norm_units(agent.worldCenter), 
						self.norm_units(block.worldCenter) )
					# print(agent.userData, agent.worldCenter*SCALE)
					# print(distance(
					# 	agent.worldCenter*SCALE, 
					# 	block.worldCenter*SCALE))

	def _blk_out_of_bounds(self):
		# OK not unitized??
		for obj in self.blocks:
			x, y = obj.worldCenter
			if x < BOUNDS or x > (VIEWPORT_W/SCALE - BOUNDS):
				return True
			elif y < BOUNDS or y > (VIEWPORT_H/SCALE - BOUNDS):
				return True
		return False

	def _agt_out_of_bounds(self):
		# OK not unitized??
		for obj in self.agents:
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
		if SIMPLE: BORDER = 0.4
		else: BORDER = 0.3
		# print(self.norm_units((VIEWPORT_W/SCALE*2/3+BORDER, VIEWPORT_W/SCALE-BORDER)))
		x = np.random.uniform(VIEWPORT_W/SCALE*2/3+BORDER, VIEWPORT_W/SCALE-BORDER)
		y = np.random.uniform(BORDER, VIEWPORT_H/SCALE-BORDER)
		# print("norm units")
		x, y = self.norm_units((x,y))
		# print(x, y)

		goal_dict[self.goal_block.userData] = (x, y, 0)
		return goal_dict

	def _generate_blocks(self):
		
		self.blocks = []

		for i, block in enumerate(self.block_names):
			if SIMPLE: x, y = VIEWPORT_W/SCALE/2, VIEWPORT_H/SCALE/2
			else:
				x = np.random.uniform(VIEWPORT_W/SCALE/3+BORDER, VIEWPORT_W/SCALE*2/3-BORDER)
				y = np.random.uniform(BORDER, VIEWPORT_H/SCALE-BORDER)
			
			block = self.world.CreateDynamicBody(
				position = (x, y),
				# angle=0, 
				angle=np.random.uniform(0, 2*np.pi), 
				linearDamping=LINEAR_DAMP, 
				angularDamping = ANG_DAMP,
				userData=block
				)
			# print("block starting position: %s" % block.position)
			block.agent_contact = False # store for contact

			if i == 0: # t_block
				block.CreatePolygonFixture(
					box=(0.1, 0.1, (0., -0.1),0), 
					density=self.block_density, 
					friction=FR, 
					restitution=RES)
				block.CreatePolygonFixture(
					box=(0.3, 0.1, (0., 0.1),0), 
					density=self.block_density, 
					friction=FR, 
					restitution=RES)
			
			self.blocks.append(block)

			print(block.mass)
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
			if ANYWHERE:
				x = np.random.uniform(BORDER, VIEWPORT_W/SCALE-BORDER)
			else:
				x = np.random.uniform(BORDER, VIEWPORT_W/SCALE/3-BORDER)
			y = np.random.uniform(BORDER, VIEWPORT_H/SCALE-BORDER)
			if SIMPLE: theta = 3/2*np.pi
			else: theta = np.random.uniform(0, 2*np.pi)

			agent = self.world.CreateDynamicBody(
				position = (x, y),
				fixtures = [
					fixtureDef(
						shape=polygonShape(vertices=[(x,y) for x,y in AGENT_POLY]),
						density=AGT_DENSE,
						friction=FR,
						restitution=RES,
						userData='body'),
					fixtureDef(
						shape=polygonShape(box=(0.005, 0.05, (0.06, 0.),0)), 
						density=0, 
						friction=FR, 
						restitution=RES,
						userData='wheel1'),
					fixtureDef(
						shape=polygonShape(box=(0.005, 0.05, (-0.06, 0.),0)),
						density=0, 
						friction=FR, 
						restitution=RES,
						userData='wheel2'),
					],
				angle=theta, 
				linearDamping=LINEAR_DAMP, 
				angularDamping=ANG_DAMP,
				userData='agent_%s'%i,
				)

			agent.goal_contact = False # Track contact with goal block
			self.agents.append(agent)

	def _generate_boundary(self):
		self.boundary = []
		borders = [(0, 1/2), (1, 1/2), (1/2, 0), (1/2, 1)]
		for i, border in enumerate(borders):
			if i < 2:
				box_shape = [BOUNDS, VIEWPORT_H/SCALE]
			else:
				box_shape = [VIEWPORT_W/SCALE, BOUNDS]

			wall = self.world.CreateStaticBody(
				position=(
					VIEWPORT_W/SCALE*border[0], 
					VIEWPORT_H/SCALE*border[1]),
				fixtures = fixtureDef(
					shape=polygonShape(box=(box_shape)),
					),
				userData = 'wall'
				)
			self.boundary.append(wall)

	def is_in_place(self, x, y, angle, block):
		
		f_x, f_y, f_angle = self.block_final_pos[block.userData] # UNITIZED
		# print('is_in_place:')
		# print("final_position:", f_x, f_y, f_angle)
		# print("current_loc:", x, y, angle)

		if abs(f_x - x) > self.scaled_epsilon:
			return False
		if abs(f_y - y) > self.scaled_epsilon:
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
		self._generate_boundary()

		
		# RESET goal block
		self.block_queue = self.block_names.copy()
		self._set_next_goal_block()
		self.block_final_pos = self._set_random_goal()

		self._calculate_distance()
		self._calculate_agent_distance()
		self.done_status = None
		# self._reset_params()

		self.drawlist = self.boundary + self.blocks + self.agents

		return self._step(self.action_space.sample())[0]

	def _step(self, action):
		# print(action)
		# CHOOSE Action 
		for i, agent in enumerate(self.agents):
			turn, vel = action[0 + i*2], action[1 + i*2]

			# print("%s: turn: %s vel: %s" % (agent.userData, turn, vel))
			# x, y = self.norm_units(agent.worldCenter)
			# print(agent.userData, self.norm_units(agent.worldCenter))

			f = agent.GetWorldVector(localVector=(0.0, 1.0))
			p = agent.GetWorldPoint(localPoint=(0.0, 2.0)) # change apply point?
			
			f = (f[0]*vel*FORCE, f[1]*vel*FORCE)

			agent.ApplyForce(f, p, True)
			updateFriction(agent)
			agent.ApplyAngularImpulse( 0.1 * agent.inertia * agent.angularVelocity, True )
			# agent.ApplyTorque(turn, True)
			# print(agent)

			max_torque = 0.0005
			# # print(turn)
			torque = abs(turn)*max_torque
			# print(torque)
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

			# APPLY soft force
			force = 10**(-self.agent_dist[agent.userData]) # CHANGE STRENGTH of soft force over time
			# force2 = 1.1**(-self.agent_dist[agent.userData]) # CHANGE STRENGTH of soft force over time
			force /= 50
			soft_vect = unitVector(agent, self.goal_block)
			# print(self.goal_block.worldCenter)
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

		# RETRIEVE Block locations + SET STATE
		in_place = []
		in_contact = False

		# BUILD state
		self.state = []

		for agent in self.agents:

			# ADD global location	
			aX, aY = self.norm_units(agent.worldCenter)
			self.state.extend([
				aX, 
				aY, 
				self.norm_angle(agent.angle), 
				])

			# ADD location relative to goal block 
			bX, bY = self.norm_units(self.goal_block.worldCenter)
			self.state.extend([
				aX - bX, 
				aY - bY,
				])

			vX, vY = agent.linearVelocity
			self.state.extend([vX, vY, agent.angularVelocity])

			self.state.append(self.agent_dist[agent.userData])
		

		for block in self.blocks:
			# CALCULATE relative location 
			# TODO: LOOK INTO ANGLE
			x, y = self.norm_units(block.worldCenter) # is actual world center - unscaled
			angle = block.angle % (2*np.pi)
			fx, fy, fangle = self.block_final_pos[block.userData]
			# print("final location: ", fx, fy)
			# print(x-fx, y-fy)
			a_diff = fangle % (2*np.pi) - angle
			a_diff /= np.pi #normalize

			in_place.append(self.is_in_place(x, y, angle, block))
			
			# STATE add relative block location
			self.state.extend([x-fx, y-fy, a_diff])
			self.state.append(distance((x,y), (fx, fy)))
			# STATE add world vertices location
			for v in self.blks_vertices[block.userData]:
				x, y = self.norm_units(block.GetWorldPoint(v))
				self.state.extend([x, y])

		if self.contact_weight:
			self.state.append(self.scaled_epsilon)



		# CALCULATE rewards
		reward = 0

		# DISTANCE of BLOCK 
		deltaDist = prev_distance[self.goal_block.userData] - self.block_distance[self.goal_block.userData]
		reward += deltaDist * self.weight_deltaBlock
		# # print("block deltadistance: %s" % (deltaDist*200)) # <0.05
		reward -= self.weight_blk_dist * self.block_distance[self.goal_block.userData]

		# FUCKED-UP TESTS!!!
		# deltaDist = prev_distance[self.goal_block.userData] - self.block_distance[self.goal_block.userData]
		# reward += deltaDist * 25
		# # # print("block deltadistance: %s" % (deltaDist*200)) # <0.05
		# reward -= 0.1 * self.block_distance[self.goal_block.userData]

		# print(0.25 * self.block_distance[self.goal_block.userData])
		
		# DISTANCE PENALTY
		for agent in self.agents:
			# print("%s's distance: %s" % (agent.userData, self.agent_dist[agent.userData]))
			# print("prev %s's distance: %s" % (agent.userData, prev_agent_dist[agent.userData]))

			# DISTANCE OF AGENT
			deltaAgent = prev_agent_dist[agent.userData] - self.agent_dist[agent.userData]
			# print("%s's deltadistance: %s" % (agent.userData, deltaAgent*100)) # <0.05
			reward += deltaAgent*self.weight_deltaAgent
			reward -= self.weight_agent_dist * self.agent_dist[agent.userData]
			# reward -= self.agent_dist[agent.userData]
			# print("distance reward: ", self.agent_dist[agent.userData])

		# CHECK if DONE
		done = False

		if self._agt_out_of_bounds():
			done = True
			reward -= self.shaped_bounds_penalty
			# print("LOSER!!!", reward)
			self.done_status = "FAIL! Agent Out Of Bounds: %s" % reward
			return np.array(self.state), reward, done, {}

		if self._blk_out_of_bounds():
			done = True
			reward -= self.shaped_blk_bounds_penalty
			# print("Nice Try!!!", reward)
			self.done_status = "Nice Try! Block Out Of Bounds: %s" % reward
			return np.array(self.state), reward, done, {}
			
		# CALCULATE new blocks in place
		self.prev_blks_in_place = self.blks_in_place 
		self.blks_in_place = 0
		for complete in in_place:
			if complete == True:
				self.blks_in_place += 1

		# ASSIGN new reward
		# reward += (self.blks_in_place-self.prev_blks_in_place) * BLOCK_REWARD 
			# -10 for moving a block out of place
			# +10 for moving a block in place
			# intended to avoid robot moving a block in and out of place to take get continuous reward
		num_in_contact = 0
		for agent in self.agents:
			if agent.goal_contact: num_in_contact +=1


		# print("agents in contact")
		# print(num_in_contact/len(self.agents))

		if self.blks_in_place == 1:
			done = True
			# reward += self.puzzle_complete_reward
			if self.contact_weight:
				reward += self.shaped_puzzle_reward * (num_in_contact/len(self.agents))
			else:
				reward += self.shaped_puzzle_reward
			# print("puzzle complete!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
			self.done_status = "\nPuzzle Complete!!!!!!!!!!!!!!!!!!!!!!!!!\n%s" % reward
			return np.array(self.state), reward, done, {}

		# print("state: ", self.state)
		# print("total reward: ", reward)
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
			(0,					0),
			(VIEWPORT_W/SCALE,	0),
			(VIEWPORT_W/SCALE,	VIEWPORT_H/SCALE),
			(0, 				VIEWPORT_H/SCALE),
			], color=(0., 0., 0.) )

		if self.human_vision:
			self._render_human_vision()
		else:
			self._render_agent_vision()
		
		return self.viewer.render(return_rgb_array = mode=='rgb_array')

	def _render_human_vision(self):
		# DRAW BOUNDARY LINES
		from gym.envs.classic_control import rendering

		# self.viewer.draw_polyline( [
		# 	(BOUNDS,					BOUNDS),
		# 	(VIEWPORT_W/SCALE - BOUNDS,	BOUNDS),
		# 	(VIEWPORT_W/SCALE - BOUNDS,	VIEWPORT_H/SCALE - BOUNDS),
		# 	(BOUNDS,					VIEWPORT_H/SCALE - BOUNDS),
		# 	(BOUNDS,					BOUNDS),
		# 	], color=(.75, 0., 0.,),
		# 	linewidth=5)

		# bound_color = (0.2, 0.2, 0.2)
		# self.viewer.draw_polyline( [
		# 	(BORDER,					BORDER),
		# 	(VIEWPORT_W/SCALE - BORDER,	BORDER),
		# 	(VIEWPORT_W/SCALE - BORDER,	VIEWPORT_H/SCALE - BORDER),
		# 	(BORDER,					VIEWPORT_H/SCALE - BORDER),
		# 	(BORDER,					BORDER),
		# 	], color=bound_color,
		# 	linewidth=3)

		# self.viewer.draw_polyline( [ 
		# 	(VIEWPORT_W/SCALE/3, BORDER),
		# 	(VIEWPORT_W/SCALE/3, VIEWPORT_H/SCALE - BORDER),
		# 	], color=bound_color,
		# 	linewidth=3)

		# self.viewer.draw_polyline( [ 
		# 	(VIEWPORT_W/SCALE*2/3, BORDER),
		# 	(VIEWPORT_W/SCALE*2/3, VIEWPORT_H/SCALE - BORDER),
		# 	], color=bound_color,
		# 	linewidth=3)

		a_cp_dim 	= 0.03/2
		v_dim 		= 0.015/2
		thin_line	= 2
		thick_line	= 5
		white 		= (1., 1., 1.)
		dark_grey 	= (0.2, 0.2, 0.2)

		# DRAW FINAL POINTS
		for f_loc in self.block_final_pos.values():
			fx, fy = self.scale_units(f_loc)
			t = rendering.Transform(translation=(fx, fy))
			self.viewer.draw_circle(v_dim, 100, color=white).add_attr(t)
			self.viewer.draw_circle(self.scaled_epsilon/RATIO, 30, color=dark_grey, linewidth=thick_line, filled=False).add_attr(t)
			# self.viewer.draw_circle(self.scaled_epsilon/RATIO, 30, color=final_color).add_attr(t)

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
				self.viewer.draw_circle(a_cp_dim, 30, color=block_color).add_attr(t)

			if 'block' in obj.userData:
				x, y = obj.worldCenter
				# print("world center: ", x, y)
				# print("local center: ", obj.localCenter)
				t = rendering.Transform(translation=(x, y))
				self.viewer.draw_circle(a_cp_dim, 30, color=cp_color).add_attr(t)
				for v in self.blks_vertices[obj.userData]:
					x, y = obj.GetWorldPoint(v)
					t = rendering.Transform(translation=(x, y))
					self.viewer.draw_circle(v_dim, 30, color=cp_color).add_attr(t)



	def _render_agent_vision(self):
		from gym.envs.classic_control import rendering
		
		a_cp_dim 	= 0.03
		v_dim 		= 0.015
		bx, by 		= 0, 0 
		thin_line	= 2*2
		thick_line	= 5*2
		white 		= (1., 1., 1.)
		dark_grey 	= (0.2, 0.2, 0.2)
		dash = rendering.LineStyle(style=True)

		# DRAW FINAL POINTS
		for f_loc in self.block_final_pos.values():
			fx, fy = self.scale_units(f_loc)
			t = rendering.Transform(translation=(fx, fy))
			self.viewer.draw_circle(v_dim, 100, color=white).add_attr(t)
			self.viewer.draw_circle(self.scaled_epsilon/RATIO, 30, color=dark_grey, linewidth=thick_line, filled=False).add_attr(t)

		# DRAW OBJECTS
		for obj in self.drawlist:
			
			# DRAW CP
			if 'agent' in obj.userData:
				x, y = obj.position

				t = rendering.Transform(translation=(x, y))
				self.viewer.draw_circle(a_cp_dim, 30, color=white).add_attr(t)
				# Draw pointer
				vx, vy = obj.GetWorldVector(localVector=(0,0.1))
				self.viewer.draw_polyline([(x, y), (x+vx, y+vy)], color=white, linewidth=thin_line)
				if by!=0 and bx!=0:
					self.viewer.draw_polyline([(x, y), (bx, by)], color=white, linewidth=thin_line).add_attr(dash)

			if 'block' in obj.userData:
				bx, by = obj.worldCenter
				t = rendering.Transform(translation=(bx, by))
				self.viewer.draw_circle(v_dim, 30, color=cp_color).add_attr(t)
				self.viewer.draw_polyline([(bx, by), (fx, fy)], color=white, linewidth=thin_line).add_attr(dash)
				for v in self.blks_vertices[obj.userData]:
					x, y = obj.GetWorldPoint(v)
					t = rendering.Transform(translation=(x, y))
					self.viewer.draw_circle(v_dim, 30, color=cp_color).add_attr(t)

####################################################################################################################

class MultiRobotPuzzleHeavy2(MultiRobotPuzzle2):
	heavy = True


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



	# env = gym.make("MultiRobotPuzzle-v0")
	env = MultiRobotPuzzle2()
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
	
