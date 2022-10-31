from typing import TYPE_CHECKING, Optional
import Box2D
from Box2D.b2 import (polygonShape, fixtureDef)
from gym.envs.classic_control import rendering

import random
import warnings



# BLOCK SETTINGS
FR 	    = 2.5 # 0.999 # friction (between bodies)
DAMP 	= 5.0 # damping

BLOCK_OPTIONS = ["T", "L", "I"]

class Block(object):
    def __init__(
        self, 
        world: Box2D.b2World, 
        init_angle: float, 
        init_x: float, 
        init_y: float,
        scale: float = 1.0,
        density: float = 5.0,
        shape: Optional [str] = None
    ) -> None:

        self.block = None
        self.block_vertices = []
        self.scale = scale
        self.density = density
        self.world = world
        self._agent_contact = False # store for contact listener
        self.block_color = (0.5, 0.5, 0.5)
        self.block_vert_color = (1., 1., 1.)

        # Choose block shape
        if isinstance(shape, str):
            shape = shape.upper()
        if shape not in BLOCK_OPTIONS:
            warnings.warn(
                f"WARN: Block shape {shape} is not supported. Choose between [T, L, I]. Choosing shape at random"
            )
            self.shape = random.choice(["T", "L", "I"])
        else:
            self.shape = shape

        self._userData = f"block_{shape.lower()}"
        self.generate_block(x=init_x, y=init_y, rot=init_angle, shape=self.shape, scale=scale)
    

    @property
    def userData(self):
        return self._userData

    @property
    def worldCenter(self):
        return self.block.worldCenter

    @property
    def angle(self):
        return self.block.angle

    @property
    def agent_contact(self):
        return self._agent_contact


    def generate_block(self, x, y, rot, shape: str = "T", scale: float = 1.0):

        block = self.world.CreateDynamicBody(
            position = (x, y),
            angle=rot, 
            linearDamping=DAMP, 
            angularDamping = DAMP,
            userData=self._userData
        )

        if shape == "T": # t_block
            block.CreatePolygonFixture(
                box=(1 * scale, 1 * scale, (0., -1 * scale),0), 
                density=self.density, 
                friction=FR, 
                restitution=0.)
            block.CreatePolygonFixture(
                box=(3 * scale, 1 * scale, (0., 1 * scale),0), 
                density=self.density, 
                friction=FR, 
                restitution=0.)
        
        elif shape == "L": # l_block
            block.CreatePolygonFixture(
                box=(1 * scale, 1 * scale, (1 * scale, 0.5 * scale), 0), 
                density=self.density,
                friction=FR, 
                restitution=0.)
            block.CreatePolygonFixture(
                box=(1 * scale, 2 * scale, (-1 * scale, -0.5 * scale), 0), 
                density=self.density, 
                friction=FR, 
                restitution=0.)
        
        else: # i_block
            block.CreatePolygonFixture(
                box=(1 * scale, 2 * scale), 
                density=self.density, 
                friction=FR, 
                restitution=0.)

        # SAVE vertices data
        for fix in block.fixtures:
            extend_v = [v for v in fix.shape.vertices if v not in self.block_vertices]
            self.block_vertices.extend(extend_v)
        
        self.block = block

    def get_vertices(self, norm_fn):
        vertices = []
        for v in self.block_vertices:
            x, y = self.block.GetWorldPoint(v)
            vertices.extend(norm_fn(x, y))
        return vertices

    def apply_soft_force(self, force):
        # apply soft force
        self.block.ApplyForce(force, self.block.worldCenter, True)
    

    def destroy(self):
        self.world.DestroyBody(self.block)
        self.block = None


    def draw(self, viewer, mode="human"):
        # Draw body
        if mode != "agent":
            for f in self.block.fixtures:
                trans = f.body.transform
                path = [trans*v for v in f.shape.vertices]
                viewer.draw_polygon(path, color=self.block_color)
   
        # Draw center point + vertices
        x, y = self.block.worldCenter
        t = rendering.Transform(translation=(x, y))
        viewer.draw_circle(0.16, 30, color=self.block_vert_color).add_attr(t)
        for v in self.block_vertices:
            x, y = self.block.GetWorldPoint(v)
            t = rendering.Transform(translation=(x, y))
            viewer.draw_circle(0.08, 30, color=self.block_vert_color).add_attr(t)
       
        return viewer