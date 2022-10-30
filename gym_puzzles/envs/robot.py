import Box2D
from Box2D.b2 import (polygonShape, fixtureDef)
from gym.envs.classic_control import rendering



AGENT_POLY = [
	(-0.039,-0.095), (0.039,-0.095), (0.095,-0.039), (0.095,0.039), 
	(0.039,0.095), (-0.039,0.095), (-0.095,0.039), (-0.095,-0.039)
]

# ROBOT SETTINGS
FR 	    = 2.5 # 0.999 # friction (between bodies)
DAMP 	= 5.0		# damping


class Robot(object):
    def __init__(
        self, 
        world: Box2D.b2World, 
        init_angle: float, 
        init_x: float, 
        init_y: float,
        name: str,
        max_speed: float,
        scale: float = 1.0,
        density: float = 5.0,
    ) -> None:
        
        self.world = world
        self.max_speed = max_speed
        self._name = name

        self.agent = self.world.CreateDynamicBody(
				position = (init_x, init_y),
				angle=init_angle, 
				fixtures = fixtureDef(
					shape=polygonShape(vertices=[(x*scale,y*scale) for x,y in AGENT_POLY]),
                    density=density,
                    restitution=0.0),
				# linearDamping=DAMP, 
				# angularDamping=DAMP,
				userData=name,
        )

        self._goal_contact = False # track contact with goal block
        self.agent.color = (1., 1., 1.)
    
    @property
    def userData(self):
        return self._name

    @property
    def worldCenter(self):
        return self.agent.worldCenter

    @property
    def goal_contact(self):
        return self._goal_contact

    def step(self, x, y, rot):
        # TODO: make the contol system more realistic
        self.agent.linearVelocity = x * self.max_speed, y * self.max_speed
        self.agent.angularVelocity = float(rot) # won't take numpy.float32 - needs to be float

    
    def destroy(self):
        self.world.DestroyBody(self.agent)
        self.agent = None


    def draw(self, viewer):
        # Draw body
        for f in self.agent.fixtures:
            trans = f.body.transform
            path = [trans*v for v in f.shape.vertices]
            viewer.draw_polygon(path, color=self.agent.color)
   
        # draw center point
        x, y = self.agent.position
        t = rendering.Transform(translation=(x, y))
        viewer.draw_circle(0.16, 30, color=(0.5, 0.5, 0.5)).add_attr(t)
        return viewer