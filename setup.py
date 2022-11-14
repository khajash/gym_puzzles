#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
      name="gym_puzzles",
      version="0.0.2",
      install_requires=[
            "gym==0.21",
            "numpy",
            "box2d-py",
            "pyglet",
      ],
      description="Custom OpenAI gym env using pybox2D where agents solve puzzles",
      author="Kate Hajash",
      url="https://github.com/khajash/gym_puzzles",
      author_email="kshajash@gmail.com",
      keywords="reinforcement-learning gym openai puzzle",
      packages=find_packages(exclude=['imgs', 'train']),
)
