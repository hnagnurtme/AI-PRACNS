# Optimization Algorithms: PSO, ACO, and RL

This directory contains implementations of various optimization algorithms used in our project:

## Particle Swarm Optimization (PSO)

PSO is a population-based stochastic optimization technique inspired by the social behavior of birds flocking or fish schooling. The algorithm maintains a population of candidate solutions (particles) and moves them around in the search space according to simple mathematical formulas.

## Ant Colony Optimization (ACO)

ACO is a probabilistic technique for solving computational problems which can be reduced to finding good paths through graphs. Inspired by the behavior of ants seeking a path between their colony and a source of food, the algorithm uses artificial "ants" to construct solutions guided by pheromone trails and heuristic information.

## Reinforcement Learning (RL)

RL is an area of machine learning concerned with how intelligent agents ought to take actions in an environment to maximize the notion of cumulative reward. Unlike supervised learning, RL learns from interaction with the environment through trial and error, rather than from a training dataset.

## Directory Structure

```
optimization/
├── pso/        # Particle Swarm Optimization implementation
├── aco/        # Ant Colony Optimization implementation
├── rl/         # Reinforcement Learning implementation
└── utils/      # Shared utilities and helper functions
```

## Usage

Details on how to use each algorithm can be found in their respective subdirectories.