# [Network-Destroy-and-Repair-Game](https://drive.google.com/file/d/1eBTJ-DKfCF_z8ZjQzMImO99z4dqZC8Ah/view?usp=sharing)
Training auto-agents to destroy or repair the complex network based on reinforcement learning<br>
This is course project for [EE382 Complex Network in Real World](https://drive.google.com/file/d/1HydBfSnyvJAxpsKcDcdSa6OlQfjouy_7/view?usp=sharing). 

## Overview of the project
How networks’ protection mechanisms respond to environmental adversity is a vital task of network design. This project will explore the protection mechanism and design a
destroy-repair game between two parties. Reinforcement learning, e.g, Markov Decision Processes or Q-learning, will be used to explore the optimal destroy-repair strategies when given certain objectives and network information, and a multi-agent game based on the learned strategies will be further developed in the physical network setting. Few existing papers investigated the dynamic interaction process incomplex networks, especially in physical networks, which makes this project novel and exciting.

## Description
**myenv.py**: Develop a game environment inherited from openai.gym which describes the rule of network, cost/reward of attacker and defender, winning condition and action set.
**NetGame.py**: Define classes 'Attacker', 'Defender' and 'Game', provides 
