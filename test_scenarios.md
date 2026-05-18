> Possible test scenarios
```bash
python3 Hex.py -p1 "agents.AZ_agent.agent1 Agent1" -p2 "agents.DefaultAgents.NaiveAgent NaiveAgent" -v
python3 Hex.py -p1 "agents.AZ_agent.agent1 Agent1" -p2 "agents.MCTSAgent.MCTSAgent MCTSAgent" -v
python Hex.py -p1 "agents.AZ_agent.agent1 Agent1" -p2 "agents.TestAgents.ValidAgent ValidAgent" -v
python Hex.py -p1 "agents.AZ_agent.agent1 Agent1" -p2 "agents.AZ_agent.agent1 Agent1" -v
python Hex_temp.py -p1 "agents.AZ_agent.agent1 Agent1" -t1 0.1 -p2 "agents.AZ_agent.agent1 Agent1" -t2 0.9 -v
python Hex_temp.py -p1 "agents.AZ_agent.agent1 Agent1" -t1 0.1 -p2 "agents.AZ_agent.agent1 Agent1" -t2 0.5 -v -turns 2 -l
```
> Use this to play against any checkpoint you put under that folder:
```bash
python Hex_temp.py -p1 "agents.AZ_agent.agent1 Agent1" -t1 0.1 -p2 "agents.AZ_agent.agent1 Agent1" -t2 0.9 -v -path2 "checkpoints_eval/checkpoint_10.pt"
```
To play with a smaller board (7x7):
```bash
python Hex_temp.py -p1 "agents.AZ_agent.agent1 Agent1" -path1 "checkpoints_small/best_model.pt" -p2 "agents.AZ_agent.agent1 Agent1" -path2 "checkpoints_small/checkpoint_10.pt" -b 7
```
> AI vs AI on 7x7 board (watch-only)
```bash
python Hex_temp.py -p1 "agents.AZ_agent.agent1 Agent1" -path1 "agents/AZ_agent/checkpoints_small/best_model.pt" -p2 "agents.AZ_agent.agent1 Agent1" -path2 "agents/AZ_agent/checkpoints_small/checkpoint_10.pt" -b 7 --web
```
> Human vs AI model with custom temperature
```bash
python Hex_temp.py -p1 "agents.AZ_agent.agent1 Agent1" -path1 "agents/AZ_agent/checkpoints_small/best_model.pt" -t1 0.3 -p2 "Human" -b 7 --web
python Hex_temp.py -p2 "agents.AZ_agent.agent1 Agent1" -path2 "agents/AZ_agent/checkpoints_small/best_model.pt" -p1 "Human" -b 7 --web
python Hex_temp.py -p2 "agents.AZ_agent.agent1 Agent1" -path2 "agents/AZ_agent/checkpoints_9x9/best_model.pt" -p1 "Human" -b 9 --web
```