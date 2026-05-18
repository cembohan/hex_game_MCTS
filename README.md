# AlphaZero for Hex

This project is a from-scratch implementation of AlphaZero applied to the board game Hex. It started as a senior-year CSE Multi-Agent Systems course project and grew into something considerably larger: a fully working self-play training pipeline, a batched MCTS engine, a two-headed residual policy-value network, a swap rule implementation, and a Flask-based web UI for watching or playing games. The trained weights for the 7x7 board are included.

If you have ever looked at the AlphaZero paper and thought "that is way too complicated for one person to build" — this repo exists to prove otherwise. It is not magic. It is just a lot of careful pieces that all have to work at the same time.

---

## Table of Contents

1. [What is Hex](#what-is-hex)
2. [Architecture Overview](#architecture-overview)
3. [The Neural Network](#the-neural-network)
4. [MCTS Implementation](#mcts-implementation)
5. [Departures from AlphaGo: No Hard Eval, No Pruning](#departures-from-alphago-no-hard-eval-no-pruning)
6. [The Swap Rule](#the-swap-rule)
7. [Training Pipeline](#training-pipeline)
8. [The Hyperparameter Problem](#the-hyperparameter-problem)
9. [Small Bits That Matter](#small-bits-that-matter)
10. [Modular Game Runner and Webhook](#modular-game-runner-and-webhook)
11. [Pretrained Weights](#pretrained-weights)
12. [How to Run](#how-to-run)
13. [Test Scenarios](#test-scenarios)
14. [Acknowledgements](#acknowledgements)

---

## What is Hex

Hex is a two-player connection game played on an N×N rhombic grid. RED tries to connect top to bottom, BLUE tries to connect left to right. Players alternate placing a single stone per turn. The game is always decided — there are no draws. On an 11×11 board the branching factor starts at 121 and the average game length is around 50 moves, putting it in a difficulty class well above Tic-Tac-Toe and closer to Go than to Chess in terms of the challenge it poses to tree search.

---

## Architecture Overview

```
Self-Play (BatchedMCTS)
        │
        ▼
  Replay Buffer  ──→  Training (HexTrainer / AdamW)
        │                        │
        └────────────────────────┘
                     │
                     ▼
              HexPVNet (ResNet)
              ┌─────────────┐
              │  Policy Head│  → π  (action distribution)
              │  Value Head │  → v  (win probability ∈ [-1, 1])
              └─────────────┘
```

Everything is driven by a single model. There is no separate opponent model, no evaluation gate, and no rollout policy network. The model generates its own training data through self-play and is updated on that data in the same epoch.

---

## The Neural Network

`HexPVNet` is a standard two-headed ResNet with approximately 2.43 million parameters. The input is a 3-channel spatial tensor of shape `(3, N, N)`:

- Channel 0: current player's stones
- Channel 1: opponent's stones  
- Channel 2: a multi-purpose signal that encodes three distinct states in one channel — 1.0 on all RED turns (current-player indicator), 0.0 on BLUE turns except turn 2, and -1.0 on turn 2 specifically (the swap-legal signal). Described in detail under the swap rule section.

The network passes this through an initial convolution block followed by 8 residual blocks with 128 hidden channels, then splits into a policy head and a value head. The policy head outputs logits over `N*N + 1` actions (all board positions plus the swap action). The value head outputs a single scalar in `[-1, 1]` via Tanh.

The input encoding is always done from the perspective of the player to move: channel 0 is always the current player's stones, channel 1 is always the opponent's. This means the network never needs to learn "I am RED" or "I am BLUE" — it always reasons from a first-person perspective, which keeps the representation clean and halves the symmetry burden on the network.

---

## MCTS Implementation

The MCTS lives in `mcts.py` and has two classes: `MCTS` for single-game inference during tournament play, and `BatchedMCTS` for parallel self-play during training.

### Lazy Node Initialization

When a node is expanded, only its prior probabilities, visit counts, and value sums are initialized — as flat NumPy arrays on the parent. No child `Node` Python objects are created at expansion time. The actual `Node` object for a child is only instantiated the first time the selection phase actually decides to visit that child:

```python
# LAZY INITIALIZATION
# Only create the Python object if we actually decided to visit it
if node.children_nodes[best_action] is None:
    node.children_nodes[best_action] = Node(
        prior=node.children_priors[best_action],
        parent=node,
        action_from_parent=best_action
    )
```

On an 11×11 board, a fully expanded root would have 121 potential children. Without lazy initialization, every expansion would allocate 121 `Node` objects immediately, most of which are never visited. This matters a lot when you are running hundreds of simulations per move across dozens of parallel games.

### Weakref for Parent References

Every node holds a reference to its parent. In a naive implementation this creates a reference cycle: parent → children array → child → parent. Python's garbage collector can handle cycles, but they are expensive and cause memory to accumulate across a long training run. The fix is one line:

```python
self.parent = weakref.ref(parent) if parent is not None else None
```

A `weakref` does not prevent garbage collection. When the parent is freed (e.g. when the tree is pruned after a move is played), the child's weak reference simply becomes dead rather than keeping the parent alive.

### Board Object Pool

Cloning board state is the most frequent operation during simulation. Rather than allocating a new `Board` object on every clone, a module-level pool (`_BOARD_POOL`) keeps freed boards available for reuse. After a simulation path is finished, `release_board()` puts the board back into the pool. The next `fast_clone_board()` call pops from the pool instead of calling `Board(size)`. This eliminates the majority of allocations during the simulation loop.

### UCB Computation Without Python Loops

The UCB scores for all children are computed in a single vectorized NumPy pass using the stored arrays on the parent node:

```python
q = np.divide(node.children_values, visits,
              out=np.zeros_like(node.children_values), where=(visits > 0))
q = np.clip(q, -1.0, 1.0)
sqrt_n = math.sqrt(max(1, node.visit_count))
u = self.c_puct * node.children_priors * sqrt_n / (1 + visits)
scores = q + u
scores[~mask] = -np.inf
best_action = int(np.argmax(scores))
```

`np.divide` with `out` and `where` avoids allocating an intermediate array for the zero-visit guard. The whole selection step for one node is three NumPy operations.

### Lazy Terminal Detection

Terminal nodes are detected on the first visit and cached on the node. On subsequent visits the cached value is returned directly, skipping the board win-check entirely:

```python
if node.is_terminal:
    value = node.terminal_value  # cached, no board.has_ended() call
    terminal_hits += 1
```

### Value Discount During Backpropagation

Values are discounted by 0.99 at each step of backpropagation: `value = -value * 0.99`. This is a subtle but meaningful detail. Without it, the value signal is equally strong regardless of how far away the terminal position is, which makes the agent indifferent between winning quickly and winning slowly. The 0.99 discount nudges it toward faster wins and away from drawn-out positions.

### Tree Reuse Between Moves

After each move is played during self-play, the subtree rooted at the chosen action is preserved and reused as the root for the next position rather than discarding the entire tree:

```python
if game['root'].children_exists[action] and game['root'].children_nodes[action] is not None:
    old_root = game['root']
    new_root = old_root.children_nodes[action]
    game['root'] = new_root
    new_root.parent = None
    old_root.children_nodes[action] = None
    recursive_free(old_root)
```

The rest of the old tree is explicitly freed via `recursive_free`, which walks the tree and nulls out all array references before deletion. This is important because without it, the garbage collector accumulates a large amount of dead tree structure during a long self-play batch.

---

## Departures from AlphaGo: No Hard Eval, No Pruning

### No Hard Evaluation Gate

AlphaGo Zero used a gating mechanism: a candidate model had to beat the current best model by a threshold (55%) before it was promoted. This makes intuitive sense but adds a lot of complexity — you need to maintain a separate "best model", run a separate evaluation match, and gate promotion. It also slows learning because you discard updates that are not yet good enough to win the gate.

The later AlphaZero paper dropped the gate entirely. Training is continuous: the single model generates self-play data and is trained on it immediately, epoch after epoch. The model is always the latest version. This implementation follows AlphaZero in this regard. The evaluation in this repo (`evaluate_vs_set_checkpoint`) is purely informational — it logs win rates against a fixed old checkpoint but never gates or reverts training.

### No Pruning

Classic MCTS variants for games like Go used heuristic pruning (RAVE, AMAF, progressive widening) to cut unpromising branches early. AlphaZero's central insight was that the policy network already does this job: if the network assigns near-zero prior probability to an action, UCB will never visit it with a reasonable simulation budget. There is no need to manually cut branches.

This implementation trusts UCB completely. The only intervention at the root is Dirichlet noise for exploration during self-play, described below. During evaluation and tournament play, even noise is disabled. The exploration–exploitation balance is handled entirely by the UCB formula and the prior probabilities from the network.

---

## The Swap Rule

Hex has a known first-player advantage. The swap rule addresses this: after RED plays the first stone, BLUE may choose to swap sides and take RED's position as their own instead of responding normally. A rational player swaps if and only if RED's opening move is stronger than the average first move.

### How It Works Here

The swap decision is made by looking at the raw MCTS visit proportion that RED's opening move received during turn 1. This proportion is stored as `game['red_opener_pi']`. On turn 2, before BLUE acts, this value is checked:

```python
SWAP_PI_THRESHOLD = 0.10

def _decide_swap_turn2(game, swap_idx, mcts_root):
    red_opener_pi = game.get('red_opener_pi', None)
    if red_opener_pi is not None and red_opener_pi > SWAP_PI_THRESHOLD:
        return swap_idx  # Force swap
    return None  # Let BLUE sample normally
```

If RED's chosen move received more than 10% of all MCTS visits (meaning the network considered it substantially better than average), BLUE swaps. Otherwise BLUE plays normally.

The threshold is intentionally low. On an 11×11 board there are 121 possible first moves, so a uniform distribution would assign ~0.8% to each. A move getting 10% or more is the network saying that move is roughly 12 times more visited than a random move would be. On a 7×7 board the first-player advantage is much stronger, so the model almost always swaps because any competent opener is dominant relative to the uniform baseline.

### The CNN Signal for Swap Legality

The network needs to know when swap is a legal action, because this only applies on turn 2. Rather than encoding turn number as a scalar, channel 2 of the input tensor carries this signal:

```python
if current_colour == Colour.RED:
    out_tensor[0, 2] = 1.0   # Player-to-move indicator (RED's turns)
else:
    if turn == 2:
        out_tensor[0, 2] = -1.0  # "SWAP IS LEGAL RIGHT NOW"
    # otherwise channel 2 stays 0.0
```

On RED's turns, channel 2 is always 1.0 (a standard current-player indicator). On BLUE's turns it is 0.0 except on turn 2, when it is set to -1.0. The negative value is a deliberate choice: it is maximally different from every other state the network sees, making it easy for the convolutional layers to detect and route to swap-related policy outputs. The comment in the code calls this a "neon sign."

---

## Training Pipeline

Training alternates between two phases per epoch:

1. **Self-Play**: `GAMES_PER_EPOCH` games are run in a single batched pass through `BatchedMCTS`. All games advance one move per iteration, with the GPU evaluating all leaf states in one forward pass. Finished games are written to the replay buffer.

2. **Training**: `TRAINING_STEPS` gradient updates are made, each sampling a random mini-batch of `BATCH_SIZE` positions from the replay buffer. Loss is the sum of policy cross-entropy, value MSE, and a small entropy bonus to prevent premature collapse.

The loss function:

```python
p_loss = -torch.mean(torch.sum(target_pis * F.log_softmax(p_logits, dim=1), dim=1))
v_loss = F.mse_loss(v.view(-1), target_vs)
total_loss = p_loss + v_loss - ENTROPY_COEF * entropy
```

Invalid move masking is applied before the softmax: illegal positions are set to -1e9 in the logits, so the network cannot assign probability to occupied squares. The replay buffer stores the valid moves list alongside each position specifically to enable this.

### Data Augmentation

Every position saved to the replay buffer is immediately doubled by applying a 180-degree rotation. Hex boards are rotationally symmetric under 180-degree rotation (the topology of the connection goal is preserved). The board state, policy target, and valid moves are all rotated consistently. This doubles effective data without any additional self-play.

### Graceful Shutdown

Training registers handlers for `SIGINT` and `SIGTERM`. If you press Ctrl+C or send a kill signal, the loop finishes the current step, saves the model and replay buffer to disk, and exits cleanly. Partial self-play games that were in progress are discarded rather than saved — saving them would label all moves as losses (z = -1 for every position in an unfinished game), which would corrupt the replay buffer.

### Configurable Training for Multiple Board Sizes

`train.py` exposes a `configure()` function that overrides any module-level constant before training starts. `train_7x7.py` is a two-line file that imports `train`, calls `configure()` with the 7×7-specific paths and hyperparameters, and calls `run_training()`. There is no code duplication. To add a 5×5 variant you write the same two lines. When the training logic changes, you change it once in `train.py`.

---

## The Hyperparameter Problem

This is the part of the project that takes the longest and is the hardest to describe to someone who has not been through it.

In supervised fine-tuning, bad data hurts but does not destroy you. If 25% of your examples are mislabeled, you still learn from the other 75%. In self-play reinforcement learning, this tolerance does not exist. The model plays against itself, so by construction it wins 50% and loses 50% regardless of how strong it is. The reward signal is not about absolute quality — it is about *relative* quality within the game. If the data is bad, the model trains on bad data, becomes slightly worse, generates worse data in the next epoch, and the whole thing collapses into a policy that plays randomly. There is no simulator providing ground-truth feedback. The model is both the student and the teacher, and if the teacher is wrong the student gets worse.

This means every hyperparameter is load-bearing in a way that is not true in supervised learning:

**Games per epoch and replay buffer capacity**: Too few games per epoch and the buffer fills with stale data from an older, weaker version of the model. Too many and you are not updating fast enough. The buffer capacity controls how far back in training history your samples can come from. If the buffer is too large, positions from 500 epochs ago (where the model played randomly) still appear in batches. If it is too small, you overfit to the current policy and lose diversity.

**Training steps per epoch**: Each gradient step is a nudge based on the current buffer. More steps per epoch means more learning from the same data, but also more risk of overfitting to recent self-play patterns before new data arrives.

**Learning rate and weight decay**: Standard tuning concerns, but with a tighter tolerance than usual. A learning rate that is too high causes the policy to oscillate and never converge. AdamW's weight decay acts as a regularizer that prevents the network from memorizing specific opening sequences and keeps the policy from becoming brittle.

**c_puct**: The UCB exploration constant. Too high and MCTS ignores the policy network and explores randomly (wasting simulations). Too low and it trusts the network too much, converging on whatever suboptimal policy the network currently has and never discovering corrections. The value 1.25 used here was arrived at by watching self-play games and judging whether the agent was exploring new lines or cycling through the same moves.

**Dirichlet alpha and epsilon**: Alpha controls the concentration of the noise distribution. A small alpha (e.g. 0.03 as used in the original AlphaZero for Go) produces sparse noise where only a few actions get meaningful noise mass. A larger alpha (0.25 used here) distributes noise more evenly, which helps on smaller boards where the action space is not as large and sparse noise would leave most positions unstimulated.

**Temperature schedule**: The temperature controls how sharp or diffuse the final action selection distribution is after MCTS. High temperature (1.0) in the early game promotes exploration and diverse game trajectories. Low temperature (0.1–0.2) in the late game makes the agent more decisive and produces cleaner win/loss signals for backpropagation. The schedule here uses high temperature for roughly the first 12 turns, then drops to the low value. For 7×7, where games are much shorter, the high-temperature phase is reduced to 2 turns — otherwise most of the game runs at high temperature and the data is too noisy.

**MCTS simulations per move**: More simulations produce better policy targets at the cost of slower self-play. The training simulations (200–400 depending on board size) are a compromise. Evaluation uses more simulations (800) because quality matters more than speed there.

The key realization: unlike a standard RL problem like Atari, where the environment provides a clear and externally grounded reward, in self-play the reward is only meaningful if the games themselves are meaningful. If the model is too weak to play recognizable Hex, the win/loss signal at the end of a game is essentially random noise. You have to get the model past a threshold of competence before the training signal becomes useful at all, and getting there requires the hyperparameters to be in the right ballpark from the start. It is all-or-nothing in a way that supervised learning never is.

---

## Small Bits That Matter

**Dirichlet noise at the root**: During self-play, noise is injected into the root node's prior probabilities before any simulation runs. This forces the agent to explore moves it would otherwise rate as suboptimal, which diversifies the training data. Noise is not added during evaluation or tournament play.

**Raw pi stored in replay buffer**: The MCTS visit distribution used as the policy target is always the raw, unsharpened distribution (temperature = 1.0 internally to MCTS). The temperature sharpening for action selection happens separately and only affects which move is actually played — it is never stored as the training target. Storing a sharpened distribution would bias the network toward greedy behavior and reduce the quality of the exploration signal.

**Invalid move masking during training**: The policy head outputs logits over all `N*N + 1` positions. During the training forward pass, illegal positions (already occupied squares, and the swap action on any turn other than turn 2) are masked to -1e9 before the softmax. Without this, the network would waste capacity learning to assign near-zero probability to illegal moves, and would receive misleading gradient signal whenever an illegal position appeared in the cross-entropy target.

**AdamW over Adam**: AdamW decouples weight decay from the gradient update, which makes the regularization more effective and more predictable. In long training runs the difference is meaningful: Adam with L2 regularization in the loss effectively scales the weight decay by the adaptive learning rate, which varies per parameter. AdamW applies it uniformly.

**Entropy bonus**: A small entropy coefficient (0.007) is subtracted from the total loss, encouraging the policy to remain somewhat diffuse rather than collapsing to a near-deterministic distribution too early. If the policy collapses before the value head has learned meaningful positional evaluations, self-play games become very repetitive and the training signal degrades.

**Separate eval logger**: Training and evaluation logs are written to separate files (`training.log` and `evals.log`) with the same formatter. This makes it practical to monitor win-rate trends over time without wading through the per-epoch loss output. The eval state (last epoch evaluated) is persisted to a JSON file so that evaluation cadence survives training restarts.

---

## Modular Game Runner and Webhook

`Hex_temp.py` extends the standard `Hex.py` runner with additional arguments: per-player temperature (`-t1`, `-t2`), checkpoint paths (`-path1`, `-path2`), board size (`-b`), a turn limit (`-turns`), and a `--web` flag that starts a Flask server instead of running a CLI game.

Any agent in the repository can be pitted against any other by specifying the module path and class name on the command line. The runner uses `importlib` to load agents dynamically at runtime and `inspect.signature` to detect whether an agent accepts `temperature` and `model_path` arguments before passing them:

```python
def instantiate_agent(agent_class, colour, temperature, model_path=None):
    sig = inspect.signature(agent_class.__init__)
    if 'temperature' in sig.parameters or "model_path" in sig.parameters:
        return agent_class(colour, temperature=temperature, model_path=model_path)
    return agent_class(colour)
```

This means naive agents and other non-MCTS agents that do not accept these parameters work without any modifications. The same command-line interface works for the Flask web UI: the `--web` flag passes the player configuration to `app.py`, which sets up a browser-based board and handles human input as a player type.

---

## Pretrained Weights

Trained weights for the 7×7 board are included at:

```
agents/AZ_agent/checkpoints_small/best_model.pt
```

These weights are from the `v1.0.0` release ("The 7x7 Breakthrough"). The checkpoint includes the model state dict, optimizer state, board size, and training iteration, so training can be resumed from this point.

No 11×11 weights are included in the repository. Training a competent 11×11 model requires significantly more compute than the 7×7 case.

---

## How to Run

### Option 1: Docker (recommended for reproducibility)

Build the image:

```
docker build --build-arg UID=$UID -t hex .
```

Run the container (CPU):

```
docker run --cpus=8 --memory=8G -v "${PWD}:/home/hex" --name hex --rm -it hex /bin/bash
```

Run the container with GPU:

```
docker run --cpus=8 --memory=8G -v "${PWD}:/home/hex" --name hex --runtime=nvidia --rm -it hex /bin/bash
```

The repo directory is mounted at `/home/hex` inside the container. Changes to files are reflected immediately.

To re-enter an existing container:

```
docker start -i hex
```

### Option 2: Local (for training)

Install dependencies:

```
pip install -r requirements.txt
```

### Playing a Game

AI vs AI on 7×7 (watch in browser):

```
python Hex_temp.py -p1 "agents.AZ_agent.agent1 Agent1" -path1 "agents/AZ_agent/checkpoints_small/best_model.pt" -p2 "agents.AZ_agent.agent1 Agent1" -path2 "agents/AZ_agent/checkpoints_small/checkpoint_10.pt" -b 7 --web
```

Human vs AI (you play RED):

```
python Hex_temp.py -p1 "Human" -p2 "agents.AZ_agent.agent1 Agent1" -path2 "agents/AZ_agent/checkpoints_small/best_model.pt" -b 7 --web
```

Human vs AI (you play BLUE):

```
python Hex_temp.py -p1 "agents.AZ_agent.agent1 Agent1" -path1 "agents/AZ_agent/checkpoints_small/best_model.pt" -p2 "Human" -b 7 --web
```

AI vs Naive agent (CLI):

```
python Hex_temp.py -p1 "agents.AZ_agent.agent1 Agent1" -p2 "agents.DefaultAgents.NaiveAgent NaiveAgent" -v
```

Compare two checkpoints at different temperatures:

```
python Hex_temp.py -p1 "agents.AZ_agent.agent1 Agent1" -t1 0.1 -p2 "agents.AZ_agent.agent1 Agent1" -t2 0.9 -v
```

Evaluate against a specific old checkpoint:

```
python Hex_temp.py -p1 "agents.AZ_agent.agent1 Agent1" -t1 0.1 -p2 "agents.AZ_agent.agent1 Agent1" -t2 0.9 -v -path2 "checkpoints_eval/checkpoint_10.pt"
```

### Training

Train the 7×7 model:

```
python -m agents.AZ_agent.train_7x7
```

Train the 11×11 model:

```
python -m agents.AZ_agent.train
```

Training saves `best_model.pt` after every epoch and a numbered checkpoint every 10 epochs. If training is interrupted, it resumes automatically from `best_model.pt` on the next run. Press Ctrl+C for a graceful shutdown that saves the current state before exiting.

### Running Tests

```
python -m unittest discover
```

---

## Test Scenarios

A `test_scenarios.md` file at the repo root contains copy-pasteable command lines for common matchups and configurations.

## Acknowledgements

The Hex game simulator (`src/`) was provided as course material for CSE4080 
Multi-Agent Systems, based on original work by King Lok Chung. 
Original documentation: https://typst.app/project/wimHW-RlEYIYkqEVJWgrXC
