# NECoRT (Nash-Equilibrium Chain of Recursive Thoughts) 🧠🔄🎮

## TL;DR: AI agents compete and collaborate to reach optimal equilibrium responses. Evolution meets Game Theory.

### What is NECoRT?

NECoRT extends the Chain of Recursive Thoughts (CoRT) framework by integrating Nash Equilibrium concepts from game theory. It creates a multi-agent ecosystem where AI instances:

1. Generate diverse responses to the same prompt
2. Evaluate each other's responses 
3. Improve their responses based on group feedback
4. Converge on a stable equilibrium where no agent would unilaterally change their strategy

The result is responses that are not just recursive improvements but represent optimal consensus points where competing strategies reach equilibrium.

### How is this different from regular CoRT?

| Feature | CoRT | NECoRT |
|---------|------|--------|
| Thinking strategy | Single agent refining own thoughts | Multiple agents competing and evaluating |
| Improvement mechanism | Generate alternatives & pick best | Game theoretic utility optimization |
| Termination condition | Fixed rounds | Dynamic convergence to equilibrium |
| Theoretical foundation | Self-reflection | Nash Equilibrium in game theory |
| Output stability | Varies with each run | Converges to stable equilibria |

## The Nash Equilibrium Advantage

In game theory, a Nash Equilibrium is a state where no player can gain advantage by changing only their own strategy, given what others are doing. NECoRT applies this to AI reasoning by:

1. **Multiple Perspectives**: Creates a utility matrix of how agents rate each other's responses
2. **Strategic Improvements**: Agents learn from highest-rated responses
3. **Convergence Detection**: Automatically identifies when the system reaches equilibrium
4. **Optimal Selection**: Chooses the response that represents the best equilibrium point

## How to Use NECoRT

### Quick Start

```bash
# On Windows
start-necort.bat

# On Linux
pip install -r requirements.txt
cd frontend && npm install
cd ..
python ./necort_web.py

# In a separate terminal
cd frontend
npm start
```

### API Usage

```python
from nash_recursive_thinking import NashEquilibriumRecursiveChat

# Initialize with your API key
necort = NashEquilibriumRecursiveChat(
    api_key="your_openrouter_api_key",
    num_agents=3,
    convergence_threshold=0.05
)

# Get an equilibrium-optimized response
result = necort.think_and_respond("Your complex question here")
print(result["response"])

# Examine the Nash Equilibrium process
print(f"Converged in {result['convergence_round']} rounds")
print(f"Final response from agent {result['final_response_agent']}")
```

## Technical Implementation

NECoRT implements:

1. **Utility Matrix Construction**: Each agent evaluates all other agents' responses
2. **Nash Equilibrium Detection**: Identifies response sets that represent stable equilibria
3. **Convergence Monitoring**: Tracks changes in utility matrix until stabilization
4. **Equilibrium Response Selection**: Picks optimal response from the equilibrium set

## Comparison to Other Methods

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| Standard LLM | Fast, single response | Limited reflection |
| Chain of Thought | Shows reasoning steps | Linear thought process |
| CoRT | Recursive improvement | Single perspective |
| NECoRT | Multi-agent equilibrium, stability, handles divergent ideas | More compute-intensive |

## Future Directions

- **Mixed Strategy Equilibria**: Allow probabilistic combinations of responses
- **Evolutionary Dynamics**: Implement replicator dynamics for response evolution
- **Coalition Formation**: Allow agent groups to form voting blocs
- **Subgame Perfection**: Extend to multi-stage reasoning games

## Contributing

Contributions are welcome! Areas particularly in need of improvement:
- Optimization of Nash Equilibrium search algorithms
- UI improvements for visualizing agent interactions
- Integration with more LLM providers

## License

MIT License - See LICENSE file for details

---

*"Let your thoughts argue, evolve, and stabilize."* 