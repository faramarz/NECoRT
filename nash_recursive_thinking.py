from recursive_thinking_ai import EnhancedRecursiveThinkingChat
import numpy as np
from typing import List, Dict, Tuple
import json

class NashEquilibriumRecursiveChat(EnhancedRecursiveThinkingChat):
    """
    NECoRT: Nash-Equilibrium Chain of Recursive Thoughts
    
    This extension implements Nash Equilibrium concepts to find an optimal
    response set where no agent would unilaterally change their strategy.
    """
    
    def __init__(self, api_key: str = None, model: str = "mistralai/mistral-small-3.1-24b-instruct:free",
                 num_agents: int = 3, convergence_threshold: float = 0.05):
        """Initialize with OpenRouter API and Nash Equilibrium parameters."""
        super().__init__(api_key, model)
        self.num_agents = num_agents
        self.convergence_threshold = convergence_threshold
        self.utility_matrix = None
        self.nash_equilibrium_log = []
    
    def _evaluate_response_utility(self, prompt: str, responses: List[str]) -> np.ndarray:
        """
        Create a utility matrix where each element [i,j] represents how agent i 
        rates the response from agent j.
        """
        n = len(responses)
        utility_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:  # Self-evaluation is biased, set to 0
                    continue
                
                eval_prompt = f"""Original message: {prompt}

I am evaluating multiple responses. Rate this response from 0-10 (10 being best):

Response being rated: {responses[j]}

Consider accuracy, relevance, clarity, and completeness.
Respond with ONLY a single number from 0-10."""
                
                messages = [{"role": "user", "content": eval_prompt}]
                evaluation = self._call_api(messages, temperature=0.1, stream=False)
                
                try:
                    score = float(''.join(filter(str.isdigit, evaluation.split('.')[0])))
                    utility_matrix[i, j] = min(score, 10)  # Ensure max is 10
                except:
                    utility_matrix[i, j] = 5  # Default to neutral if parsing fails
        
        return utility_matrix
    
    def _find_nash_equilibrium(self, utility_matrix: np.ndarray) -> List[int]:
        """
        Find the Nash Equilibrium in the utility matrix.
        Returns indices of responses that are part of the equilibrium.
        """
        n = utility_matrix.shape[0]
        
        # For each agent, find their best response to others
        best_responses = []
        for i in range(n):
            # Each agent's utility for each response
            agent_utilities = utility_matrix[i, :]
            best_response = np.argmax(agent_utilities)
            best_responses.append(best_response)
        
        # Find responses that are mutual best responses
        equilibrium_indices = []
        for i in range(n):
            is_equilibrium = True
            for j in range(n):
                if j != i and best_responses[j] != i:
                    is_equilibrium = False
                    break
            
            if is_equilibrium:
                equilibrium_indices.append(i)
        
        # If no pure Nash equilibrium is found, return the response with highest average utility
        if not equilibrium_indices:
            avg_utilities = np.mean(utility_matrix, axis=0)
            equilibrium_indices = [np.argmax(avg_utilities)]
        
        return equilibrium_indices
    
    def think_and_respond(self, user_input: str, verbose: bool = True) -> Dict:
        """Process user input with Nash Equilibrium recursive thinking."""
        print("\n" + "=" * 50)
        print("🤔 NASH EQUILIBRIUM RECURSIVE THINKING PROCESS STARTING")
        print("=" * 50)
        
        thinking_rounds = self._determine_thinking_rounds(user_input)
        
        if verbose:
            print(f"\n🤔 Thinking... ({thinking_rounds} rounds needed)")
        
        # Generate initial responses from each agent
        print("\n=== GENERATING INITIAL RESPONSES FROM ALL AGENTS ===")
        agent_responses = []
        messages = self.conversation_history + [{"role": "user", "content": user_input}]
        
        for i in range(self.num_agents):
            temperature = 0.7 + (i * 0.1)  # Vary temperature to get diverse responses
            response = self._call_api(messages, temperature=temperature, stream=True)
            agent_responses.append(response)
            print(f"\nAgent {i+1} initial response complete.")
        print("=" * 50)
        
        nash_thinking_history = [{
            "round": 0, 
            "agent_responses": agent_responses.copy(),
            "equilibrium_indices": [],
            "utility_matrix": None
        }]
        
        # Iterative improvement and equilibrium finding
        converged = False
        for round_num in range(1, thinking_rounds + 1):
            if verbose:
                print(f"\n=== ROUND {round_num}/{thinking_rounds} ===")
            
            if converged:
                print("Nash Equilibrium converged. Stopping early.")
                break
            
            # Evaluate utility of each response for each agent
            utility_matrix = self._evaluate_response_utility(user_input, agent_responses)
            
            if verbose:
                print("\nUtility Matrix:")
                print(utility_matrix)
            
            # Find Nash Equilibrium
            equilibrium_indices = self._find_nash_equilibrium(utility_matrix)
            
            if verbose:
                print(f"\nNash Equilibrium Indices: {equilibrium_indices}")
            
            # Check if we've converged
            if round_num > 1:
                prev_matrix = nash_thinking_history[-1]["utility_matrix"]
                if prev_matrix is not None:
                    diff = np.mean(np.abs(utility_matrix - prev_matrix))
                    if diff < self.convergence_threshold:
                        converged = True
                        print(f"\nConverged with difference: {diff:.4f}")
            
            # Generate improved responses based on utility
            if not converged:
                for i in range(self.num_agents):
                    # Agent improves by learning from responses with high utility
                    highest_utility_response_idx = np.argmax(utility_matrix[i, :])
                    highest_utility_response = agent_responses[highest_utility_response_idx]
                    
                    improve_prompt = f"""Original message: {user_input}

Highly rated response: {highest_utility_response}

Your current response: {agent_responses[i]}

Improve your response by learning from the highly rated response while maintaining your unique perspective.
Improved response:"""
                    
                    improve_messages = [{"role": "user", "content": improve_prompt}]
                    improved_response = self._call_api(improve_messages, temperature=0.5, stream=True)
                    agent_responses[i] = improved_response
            
            # Store history
            nash_thinking_history.append({
                "round": round_num,
                "agent_responses": agent_responses.copy(),
                "equilibrium_indices": equilibrium_indices,
                "utility_matrix": utility_matrix.tolist() if utility_matrix is not None else None
            })
        
        # Select final response from equilibrium
        final_round = nash_thinking_history[-1]
        equilibrium_indices = final_round["equilibrium_indices"]
        
        if not equilibrium_indices:
            # If no equilibrium, use response with highest average utility
            utility_matrix = np.array(final_round["utility_matrix"])
            avg_utilities = np.mean(utility_matrix, axis=0)
            final_response_idx = np.argmax(avg_utilities)
        else:
            # If multiple equilibria, choose the one with highest overall utility
            if len(equilibrium_indices) > 1:
                utility_matrix = np.array(final_round["utility_matrix"])
                avg_utilities = np.mean(utility_matrix, axis=0)
                max_utility = -1
                final_response_idx = equilibrium_indices[0]
                
                for idx in equilibrium_indices:
                    if avg_utilities[idx] > max_utility:
                        max_utility = avg_utilities[idx]
                        final_response_idx = idx
            else:
                final_response_idx = equilibrium_indices[0]
        
        final_response = agent_responses[final_response_idx]
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": final_response})
        
        # Add to full thinking log
        self.full_thinking_log.append({
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "nash_thinking_history": nash_thinking_history,
            "final_response": final_response,
            "final_response_agent": final_response_idx,
            "thinking_rounds": round_num if converged else thinking_rounds,
            "converged": converged,
            "convergence_round": round_num if converged else None
        })
        
        return {
            "response": final_response,
            "thinking_rounds": round_num if converged else thinking_rounds,
            "thinking_history": nash_thinking_history,
            "converged": converged,
            "convergence_round": round_num if converged else None,
            "final_response_agent": final_response_idx
        }
    
    def save_nash_equilibrium_log(self, filename: str = None):
        """Save the full Nash Equilibrium thinking log to a file."""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"nash_equilibrium_log_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.full_thinking_log, f, indent=2)
        
        print(f"Nash Equilibrium log saved to {filename}")

# Test function
def main():
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        sys.exit(1)
    
    chat = NashEquilibriumRecursiveChat(api_key=api_key)
    
    while True:
        user_input = input("\n> ")
        if user_input.lower() in ['exit', 'quit', 'q']:
            break
        
        result = chat.think_and_respond(user_input)
        print("\n🤖 FINAL RESPONSE:")
        print(result["response"])
        
        # Option to save logs
        if input("\nSave logs? (y/n): ").lower() == 'y':
            chat.save_nash_equilibrium_log()

if __name__ == "__main__":
    import os
    import sys
    main() 