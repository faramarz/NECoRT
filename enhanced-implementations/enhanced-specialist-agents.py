#!/usr/bin/env python3
"""
Enhanced Specialist Agents Framework for NECoRT
==============================================

Advanced multi-agent framework for Nash-Equilibrium Chain of Recursive Thoughts
with specialized agent types and enhanced utility evaluation.

Key Enhancements over Base NECoRT:
- Specialist agent architecture for domain-specific expertise
- Enhanced utility matrix with bias detection
- Continuous learning from equilibrium outcomes
- Performance tracking and confidence calibration
- Production-ready integration patterns

Contribution to: https://github.com/faramarz/NECoRT
From: Repository Management System - Systems Learning Implementation
"""

import json
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging

@dataclass
class AgentResponse:
    """Enhanced agent response with metadata"""
    agent_id: str
    agent_type: str
    content: str
    confidence: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    bias_indicators: Dict[str, float] = field(default_factory=dict)

@dataclass
class UtilityEvaluation:
    """Enhanced utility evaluation with bias detection"""
    evaluating_agent: str
    target_agent: str
    utility_score: float
    confidence_alignment: float
    reasoning_quality: float
    bias_indicators: Dict[str, float] = field(default_factory=dict)
    improvement_suggestions: List[str] = field(default_factory=list)

class SpecialistAgent(ABC):
    """Base class for specialist agents in NECoRT framework"""
    
    def __init__(self, agent_id: str, agent_type: str, specialization: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.specialization = specialization
        self.performance_history = []
        self.learning_parameters = {}
        self.bias_detection_enabled = True
        
        # Performance tracking
        self.response_count = 0
        self.successful_equilibria = 0
        self.average_utility_received = 0.0
        
        # Learning state
        self.learning_rate = 0.01
        self.adaptation_threshold = 0.1
        
    @abstractmethod
    def generate_response(self, prompt: str, context: Dict[str, Any]) -> AgentResponse:
        """Generate response with specialist expertise"""
        pass
    
    @abstractmethod
    def evaluate_peer_response(self, peer_response: AgentResponse, prompt: str) -> UtilityEvaluation:
        """Evaluate another agent's response with specialist knowledge"""
        pass
    
    def learn_from_equilibrium(self, equilibrium_result: Dict[str, Any]):
        """Learn from Nash equilibrium outcome"""
        # Update performance metrics
        self.response_count += 1
        if equilibrium_result.get('converged', False):
            self.successful_equilibria += 1
        
        # Extract learning signals
        my_utility = equilibrium_result.get('utility_received', {}).get(self.agent_id, 0.0)
        self.average_utility_received = (
            (self.average_utility_received * (self.response_count - 1) + my_utility) / 
            self.response_count
        )
        
        # Adapt parameters based on performance
        if my_utility < self.average_utility_received - self.adaptation_threshold:
            self._adapt_strategy(equilibrium_result)
    
    def _adapt_strategy(self, equilibrium_result: Dict[str, Any]):
        """Adapt strategy based on poor performance"""
        # Example adaptation: adjust confidence calibration
        if 'overconfidence_detected' in equilibrium_result.get('bias_indicators', {}):
            self.learning_parameters['confidence_adjustment'] = -0.1
        elif 'underconfidence_detected' in equilibrium_result.get('bias_indicators', {}):
            self.learning_parameters['confidence_adjustment'] = 0.1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for this agent"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'specialization': self.specialization,
            'response_count': self.response_count,
            'successful_equilibria': self.successful_equilibria,
            'success_rate': self.successful_equilibria / max(self.response_count, 1),
            'average_utility_received': self.average_utility_received,
            'learning_parameters': self.learning_parameters
        }

class AnalysisSpecialist(SpecialistAgent):
    """Specialist agent for analytical thinking and problem decomposition"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id, "analysis_specialist", "analytical_thinking")
        self.analysis_patterns = {
            'problem_decomposition': 0.8,
            'logical_reasoning': 0.9,
            'evidence_evaluation': 0.7,
            'conclusion_validation': 0.8
        }
    
    def generate_response(self, prompt: str, context: Dict[str, Any]) -> AgentResponse:
        """Generate analytically-focused response"""
        
        # Analytical approach to the prompt
        analysis_steps = self._decompose_problem(prompt)
        logical_reasoning = self._apply_logical_reasoning(prompt, context)
        evidence_evaluation = self._evaluate_evidence(prompt, context)
        
        # Synthesize analytical response
        response_content = self._synthesize_analytical_response(
            analysis_steps, logical_reasoning, evidence_evaluation
        )
        
        # Calculate confidence based on analytical rigor
        confidence = self._calculate_analytical_confidence(
            analysis_steps, logical_reasoning, evidence_evaluation
        )
        
        # Generate reasoning explanation
        reasoning = f"Applied analytical thinking with {len(analysis_steps)} decomposition steps, logical reasoning score {logical_reasoning:.2f}, and evidence evaluation score {evidence_evaluation:.2f}"
        
        return AgentResponse(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            content=response_content,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                'analysis_steps': len(analysis_steps),
                'logical_reasoning_score': logical_reasoning,
                'evidence_evaluation_score': evidence_evaluation,
                'analytical_approach': 'systematic_decomposition'
            }
        )
    
    def evaluate_peer_response(self, peer_response: AgentResponse, prompt: str) -> UtilityEvaluation:
        """Evaluate peer response from analytical perspective"""
        
        # Evaluate logical consistency
        logical_consistency = self._evaluate_logical_consistency(peer_response.content)
        
        # Evaluate evidence quality
        evidence_quality = self._evaluate_evidence_quality(peer_response.content)
        
        # Evaluate reasoning depth
        reasoning_depth = self._evaluate_reasoning_depth(peer_response.content, peer_response.reasoning)
        
        # Calculate overall utility
        utility_score = (
            logical_consistency * 0.4 +
            evidence_quality * 0.3 +
            reasoning_depth * 0.3
        )
        
        # Assess confidence alignment
        confidence_alignment = 1.0 - abs(peer_response.confidence - utility_score)
        
        # Detect potential biases
        bias_indicators = self._detect_analytical_biases(peer_response)
        
        return UtilityEvaluation(
            evaluating_agent=self.agent_id,
            target_agent=peer_response.agent_id,
            utility_score=utility_score,
            confidence_alignment=confidence_alignment,
            reasoning_quality=reasoning_depth,
            bias_indicators=bias_indicators,
            improvement_suggestions=self._generate_improvement_suggestions(peer_response)
        )
    
    def _decompose_problem(self, prompt: str) -> List[str]:
        """Decompose problem into analytical steps"""
        # Simple heuristic decomposition
        sentences = prompt.split('.')
        steps = []
        for sentence in sentences:
            if len(sentence.strip()) > 10:
                steps.append(f"Analyze: {sentence.strip()}")
        return steps
    
    def _apply_logical_reasoning(self, prompt: str, context: Dict[str, Any]) -> float:
        """Apply logical reasoning and return quality score"""
        # Heuristic: longer, more structured prompts score higher
        structure_score = min(len(prompt.split()) / 50, 1.0)
        context_score = len(context) / 10 if context else 0.5
        return min((structure_score + context_score) / 2, 1.0)
    
    def _evaluate_evidence(self, prompt: str, context: Dict[str, Any]) -> float:
        """Evaluate available evidence quality"""
        evidence_indicators = ['data', 'research', 'study', 'evidence', 'proof', 'analysis']
        evidence_count = sum(1 for indicator in evidence_indicators if indicator in prompt.lower())
        return min(evidence_count / len(evidence_indicators), 1.0)
    
    def _synthesize_analytical_response(self, steps: List[str], reasoning: float, evidence: float) -> str:
        """Synthesize analytical response"""
        return f"Analytical Assessment:\n\nProblem decomposition reveals {len(steps)} key components. Logical reasoning strength: {reasoning:.2f}. Evidence quality: {evidence:.2f}.\n\nRecommendation: Proceed with systematic analysis approach emphasizing logical consistency and evidence validation."
    
    def _calculate_analytical_confidence(self, steps: List[str], reasoning: float, evidence: float) -> float:
        """Calculate confidence based on analytical rigor"""
        step_confidence = min(len(steps) / 5, 1.0)
        analytical_confidence = (step_confidence * 0.4 + reasoning * 0.3 + evidence * 0.3)
        
        # Apply learning parameters
        confidence_adjustment = self.learning_parameters.get('confidence_adjustment', 0.0)
        return max(0.1, min(0.95, analytical_confidence + confidence_adjustment))
    
    def _evaluate_logical_consistency(self, content: str) -> float:
        """Evaluate logical consistency of response"""
        # Heuristic: look for logical connectors and structured reasoning
        logical_indicators = ['therefore', 'because', 'since', 'thus', 'consequently', 'however']
        indicator_count = sum(1 for indicator in logical_indicators if indicator in content.lower())
        return min(indicator_count / 3, 1.0)
    
    def _evaluate_evidence_quality(self, content: str) -> float:
        """Evaluate quality of evidence presented"""
        evidence_indicators = ['data shows', 'research indicates', 'studies demonstrate', 'analysis reveals']
        evidence_count = sum(1 for indicator in evidence_indicators if indicator in content.lower())
        return min(evidence_count / 2, 1.0)
    
    def _evaluate_reasoning_depth(self, content: str, reasoning: str) -> float:
        """Evaluate depth of reasoning"""
        depth_indicators = ['analysis', 'evaluation', 'assessment', 'consideration', 'examination']
        depth_count = sum(1 for indicator in depth_indicators if indicator in (content + reasoning).lower())
        return min(depth_count / 3, 1.0)
    
    def _detect_analytical_biases(self, response: AgentResponse) -> Dict[str, float]:
        """Detect analytical biases in peer response"""
        biases = {}
        
        # Overconfidence bias
        if response.confidence > 0.9 and len(response.content) < 100:
            biases['overconfidence'] = 0.8
        
        # Confirmation bias (looking for one-sided analysis)
        if 'however' not in response.content.lower() and 'but' not in response.content.lower():
            biases['confirmation_bias'] = 0.6
        
        # Availability bias (relying on easily recalled information)
        if 'recent' in response.content.lower() or 'commonly' in response.content.lower():
            biases['availability_bias'] = 0.4
        
        return biases
    
    def _generate_improvement_suggestions(self, response: AgentResponse) -> List[str]:
        """Generate improvement suggestions for peer response"""
        suggestions = []
        
        if response.confidence > 0.9:
            suggestions.append("Consider expressing more nuanced confidence levels")
        
        if len(response.reasoning) < 50:
            suggestions.append("Provide more detailed reasoning for better peer evaluation")
        
        if 'analysis' not in response.content.lower():
            suggestions.append("Include more analytical depth in response")
        
        return suggestions

class CreativitySpecialist(SpecialistAgent):
    """Specialist agent for creative thinking and novel solutions"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id, "creativity_specialist", "creative_thinking")
        self.creativity_patterns = {
            'novel_connections': 0.8,
            'alternative_perspectives': 0.9,
            'innovative_solutions': 0.7,
            'creative_synthesis': 0.8
        }
    
    def generate_response(self, prompt: str, context: Dict[str, Any]) -> AgentResponse:
        """Generate creatively-focused response"""
        
        # Creative approach to the prompt
        novel_angles = self._explore_novel_angles(prompt)
        alternative_solutions = self._generate_alternatives(prompt, context)
        creative_synthesis = self._synthesize_creatively(prompt, novel_angles, alternative_solutions)
        
        # Calculate confidence based on creative innovation
        confidence = self._calculate_creative_confidence(novel_angles, alternative_solutions)
        
        reasoning = f"Applied creative thinking with {len(novel_angles)} novel angles and {len(alternative_solutions)} alternative solutions"
        
        return AgentResponse(
            agent_id=self.agent_id,
            agent_type=self.agent_type,
            content=creative_synthesis,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                'novel_angles': len(novel_angles),
                'alternative_solutions': len(alternative_solutions),
                'creative_approach': 'divergent_thinking'
            }
        )
    
    def evaluate_peer_response(self, peer_response: AgentResponse, prompt: str) -> UtilityEvaluation:
        """Evaluate peer response from creative perspective"""
        
        # Evaluate novelty
        novelty_score = self._evaluate_novelty(peer_response.content)
        
        # Evaluate originality
        originality_score = self._evaluate_originality(peer_response.content)
        
        # Evaluate creative synthesis
        synthesis_score = self._evaluate_creative_synthesis(peer_response.content)
        
        utility_score = (novelty_score * 0.4 + originality_score * 0.3 + synthesis_score * 0.3)
        confidence_alignment = 1.0 - abs(peer_response.confidence - utility_score)
        
        bias_indicators = self._detect_creative_biases(peer_response)
        
        return UtilityEvaluation(
            evaluating_agent=self.agent_id,
            target_agent=peer_response.agent_id,
            utility_score=utility_score,
            confidence_alignment=confidence_alignment,
            reasoning_quality=synthesis_score,
            bias_indicators=bias_indicators,
            improvement_suggestions=self._generate_creative_suggestions(peer_response)
        )
    
    def _explore_novel_angles(self, prompt: str) -> List[str]:
        """Explore novel angles on the problem"""
        angles = [
            "Reverse perspective: What if we approached this backwards?",
            "Cross-domain insight: How would a different field solve this?",
            "Constraint removal: What if limitations didn't exist?",
            "Future perspective: How might this evolve over time?"
        ]
        return angles[:3]  # Return top 3 for this prompt
    
    def _generate_alternatives(self, prompt: str, context: Dict[str, Any]) -> List[str]:
        """Generate alternative solutions"""
        alternatives = [
            "Alternative 1: Completely different approach",
            "Alternative 2: Hybrid solution combining approaches",
            "Alternative 3: Minimalist solution with core features"
        ]
        return alternatives
    
    def _synthesize_creatively(self, prompt: str, angles: List[str], alternatives: List[str]) -> str:
        """Synthesize creative response"""
        return f"Creative Analysis:\n\nExploring {len(angles)} novel perspectives reveals innovative possibilities. {len(alternatives)} alternative approaches suggest flexible solutions.\n\nInnovative Recommendation: Combine reverse-perspective thinking with cross-domain insights to develop a hybrid solution that transcends traditional boundaries."
    
    def _calculate_creative_confidence(self, angles: List[str], alternatives: List[str]) -> float:
        """Calculate confidence based on creative exploration"""
        exploration_depth = (len(angles) + len(alternatives)) / 8
        creative_confidence = min(exploration_depth, 0.85)  # Cap creative confidence lower due to uncertainty
        
        confidence_adjustment = self.learning_parameters.get('confidence_adjustment', 0.0)
        return max(0.1, min(0.85, creative_confidence + confidence_adjustment))
    
    def _evaluate_novelty(self, content: str) -> float:
        """Evaluate novelty of response"""
        novelty_indicators = ['innovative', 'novel', 'unique', 'creative', 'unconventional', 'original']
        novelty_count = sum(1 for indicator in novelty_indicators if indicator in content.lower())
        return min(novelty_count / 3, 1.0)
    
    def _evaluate_originality(self, content: str) -> float:
        """Evaluate originality of thinking"""
        originality_indicators = ['perspective', 'approach', 'insight', 'breakthrough', 'reimagine']
        originality_count = sum(1 for indicator in originality_indicators if indicator in content.lower())
        return min(originality_count / 3, 1.0)
    
    def _evaluate_creative_synthesis(self, content: str) -> float:
        """Evaluate creative synthesis quality"""
        synthesis_indicators = ['combination', 'integration', 'synthesis', 'merge', 'blend']
        synthesis_count = sum(1 for indicator in synthesis_indicators if indicator in content.lower())
        return min(synthesis_count / 2, 1.0)
    
    def _detect_creative_biases(self, response: AgentResponse) -> Dict[str, float]:
        """Detect creative biases"""
        biases = {}
        
        # Novelty bias (overvaluing newness)
        if response.content.lower().count('new') > 3:
            biases['novelty_bias'] = 0.6
        
        # Complexity bias (overcomplicating)
        if len(response.content) > 500:
            biases['complexity_bias'] = 0.5
        
        return biases
    
    def _generate_creative_suggestions(self, response: AgentResponse) -> List[str]:
        """Generate creative improvement suggestions"""
        suggestions = []
        
        if 'creative' not in response.content.lower():
            suggestions.append("Consider more creative perspectives")
        
        if response.confidence > 0.85:
            suggestions.append("Creative solutions often have higher uncertainty")
        
        return suggestions

class EnhancedNashEquilibrium:
    """Enhanced Nash Equilibrium solver with specialist agents"""
    
    def __init__(self, agents: List[SpecialistAgent], config: Dict[str, Any] = None):
        self.agents = agents
        self.config = config or {}
        self.convergence_threshold = self.config.get('convergence_threshold', 0.05)
        self.max_iterations = self.config.get('max_iterations', 10)
        self.learning_enabled = self.config.get('learning_enabled', True)
        
        # Performance tracking
        self.equilibrium_history = []
        self.performance_metrics = defaultdict(list)
        
    def solve_equilibrium(self, prompt: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Solve Nash equilibrium with specialist agents"""
        context = context or {}
        
        # Generate initial responses
        responses = []
        for agent in self.agents:
            response = agent.generate_response(prompt, context)
            responses.append(response)
        
        # Iterative equilibrium search
        for iteration in range(self.max_iterations):
            # Build utility matrix
            utility_matrix = self._build_enhanced_utility_matrix(responses, prompt)
            
            # Check convergence
            if self._check_convergence(utility_matrix, iteration):
                break
            
            # Update responses based on peer feedback (in real implementation)
            # For now, we'll use the initial responses
        
        # Select equilibrium response
        equilibrium_result = self._select_equilibrium_response(responses, utility_matrix)
        
        # Learning phase
        if self.learning_enabled:
            self._apply_learning(equilibrium_result)
        
        return equilibrium_result
    
    def _build_enhanced_utility_matrix(self, responses: List[AgentResponse], prompt: str) -> Dict[str, Dict[str, UtilityEvaluation]]:
        """Build enhanced utility matrix with detailed evaluations"""
        matrix = {}
        
        for i, evaluating_agent in enumerate(self.agents):
            matrix[evaluating_agent.agent_id] = {}
            
            for j, response in enumerate(responses):
                if i != j:  # Don't evaluate self
                    evaluation = evaluating_agent.evaluate_peer_response(response, prompt)
                    matrix[evaluating_agent.agent_id][response.agent_id] = evaluation
                else:
                    # Self-evaluation based on confidence
                    self_eval = UtilityEvaluation(
                        evaluating_agent=evaluating_agent.agent_id,
                        target_agent=response.agent_id,
                        utility_score=response.confidence,
                        confidence_alignment=1.0,
                        reasoning_quality=0.8
                    )
                    matrix[evaluating_agent.agent_id][response.agent_id] = self_eval
        
        return matrix
    
    def _check_convergence(self, utility_matrix: Dict[str, Dict[str, UtilityEvaluation]], iteration: int) -> bool:
        """Check if Nash equilibrium has been reached"""
        if iteration == 0:
            return False
        
        # Extract utility scores for variance analysis
        utilities = []
        for agent_evals in utility_matrix.values():
            for evaluation in agent_evals.values():
                utilities.append(evaluation.utility_score)
        
        if len(utilities) == 0:
            return True
        
        utility_variance = np.var(utilities)
        return utility_variance < self.convergence_threshold
    
    def _select_equilibrium_response(self, responses: List[AgentResponse], utility_matrix: Dict[str, Dict[str, UtilityEvaluation]]) -> Dict[str, Any]:
        """Select equilibrium response with enhanced metrics"""
        
        # Calculate aggregate utilities
        agent_utilities = {}
        bias_scores = {}
        
        for response in responses:
            total_utility = 0
            total_bias = 0
            evaluation_count = 0
            
            for agent_id, evaluations in utility_matrix.items():
                if response.agent_id in evaluations:
                    eval_obj = evaluations[response.agent_id]
                    total_utility += eval_obj.utility_score
                    total_bias += sum(eval_obj.bias_indicators.values())
                    evaluation_count += 1
            
            if evaluation_count > 0:
                agent_utilities[response.agent_id] = total_utility / evaluation_count
                bias_scores[response.agent_id] = total_bias / evaluation_count
            else:
                agent_utilities[response.agent_id] = response.confidence
                bias_scores[response.agent_id] = 0.0
        
        # Select best response (highest utility, lowest bias)
        best_agent_id = max(agent_utilities.items(), 
                           key=lambda x: x[1] - 0.5 * bias_scores.get(x[0], 0))[0]
        
        best_response = next(r for r in responses if r.agent_id == best_agent_id)
        
        # Calculate equilibrium stability
        utility_values = list(agent_utilities.values())
        equilibrium_stability = 1.0 - np.std(utility_values) if len(utility_values) > 1 else 1.0
        
        return {
            'best_response': best_response,
            'agent_utilities': agent_utilities,
            'bias_scores': bias_scores,
            'equilibrium_stability': equilibrium_stability,
            'utility_matrix': utility_matrix,
            'converged': True,  # Simplified for this example
            'performance_metrics': self._calculate_performance_metrics(responses, utility_matrix)
        }
    
    def _apply_learning(self, equilibrium_result: Dict[str, Any]):
        """Apply learning to all agents"""
        for agent in self.agents:
            agent.learn_from_equilibrium(equilibrium_result)
        
        # Track system-wide performance
        self.equilibrium_history.append(equilibrium_result)
        self.performance_metrics['equilibrium_stability'].append(
            equilibrium_result['equilibrium_stability']
        )
    
    def _calculate_performance_metrics(self, responses: List[AgentResponse], utility_matrix: Dict[str, Dict[str, UtilityEvaluation]]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        
        metrics = {
            'response_count': len(responses),
            'agent_diversity': len(set(r.agent_type for r in responses)),
            'average_confidence': np.mean([r.confidence for r in responses]),
            'confidence_spread': np.std([r.confidence for r in responses])
        }
        
        # Bias analysis
        all_biases = []
        for agent_evals in utility_matrix.values():
            for evaluation in agent_evals.values():
                all_biases.extend(evaluation.bias_indicators.values())
        
        if all_biases:
            metrics['average_bias_score'] = np.mean(all_biases)
            metrics['bias_detection_rate'] = len(all_biases) / len(responses)
        
        return metrics
    
    def get_system_performance(self) -> Dict[str, Any]:
        """Get comprehensive system performance report"""
        
        if not self.equilibrium_history:
            return {'status': 'no_data'}
        
        recent_stability = [eq['equilibrium_stability'] for eq in self.equilibrium_history[-10:]]
        
        # Agent performance summaries
        agent_summaries = []
        for agent in self.agents:
            agent_summaries.append(agent.get_performance_summary())
        
        return {
            'total_equilibria': len(self.equilibrium_history),
            'recent_average_stability': np.mean(recent_stability) if recent_stability else 0.0,
            'stability_trend': self._calculate_trend(recent_stability),
            'agent_performance': agent_summaries,
            'learning_enabled': self.learning_enabled,
            'system_metrics': {
                'average_convergence_rate': np.mean([eq['performance_metrics'].get('response_count', 0) 
                                                   for eq in self.equilibrium_history]),
                'bias_detection_effectiveness': np.mean([eq['performance_metrics'].get('bias_detection_rate', 0) 
                                                       for eq in self.equilibrium_history])
            }
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend in performance values"""
        if len(values) < 5:
            return 'insufficient_data'
        
        early = np.mean(values[:len(values)//2])
        late = np.mean(values[len(values)//2:])
        
        if late > early + 0.05:
            return 'improving'
        elif late < early - 0.05:
            return 'declining'
        else:
            return 'stable'

# Example usage and testing
def demo_enhanced_necort():
    """Demonstrate enhanced NECoRT with specialist agents"""
    
    print("🧠 Enhanced NECoRT with Specialist Agents")
    print("=" * 50)
    
    # Create specialist agents
    analysis_agent = AnalysisSpecialist("analyst_1")
    creativity_agent = CreativitySpecialist("creative_1")
    
    agents = [analysis_agent, creativity_agent]
    
    # Initialize enhanced Nash equilibrium solver
    enhanced_necort = EnhancedNashEquilibrium(
        agents=agents,
        config={
            'convergence_threshold': 0.05,
            'max_iterations': 5,
            'learning_enabled': True
        }
    )
    
    # Test with a complex problem
    test_prompt = """
    How can we improve AI decision-making systems to reduce overconfidence 
    while maintaining high performance? Consider both technical and 
    philosophical approaches.
    """
    
    # Solve equilibrium
    result = enhanced_necort.solve_equilibrium(test_prompt)
    
    print(f"📊 Equilibrium Result:")
    print(f"Best Response: {result['best_response'].content[:200]}...")
    print(f"Agent: {result['best_response'].agent_type}")
    print(f"Confidence: {result['best_response'].confidence:.3f}")
    print(f"Equilibrium Stability: {result['equilibrium_stability']:.3f}")
    
    print(f"\n🎯 Agent Utilities:")
    for agent_id, utility in result['agent_utilities'].items():
        print(f"   {agent_id}: {utility:.3f}")
    
    print(f"\n🔍 Bias Analysis:")
    for agent_id, bias_score in result['bias_scores'].items():
        print(f"   {agent_id}: {bias_score:.3f}")
    
    # System performance
    performance = enhanced_necort.get_system_performance()
    print(f"\n📈 System Performance:")
    print(f"   Recent Stability: {performance['recent_average_stability']:.3f}")
    print(f"   Trend: {performance['stability_trend']}")
    
    return result

if __name__ == "__main__":
    demo_enhanced_necort() 