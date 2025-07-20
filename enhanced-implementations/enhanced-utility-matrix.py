#!/usr/bin/env python3
"""
Enhanced Utility Matrix Design for NECoRT
=========================================

Advanced utility matrix implementation with bias detection, confidence calibration,
and multi-dimensional agent evaluation for Nash-Equilibrium systems.

Key Enhancements:
- Multi-dimensional utility evaluation beyond simple scoring
- Bias detection and mitigation in agent evaluations
- Confidence calibration and alignment scoring
- Dynamic weighting based on agent performance history
- Real-time adaptation and learning integration

Contribution to: https://github.com/faramarz/NECoRT
From: Repository Management System - Advanced Nash Equilibrium Implementation
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
import json
from datetime import datetime

@dataclass
class UtilityDimension:
    """Single dimension of utility evaluation"""
    name: str
    value: float
    weight: float
    confidence: float
    bias_indicators: Dict[str, float]
    explanation: str

@dataclass
class EnhancedUtilityScore:
    """Comprehensive utility score with multiple dimensions"""
    evaluating_agent: str
    target_agent: str
    overall_score: float
    dimensions: List[UtilityDimension]
    confidence_alignment: float
    bias_score: float
    reliability_score: float
    temporal_consistency: float
    improvement_vector: Dict[str, float]

class UtilityMatrixCalculator:
    """Enhanced utility matrix calculator with bias detection and learning"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Core dimensions for utility evaluation
        self.core_dimensions = {
            'relevance': {'weight': 0.25, 'description': 'Response relevance to prompt'},
            'quality': {'weight': 0.20, 'description': 'Overall response quality'},
            'novelty': {'weight': 0.15, 'description': 'Novel insights and creativity'},
            'logical_consistency': {'weight': 0.15, 'description': 'Logical coherence and consistency'},
            'completeness': {'weight': 0.10, 'description': 'Completeness of response'},
            'clarity': {'weight': 0.10, 'description': 'Clarity and comprehensibility'},
            'actionability': {'weight': 0.05, 'description': 'Practical actionability'}
        }
        
        # Bias detection parameters
        self.bias_thresholds = {
            'overconfidence_threshold': 0.15,  # Confidence > utility score difference
            'underconfidence_threshold': 0.15,  # Utility score > confidence difference
            'consistency_threshold': 0.20,     # Variation in repeated evaluations
            'halo_effect_threshold': 0.25,     # Correlation between dimensions
            'anchoring_threshold': 0.30        # First response advantage
        }
        
        # Learning and adaptation
        self.evaluation_history = defaultdict(list)
        self.agent_reliability_scores = defaultdict(float)
        self.dimension_performance = defaultdict(list)
        
    def calculate_enhanced_utility(self, 
                                 evaluating_agent: str,
                                 target_response: str,
                                 target_confidence: float,
                                 target_agent: str,
                                 prompt: str,
                                 context: Dict[str, Any] = None) -> EnhancedUtilityScore:
        """Calculate comprehensive utility score with bias detection"""
        
        context = context or {}
        
        # Calculate utility dimensions
        dimensions = []
        for dim_name, dim_config in self.core_dimensions.items():
            dimension = self._evaluate_dimension(
                dim_name, target_response, prompt, evaluating_agent, context
            )
            dimensions.append(dimension)
        
        # Calculate overall score
        overall_score = sum(dim.value * dim.weight for dim in dimensions)
        
        # Confidence alignment analysis
        confidence_alignment = self._calculate_confidence_alignment(
            overall_score, target_confidence, evaluating_agent
        )
        
        # Bias detection
        bias_score = self._detect_evaluation_biases(
            dimensions, target_confidence, evaluating_agent, target_agent
        )
        
        # Reliability assessment
        reliability_score = self._calculate_reliability_score(evaluating_agent, dimensions)
        
        # Temporal consistency
        temporal_consistency = self._calculate_temporal_consistency(
            evaluating_agent, target_agent, overall_score
        )
        
        # Improvement recommendations
        improvement_vector = self._generate_improvement_vector(dimensions, target_response)
        
        # Create enhanced utility score
        utility_score = EnhancedUtilityScore(
            evaluating_agent=evaluating_agent,
            target_agent=target_agent,
            overall_score=overall_score,
            dimensions=dimensions,
            confidence_alignment=confidence_alignment,
            bias_score=bias_score,
            reliability_score=reliability_score,
            temporal_consistency=temporal_consistency,
            improvement_vector=improvement_vector
        )
        
        # Store for learning
        self._record_evaluation(utility_score)
        
        return utility_score
    
    def _evaluate_dimension(self, dimension_name: str, response: str, prompt: str, 
                          evaluating_agent: str, context: Dict[str, Any]) -> UtilityDimension:
        """Evaluate a single utility dimension"""
        
        if dimension_name == 'relevance':
            value = self._calculate_relevance(response, prompt)
        elif dimension_name == 'quality':
            value = self._calculate_quality(response)
        elif dimension_name == 'novelty':
            value = self._calculate_novelty(response, context)
        elif dimension_name == 'logical_consistency':
            value = self._calculate_logical_consistency(response)
        elif dimension_name == 'completeness':
            value = self._calculate_completeness(response, prompt)
        elif dimension_name == 'clarity':
            value = self._calculate_clarity(response)
        elif dimension_name == 'actionability':
            value = self._calculate_actionability(response)
        else:
            value = 0.5  # Default neutral score
        
        # Detect dimension-specific biases
        bias_indicators = self._detect_dimension_biases(
            dimension_name, value, evaluating_agent, response
        )
        
        # Calculate confidence in this dimension evaluation
        dimension_confidence = self._calculate_dimension_confidence(
            dimension_name, value, evaluating_agent
        )
        
        return UtilityDimension(
            name=dimension_name,
            value=value,
            weight=self.core_dimensions[dimension_name]['weight'],
            confidence=dimension_confidence,
            bias_indicators=bias_indicators,
            explanation=f"{dimension_name.replace('_', ' ').title()}: {value:.3f}"
        )
    
    def _calculate_relevance(self, response: str, prompt: str) -> float:
        """Calculate relevance of response to prompt"""
        # Simple keyword overlap heuristic
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        
        if len(prompt_words) == 0:
            return 0.5
        
        overlap = len(prompt_words.intersection(response_words))
        relevance = min(overlap / len(prompt_words), 1.0)
        
        # Boost for direct question answering
        if '?' in prompt and any(word in response.lower() for word in ['answer', 'solution', 'approach']):
            relevance += 0.1
        
        return min(relevance, 1.0)
    
    def _calculate_quality(self, response: str) -> float:
        """Calculate overall quality of response"""
        quality_indicators = {
            'length_appropriateness': min(len(response) / 200, 1.0),  # Appropriate length
            'structure_presence': 1.0 if any(marker in response for marker in ['\n', '.', ':', ';']) else 0.3,
            'vocabulary_richness': min(len(set(response.lower().split())) / max(len(response.split()), 1), 1.0),
            'professional_tone': 0.8 if not any(word in response.lower() for word in ['um', 'uh', 'like']) else 0.4
        }
        
        return sum(quality_indicators.values()) / len(quality_indicators)
    
    def _calculate_novelty(self, response: str, context: Dict[str, Any]) -> float:
        """Calculate novelty and creativity of response"""
        novelty_indicators = [
            'innovative', 'novel', 'creative', 'unique', 'breakthrough', 'unconventional',
            'original', 'pioneering', 'cutting-edge', 'revolutionary'
        ]
        
        novelty_count = sum(1 for indicator in novelty_indicators if indicator in response.lower())
        base_novelty = min(novelty_count / 3, 0.8)
        
        # Check against previous responses for uniqueness
        previous_responses = context.get('previous_responses', [])
        if previous_responses:
            similarity_scores = []
            for prev_response in previous_responses:
                similarity = self._calculate_text_similarity(response, prev_response)
                similarity_scores.append(similarity)
            
            if similarity_scores:
                uniqueness = 1.0 - max(similarity_scores)
                base_novelty = (base_novelty + uniqueness) / 2
        
        return base_novelty
    
    def _calculate_logical_consistency(self, response: str) -> float:
        """Calculate logical consistency of response"""
        consistency_indicators = {
            'logical_connectors': sum(1 for connector in ['therefore', 'because', 'since', 'thus', 'however'] 
                                    if connector in response.lower()) / 5,
            'contradiction_absence': 0.0 if any(word in response.lower() for word in ['contradict', 'however not', 'but not']) else 1.0,
            'argument_structure': 0.8 if any(word in response.lower() for word in ['first', 'second', 'finally', 'conclusion']) else 0.4
        }
        
        return min(sum(consistency_indicators.values()) / len(consistency_indicators), 1.0)
    
    def _calculate_completeness(self, response: str, prompt: str) -> float:
        """Calculate completeness of response relative to prompt"""
        # Count question words in prompt
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        questions_asked = sum(1 for word in question_words if word in prompt.lower())
        
        if questions_asked == 0:
            return 0.7  # Neutral for non-question prompts
        
        # Check if response addresses questions
        addressing_indicators = ['answer', 'solution', 'because', 'by', 'through', 'via']
        addresses_count = sum(1 for indicator in addressing_indicators if indicator in response.lower())
        
        completeness = min(addresses_count / questions_asked, 1.0)
        
        # Boost for comprehensive responses
        if len(response) > 150 and addresses_count >= questions_asked:
            completeness += 0.1
        
        return min(completeness, 1.0)
    
    def _calculate_clarity(self, response: str) -> float:
        """Calculate clarity and comprehensibility"""
        sentences = response.split('.')
        if len(sentences) == 0:
            return 0.3
        
        # Average sentence length (optimal around 15-20 words)
        avg_sentence_length = sum(len(sentence.split()) for sentence in sentences) / len(sentences)
        length_score = 1.0 - min(abs(avg_sentence_length - 17.5) / 17.5, 0.5)
        
        # Clarity indicators
        clarity_indicators = {
            'simple_language': 0.8 if not any(len(word) > 12 for word in response.split()) else 0.4,
            'clear_structure': 0.9 if any(marker in response for marker in ['\n', '1.', '2.', '-', '*']) else 0.5,
            'jargon_absence': 0.7 if response.count('(') < 3 else 0.3  # Minimal parenthetical explanations
        }
        
        clarity_score = (length_score + sum(clarity_indicators.values()) / len(clarity_indicators)) / 2
        return min(clarity_score, 1.0)
    
    def _calculate_actionability(self, response: str) -> float:
        """Calculate practical actionability of response"""
        action_indicators = [
            'implement', 'apply', 'use', 'try', 'consider', 'adopt', 'integrate',
            'start', 'begin', 'create', 'develop', 'build', 'establish'
        ]
        
        action_count = sum(1 for indicator in action_indicators if indicator in response.lower())
        base_actionability = min(action_count / 3, 0.8)
        
        # Boost for specific recommendations
        if any(phrase in response.lower() for phrase in ['recommend', 'suggest', 'should', 'steps']):
            base_actionability += 0.1
        
        # Check for concrete examples
        if any(phrase in response.lower() for phrase in ['example', 'instance', 'case', 'such as']):
            base_actionability += 0.1
        
        return min(base_actionability, 1.0)
    
    def _calculate_confidence_alignment(self, utility_score: float, target_confidence: float, 
                                     evaluating_agent: str) -> float:
        """Calculate how well confidence aligns with utility"""
        alignment = 1.0 - abs(utility_score - target_confidence)
        
        # Apply agent-specific calibration
        agent_reliability = self.agent_reliability_scores.get(evaluating_agent, 0.5)
        calibrated_alignment = alignment * agent_reliability + (1 - agent_reliability) * 0.5
        
        return calibrated_alignment
    
    def _detect_evaluation_biases(self, dimensions: List[UtilityDimension], 
                                target_confidence: float, evaluating_agent: str, 
                                target_agent: str) -> float:
        """Detect various evaluation biases"""
        bias_score = 0.0
        bias_count = 0
        
        # Overconfidence bias
        avg_dimension_value = np.mean([dim.value for dim in dimensions])
        if target_confidence - avg_dimension_value > self.bias_thresholds['overconfidence_threshold']:
            bias_score += 0.3
            bias_count += 1
        
        # Underconfidence bias
        if avg_dimension_value - target_confidence > self.bias_thresholds['underconfidence_threshold']:
            bias_score += 0.2
            bias_count += 1
        
        # Halo effect (high correlation between dimensions)
        dimension_values = [dim.value for dim in dimensions]
        if len(dimension_values) > 1:
            correlation_matrix = np.corrcoef(dimension_values)
            if np.mean(correlation_matrix[correlation_matrix != 1.0]) > self.bias_thresholds['halo_effect_threshold']:
                bias_score += 0.25
                bias_count += 1
        
        # Agent favoritism (consistent overrating of specific agents)
        agent_history = [eval_record for eval_record in self.evaluation_history[evaluating_agent] 
                        if eval_record.get('target_agent') == target_agent]
        if len(agent_history) > 3:
            recent_scores = [record['overall_score'] for record in agent_history[-3:]]
            if all(score > 0.8 for score in recent_scores):
                bias_score += 0.2
                bias_count += 1
        
        return bias_score / max(bias_count, 1)
    
    def _calculate_reliability_score(self, evaluating_agent: str, 
                                   dimensions: List[UtilityDimension]) -> float:
        """Calculate reliability of the evaluating agent"""
        base_reliability = self.agent_reliability_scores.get(evaluating_agent, 0.5)
        
        # Adjust based on dimension confidence
        avg_dimension_confidence = np.mean([dim.confidence for dim in dimensions])
        
        # Consistency with past evaluations
        consistency_score = 1.0
        if evaluating_agent in self.evaluation_history:
            recent_evaluations = self.evaluation_history[evaluating_agent][-5:]
            if len(recent_evaluations) > 1:
                recent_scores = [eval_record['overall_score'] for eval_record in recent_evaluations]
                consistency_score = 1.0 - min(np.std(recent_scores), 0.3) / 0.3
        
        reliability = (base_reliability + avg_dimension_confidence + consistency_score) / 3
        return min(reliability, 1.0)
    
    def _calculate_temporal_consistency(self, evaluating_agent: str, target_agent: str, 
                                      current_score: float) -> float:
        """Calculate temporal consistency of evaluations"""
        if evaluating_agent not in self.evaluation_history:
            return 0.5  # Neutral for new agents
        
        # Find previous evaluations of same target agent
        previous_evaluations = [
            eval_record for eval_record in self.evaluation_history[evaluating_agent]
            if eval_record.get('target_agent') == target_agent
        ]
        
        if len(previous_evaluations) < 2:
            return 0.5
        
        # Calculate consistency
        previous_scores = [record['overall_score'] for record in previous_evaluations[-3:]]
        score_variance = np.var(previous_scores + [current_score])
        
        consistency = 1.0 - min(score_variance * 5, 1.0)  # Scale variance to 0-1
        return consistency
    
    def _generate_improvement_vector(self, dimensions: List[UtilityDimension], 
                                   response: str) -> Dict[str, float]:
        """Generate improvement recommendations"""
        improvements = {}
        
        for dimension in dimensions:
            if dimension.value < 0.6:  # Room for improvement
                improvement_potential = 0.8 - dimension.value
                improvements[dimension.name] = improvement_potential * dimension.weight
        
        return improvements
    
    def _detect_dimension_biases(self, dimension_name: str, value: float, 
                               evaluating_agent: str, response: str) -> Dict[str, float]:
        """Detect biases specific to individual dimensions"""
        biases = {}
        
        # Length bias (overvaluing longer responses)
        if dimension_name in ['quality', 'completeness'] and len(response) > 300:
            if value > 0.8:
                biases['length_bias'] = 0.3
        
        # Complexity bias (overvaluing complex language)
        if dimension_name == 'quality':
            complex_words = sum(1 for word in response.split() if len(word) > 10)
            if complex_words > 5 and value > 0.8:
                biases['complexity_bias'] = 0.25
        
        # Novelty bias (overvaluing anything that appears new)
        if dimension_name == 'novelty' and value > 0.9:
            biases['novelty_bias'] = 0.2
        
        return biases
    
    def _calculate_dimension_confidence(self, dimension_name: str, value: float, 
                                      evaluating_agent: str) -> float:
        """Calculate confidence in dimension evaluation"""
        # Base confidence varies by dimension type
        base_confidences = {
            'relevance': 0.8,      # Relatively objective
            'clarity': 0.7,        # Mostly objective
            'completeness': 0.6,   # Somewhat subjective
            'quality': 0.5,        # Subjective
            'novelty': 0.4,        # Highly subjective
            'logical_consistency': 0.7,
            'actionability': 0.6
        }
        
        base_confidence = base_confidences.get(dimension_name, 0.5)
        
        # Adjust based on agent reliability
        agent_reliability = self.agent_reliability_scores.get(evaluating_agent, 0.5)
        
        # Adjust based on value extremeness (extreme values often less reliable)
        extremeness_penalty = abs(value - 0.5) * 0.2
        
        confidence = base_confidence * agent_reliability - extremeness_penalty
        return max(0.1, min(confidence, 0.95))
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if len(words1) == 0 and len(words2) == 0:
            return 1.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if len(union) > 0 else 0.0
    
    def _record_evaluation(self, utility_score: EnhancedUtilityScore):
        """Record evaluation for learning and bias detection"""
        evaluation_record = {
            'timestamp': datetime.now().isoformat(),
            'target_agent': utility_score.target_agent,
            'overall_score': utility_score.overall_score,
            'confidence_alignment': utility_score.confidence_alignment,
            'bias_score': utility_score.bias_score,
            'reliability_score': utility_score.reliability_score,
            'dimensions': {dim.name: dim.value for dim in utility_score.dimensions}
        }
        
        self.evaluation_history[utility_score.evaluating_agent].append(evaluation_record)
        
        # Update agent reliability
        self._update_agent_reliability(utility_score.evaluating_agent, utility_score)
    
    def _update_agent_reliability(self, agent_id: str, utility_score: EnhancedUtilityScore):
        """Update agent reliability based on evaluation quality"""
        current_reliability = self.agent_reliability_scores.get(agent_id, 0.5)
        
        # Factors that increase reliability
        reliability_factors = [
            utility_score.confidence_alignment,
            1.0 - utility_score.bias_score,
            utility_score.temporal_consistency,
            np.mean([dim.confidence for dim in utility_score.dimensions])
        ]
        
        new_reliability_signal = np.mean(reliability_factors)
        
        # Update with exponential smoothing
        alpha = 0.1  # Learning rate
        updated_reliability = current_reliability * (1 - alpha) + new_reliability_signal * alpha
        
        self.agent_reliability_scores[agent_id] = min(max(updated_reliability, 0.1), 0.95)

def build_enhanced_utility_matrix(agents: List[str], responses: List[Dict[str, Any]], 
                                prompt: str, context: Dict[str, Any] = None) -> Dict[str, Dict[str, EnhancedUtilityScore]]:
    """Build enhanced utility matrix with bias detection"""
    
    calculator = UtilityMatrixCalculator()
    matrix = {}
    
    for i, evaluating_agent in enumerate(agents):
        matrix[evaluating_agent] = {}
        
        for j, response_data in enumerate(responses):
            if i != j:  # Don't evaluate self
                utility_score = calculator.calculate_enhanced_utility(
                    evaluating_agent=evaluating_agent,
                    target_response=response_data['content'],
                    target_confidence=response_data['confidence'],
                    target_agent=response_data['agent'],
                    prompt=prompt,
                    context=context
                )
                matrix[evaluating_agent][response_data['agent']] = utility_score
    
    return matrix

# Example usage
def demo_enhanced_utility_matrix():
    """Demonstrate enhanced utility matrix with bias detection"""
    
    print("🎯 Enhanced Utility Matrix with Bias Detection")
    print("=" * 50)
    
    # Sample data
    agents = ["analyst", "creative", "pragmatic"]
    responses = [
        {
            'agent': 'analyst', 
            'content': 'A systematic analysis reveals three key factors: data quality, algorithmic bias, and validation methodology. Each requires specific interventions.',
            'confidence': 0.8
        },
        {
            'agent': 'creative', 
            'content': 'Imagine AI systems as collaborative orchestras - each instrument (algorithm) must harmonize with others while maintaining its unique voice.',
            'confidence': 0.7
        },
        {
            'agent': 'pragmatic', 
            'content': 'Implement robust testing frameworks, establish clear performance benchmarks, and create feedback loops for continuous improvement.',
            'confidence': 0.9
        }
    ]
    
    prompt = "How can we improve AI decision-making systems to reduce overconfidence?"
    
    # Build enhanced utility matrix
    matrix = build_enhanced_utility_matrix(agents, responses, prompt)
    
    # Display results
    for evaluating_agent, evaluations in matrix.items():
        print(f"\n👤 {evaluating_agent.upper()} Evaluations:")
        
        for target_agent, utility_score in evaluations.items():
            print(f"   🎯 {target_agent}: {utility_score.overall_score:.3f}")
            print(f"      Confidence Alignment: {utility_score.confidence_alignment:.3f}")
            print(f"      Bias Score: {utility_score.bias_score:.3f}")
            print(f"      Reliability: {utility_score.reliability_score:.3f}")
            
            # Top dimensions
            top_dims = sorted(utility_score.dimensions, key=lambda x: x.value, reverse=True)[:3]
            print(f"      Top Dimensions: {', '.join([f'{d.name}({d.value:.2f})' for d in top_dims])}")
            
            if utility_score.improvement_vector:
                improvements = sorted(utility_score.improvement_vector.items(), 
                                    key=lambda x: x[1], reverse=True)[:2]
                print(f"      Improvements: {', '.join([f'{k}({v:.2f})' for k, v in improvements])}")

if __name__ == "__main__":
    demo_enhanced_utility_matrix() 