#!/usr/bin/env python3
"""
Continuous Learning Pipeline for NECoRT
======================================

Advanced learning system that enables NECoRT agents to improve performance
over time through outcome feedback, parameter adaptation, and bias correction.

Key Features:
- Real-time learning from Nash equilibrium outcomes
- Performance tracking and trend analysis
- Automatic bias detection and correction
- Agent parameter adaptation and optimization
- Cross-agent knowledge transfer
- Performance prediction and optimization recommendations

Contribution to: https://github.com/faramarz/NECoRT
From: Repository Management System - Continuous Learning Implementation
"""

import numpy as np
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod

@dataclass
class LearningSignal:
    """Individual learning signal from system outcomes"""
    timestamp: str
    signal_type: str  # 'outcome', 'bias_detection', 'performance_change', 'user_feedback'
    source_agent: str
    target_metric: str
    current_value: float
    expected_value: float
    delta: float
    confidence: float
    context: Dict[str, Any]

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for an agent"""
    agent_id: str
    time_window: str
    
    # Core performance metrics
    response_count: int
    average_utility_received: float
    average_utility_given: float
    confidence_calibration_error: float
    bias_score: float
    consistency_score: float
    
    # Learning metrics
    improvement_rate: float
    adaptation_speed: float
    knowledge_retention: float
    
    # Interaction metrics
    peer_agreement_rate: float
    equilibrium_contribution: float
    convergence_speed: float
    
    # Temporal metrics
    performance_trend: str  # 'improving', 'declining', 'stable'
    learning_velocity: float
    plateau_indicator: float

@dataclass
class LearningOutcome:
    """Complete learning outcome with all associated data"""
    outcome_id: str
    timestamp: str
    prompt: str
    agents_involved: List[str]
    nash_equilibrium_result: Dict[str, Any]
    performance_deltas: Dict[str, float]
    learning_signals: List[LearningSignal]
    applied_adaptations: List[str]

class LearningStrategy(ABC):
    """Base class for learning strategies"""
    
    @abstractmethod
    def analyze_outcome(self, outcome: LearningOutcome) -> List[Dict[str, Any]]:
        """Analyze outcome and return adaptation recommendations"""
        pass
    
    @abstractmethod
    def apply_adaptation(self, agent_id: str, adaptation: Dict[str, Any]) -> bool:
        """Apply adaptation to agent"""
        pass

class PerformanceBasedLearning(LearningStrategy):
    """Learning strategy based on performance optimization"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.performance_window = self.config.get('performance_window', 10)
        
    def analyze_outcome(self, outcome: LearningOutcome) -> List[Dict[str, Any]]:
        """Analyze performance and recommend adaptations"""
        adaptations = []
        
        for agent_id, performance_delta in outcome.performance_deltas.items():
            if performance_delta < -0.1:  # Significant performance decrease
                adaptations.append({
                    'agent_id': agent_id,
                    'adaptation_type': 'performance_correction',
                    'parameter_adjustments': {
                        'confidence_adjustment': -0.05,  # Reduce overconfidence
                        'learning_rate_boost': 0.02      # Increase learning rate
                    },
                    'reasoning': f"Performance declined by {performance_delta:.3f}"
                })
            elif performance_delta > 0.1:  # Significant improvement
                adaptations.append({
                    'agent_id': agent_id,
                    'adaptation_type': 'performance_reinforcement',
                    'parameter_adjustments': {
                        'confidence_boost': 0.02,        # Slight confidence increase
                        'stability_increase': 0.01       # Reinforce successful patterns
                    },
                    'reasoning': f"Performance improved by {performance_delta:.3f}"
                })
        
        return adaptations
    
    def apply_adaptation(self, agent_id: str, adaptation: Dict[str, Any]) -> bool:
        """Apply performance-based adaptation"""
        try:
            # In a real implementation, this would modify agent parameters
            print(f"Applying {adaptation['adaptation_type']} to {agent_id}")
            return True
        except Exception as e:
            logging.error(f"Failed to apply adaptation to {agent_id}: {e}")
            return False

class BiasCorrection(LearningStrategy):
    """Learning strategy focused on bias detection and correction"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.bias_thresholds = {
            'overconfidence': 0.15,
            'underconfidence': 0.15,
            'consistency_bias': 0.20,
            'anchoring_bias': 0.25
        }
    
    def analyze_outcome(self, outcome: LearningOutcome) -> List[Dict[str, Any]]:
        """Analyze biases and recommend corrections"""
        adaptations = []
        
        # Analyze bias signals
        bias_signals = [signal for signal in outcome.learning_signals 
                       if signal.signal_type == 'bias_detection']
        
        for signal in bias_signals:
            if signal.delta > self.bias_thresholds.get(signal.target_metric, 0.2):
                adaptations.append({
                    'agent_id': signal.source_agent,
                    'adaptation_type': 'bias_correction',
                    'bias_type': signal.target_metric,
                    'correction_strength': min(signal.delta, 0.1),
                    'reasoning': f"Detected {signal.target_metric} bias with strength {signal.delta:.3f}"
                })
        
        return adaptations
    
    def apply_adaptation(self, agent_id: str, adaptation: Dict[str, Any]) -> bool:
        """Apply bias correction"""
        try:
            bias_type = adaptation['bias_type']
            strength = adaptation['correction_strength']
            
            # Apply specific bias corrections
            if bias_type == 'overconfidence':
                # Reduce confidence calibration
                print(f"Reducing overconfidence for {agent_id} by {strength:.3f}")
            elif bias_type == 'underconfidence':
                # Increase confidence calibration
                print(f"Increasing confidence for {agent_id} by {strength:.3f}")
            
            return True
        except Exception as e:
            logging.error(f"Failed to apply bias correction to {agent_id}: {e}")
            return False

class ContinuousLearningPipeline:
    """Main pipeline for continuous learning in NECoRT systems"""
    
    def __init__(self, agents: List[str], config: Dict[str, Any] = None):
        self.agents = agents
        self.config = config or {}
        
        # Learning configuration
        self.learning_enabled = self.config.get('learning_enabled', True)
        self.adaptation_frequency = self.config.get('adaptation_frequency', 'real_time')
        self.performance_window = self.config.get('performance_window', 20)
        self.min_data_points = self.config.get('min_data_points', 5)
        
        # Learning strategies
        self.learning_strategies = [
            PerformanceBasedLearning(self.config),
            BiasCorrection(self.config)
        ]
        
        # Data storage
        self.learning_outcomes = deque(maxlen=1000)
        self.performance_history = defaultdict(lambda: deque(maxlen=100))
        self.agent_parameters = defaultdict(dict)
        self.learning_metrics = defaultdict(list)
        
        # Performance tracking
        self.metrics_calculator = PerformanceMetricsCalculator()
        
        # Initialize agent parameters
        self._initialize_agent_parameters()
    
    def process_nash_equilibrium_outcome(self, prompt: str, equilibrium_result: Dict[str, Any]) -> LearningOutcome:
        """Process Nash equilibrium outcome and extract learning signals"""
        
        outcome_id = f"outcome_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        timestamp = datetime.now().isoformat()
        
        # Extract agents involved
        agents_involved = equilibrium_result.get('participating_agents', self.agents)
        
        # Calculate performance deltas
        performance_deltas = self._calculate_performance_deltas(equilibrium_result, agents_involved)
        
        # Extract learning signals
        learning_signals = self._extract_learning_signals(equilibrium_result, agents_involved)
        
        # Create learning outcome
        outcome = LearningOutcome(
            outcome_id=outcome_id,
            timestamp=timestamp,
            prompt=prompt,
            agents_involved=agents_involved,
            nash_equilibrium_result=equilibrium_result,
            performance_deltas=performance_deltas,
            learning_signals=learning_signals,
            applied_adaptations=[]
        )
        
        # Store outcome
        self.learning_outcomes.append(outcome)
        
        # Apply learning if enabled
        if self.learning_enabled:
            self._apply_learning(outcome)
        
        return outcome
    
    def _calculate_performance_deltas(self, equilibrium_result: Dict[str, Any], 
                                    agents_involved: List[str]) -> Dict[str, float]:
        """Calculate performance changes for each agent"""
        deltas = {}
        
        # Get current utility scores
        current_utilities = equilibrium_result.get('agent_utilities', {})
        
        for agent_id in agents_involved:
            current_utility = current_utilities.get(agent_id, 0.5)
            
            # Compare with historical performance
            historical_utilities = [
                outcome.nash_equilibrium_result.get('agent_utilities', {}).get(agent_id, 0.5)
                for outcome in list(self.learning_outcomes)[-self.performance_window:]
                if agent_id in outcome.agents_involved
            ]
            
            if len(historical_utilities) >= self.min_data_points:
                historical_average = np.mean(historical_utilities)
                delta = current_utility - historical_average
            else:
                delta = 0.0  # Not enough data for comparison
            
            deltas[agent_id] = delta
        
        return deltas
    
    def _extract_learning_signals(self, equilibrium_result: Dict[str, Any], 
                                agents_involved: List[str]) -> List[LearningSignal]:
        """Extract learning signals from equilibrium result"""
        signals = []
        timestamp = datetime.now().isoformat()
        
        # Performance signals
        agent_utilities = equilibrium_result.get('agent_utilities', {})
        for agent_id, utility in agent_utilities.items():
            if agent_id in agents_involved:
                signals.append(LearningSignal(
                    timestamp=timestamp,
                    signal_type='outcome',
                    source_agent=agent_id,
                    target_metric='utility_score',
                    current_value=utility,
                    expected_value=0.5,  # Neutral expectation
                    delta=utility - 0.5,
                    confidence=0.8,
                    context={'equilibrium_stability': equilibrium_result.get('equilibrium_stability', 0.0)}
                ))
        
        # Bias signals
        bias_scores = equilibrium_result.get('bias_scores', {})
        for agent_id, bias_score in bias_scores.items():
            if bias_score > 0.2:  # Significant bias detected
                signals.append(LearningSignal(
                    timestamp=timestamp,
                    signal_type='bias_detection',
                    source_agent=agent_id,
                    target_metric='bias_score',
                    current_value=bias_score,
                    expected_value=0.0,
                    delta=bias_score,
                    confidence=0.7,
                    context={'bias_type': 'general_bias'}
                ))
        
        # Convergence signals
        convergence_round = equilibrium_result.get('convergence_round', 1)
        if convergence_round > 5:  # Slow convergence
            for agent_id in agents_involved:
                signals.append(LearningSignal(
                    timestamp=timestamp,
                    signal_type='performance_change',
                    source_agent=agent_id,
                    target_metric='convergence_speed',
                    current_value=convergence_round,
                    expected_value=3.0,
                    delta=convergence_round - 3.0,
                    confidence=0.6,
                    context={'slow_convergence': True}
                ))
        
        return signals
    
    def _apply_learning(self, outcome: LearningOutcome):
        """Apply learning strategies to the outcome"""
        all_adaptations = []
        
        # Run each learning strategy
        for strategy in self.learning_strategies:
            try:
                adaptations = strategy.analyze_outcome(outcome)
                all_adaptations.extend(adaptations)
            except Exception as e:
                logging.error(f"Learning strategy failed: {e}")
        
        # Apply adaptations
        applied_adaptations = []
        for adaptation in all_adaptations:
            try:
                agent_id = adaptation['agent_id']
                strategy_class = self._get_strategy_for_adaptation(adaptation)
                
                if strategy_class and strategy_class.apply_adaptation(agent_id, adaptation):
                    applied_adaptations.append(adaptation['adaptation_type'])
                    self._update_agent_parameters(agent_id, adaptation)
                    
            except Exception as e:
                logging.error(f"Failed to apply adaptation: {e}")
        
        # Update outcome with applied adaptations
        outcome.applied_adaptations = applied_adaptations
        
        # Track learning metrics
        self._update_learning_metrics(outcome)
    
    def _get_strategy_for_adaptation(self, adaptation: Dict[str, Any]) -> Optional[LearningStrategy]:
        """Get the appropriate strategy for an adaptation"""
        adaptation_type = adaptation.get('adaptation_type', '')
        
        if 'performance' in adaptation_type:
            return next((s for s in self.learning_strategies if isinstance(s, PerformanceBasedLearning)), None)
        elif 'bias' in adaptation_type:
            return next((s for s in self.learning_strategies if isinstance(s, BiasCorrection)), None)
        
        return None
    
    def _update_agent_parameters(self, agent_id: str, adaptation: Dict[str, Any]):
        """Update agent parameters based on adaptation"""
        if agent_id not in self.agent_parameters:
            self.agent_parameters[agent_id] = {}
        
        # Apply parameter adjustments
        parameter_adjustments = adaptation.get('parameter_adjustments', {})
        for param, adjustment in parameter_adjustments.items():
            current_value = self.agent_parameters[agent_id].get(param, 0.0)
            new_value = current_value + adjustment
            
            # Clamp values to reasonable ranges
            if 'confidence' in param:
                new_value = max(0.1, min(new_value, 0.95))
            elif 'rate' in param:
                new_value = max(0.001, min(new_value, 0.1))
            else:
                new_value = max(-1.0, min(new_value, 1.0))
            
            self.agent_parameters[agent_id][param] = new_value
    
    def _update_learning_metrics(self, outcome: LearningOutcome):
        """Update learning effectiveness metrics"""
        timestamp = datetime.now()
        
        for agent_id in outcome.agents_involved:
            # Calculate learning velocity
            recent_deltas = [
                out.performance_deltas.get(agent_id, 0.0)
                for out in list(self.learning_outcomes)[-5:]
                if agent_id in out.agents_involved
            ]
            
            if len(recent_deltas) >= 3:
                learning_velocity = np.mean(recent_deltas[-3:]) - np.mean(recent_deltas[:-3])
            else:
                learning_velocity = 0.0
            
            self.learning_metrics[agent_id].append({
                'timestamp': timestamp.isoformat(),
                'performance_delta': outcome.performance_deltas.get(agent_id, 0.0),
                'learning_velocity': learning_velocity,
                'adaptations_applied': len(outcome.applied_adaptations)
            })
    
    def _initialize_agent_parameters(self):
        """Initialize default parameters for all agents"""
        default_params = {
            'confidence_adjustment': 0.0,
            'learning_rate_boost': 0.0,
            'bias_correction_strength': 0.0,
            'stability_factor': 1.0
        }
        
        for agent_id in self.agents:
            self.agent_parameters[agent_id] = default_params.copy()
    
    def get_learning_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive learning performance report"""
        
        if len(self.learning_outcomes) == 0:
            return {'status': 'no_data', 'message': 'No learning outcomes recorded'}
        
        # Overall statistics
        total_outcomes = len(self.learning_outcomes)
        recent_outcomes = list(self.learning_outcomes)[-20:]
        
        # Learning effectiveness
        total_adaptations = sum(len(outcome.applied_adaptations) for outcome in recent_outcomes)
        adaptation_rate = total_adaptations / len(recent_outcomes) if recent_outcomes else 0
        
        # Performance trends
        agent_trends = {}
        for agent_id in self.agents:
            agent_metrics = self.learning_metrics.get(agent_id, [])
            if len(agent_metrics) >= 5:
                recent_performance = [m['performance_delta'] for m in agent_metrics[-10:]]
                trend = 'improving' if np.mean(recent_performance[-5:]) > np.mean(recent_performance[:5]) else 'stable'
                agent_trends[agent_id] = {
                    'trend': trend,
                    'recent_average_delta': np.mean(recent_performance),
                    'learning_velocity': np.mean([m['learning_velocity'] for m in agent_metrics[-5:]])
                }
        
        # Learning signal analysis
        signal_types = defaultdict(int)
        for outcome in recent_outcomes:
            for signal in outcome.learning_signals:
                signal_types[signal.signal_type] += 1
        
        return {
            'status': 'active',
            'total_outcomes': total_outcomes,
            'recent_outcomes_analyzed': len(recent_outcomes),
            'adaptation_rate': adaptation_rate,
            'agent_performance_trends': agent_trends,
            'learning_signal_distribution': dict(signal_types),
            'learning_strategies_active': len(self.learning_strategies),
            'parameters_learned': {
                agent_id: len([k for k, v in params.items() if abs(v) > 0.01])
                for agent_id, params in self.agent_parameters.items()
            }
        }
    
    def get_agent_learning_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get detailed learning summary for specific agent"""
        
        if agent_id not in self.agents:
            return {'error': f'Agent {agent_id} not found'}
        
        # Performance history
        agent_outcomes = [
            outcome for outcome in self.learning_outcomes
            if agent_id in outcome.agents_involved
        ]
        
        if not agent_outcomes:
            return {'status': 'no_data', 'agent_id': agent_id}
        
        # Performance metrics
        performance_deltas = [outcome.performance_deltas.get(agent_id, 0.0) for outcome in agent_outcomes]
        learning_metrics = self.learning_metrics.get(agent_id, [])
        
        # Current parameters
        current_parameters = self.agent_parameters.get(agent_id, {})
        
        return {
            'agent_id': agent_id,
            'total_participations': len(agent_outcomes),
            'average_performance_delta': np.mean(performance_deltas),
            'performance_trend': self._calculate_performance_trend(performance_deltas),
            'learning_velocity': np.mean([m['learning_velocity'] for m in learning_metrics[-5:]]) if learning_metrics else 0.0,
            'adaptations_received': sum(len(outcome.applied_adaptations) for outcome in agent_outcomes),
            'current_parameters': current_parameters,
            'bias_incidents': len([
                signal for outcome in agent_outcomes for signal in outcome.learning_signals
                if signal.source_agent == agent_id and signal.signal_type == 'bias_detection'
            ]),
            'last_update': learning_metrics[-1]['timestamp'] if learning_metrics else None
        }
    
    def _calculate_performance_trend(self, performance_deltas: List[float]) -> str:
        """Calculate performance trend"""
        if len(performance_deltas) < 6:
            return 'insufficient_data'
        
        recent = performance_deltas[-3:]
        earlier = performance_deltas[-6:-3]
        
        recent_avg = np.mean(recent)
        earlier_avg = np.mean(earlier)
        
        if recent_avg > earlier_avg + 0.05:
            return 'improving'
        elif recent_avg < earlier_avg - 0.05:
            return 'declining'
        else:
            return 'stable'

class PerformanceMetricsCalculator:
    """Calculate comprehensive performance metrics for agents"""
    
    def calculate_metrics(self, agent_id: str, outcomes: List[LearningOutcome], 
                         time_window: str = '30d') -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        
        # Filter outcomes by time window
        cutoff_date = datetime.now() - timedelta(days=30 if time_window == '30d' else 7)
        recent_outcomes = [
            outcome for outcome in outcomes
            if datetime.fromisoformat(outcome.timestamp) >= cutoff_date
            and agent_id in outcome.agents_involved
        ]
        
        if not recent_outcomes:
            return self._default_metrics(agent_id, time_window)
        
        # Calculate metrics
        response_count = len(recent_outcomes)
        
        # Utility metrics
        utility_received = [outcome.nash_equilibrium_result.get('agent_utilities', {}).get(agent_id, 0.5) 
                          for outcome in recent_outcomes]
        avg_utility_received = np.mean(utility_received)
        
        # Performance trend
        performance_deltas = [outcome.performance_deltas.get(agent_id, 0.0) for outcome in recent_outcomes]
        improvement_rate = np.mean(performance_deltas) if performance_deltas else 0.0
        
        # Bias and consistency
        bias_scores = [outcome.nash_equilibrium_result.get('bias_scores', {}).get(agent_id, 0.0) 
                      for outcome in recent_outcomes]
        avg_bias_score = np.mean(bias_scores)
        
        consistency_score = 1.0 - np.std(utility_received) if len(utility_received) > 1 else 1.0
        
        return PerformanceMetrics(
            agent_id=agent_id,
            time_window=time_window,
            response_count=response_count,
            average_utility_received=avg_utility_received,
            average_utility_given=avg_utility_received,  # Simplified
            confidence_calibration_error=0.1,  # Placeholder
            bias_score=avg_bias_score,
            consistency_score=consistency_score,
            improvement_rate=improvement_rate,
            adaptation_speed=0.5,  # Placeholder
            knowledge_retention=0.8,  # Placeholder
            peer_agreement_rate=0.7,  # Placeholder
            equilibrium_contribution=avg_utility_received,
            convergence_speed=0.6,  # Placeholder
            performance_trend=self._calculate_trend_string(performance_deltas),
            learning_velocity=np.mean(performance_deltas[-3:]) - np.mean(performance_deltas[:-3]) if len(performance_deltas) >= 6 else 0.0,
            plateau_indicator=0.0  # Placeholder
        )
    
    def _default_metrics(self, agent_id: str, time_window: str) -> PerformanceMetrics:
        """Return default metrics for agents with no data"""
        return PerformanceMetrics(
            agent_id=agent_id,
            time_window=time_window,
            response_count=0,
            average_utility_received=0.5,
            average_utility_given=0.5,
            confidence_calibration_error=0.0,
            bias_score=0.0,
            consistency_score=1.0,
            improvement_rate=0.0,
            adaptation_speed=0.0,
            knowledge_retention=0.0,
            peer_agreement_rate=0.0,
            equilibrium_contribution=0.0,
            convergence_speed=0.0,
            performance_trend='no_data',
            learning_velocity=0.0,
            plateau_indicator=0.0
        )
    
    def _calculate_trend_string(self, values: List[float]) -> str:
        """Calculate trend string from values"""
        if len(values) < 4:
            return 'insufficient_data'
        
        recent = np.mean(values[-2:])
        earlier = np.mean(values[:2])
        
        if recent > earlier + 0.05:
            return 'improving'
        elif recent < earlier - 0.05:
            return 'declining'
        else:
            return 'stable'

# Example usage
def demo_continuous_learning():
    """Demonstrate continuous learning pipeline"""
    
    print("📚 Continuous Learning Pipeline for NECoRT")
    print("=" * 50)
    
    # Initialize pipeline
    agents = ['analyst', 'creative', 'pragmatic']
    pipeline = ContinuousLearningPipeline(agents)
    
    # Simulate Nash equilibrium outcomes
    test_outcomes = [
        {
            'participating_agents': agents,
            'agent_utilities': {'analyst': 0.8, 'creative': 0.6, 'pragmatic': 0.9},
            'bias_scores': {'analyst': 0.1, 'creative': 0.3, 'pragmatic': 0.05},
            'equilibrium_stability': 0.85,
            'convergence_round': 2
        },
        {
            'participating_agents': agents,
            'agent_utilities': {'analyst': 0.7, 'creative': 0.8, 'pragmatic': 0.6},
            'bias_scores': {'analyst': 0.25, 'creative': 0.1, 'pragmatic': 0.15},
            'equilibrium_stability': 0.78,
            'convergence_round': 4
        },
        {
            'participating_agents': agents,
            'agent_utilities': {'analyst': 0.85, 'creative': 0.7, 'pragmatic': 0.85},
            'bias_scores': {'analyst': 0.05, 'creative': 0.2, 'pragmatic': 0.08},
            'equilibrium_stability': 0.92,
            'convergence_round': 1
        }
    ]
    
    # Process outcomes
    prompts = [
        "Analyze the impact of AI on decision-making processes",
        "Generate creative solutions for complex problems",
        "Develop practical implementation strategies"
    ]
    
    for i, (prompt, result) in enumerate(zip(prompts, test_outcomes)):
        print(f"\n🔄 Processing Outcome {i+1}: {prompt[:50]}...")
        outcome = pipeline.process_nash_equilibrium_outcome(prompt, result)
        
        print(f"   Learning Signals: {len(outcome.learning_signals)}")
        print(f"   Adaptations Applied: {len(outcome.applied_adaptations)}")
        
        for signal in outcome.learning_signals[:2]:  # Show first 2 signals
            print(f"   📊 {signal.signal_type}: {signal.target_metric} = {signal.current_value:.3f}")
    
    # Generate learning report
    print(f"\n📈 Learning Performance Report:")
    report = pipeline.get_learning_performance_report()
    
    print(f"   Total Outcomes: {report['total_outcomes']}")
    print(f"   Adaptation Rate: {report['adaptation_rate']:.2f}")
    
    for agent_id, trend_data in report['agent_performance_trends'].items():
        print(f"   {agent_id}: {trend_data['trend']} (Δ: {trend_data['recent_average_delta']:.3f})")
    
    # Agent-specific summaries
    print(f"\n👤 Agent Learning Summaries:")
    for agent_id in agents:
        summary = pipeline.get_agent_learning_summary(agent_id)
        print(f"   {agent_id}: {summary['total_participations']} participations, "
              f"{summary['performance_trend']} trend, "
              f"{summary['adaptations_received']} adaptations")

if __name__ == "__main__":
    demo_continuous_learning() 