"""
Data processing module for quantum sensor analysis
Handles survey data processing, text analysis, and data preparation for BigQuery
"""

import pandas as pd
import numpy as np
import re
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
from dataclasses import dataclass

from config.config import config, QUANTUM_SENSOR_APPLICATIONS

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QuantumTextAnalysis:
    """Results from quantum text analysis"""
    quantum_keywords: List[str]
    application_matches: List[str]
    complexity_score: float
    market_potential: str
    business_domains: List[str]
    sentiment_score: float

class QuantumDataProcessor:
    """Processes quantum sensor data and survey responses"""
    
    def __init__(self):
        self.quantum_keywords = [
            'quantum', 'sensor', 'interferometry', 'entanglement', 'superposition',
            'coherence', 'decoherence', 'qubit', 'quantum computing', 'quantum mechanics',
            'quantum optics', 'quantum field', 'quantum state', 'quantum algorithm',
            'quantum advantage', 'quantum supremacy', 'quantum error correction',
            'quantum annealing', 'adiabatic', 'variational', 'optimization'
        ]
        
        self.business_keywords = [
            'market', 'commercial', 'business', 'industry', 'application',
            'revenue', 'profit', 'investment', 'funding', 'startup',
            'enterprise', 'customer', 'product', 'service', 'solution'
        ]
        
        self.complexity_indicators = {
            'high': ['quantum supremacy', 'error correction', 'coherence', 'entanglement'],
            'medium': ['quantum computing', 'quantum algorithm', 'optimization'],
            'low': ['quantum sensor', 'quantum optics', 'quantum field']
        }

    def load_survey_data(self, file_path: str) -> str:
        """Load survey data from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            logger.info(f"Loaded survey data from {file_path}")
            return content
        except FileNotFoundError:
            logger.error(f"Survey file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading survey data: {e}")
            raise

    def extract_quantum_concepts(self, text: str) -> List[str]:
        """Extract quantum-related concepts from text"""
        text_lower = text.lower()
        found_concepts = []
        
        for keyword in self.quantum_keywords:
            if keyword in text_lower:
                found_concepts.append(keyword)
        
        # Extract additional quantum concepts using regex
        quantum_patterns = [
            r'quantum\s+\w+',
            r'\w+\s+quantum',
            r'quantum\s+[A-Z]\w+',
            r'[A-Z]\w+\s+quantum'
        ]
        
        for pattern in quantum_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            found_concepts.extend([match.strip() for match in matches])
        
        return list(set(found_concepts))

    def analyze_business_potential(self, text: str) -> Tuple[float, str]:
        """Analyze business potential of quantum concepts"""
        text_lower = text.lower()
        
        # Count business-related keywords
        business_score = sum(1 for keyword in self.business_keywords if keyword in text_lower)
        
        # Normalize score
        max_possible_score = len(self.business_keywords)
        normalized_score = business_score / max_possible_score if max_possible_score > 0 else 0
        
        # Determine market potential
        if normalized_score > 0.7:
            market_potential = 'high'
        elif normalized_score > 0.4:
            market_potential = 'medium'
        else:
            market_potential = 'low'
        
        return normalized_score, market_potential

    def calculate_complexity_score(self, text: str) -> float:
        """Calculate technical complexity score"""
        text_lower = text.lower()
        complexity_score = 0.0
        
        for level, indicators in self.complexity_indicators.items():
            weight = {'high': 3, 'medium': 2, 'low': 1}[level]
            matches = sum(1 for indicator in indicators if indicator in text_lower)
            complexity_score += matches * weight
        
        # Normalize to 0-1 scale
        max_possible = sum(len(indicators) * {'high': 3, 'medium': 2, 'low': 1}[level] 
                          for level, indicators in self.complexity_indicators.items())
        
        return min(complexity_score / max_possible, 1.0) if max_possible > 0 else 0.0

    def match_quantum_applications(self, concepts: List[str]) -> List[str]:
        """Match extracted concepts to known quantum applications"""
        matched_applications = []
        
        for concept in concepts:
            concept_lower = concept.lower()
            for app_name, app_data in QUANTUM_SENSOR_APPLICATIONS.items():
                if any(keyword in concept_lower for keyword in app_data['description'].lower().split()):
                    matched_applications.append(app_name)
        
        return list(set(matched_applications))

    def analyze_sentiment(self, text: str) -> float:
        """Simple sentiment analysis for quantum text"""
        positive_words = ['breakthrough', 'innovation', 'potential', 'advancement', 'revolutionary', 'promising']
        negative_words = ['challenge', 'difficult', 'limitation', 'barrier', 'complex', 'expensive']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        sentiment = (positive_count - negative_count) / total_words
        return max(-1.0, min(1.0, sentiment))  # Clamp between -1 and 1

    def process_survey_data(self, file_path: str) -> QuantumTextAnalysis:
        """Process survey data and extract quantum insights"""
        logger.info("Processing survey data...")
        
        # Load survey data
        survey_text = self.load_survey_data(file_path)
        
        # Extract quantum concepts
        quantum_concepts = self.extract_quantum_concepts(survey_text)
        logger.info(f"Found {len(quantum_concepts)} quantum concepts")
        
        # Analyze business potential
        business_score, market_potential = self.analyze_business_potential(survey_text)
        
        # Calculate complexity
        complexity_score = self.calculate_complexity_score(survey_text)
        
        # Match applications
        matched_applications = self.match_quantum_applications(quantum_concepts)
        
        # Extract business domains
        business_domains = []
        for app in matched_applications:
            if app in QUANTUM_SENSOR_APPLICATIONS:
                business_domains.extend(QUANTUM_SENSOR_APPLICATIONS[app]['domains'])
        business_domains = list(set(business_domains))
        
        # Analyze sentiment
        sentiment_score = self.analyze_sentiment(survey_text)
        
        return QuantumTextAnalysis(
            quantum_keywords=quantum_concepts,
            application_matches=matched_applications,
            complexity_score=complexity_score,
            market_potential=market_potential,
            business_domains=business_domains,
            sentiment_score=sentiment_score
        )

    def create_quantum_dataset(self, analysis: QuantumTextAnalysis) -> pd.DataFrame:
        """Create structured dataset for BigQuery"""
        logger.info("Creating quantum dataset for BigQuery...")
        
        data = []
        
        # Create entries for each quantum concept
        for i, concept in enumerate(analysis.quantum_keywords):
            data.append({
                'id': f"concept_{i}",
                'type': 'quantum_concept',
                'text': concept,
                'complexity_score': analysis.complexity_score,
                'market_potential': analysis.market_potential,
                'sentiment_score': analysis.sentiment_score,
                'business_domains': json.dumps(analysis.business_domains),
                'timestamp': pd.Timestamp.now().isoformat()
            })
        
        # Create entries for each application match
        for i, app in enumerate(analysis.application_matches):
            if app in QUANTUM_SENSOR_APPLICATIONS:
                app_data = QUANTUM_SENSOR_APPLICATIONS[app]
                data.append({
                    'id': f"app_{i}",
                    'type': 'quantum_application',
                    'text': app_data['description'],
                    'complexity_score': analysis.complexity_score,
                    'market_potential': app_data['market_potential'],
                    'sentiment_score': analysis.sentiment_score,
                    'business_domains': json.dumps(app_data['domains']),
                    'timestamp': pd.Timestamp.now().isoformat()
                })
        
        return pd.DataFrame(data)

    def prepare_bigquery_data(self, analysis: QuantumTextAnalysis) -> Dict[str, pd.DataFrame]:
        """Prepare all data for BigQuery ingestion"""
        logger.info("Preparing data for BigQuery...")
        
        # Create main quantum dataset
        quantum_df = self.create_quantum_dataset(analysis)
        
        # Create business insights dataset
        business_insights = pd.DataFrame([{
            'insight_id': 'quantum_market_analysis',
            'insight_type': 'market_analysis',
            'description': f"Quantum sensor market analysis with {len(analysis.quantum_keywords)} concepts identified",
            'market_potential': analysis.market_potential,
            'complexity_level': 'high' if analysis.complexity_score > 0.7 else 'medium' if analysis.complexity_score > 0.4 else 'low',
            'business_domains': json.dumps(analysis.business_domains),
            'confidence_score': analysis.sentiment_score,
            'timestamp': pd.Timestamp.now().isoformat()
        }])
        
        return {
            'quantum_data': quantum_df,
            'business_insights': business_insights
        }

def main():
    """Test the data processor"""
    processor = QuantumDataProcessor()
    
    # Process survey data
    analysis = processor.process_survey_data(config.survey_file)
    
    print("Quantum Text Analysis Results:")
    print(f"Quantum Keywords: {analysis.quantum_keywords}")
    print(f"Application Matches: {analysis.application_matches}")
    print(f"Complexity Score: {analysis.complexity_score:.2f}")
    print(f"Market Potential: {analysis.market_potential}")
    print(f"Business Domains: {analysis.business_domains}")
    print(f"Sentiment Score: {analysis.sentiment_score:.2f}")
    
    # Prepare BigQuery data
    bq_data = processor.prepare_bigquery_data(analysis)
    print(f"\nPrepared {len(bq_data['quantum_data'])} quantum data records")
    print(f"Prepared {len(bq_data['business_insights'])} business insight records")

if __name__ == "__main__":
    main()
