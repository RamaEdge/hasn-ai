#!/usr/bin/env python3
"""
Real-World Experience Collector - Captures new experiences from external sources
Replaces static templates with dynamic discovery from internet, APIs, sensors, etc.
"""

import asyncio
import aiohttp
import json
import time
import logging
import sys
import os
import re
import hashlib
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Generator
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
from urllib.parse import urlparse, urljoin, quote_plus
import feedparser
import requests
from bs4 import BeautifulSoup

# Add path for brain imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.cognitive_brain_network import CognitiveBrainNetwork, CognitiveConfig


@dataclass
class ExternalExperience:
    """Represents a real-world experience captured from external sources"""
    experience_id: str
    source_type: str  # 'web', 'api', 'rss', 'sensor', 'social'
    source_url: str
    content: str
    metadata: Dict[str, Any]
    timestamp: float
    quality_score: float
    discovered_concepts: List[str] = field(default_factory=list)
    neural_pattern: Dict[int, bool] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


class ConceptExtractor:
    """Extracts concepts and patterns from raw external data"""
    
    def __init__(self):
        self.concept_patterns = {
            # Scientific concepts
            'scientific': [
                r'\b(?:research|study|experiment|hypothesis|theory|discovery)\b',
                r'\b(?:DNA|protein|molecule|atom|electron|quantum)\b',
                r'\b(?:climate|temperature|weather|atmosphere|ocean)\b'
            ],
            # Technology concepts  
            'technology': [
                r'\b(?:AI|artificial intelligence|machine learning|neural network)\b',
                r'\b(?:blockchain|cryptocurrency|bitcoin|ethereum)\b',
                r'\b(?:software|hardware|algorithm|programming|code)\b'
            ],
            # Social concepts
            'social': [
                r'\b(?:society|community|culture|tradition|custom)\b',
                r'\b(?:politics|government|democracy|election|vote)\b',
                r'\b(?:economy|market|trade|business|finance)\b'
            ],
            # Natural concepts
            'nature': [
                r'\b(?:animal|plant|tree|forest|jungle|desert)\b',
                r'\b(?:evolution|species|habitat|ecosystem|biodiversity)\b',
                r'\b(?:water|river|ocean|mountain|valley|island)\b'
            ],
            # Human concepts
            'human': [
                r'\b(?:emotion|feeling|happiness|sadness|anger|fear)\b',
                r'\b(?:learning|education|knowledge|skill|talent)\b',
                r'\b(?:health|medicine|disease|treatment|cure)\b'
            ]
        }
        
        self.relationship_patterns = {
            'causation': [r'because', r'due to', r'caused by', r'results in', r'leads to'],
            'comparison': [r'similar to', r'different from', r'compared to', r'unlike', r'like'],
            'temporal': [r'before', r'after', r'during', r'while', r'when', r'then'],
            'spatial': [r'above', r'below', r'near', r'far', r'inside', r'outside']
        }
        
        # Dynamic concept discovery
        self.discovered_concepts = defaultdict(int)
        self.concept_relationships = defaultdict(list)
    
    def extract_concepts(self, text: str) -> Tuple[List[str], Dict[str, List[str]]]:
        """Extract concepts and relationships from text"""
        text_lower = text.lower()
        
        # Extract known concepts
        found_concepts = []
        concept_categories = {}
        
        for category, patterns in self.concept_patterns.items():
            category_concepts = []
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                category_concepts.extend(matches)
            
            if category_concepts:
                concept_categories[category] = list(set(category_concepts))
                found_concepts.extend(category_concepts)
        
        # Discover new concepts (nouns that appear frequently)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text)
        for word in words:
            if word.lower() not in [c.lower() for c in found_concepts]:
                self.discovered_concepts[word.lower()] += 1
                if self.discovered_concepts[word.lower()] >= 3:  # Threshold for new concept
                    found_concepts.append(word.lower())
        
        # Extract relationships
        relationships = {}
        for rel_type, patterns in self.relationship_patterns.items():
            rel_matches = []
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    rel_matches.append(pattern)
            if rel_matches:
                relationships[rel_type] = rel_matches
        
        return list(set(found_concepts)), relationships
    
    def create_neural_pattern(self, concepts: List[str], relationships: Dict) -> Dict[int, bool]:
        """Convert concepts and relationships to neural activation pattern"""
        pattern = {}
        
        # Encode concepts as neuron activations
        for i, concept in enumerate(concepts[:50]):  # Limit to 50 concepts
            # Hash concept to consistent neuron ID
            concept_hash = int(hashlib.md5(concept.encode()).hexdigest()[:8], 16)
            neuron_id = concept_hash % 1000  # Map to neuron 0-999
            pattern[neuron_id] = True
        
        # Encode relationships
        for rel_type, rel_patterns in relationships.items():
            rel_hash = int(hashlib.md5(rel_type.encode()).hexdigest()[:8], 16)
            rel_neuron = 1000 + (rel_hash % 200)  # Map to neurons 1000-1199
            pattern[rel_neuron] = True
        
        return pattern


class ExternalDataCollector:
    """Collects experiences from various external sources"""
    
    def __init__(self):
        self.concept_extractor = ConceptExtractor()
        self.experience_cache = deque(maxlen=10000)
        self.source_quality = defaultdict(float)
        self.collection_stats = {
            'total_collected': 0,
            'sources_active': 0,
            'avg_quality': 0.0,
            'concepts_discovered': 0
        }
        
        # External data sources
        self.rss_feeds = [
            'http://rss.cnn.com/rss/cnn_latest.rss',
            'https://feeds.bbci.co.uk/news/rss.xml',
            'https://rss.reuters.com/news/world.xml',
            'https://feeds.nature.com/nature/rss/current',
            'https://feeds.sciencedaily.com/sciencedaily/top_news.xml'
        ]
        
        self.api_sources = {
            'wikipedia_random': 'https://en.wikipedia.org/api/rest_v1/page/random/summary',
            'quote_api': 'https://api.quotable.io/random',
            'fact_api': 'http://numbersapi.com/random/trivia',
            'word_api': 'https://api.wordnik.com/v4/words.json/randomWord?api_key=demo'
        }
        
        # Web scraping targets (educational/informational sites)
        self.web_sources = [
            'https://www.khanacademy.org',
            'https://www.coursera.org',
            'https://www.ted.com/talks',
            'https://stackoverflow.com',
            'https://www.reddit.com/r/todayilearned'
        ]
    
    async def collect_from_rss(self, feed_url: str, max_items: int = 5) -> List[ExternalExperience]:
        """Collect experiences from RSS feeds"""
        experiences = []
        
        try:
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries[:max_items]:
                content = entry.get('summary', entry.get('description', ''))
                if len(content) < 50:  # Skip very short content
                    continue
                
                # Extract concepts and create neural pattern
                concepts, relationships = self.concept_extractor.extract_concepts(content)
                neural_pattern = self.concept_extractor.create_neural_pattern(concepts, relationships)
                
                # Calculate quality score
                quality_score = self._calculate_quality_score(content, concepts, relationships)
                
                experience = ExternalExperience(
                    experience_id=f"rss_{int(time.time() * 1000)}_{len(experiences)}",
                    source_type='rss',
                    source_url=feed_url,
                    content=content[:500],  # Limit content length
                    metadata={
                        'title': entry.get('title', ''),
                        'published': entry.get('published', ''),
                        'link': entry.get('link', '')
                    },
                    timestamp=time.time(),
                    quality_score=quality_score,
                    discovered_concepts=concepts,
                    neural_pattern=neural_pattern,
                    context={
                        'source_type': 'news',
                        'concepts': concepts,
                        'relationships': relationships,
                        'category': self._categorize_content(concepts)
                    }
                )
                
                experiences.append(experience)
                
        except Exception as e:
            logging.warning(f"Failed to collect from RSS {feed_url}: {e}")
        
        return experiences
    
    async def collect_from_api(self, api_name: str, api_url: str) -> Optional[ExternalExperience]:
        """Collect experience from API endpoint"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(api_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Extract content based on API type
                        content = self._extract_api_content(api_name, data)
                        if not content or len(content) < 20:
                            return None
                        
                        # Extract concepts and create neural pattern
                        concepts, relationships = self.concept_extractor.extract_concepts(content)
                        neural_pattern = self.concept_extractor.create_neural_pattern(concepts, relationships)
                        
                        quality_score = self._calculate_quality_score(content, concepts, relationships)
                        
                        experience = ExternalExperience(
                            experience_id=f"api_{api_name}_{int(time.time() * 1000)}",
                            source_type='api',
                            source_url=api_url,
                            content=content,
                            metadata=data,
                            timestamp=time.time(),
                            quality_score=quality_score,
                            discovered_concepts=concepts,
                            neural_pattern=neural_pattern,
                            context={
                                'source_type': 'api',
                                'api_name': api_name,
                                'concepts': concepts,
                                'relationships': relationships,
                                'category': self._categorize_content(concepts)
                            }
                        )
                        
                        return experience
                        
        except Exception as e:
            logging.warning(f"Failed to collect from API {api_name}: {e}")
        
        return None
    
    async def collect_from_web(self, url: str) -> List[ExternalExperience]:
        """Collect experiences from web scraping"""
        experiences = []
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        html = await response.text()
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Extract text content
                        text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'article'])
                        for element in text_elements[:10]:  # Limit to 10 elements
                            content = element.get_text().strip()
                            if len(content) < 50:  # Skip short content
                                continue
                            
                            concepts, relationships = self.concept_extractor.extract_concepts(content)
                            neural_pattern = self.concept_extractor.create_neural_pattern(concepts, relationships)
                            
                            quality_score = self._calculate_quality_score(content, concepts, relationships)
                            
                            experience = ExternalExperience(
                                experience_id=f"web_{int(time.time() * 1000)}_{len(experiences)}",
                                source_type='web',
                                source_url=url,
                                content=content[:500],
                                metadata={'element_tag': element.name},
                                timestamp=time.time(),
                                quality_score=quality_score,
                                discovered_concepts=concepts,
                                neural_pattern=neural_pattern,
                                context={
                                    'source_type': 'web',
                                    'concepts': concepts,
                                    'relationships': relationships,
                                    'category': self._categorize_content(concepts)
                                }
                            )
                            
                            experiences.append(experience)
                            
        except Exception as e:
            logging.warning(f"Failed to collect from web {url}: {e}")
        
        return experiences
    
    def _extract_api_content(self, api_name: str, data: Dict) -> str:
        """Extract relevant content from API response"""
        if api_name == 'wikipedia_random':
            return data.get('extract', '')
        elif api_name == 'quote_api':
            quote = data.get('content', '')
            author = data.get('author', '')
            return f"{quote} - {author}"
        elif api_name == 'fact_api':
            return str(data) if isinstance(data, str) else ''
        elif api_name == 'word_api':
            return data.get('word', '') + ': ' + data.get('definition', '')
        return str(data)
    
    def _calculate_quality_score(self, content: str, concepts: List[str], relationships: Dict) -> float:
        """Calculate quality score for an experience"""
        score = 0.0
        
        # Content length score
        length_score = min(len(content) / 200, 1.0)  # Normalize to 0-1
        score += 0.3 * length_score
        
        # Concept richness score
        concept_score = min(len(concepts) / 10, 1.0)  # Normalize to 0-1
        score += 0.4 * concept_score
        
        # Relationship complexity score
        relationship_score = min(len(relationships) / 4, 1.0)  # Normalize to 0-1
        score += 0.3 * relationship_score
        
        return score
    
    def _categorize_content(self, concepts: List[str]) -> str:
        """Categorize content based on discovered concepts"""
        category_scores = defaultdict(int)
        
        for concept in concepts:
            for category, patterns in self.concept_extractor.concept_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, concept, re.IGNORECASE):
                        category_scores[category] += 1
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        return 'general'
    
    async def collect_batch(self, batch_size: int = 20) -> List[ExternalExperience]:
        """Collect a batch of experiences from all sources"""
        all_experiences = []
        
        # Collect from RSS feeds
        for feed_url in self.rss_feeds[:2]:  # Limit to 2 feeds per batch
            rss_experiences = await self.collect_from_rss(feed_url, max_items=3)
            all_experiences.extend(rss_experiences)
        
        # Collect from APIs
        for api_name, api_url in list(self.api_sources.items())[:3]:  # Limit to 3 APIs
            api_experience = await self.collect_from_api(api_name, api_url)
            if api_experience:
                all_experiences.append(api_experience)
        
        # Collect from web (limited for demo)
        if len(all_experiences) < batch_size:
            web_url = random.choice(self.web_sources)
            web_experiences = await self.collect_from_web(web_url)
            all_experiences.extend(web_experiences[:5])  # Limit web scraping
        
        # Filter by quality and limit batch size
        quality_filtered = [exp for exp in all_experiences if exp.quality_score > 0.3]
        quality_filtered.sort(key=lambda x: x.quality_score, reverse=True)
        
        final_batch = quality_filtered[:batch_size]
        
        # Update stats
        self.collection_stats['total_collected'] += len(final_batch)
        self.collection_stats['concepts_discovered'] = len(self.concept_extractor.discovered_concepts)
        
        if final_batch:
            self.collection_stats['avg_quality'] = sum(exp.quality_score for exp in final_batch) / len(final_batch)
        
        return final_batch


class RealWorldExperienceTrainer:
    """Trains brain network with real-world experiences from external sources"""
    
    def __init__(self, network: CognitiveBrainNetwork):
        self.network = network
        self.data_collector = ExternalDataCollector()
        self.training_metrics = {
            'experiences_processed': 0,
            'external_sources_used': 0,
            'new_concepts_learned': 0,
            'memories_formed': 0,
            'session_start': time.time()
        }
        self.performance_history = deque(maxlen=100)
    
    async def run_real_world_training(self, duration_minutes: int = 5, experiences_per_minute: int = 4):
        """Run training session with real-world external experiences"""
        print(f"🌍 Starting Real-World Experience Training ({duration_minutes} minutes)")
        print(f"   🎯 Target: {experiences_per_minute} external experiences per minute")
        
        session_start = time.time()
        session_end = session_start + (duration_minutes * 60)
        experience_interval = 60.0 / experiences_per_minute
        
        last_experience_time = session_start
        
        while time.time() < session_end:
            current_time = time.time()
            
            if current_time - last_experience_time >= experience_interval:
                await self._process_external_batch()
                last_experience_time = current_time
            
            await asyncio.sleep(1)  # Non-blocking wait
        
        self._print_session_summary()
    
    async def _process_external_batch(self):
        """Process a batch of external experiences"""
        print(f"🔍 Collecting external experiences...")
        
        # Collect batch of real-world experiences
        experiences = await self.data_collector.collect_batch(batch_size=5)
        
        if not experiences:
            print("   ⚠️  No experiences collected this batch")
            return
        
        print(f"   📊 Collected {len(experiences)} experiences from external sources")
        
        # Process each experience through the cognitive network
        for experience in experiences:
            # Add inference generation request
            experience.context['generate_inferences'] = True
            
            # Process through network
            result = self.network.step_with_cognition(
                experience.neural_pattern, 
                experience.context
            )
            
            # Update metrics
            self.training_metrics['experiences_processed'] += 1
            if result.get('memory_id'):
                self.training_metrics['memories_formed'] += 1
            
            # Track new concepts
            new_concepts = set(experience.discovered_concepts) - self.network.episodic_memories.keys()
            self.training_metrics['new_concepts_learned'] += len(new_concepts)
            
            # Print progress for interesting experiences
            if experience.quality_score > 0.6:
                print(f"   🧠 High-quality experience: {experience.context.get('category', 'unknown')}")
                print(f"      Concepts: {experience.discovered_concepts[:3]}...")
                print(f"      Quality: {experience.quality_score:.2f}")
                if result.get('inferences'):
                    print(f"      Inferences: {len(result['inferences'])}")
        
        # Update source tracking
        sources_used = set(exp.source_type for exp in experiences)
        self.training_metrics['external_sources_used'] = len(sources_used)
    
    def _print_session_summary(self):
        """Print comprehensive session summary"""
        duration = time.time() - self.training_metrics['session_start']
        cognitive_state = self.network.get_cognitive_state()
        collection_stats = self.data_collector.collection_stats
        
        print(f"\n🎉 Real-World Training Complete!")
        print(f"=" * 60)
        print(f"Duration: {duration/60:.1f} minutes")
        print(f"External experiences processed: {self.training_metrics['experiences_processed']}")
        print(f"External sources used: {self.training_metrics['external_sources_used']}")
        print(f"New concepts discovered: {collection_stats['concepts_discovered']}")
        print(f"Memories formed: {self.training_metrics['memories_formed']}")
        print(f"Average experience quality: {collection_stats.get('avg_quality', 0):.2f}")
        
        print(f"\n🧠 Cognitive State After Real-World Training:")
        for key, value in cognitive_state.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
        
        print(f"\n🌍 Top Discovered Concepts:")
        top_concepts = sorted(
            self.data_collector.concept_extractor.discovered_concepts.items(),
            key=lambda x: x[1], reverse=True
        )[:10]
        for concept, count in top_concepts:
            print(f"   {concept}: {count} occurrences")


async def demo_real_world_training():
    """Demonstrate real-world experience training"""
    print("🌍 Real-World Experience Training Demo")
    print("=" * 60)
    
    # Create cognitive network
    config = CognitiveConfig(
        learning_rate=0.05,
        learning_probability=0.7,
        max_episodic_memories=500,
        association_strength_threshold=0.15,
        max_inference_depth=4
    )
    
    network = CognitiveBrainNetwork(num_neurons=200, config=config)
    
    # Create real-world trainer
    trainer = RealWorldExperienceTrainer(network)
    
    # Run real-world training session
    await trainer.run_real_world_training(duration_minutes=3, experiences_per_minute=3)
    
    return network


if __name__ == "__main__":
    asyncio.run(demo_real_world_training())