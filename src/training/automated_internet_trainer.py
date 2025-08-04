"""
Automated Internet Training Module for HASN Brain Network
Continuously trains the brain using information scraped from the internet
"""

import asyncio
import aiohttp
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import numpy as np
import re
import hashlib
from datetime import datetime, timedelta
import sys
import os
from urllib.parse import urlparse, urljoin, quote_plus
import random

# Add path for brain imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.simplified_brain_network import SimpleBrainNetwork
    from training.interactive_brain_trainer import InteractiveBrainTrainer
except ImportError:
    print("⚠️  Brain modules not found. Make sure you're running from the correct directory.")
    sys.exit(1)

@dataclass
class TrainingConfig:
    """Configuration for automated internet training"""
    # Data collection settings
    max_articles_per_session: int = 50
    max_article_length: int = 5000  # chars
    collection_interval: int = 3600  # seconds (1 hour)
    sources: List[str] = None
    
    # Training settings
    training_batch_size: int = 10
    learning_rate: float = 0.02
    pattern_consolidation_threshold: float = 0.7
    max_memory_patterns: int = 1000
    
    # Quality control
    min_article_quality_score: float = 0.6
    language_filter: str = 'en'
    content_filters: List[str] = None
    
    # System settings
    max_concurrent_requests: int = 5
    request_delay: float = 1.0  # seconds between requests
    save_interval: int = 1800  # seconds (30 minutes)
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = [
                'https://en.wikipedia.org/wiki/Special:Random',
                'https://www.reddit.com/r/todayilearned/top/.json?limit=25',
                'https://news.ycombinator.com/news',
                'https://www.nature.com/nature/articles',
                'https://arxiv.org/list/cs.AI/recent'
            ]
        
        if self.content_filters is None:
            self.content_filters = [
                'advertisement', 'cookie policy', 'privacy policy',
                'subscribe now', 'click here', 'buy now'
            ]

class WebContentCollector:
    """Collects and preprocesses content from various web sources"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.collected_urls = set()  # Avoid duplicates
        self.content_cache = {}
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Headers to appear more human-like
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(limit=self.config.max_concurrent_requests)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            connector=connector,
            timeout=timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def collect_from_wikipedia(self) -> List[Dict[str, Any]]:
        """Collect random Wikipedia articles"""
        articles = []
        
        for _ in range(self.config.max_articles_per_session // 5):  # 1/5 from Wikipedia
            try:
                # Get random article
                async with self.session.get('https://en.wikipedia.org/api/rest_v1/page/random/summary') as response:
                    if response.status == 200:
                        data = await response.json()
                        title = data.get('title', '')
                        extract = data.get('extract', '')
                        
                        if len(extract) > 100:  # Minimum content length
                            articles.append({
                                'title': title,
                                'content': extract,
                                'source': 'wikipedia',
                                'url': data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                                'timestamp': datetime.now().isoformat(),
                                'quality_score': self._calculate_quality_score(extract)
                            })
                
                await asyncio.sleep(self.config.request_delay)
                
            except Exception as e:
                self.logger.warning(f"Error collecting from Wikipedia: {e}")
        
        return articles
    
    async def collect_from_reddit(self) -> List[Dict[str, Any]]:
        """Collect from Reddit TIL (Today I Learned)"""
        articles = []
        
        try:
            url = 'https://www.reddit.com/r/todayilearned/top/.json?limit=25&t=day'
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    posts = data.get('data', {}).get('children', [])
                    
                    for post in posts[:self.config.max_articles_per_session // 5]:
                        post_data = post.get('data', {})
                        title = post_data.get('title', '')
                        selftext = post_data.get('selftext', '')
                        
                        # Reddit TIL posts often have the learning in the title
                        content = f"{title}\n\n{selftext}" if selftext else title
                        
                        if len(content) > 50:
                            articles.append({
                                'title': title,
                                'content': content,
                                'source': 'reddit_til',
                                'url': f"https://reddit.com{post_data.get('permalink', '')}",
                                'timestamp': datetime.now().isoformat(),
                                'quality_score': self._calculate_quality_score(content)
                            })
        
        except Exception as e:
            self.logger.warning(f"Error collecting from Reddit: {e}")
        
        return articles
    
    async def collect_from_news_sources(self) -> List[Dict[str, Any]]:
        """Collect from various news and educational sources"""
        articles = []
        
        # Simple RSS/feed endpoints that don't require API keys
        feeds = [
            'http://rss.cnn.com/rss/cnn_latest.rss/',
            'https://feeds.bbci.co.uk/news/rss.xml',
            'https://www.nasa.gov/rss/dyn/breaking_news.rss'
        ]
        
        for feed_url in feeds:
            try:
                async with self.session.get(feed_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        # Simple XML parsing for RSS feeds
                        items = self._parse_rss_content(content)
                        
                        for item in items[:5]:  # Limit per source
                            if item['content']:
                                articles.append({
                                    'title': item['title'],
                                    'content': item['content'],
                                    'source': f'rss_{urlparse(feed_url).netloc}',
                                    'url': item.get('link', ''),
                                    'timestamp': datetime.now().isoformat(),
                                    'quality_score': self._calculate_quality_score(item['content'])
                                })
                
                await asyncio.sleep(self.config.request_delay)
                
            except Exception as e:
                self.logger.warning(f"Error collecting from {feed_url}: {e}")
        
        return articles
    
    def _parse_rss_content(self, xml_content: str) -> List[Dict[str, str]]:
        """Simple RSS parser - extracts title and description"""
        items = []
        
        # Very basic XML parsing using regex (not robust, but works for demo)
        item_pattern = r'<item>(.*?)</item>'
        title_pattern = r'<title><!\[CDATA\[(.*?)\]\]></title>|<title>(.*?)</title>'
        desc_pattern = r'<description><!\[CDATA\[(.*?)\]\]></description>|<description>(.*?)</description>'
        link_pattern = r'<link>(.*?)</link>'
        
        item_matches = re.findall(item_pattern, xml_content, re.DOTALL)
        
        for item_content in item_matches[:10]:  # Limit items
            title_match = re.search(title_pattern, item_content)
            desc_match = re.search(desc_pattern, item_content)
            link_match = re.search(link_pattern, item_content)
            
            title = ''
            if title_match:
                title = title_match.group(1) or title_match.group(2) or ''
            
            description = ''
            if desc_match:
                description = desc_match.group(1) or desc_match.group(2) or ''
            
            link = link_match.group(1) if link_match else ''
            
            # Clean HTML tags from description
            description = re.sub(r'<[^>]+>', '', description)
            
            if title and description:
                items.append({
                    'title': title.strip(),
                    'content': description.strip(),
                    'link': link.strip()
                })
        
        return items
    
    def _calculate_quality_score(self, content: str) -> float:
        """Calculate content quality score based on various factors"""
        if not content:
            return 0.0
        
        score = 0.0
        
        # Length factor (optimal around 200-2000 characters)
        length = len(content)
        if 200 <= length <= 2000:
            score += 0.3
        elif 100 <= length < 200 or 2000 < length <= 5000:
            score += 0.2
        elif length > 5000:
            score += 0.1
        
        # Sentence structure
        sentences = content.split('.')
        if 3 <= len(sentences) <= 20:
            score += 0.2
        
        # Word diversity
        words = content.lower().split()
        unique_words = len(set(words))
        if len(words) > 0:
            diversity = unique_words / len(words)
            score += diversity * 0.3
        
        # Filter out low-quality indicators
        low_quality_patterns = [
            r'\b(click here|subscribe|buy now|advertisement)\b',
            r'^(cookie|privacy policy)',
            r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',  # Timestamps
        ]
        
        for pattern in low_quality_patterns:
            if re.search(pattern, content.lower()):
                score -= 0.2
        
        # Educational/factual content indicators
        educational_patterns = [
            r'\b(research|study|scientists?|discovered?|theory|evidence)\b',
            r'\b(according to|researchers|university|institute)\b',
            r'\b(analysis|data|statistics|findings)\b'
        ]
        
        for pattern in educational_patterns:
            if re.search(pattern, content.lower()):
                score += 0.1
        
        return max(0.0, min(1.0, score))
    
    async def collect_all_sources(self) -> List[Dict[str, Any]]:
        """Collect content from all configured sources"""
        all_articles = []
        
        self.logger.info("🔍 Starting content collection from internet sources...")
        
        # Collect from different sources
        tasks = [
            self.collect_from_wikipedia(),
            self.collect_from_reddit(),
            self.collect_from_news_sources()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
            elif isinstance(result, Exception):
                self.logger.error(f"Collection error: {result}")
        
        # Filter by quality
        quality_articles = [
            article for article in all_articles 
            if article['quality_score'] >= self.config.min_article_quality_score
        ]
        
        # Remove duplicates based on content hash
        unique_articles = self._remove_duplicates(quality_articles)
        
        self.logger.info(f"📊 Collected {len(unique_articles)} quality articles from {len(all_articles)} total")
        
        return unique_articles[:self.config.max_articles_per_session]
    
    def _remove_duplicates(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles based on content similarity"""
        unique_articles = []
        seen_hashes = set()
        
        for article in articles:
            # Create hash of content
            content_hash = hashlib.md5(article['content'].lower().encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_articles.append(article)
        
        return unique_articles

class NeuralPatternConverter:
    """Converts web content into neural patterns compatible with HASN"""
    
    def __init__(self, brain_network):
        self.brain = brain_network
        self.vocabulary = {}  # word -> neural pattern mapping
        self.concept_patterns = {}  # concept -> pattern mapping
        self.pattern_history = deque(maxlen=1000)
        
    def text_to_neural_pattern(self, text: str, context: str = "") -> Dict[int, Dict[int, bool]]:
        """Convert text content to neural activation pattern"""
        # Preprocess text
        cleaned_text = self._preprocess_text(text)
        words = cleaned_text.split()
        
        # Create base pattern from text
        pattern = {}
        
        # Distribute words across brain modules
        num_modules = 4  # sensory, memory, executive, motor
        
        for i, word in enumerate(words[:20]):  # Limit to prevent overwhelming
            module_id = i % num_modules
            
            if module_id not in pattern:
                pattern[module_id] = {}
            
            # Convert word to neural activation
            word_neurons = self._word_to_neurons(word, module_id)
            pattern[module_id].update(word_neurons)
        
        # Add contextual activation based on content type
        context_activation = self._generate_context_pattern(context, text)
        
        # Merge context with content pattern
        for module_id, neurons in context_activation.items():
            if module_id not in pattern:
                pattern[module_id] = {}
            pattern[module_id].update(neurons)
        
        return pattern
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for neural conversion"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:-]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove stop words (basic set)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        return ' '.join(filtered_words)
    
    def _word_to_neurons(self, word: str, module_id: int) -> Dict[int, bool]:
        """Convert a word to specific neuron activations in a module"""
        if word not in self.vocabulary:
            # Create new pattern for word
            hash_val = hash(word) % 1000000  # Deterministic but distributed
            
            # Generate 3-5 active neurons per word
            num_active = 3 + (hash_val % 3)
            base_neuron = hash_val % 50  # Assume 50 neurons per module
            
            active_neurons = {}
            for i in range(num_active):
                neuron_id = (base_neuron + i * 7) % 50  # Spread activation
                active_neurons[neuron_id] = True
            
            self.vocabulary[word] = active_neurons
        
        return self.vocabulary[word].copy()
    
    def _generate_context_pattern(self, context: str, content: str) -> Dict[int, Dict[int, bool]]:
        """Generate contextual neural activation based on content type and source"""
        context_pattern = {}
        
        # Analyze content for contextual cues
        content_lower = content.lower()
        
        # Science/Research context - activate memory module more
        if any(word in content_lower for word in ['research', 'study', 'scientist', 'discovery', 'theory', 'evidence']):
            context_pattern[1] = {10: True, 11: True, 12: True}  # Memory module
        
        # News/Current events - activate sensory module
        if any(word in content_lower for word in ['today', 'yesterday', 'recently', 'breaking', 'report']):
            context_pattern[0] = {5: True, 6: True, 7: True}  # Sensory module
        
        # Educational content - activate executive module
        if any(word in content_lower for word in ['learn', 'education', 'explain', 'understand', 'concept']):
            context_pattern[2] = {15: True, 16: True, 17: True}  # Executive module
        
        # Action/procedural content - activate motor module
        if any(word in content_lower for word in ['how to', 'method', 'process', 'procedure', 'steps']):
            context_pattern[3] = {20: True, 21: True, 22: True}  # Motor module
        
        return context_pattern
    
    def create_learning_objective(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Create a learning objective from an article for training"""
        return {
            'input_pattern': self.text_to_neural_pattern(article['content'], article['source']),
            'expected_response': f"Knowledge about: {article['title']}",
            'concept': self._extract_main_concept(article['title'], article['content']),
            'source_info': {
                'title': article['title'],
                'source': article['source'],
                'url': article.get('url', ''),
                'quality_score': article.get('quality_score', 0.0)
            }
        }
    
    def _extract_main_concept(self, title: str, content: str) -> str:
        """Extract the main concept/topic from the content"""
        # Simple keyword extraction
        combined_text = f"{title} {content}".lower()
        
        # Common concept patterns
        concept_patterns = [
            r'\b(artificial intelligence|machine learning|neural network)\b',
            r'\b(climate change|global warming|environment)\b',
            r'\b(space|astronomy|planet|galaxy|universe)\b',
            r'\b(medicine|health|disease|treatment|drug)\b',
            r'\b(history|historical|ancient|civilization)\b',
            r'\b(technology|computer|software|algorithm)\b',
            r'\b(science|scientific|physics|chemistry|biology)\b',
        ]
        
        for pattern in concept_patterns:
            match = re.search(pattern, combined_text)
            if match:
                return match.group(1)
        
        # Fallback: use first significant word from title
        title_words = title.lower().split()
        significant_words = [w for w in title_words if len(w) > 4]
        
        return significant_words[0] if significant_words else "general_knowledge"

class AutomatedInternetTrainer:
    """Main automated training system that orchestrates internet-based learning"""
    
    def __init__(self, config: TrainingConfig = None):
        self.config = config or TrainingConfig()
        
        # Initialize brain components
        # Using SimpleBrainNetwork - proven 2.3x faster than alternatives
        self.brain = SimpleBrainNetwork(num_neurons=100, connectivity_prob=0.1)
        self.trainer = InteractiveBrainTrainer()
        
        # Initialize training components
        self.collector = None  # Will be initialized in async context
        self.converter = NeuralPatternConverter(self.brain)
        
        # Training state
        self.training_active = False
        self.total_articles_processed = 0
        self.training_metrics = {
            'articles_collected': 0,
            'patterns_learned': 0,
            'concepts_discovered': 0,
            'training_sessions': 0,
            'last_save_time': time.time(),
            'quality_scores': deque(maxlen=100)
        }
        
        # Knowledge validation
        self.learned_concepts = set()
        self.pattern_effectiveness = {}
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    async def start_training(self, continuous: bool = True):
        """Start the automated training process"""
        self.training_active = True
        self.logger.info("🧠 Starting Automated Internet Training...")
        
        async with WebContentCollector(self.config) as collector:
            self.collector = collector
            
            try:
                if continuous:
                    await self._continuous_training_loop()
                else:
                    await self._single_training_session()
                    
            except KeyboardInterrupt:
                self.logger.info("🛑 Training interrupted by user")
            except Exception as e:
                self.logger.error(f"❌ Training error: {e}")
            finally:
                await self._save_training_state()
                self.training_active = False
    
    async def _continuous_training_loop(self):
        """Main continuous training loop"""
        while self.training_active:
            try:
                # Collect new information
                articles = await self.collector.collect_all_sources()
                
                if articles:
                    # Process articles into learning objectives
                    learning_objectives = [
                        self.converter.create_learning_objective(article) 
                        for article in articles
                    ]
                    
                    # Train the brain on new information
                    await self._train_on_objectives(learning_objectives)
                    
                    # Update metrics
                    self.training_metrics['articles_collected'] += len(articles)
                    self.training_metrics['training_sessions'] += 1
                    
                    # Log progress
                    self.logger.info(f"🎓 Training session complete: {len(articles)} articles processed")
                    self._log_training_progress()
                
                # Periodic saves
                if time.time() - self.training_metrics['last_save_time'] > self.config.save_interval:
                    await self._save_training_state()
                
                # Wait before next collection
                await asyncio.sleep(self.config.collection_interval)
                
            except Exception as e:
                self.logger.error(f"❌ Error in training loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _single_training_session(self):
        """Run a single training session"""
        articles = await self.collector.collect_all_sources()
        
        if articles:
            learning_objectives = [
                self.converter.create_learning_objective(article) 
                for article in articles
            ]
            
            await self._train_on_objectives(learning_objectives)
            
            self.logger.info(f"🎓 Single training session complete: {len(articles)} articles processed")
            self._log_training_progress()
    
    async def _train_on_objectives(self, objectives: List[Dict[str, Any]]):
        """Train the brain on learning objectives"""
        for objective in objectives:
            try:
                # Extract training components
                input_pattern = objective['input_pattern']
                concept = objective['concept']
                source_info = objective['source_info']
                
                # Train the brain on this pattern
                self.trainer.train_on_pattern(
                    pattern_name=f"{concept}_{self.total_articles_processed}",
                    input_pattern=input_pattern,
                    expected_response=objective['expected_response']
                )
                
                # Track concept learning
                if concept not in self.learned_concepts:
                    self.learned_concepts.add(concept)
                    self.training_metrics['concepts_discovered'] += 1
                
                # Update effectiveness tracking
                quality_score = source_info.get('quality_score', 0.0)
                self.training_metrics['quality_scores'].append(quality_score)
                
                if concept not in self.pattern_effectiveness:
                    self.pattern_effectiveness[concept] = []
                self.pattern_effectiveness[concept].append(quality_score)
                
                self.training_metrics['patterns_learned'] += 1
                self.total_articles_processed += 1
                
                # Process through advanced brain for cognitive integration
                brain_result = self.brain.process_pattern(input_pattern)
                
                # Log interesting brain states
                if brain_result.get('total_activity', 0) > 0.8:
                    self.logger.info(f"🔥 High brain activity for concept: {concept}")
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.warning(f"⚠️  Error training on objective: {e}")
    
    def _log_training_progress(self):
        """Log current training progress and metrics"""
        metrics = self.training_metrics
        avg_quality = np.mean(list(metrics['quality_scores'])) if metrics['quality_scores'] else 0.0
        
        self.logger.info("📊 Training Progress:")
        self.logger.info(f"   Total Articles: {metrics['articles_collected']}")
        self.logger.info(f"   Patterns Learned: {metrics['patterns_learned']}")
        self.logger.info(f"   Concepts Discovered: {metrics['concepts_discovered']}")
        self.logger.info(f"   Training Sessions: {metrics['training_sessions']}")
        self.logger.info(f"   Average Quality Score: {avg_quality:.3f}")
        self.logger.info(f"   Learned Concepts: {len(self.learned_concepts)}")
        
        # Show some learned concepts
        if self.learned_concepts:
            recent_concepts = list(self.learned_concepts)[-5:]
            self.logger.info(f"   Recent Concepts: {', '.join(recent_concepts)}")
    
    async def _save_training_state(self):
        """Save current training state and learned information"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save trainer state (includes patterns and concepts)
            trainer_file = f"output/automated_training_state_{timestamp}.json"
            self.trainer.save_training_state(trainer_file)
            
            # Save brain network state
            brain_file = f"output/automated_brain_state_{timestamp}.json"
            self.brain.save_network_state(brain_file)
            
            # Save automation metrics
            metrics_file = f"output/automated_training_metrics_{timestamp}.json"
            metrics_data = {
                'training_metrics': dict(self.training_metrics),
                'learned_concepts': list(self.learned_concepts),
                'pattern_effectiveness': {
                    concept: {
                        'count': len(scores),
                        'avg_quality': np.mean(scores),
                        'effectiveness': np.mean(scores) * len(scores)  # Quality * frequency
                    }
                    for concept, scores in self.pattern_effectiveness.items()
                },
                'config': {
                    'max_articles_per_session': self.config.max_articles_per_session,
                    'collection_interval': self.config.collection_interval,
                    'min_quality_score': self.config.min_article_quality_score
                },
                'timestamp': timestamp
            }
            
            os.makedirs('output', exist_ok=True)
            with open(metrics_file, 'w') as f:
                json.dump(metrics_data, f, indent=2, default=str)
            
            self.training_metrics['last_save_time'] = time.time()
            self.logger.info(f"💾 Training state saved at {timestamp}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving training state: {e}")
    
    async def load_training_state(self, state_file: str):
        """Load previous training state"""
        try:
            self.trainer.load_training_state(state_file)
            
            # Load metrics if available
            metrics_file = state_file.replace('training_state', 'training_metrics')
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    self.training_metrics.update(data.get('training_metrics', {}))
                    self.learned_concepts = set(data.get('learned_concepts', []))
                    
                    effectiveness_data = data.get('pattern_effectiveness', {})
                    self.pattern_effectiveness = {
                        concept: info.get('effectiveness', 0.0)
                        for concept, info in effectiveness_data.items()
                    }
            
            self.logger.info(f"📂 Training state loaded from: {state_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Error loading training state: {e}")
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get a summary of learned knowledge"""
        return {
            'total_concepts': len(self.learned_concepts),
            'concepts': list(self.learned_concepts),
            'most_effective_concepts': sorted(
                self.pattern_effectiveness.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            'training_metrics': dict(self.training_metrics),
            'brain_state': self.brain.get_brain_state(),
            'trainer_patterns': len(self.trainer.pattern_memory)
        }

async def main():
    """Main function to demonstrate automated internet training"""
    print("🧠 HASN Automated Internet Training System")
    print("=" * 50)
    
    # Create configuration
    config = TrainingConfig(
        max_articles_per_session=20,  # Start small for demo
        collection_interval=300,  # 5 minutes for demo
        min_article_quality_score=0.5,
        save_interval=600  # 10 minutes
    )
    
    # Create trainer
    trainer = AutomatedInternetTrainer(config)
    
    print("🚀 Starting automated training...")
    print("   This will continuously collect information from the internet")
    print("   and train the HASN brain network on it.")
    print("   Press Ctrl+C to stop training and save state.")
    print()
    
    # Start training (single session for demo)
    await trainer.start_training(continuous=False)
    
    # Show knowledge summary
    summary = trainer.get_knowledge_summary()
    print("\n📊 Knowledge Summary:")
    print(f"   Concepts learned: {summary['total_concepts']}")
    print(f"   Total patterns: {summary['trainer_patterns']}")
    
    if summary['concepts']:
        print("   Recent concepts:")
        for concept in summary['concepts'][-10:]:
            print(f"      • {concept}")

if __name__ == "__main__":
    asyncio.run(main())