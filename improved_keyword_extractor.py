import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import spacy
from spacy.matcher import Matcher
from nltk.corpus import wordnet as wn
import sys
import json
# Download necessary NLTK resources - run once

try:
    #nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    #nltk.download('punkt')
    nltk.download('wordnet')
    #nltk.download('stopwords')
    
question = sys.argv[1]

class ImprovedExclusionDetector:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")
        self.negation_cues = {
            "not", "no", "never", "without", "excluding", "exclude", "minus", 
            "except", "barring", "omitting", "skipping", "avoiding", "eliminating", 
            "removing", "devoid", "free", "lack", "lacking", "instead", 
            "rather", "alternative", "non", "un", "absence", "absent"
        }
        
    def detect_negation_patterns(self, text):
        """
        Improved negation detection to capture exact phrases being negated
        """
        doc = self.nlp(text)
        exclusion_terms = []
        exclusion_phrases = []  # Store the actual phrases for keyword filtering
        
        # Define specific negation patterns to capture the full negated phrase
        patterns = [
            # Pattern 1: "not directly [verb/adj] to [object]" - captures only the negated part, stops at "but"
            (r'\bnot\s+directly\s+([^,.?!\n]*?)(?:\s+but|$)', 'not_directly'),
            
            # Pattern 2: "do not [verb] [object]" - captures only up to conjunctions
            (r'\b(?:do|does|did)\s+not\s+([^,.?!\n]*?)(?:\s+but|$)', 'do_not'),
            
            (r'\b(?:is|are|was|were)\s+not\s+([^,.?!\n]*?)(?:\s+but|$)', 'is_not'),
            
            # Pattern 3: "without [object/action]" - captures what's excluded, stops at conjunctions
            (r'\bwithout\s+(?:any\s+|the\s+)?([^,.?!\n]*?)(?:\s+but|$)', 'without'),
            
            # Pattern 4: "excluding [object]" - direct exclusion
            (r'\bexcluding\s+(?:any\s+|all\s+)?([^,.?!\n]*?)(?:\s+but|$)', 'excluding'),
            
            # Pattern 5: "not [adjective/past participle] as [category]" - stops at conjunctions
            (r'\bnot\s+(?:classified|categorized|considered)\s+as\s+([^,.?!\n]*?)(?:\s+but|$)', 'not_as'),
            
            # Pattern 6: "[object] that do not [verb]" - relative clause negation (2 groups)
            (r'([^,.?!\n]+)\s+that\s+(?:do\s+)?not\s+([^,.?!\n]*?)(?:\s+but|$)', 'that_do_not'),
            
            # Pattern 7: "not include/contain [object]" - stops at conjunctions
            (r'\bnot\s+(?:include|contain|have)\s+([^,.?!\n]*?)(?:\s+but|$)', 'not_include'),
            
            # Pattern 8: "[object]-free" or "free from [object]"
            (r'\b(\w+)-free\b', 'suffix_free'),
            (r'\bfree\s+(?:from|of)\s+([^,.?!\n]*?)(?:\s+but|$)', 'free_from'),
            
            # Pattern 9: "instead of [object]" or "rather than [object]" - stops at conjunctions
            (r'\b(?:instead\s+of|rather\s+than)\s+([^,.?!\n]*?)(?:\s+but|$)', 'instead_of'),
            
            # Pattern 10: "not yet [action/state]" - stops at conjunctions
            (r'\bnot\s+yet\s+([^,.?!\n]*?)(?:\s+but|$)', 'not_yet'),
            
            # Pattern 11: "but do not [verb/action]" - stops at conjunctions
            (r'\bbut\s+(?:do\s+)?not\s+([^,.?!\n]*?)(?:\s+but|$)', 'but_not'),
            
            # Fallback patterns without the "but" constraint for cases that don't have conjunctions
            (r'\bnot\s+directly\s+([^,.?!\n]+)', 'not_directly_fallback'),
            (r'\b(?:do|does|did)\s+not\s+([^,.?!\n]+)', 'do_not_fallback'),
            (r'\bwithout\s+(?:any\s+|the\s+)?([^,.?!\n]+)', 'without_fallback'),
            (r'\b(?:is|are|was|were)\s+not\s+([^,.?!\n]+)', 'is_not_fallback'),
        ]
        
        text_lower = text.lower()
        
        for pattern, pattern_type in patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                if pattern_type == 'not_directly':
                    # Capture the full phrase after "not directly", but stop at "but"
                    full_phrase = f"not directly {match.group(1).strip()}"
                    exclusion_terms.append(f"exclude_{self._clean_excluded_content(full_phrase)}")
                    exclusion_phrases.append(full_phrase)
                    
                elif pattern_type == 'not_directly_fallback':
                    # Only use fallback if we haven't already found a match
                    if not any('not_directly' in term for term in exclusion_terms):
                        full_phrase = f"not directly {match.group(1).strip()}"
                        # For fallback, try to intelligently stop at conjunctions
                        if ' but ' in full_phrase:
                            full_phrase = full_phrase.split(' but ')[0]
                        exclusion_terms.append(f"exclude_{self._clean_excluded_content(full_phrase)}")
                        exclusion_phrases.append(full_phrase)
                    
                elif pattern_type == 'do_not':
                    # Capture the full phrase after "do not"
                    full_phrase = f"do not {match.group(1).strip()}"
                    exclusion_terms.append(f"exclude_{self._clean_excluded_content(full_phrase)}")
                    exclusion_phrases.append(full_phrase)
                    
                elif pattern_type == 'is_not':
                    # Capture the full phrase after "is/are/was/were not"
                    full_phrase = f"is not {match.group(1).strip()}"
                    exclusion_terms.append(f"exclude_{self._clean_excluded_content(full_phrase)}")
                    exclusion_phrases.append(full_phrase)
                    
                elif pattern_type == 'do_not_fallback':
                    # Only use fallback if we haven't already found a match
                    if not any('do_not' in term for term in exclusion_terms):
                        full_phrase = f"do not {match.group(1).strip()}"
                        if ' but ' in full_phrase:
                            full_phrase = full_phrase.split(' but ')[0]
                        exclusion_terms.append(f"exclude_{self._clean_excluded_content(full_phrase)}")
                        exclusion_phrases.append(full_phrase)
                
                elif pattern_type == 'is_not_fallback':
                    # Only use fallback if we haven't already found a match
                    if not any('is_not' in term for term in exclusion_terms):
                        full_phrase = f"is not {match.group(1).strip()}"
                        if ' but ' in full_phrase:
                            full_phrase = full_phrase.split(' but ')[0]
                        exclusion_terms.append(f"exclude_{self._clean_excluded_content(full_phrase)}")
                        exclusion_phrases.append(full_phrase)
                
                elif pattern_type == 'that_do_not':
                    # This pattern has 2 groups - handle it separately
                    subject = match.group(1).strip()
                    negated_part = match.group(2).strip() 
                    full_phrase = f"{subject} that do not {negated_part}"
                    exclusion_terms.append(f"exclude_{self._clean_excluded_content(f'do not {negated_part}')}")
                    exclusion_phrases.append(f"do not {negated_part}")
                    
                elif pattern_type == 'suffix_free':
                    # For "caffeine-free"
                    excluded_item = match.group(1).strip()
                    full_phrase = f"{excluded_item}-free"
                    exclusion_terms.append(f"exclude_{excluded_item}")
                    exclusion_phrases.append(excluded_item)
                    
                elif pattern_type in ['without', 'without_fallback']:
                    # For "without" patterns
                    excluded_content = match.group(1).strip()
                    if pattern_type == 'without_fallback' and any('without' in term for term in exclusion_terms):
                        continue  # Skip fallback if we already have a match
                    if ' but ' in excluded_content:
                        excluded_content = excluded_content.split(' but ')[0]
                    full_phrase = f"without {excluded_content}"
                    exclusion_phrases.append(full_phrase)
                    cleaned_content = self._clean_excluded_content(excluded_content)
                    if cleaned_content and len(cleaned_content) > 2:
                        exclusion_terms.append(f"exclude_{cleaned_content}")
                        
                else:
                    # For other patterns, extract the excluded content
                    excluded_content = match.group(1).strip()
                    
                    # Handle fallback patterns
                    if pattern_type.endswith('_fallback'):
                        base_type = pattern_type.replace('_fallback', '')
                        if any(base_type in term for term in exclusion_terms):
                            continue  # Skip if we already have a non-fallback match
                        if ' but ' in excluded_content:
                            excluded_content = excluded_content.split(' but ')[0].strip()
                    
                    prefix_map = {
                        'excluding': 'excluding', 
                        'not_as': 'not classified as',
                        'not_include': 'not include',
                        'free_from': 'free from',
                        'instead_of': 'instead of',
                        'not_yet': 'not yet',
                        'but_not': 'but not',
                        # Add fallback versions
                        'excluding_fallback': 'excluding',
                        'not_as_fallback': 'not classified as',
                        'not_include_fallback': 'not include',
                        'free_from_fallback': 'free from',
                        'instead_of_fallback': 'instead of',
                        'not_yet_fallback': 'not yet',
                        'but_not_fallback': 'but not'
                    }
                    
                    prefix = prefix_map.get(pattern_type, '')
                    if prefix:
                        full_phrase = f"{prefix} {excluded_content}"
                        exclusion_phrases.append(full_phrase)
                    else:
                        exclusion_phrases.append(excluded_content)
                    
                    cleaned_content = self._clean_excluded_content(excluded_content)
                    if cleaned_content and len(cleaned_content) > 2:
                        exclusion_terms.append(f"exclude_{cleaned_content}")
        
        # Store exclusion phrases for later use in keyword filtering
        self.exclusion_phrases = exclusion_phrases
        
        # Remove duplicates and clean up
        exclusion_terms = list(set(exclusion_terms))
        exclusion_terms = [term for term in exclusion_terms if len(term) > 10]
        
        return exclusion_terms
    
    def _clean_excluded_content(self, content):
        """Clean and normalize excluded content"""
        # Remove common stop words and articles from the beginning/end
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        words = content.split()
        # Remove stop words from beginning
        while words and words[0].lower() in stop_words:
            words.pop(0)
        # Remove stop words from end
        while words and words[-1].lower() in stop_words:
            words.pop()
        
        if not words:
            return None
            
        cleaned = '_'.join(words)
        # Remove punctuation
        cleaned = re.sub(r'[^\w\s_]', '', cleaned)
        # Replace multiple underscores with single
        cleaned = re.sub(r'_+', '_', cleaned)
        
        return cleaned.lower() if cleaned else None
    
    def _detect_spacy_negations(self, doc):
        """Use spaCy dependencies to detect negation patterns"""
        exclusions = []
        
        for token in doc:
            # Look for negation dependencies
            if token.dep_ == "neg":
                # Find what this negation modifies
                head = token.head
                if head.pos_ in ["VERB", "AUX"]:
                    # Look for objects of the negated verb
                    for child in head.children:
                        if child.dep_ in ["dobj", "pobj", "attr"]:
                            # Get the noun phrase
                            noun_phrase = self._extract_noun_phrase(child)
                            if noun_phrase:
                                exclusions.append(f"exclude_{noun_phrase}")
                                
            # Look for "without" constructions
            elif token.lower_ == "without":
                for child in token.children:
                    if child.dep_ == "pobj":
                        noun_phrase = self._extract_noun_phrase(child)
                        if noun_phrase:
                            exclusions.append(f"exclude_{noun_phrase}")
        
        return exclusions
    
    def _extract_noun_phrase(self, token):
        """Extract a clean noun phrase from a token"""
        # Get the subtree of the token
        subtree = list(token.subtree)
        # Filter to keep only relevant parts
        relevant_tokens = [t for t in subtree if t.pos_ in ["NOUN", "PROPN", "ADJ"] and len(t.text) > 1]
        
        if not relevant_tokens:
            return None
            
        phrase = "_".join([t.text.lower() for t in relevant_tokens])
        return phrase if len(phrase) > 2 else None


class ImprovedKeywordExtractor(ImprovedExclusionDetector):
    def __init__(self):
        super().__init__()
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        # Domain-specific terms
        self.domain_terms = {
            "health": ["gut", "microbiome", "barrier", "digestion", "flora", "bacteria", 
                      "probiotic", "prebiotic", "immune", "intestinal", "intestine", "bowel", 
                      "colon", "microbiota", "biome", "health", "inflammation", "integrity"],
            "action": ["strengthening", "improve", "enhance", "boost", "support", "promote", 
                      "maintain", "restore", "balance", "regulation", "reducing"],
            "compounds": ["glp", "berberine", "mulberry", "incretin", "insulin", "glucose"]
        }
        
        # Enhanced stop words
        self.contextual_stops = {
            "function", "functions", "functioning", "focus", "focuses", 
            "opportunity", "product", "products", "supplement", "supplements",
            "help", "helps", "helping", "effect", "effects", "impact", "study", "studies",
            "ingredient", "ingredients", "benefit", "benefits"
        }
        
        self.additional_stops = {
            "which", "what", "how", "when", "where", "why", "does", 
            "do", "also", "can", "will", "would", "could", "should",
            "there", "that", "is", "are", "and", "or", "the", "a", "an", 
            "with", "through", "for", "on", "in", "to", "from", "by", 
            "about", "as", "at", "into", "but", "still", "yet", "any"
        }
        self.stop_words.update(self.additional_stops)
    
    def extract_keywords_smart(self, text, max_keywords=7):
        """Extract keywords with improved negation detection"""
        # First, detect exclusion terms with improved method
        exclusion_terms = self.detect_negation_patterns(text)
        #print("Improved exclusion_terms:", exclusion_terms)
        
        # Clean text (don't remove verbs completely, just reduce their weight)
        cleaned_text = self.clean_text(text)
        
        # Tokenize and process
        tokens = word_tokenize(cleaned_text)
        filtered_tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]
        
        # Score terms with domain relevance
        token_scores = self.score_terms_improved(lemmatized_tokens, text)
        
        # Sort by score and get top keywords
        sorted_tokens = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)
        # Add meaningful exclusion terms (not all, just the most relevant)
        meaningful_exclusions = self._filter_meaningful_exclusions(exclusion_terms)
        
        if len(meaningful_exclusions) == 0:
            meaningful_exclusions = exclusion_terms
        
        keywords = [token for token, _ in sorted_tokens[:max_keywords]]
        
        # Filter keywords to remove any that are in meaningful_exclusions
        filtered_keywords = [kw for kw in keywords if not self.keyword_in_exclusions(kw, meaningful_exclusions)]
        
        
        #filtered_keywords.extend(meaningful_exclusions)
        
        return filtered_keywords[:max_keywords], meaningful_exclusions
        
    def keyword_in_exclusions(self, keyword, exclusions):
        # Check if keyword or phrase contains or equals any exclusion phrase
        keyword_lower = keyword.lower()
        #print(keyword, exclusions)
        for excl in exclusions:
            #print(excl, keyword_lower)
            if excl in keyword_lower or keyword_lower in excl:
                return True
        return False
    
    def _filter_meaningful_exclusions(self, exclusion_terms):
        """Filter exclusion terms to keep only the most meaningful ones"""
        meaningful = []
        for term in exclusion_terms:
            # Skip very generic exclusions
            if any(generic in term.lower() for generic in ['exclude_action', 'exclude_object']):
                continue
            # Keep specific exclusions about content, substances, or categories
            if any(specific in term.lower() for specific in ['berberine', 'caffeine', 'animal', 'powder', 'microbiota', 'overweight']):
                meaningful.append(term)
            elif len(term.split('_')) >= 3:  # Multi-word exclusions are often meaningful
                meaningful.append(term)
        
        return meaningful[:2]  # Limit to 2 most meaningful exclusions
    
    def score_terms_improved(self, tokens, original_text):
        """Improved scoring that considers context and domain relevance"""
        all_domain_terms = []
        for category in self.domain_terms.values():
            all_domain_terms.extend(category)
        
        token_scores = {}
        doc = self.nlp(original_text.lower())
        
        for i, token in enumerate(tokens):
            score = 1.0  # Base score
            
            # Domain relevance boost
            if token in all_domain_terms:
                score += 2.0
                if token in self.domain_terms["compounds"]:
                    score += 1.0  # Extra boost for specific compounds
            
            # Length bonus for specific terms
            if len(token) > 5:
                score += 0.5
            
            # Position bias (early terms slightly more important)
            position_bias = 1 - (i / (len(tokens) + 1))
            score += position_bias * 0.3
            
            # Check if token appears in important contexts in original text
            for sent_token in doc:
                if sent_token.text.lower() == token:
                    # Boost if it's a subject or object
                    if sent_token.dep_ in ["nsubj", "dobj", "pobj"]:
                        score += 0.5
                    # Boost if it's modified by important adjectives
                    for child in sent_token.children:
                        if child.pos_ == "ADJ" and child.text.lower() in ["clinical", "natural", "branded"]:
                            score += 0.3
            
            # Penalize very common terms even if not in stop words
            if token in self.contextual_stops:
                score *= 0.3
            
            token_scores[token] = score
        
        return token_scores
    
    def clean_text(self, text):
        """Clean text while preserving important context"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation but keep hyphens for compound words
        punctuation = string.punctuation.replace('-', '')
        translation_table = str.maketrans({'-': ' ', **{p: ' ' for p in punctuation}})
        text = text.translate(translation_table)
        
        # Remove numbers but keep important compound identifiers
        text = re.sub(r'\b\d+\b', '', text)  # Remove standalone numbers
        text = re.sub(r'glp-?\d*', 'glp', text)  # Normalize GLP-1 to GLP
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


# Test the improved version
if __name__ == "__main__":
    extractor = ImprovedKeywordExtractor()
    keywords, exclusion = extractor.extract_keywords_smart(question, max_keywords=100)
    print(json.dumps({"keywords": keywords, "exclusion": exclusion}))
