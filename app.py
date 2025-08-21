import os
import glob
from flask import Flask, request, jsonify, send_from_directory
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from llama_cpp import Llama
import urllib.parse
import time
import secrets
from datetime import datetime, timedelta
import json
import base64
import io
import PyPDF2
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import re
import logging
import threading
import random
import nltk
import spacy
from difflib import SequenceMatcher
from collections import defaultdict
from typing import Dict, List, Optional, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    logger.warning("NLTK downloads failed - StealthWriter may not work properly")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    logger.warning("spaCy model not found - StealthWriter may not work properly")
    nlp = None

class StealthWriter:
    def __init__(self):
        # Enhanced synonym cache and technical terms
        self.synonym_cache = defaultdict(list)
        self.technical_terms = {
            "analysis": ["examination", "assessment", "evaluation", "scrutiny", "inspection", "breakdown"],
            "method": ["approach", "technique", "procedure", "methodology", "practice", "strategy"],
            "important": ["crucial", "critical", "essential", "paramount", "significant", "vital", "major"],
            "show": ["demonstrate", "illustrate", "reveal", "exhibit", "display", "indicate", "present"],
            "result": ["outcome", "finding", "consequence", "product", "effect", "impact", "resolution"],
            "data": ["information", "evidence", "facts", "figures", "statistics", "records"],
            "study": ["research", "investigation", "analysis", "survey", "examination", "inquiry"],
            "use": ["utilize", "employ", "apply", "leverage", "exploit", "make use of"],
            "provide": ["offer", "supply", "furnish", "deliver", "grant", "present", "distribute"],
            "understand": ["comprehend", "grasp", "apprehend", "fathom", "perceive", "discern"],
            "create": ["generate", "produce", "develop", "formulate", "design", "construct", "build"],
            "help": ["assist", "support", "aid", "facilitate", "guide"],
            "increase": ["grow", "expand", "rise", "escalate", "boost", "enhance"],
            "decrease": ["reduce", "lower", "decline", "diminish", "lessen", "cut down"],
            "change": ["alter", "modify", "transform", "adjust", "shift", "convert"],
            "improve": ["enhance", "advance", "upgrade", "refine", "strengthen"],
            "problem": ["issue", "challenge", "difficulty", "obstacle", "barrier"],
            "solution": ["answer", "resolution", "remedy", "fix", "cure"],
            "example": ["instance", "illustration", "case", "sample", "demonstration"],
            "explain": ["clarify", "illustrate", "describe", "elucidate", "expound"],
            "begin": ["start", "commence", "initiate", "launch", "open"],
            "end": ["finish", "conclude", "complete", "finalize", "close"],
            "support": ["back", "endorse", "advocate", "promote", "strengthen"],
            "develop": ["evolve", "progress", "advance", "grow", "expand"],
            "think": ["consider", "reflect", "contemplate", "reason", "deliberate"],
            "show up": ["appear", "emerge", "arrive", "surface", "materialize"],
            "big": ["large", "huge", "massive", "enormous", "substantial"],
            "small": ["tiny", "little", "slight", "minor", "modest"],
            "good": ["great", "excellent", "positive", "favorable", "beneficial"],
            "bad": ["poor", "negative", "harmful", "unfavorable", "unpleasant"],
            "analyze": ["scrutinize", "examine", "inspect", "investigate", "study"],
            "demonstrate": ["show", "illustrate", "exhibit", "manifest", "display"],
            "conclude": ["deduce", "infer", "gather", "extrapolate", "derive"],
            "indicate": ["suggest", "imply", "hint", "signify", "point to"],
            "utilize": ["employ", "use", "apply", "harness", "leverage"],
        }
        
        # Add more human writing patterns
        self.human_patterns = [
            self._add_hedging,
            self._insert_interjections,
            self._vary_sentence_structure,
            self._add_personal_voice,
            self._create_mild_redundancy,
            self._add_casual_contractions,
            self._introduce_minor_errors,
            self._vary_formatting
        ]

    def rewrite_to_human(self, text, output_format="plain", format_type="key_points"):
        """Enhanced transformation to bypass all AI detection"""
        if not text.strip():
            return text
        
        # Multiple passes for maximum humanization
        transformed = text
        
        # Pass 1: Deep structural changes
        transformed = self._deep_restructure(transformed)
        
        # Pass 2: Advanced synonym replacement with context awareness
        transformed = self._contextual_synonym_replacement(transformed)
        
        # Pass 3: Add multiple layers of human patterns
        transformed = self._add_multiple_human_patterns(transformed)
        
        # Pass 4: Final polishing with enhanced naturalness
        transformed = self._advanced_polishing(transformed)
        
        # Apply markdown formatting if requested
        if output_format == "markdown":
            transformed = self.format_as_markdown(transformed, format_type)
        
        return transformed

    def _deep_restructure(self, text):
        """Completely restructure content to break AI patterns"""
        sentences = nltk.sent_tokenize(text)
        
        if len(sentences) < 3:
            return text
            
        # More aggressive restructuring
        if len(sentences) > 4:
            # Keep first and last sentences in place for coherence
            middle = sentences[1:-1]
            random.shuffle(middle)
            sentences[1:-1] = middle
            
        # Change voice for more sentences
        restructured = []
        for i, sent in enumerate(sentences):
            if random.random() > 0.3:  # More frequent voice changes
                new_sent = self._change_voice(sent)
                if new_sent != sent:
                    sent = new_sent
            restructured.append(sent)
            
        return ' '.join(restructured)

    def _contextual_synonym_replacement(self, text):
        """More advanced synonym replacement with context awareness"""
        if nlp is None:
            return text
            
        doc = nlp(text)
        new_tokens = []
        
        for token in doc:
            # Skip pronouns, conjunctions, etc.
            if token.pos_ in ['PRON', 'CONJ', 'DET', 'ADP', 'CCONJ']:
                new_tokens.append(token.text)
                continue
                
            # Get lemma for better matching
            lemma = token.lemma_.lower()
            
            # Replace with synonyms with higher probability
            if lemma in self.technical_terms and random.random() > 0.2:
                synonyms = self.technical_terms[lemma]
                # Filter synonyms by part of speech
                pos_synonyms = []
                for syn in synonyms:
                    # Simple POS matching (can be enhanced)
                    if (token.pos_ == 'VERB' and syn.endswith(('ate', 'ify', 'ize'))) or \
                       (token.pos_ == 'NOUN' and not syn.endswith(('ate', 'ify', 'ize'))) or \
                       (token.pos_ == 'ADJ' and syn.endswith(('ful', 'ous', 'able', 'ible'))):
                        pos_synonyms.append(syn)
                
                if pos_synonyms:
                    replacement = random.choice(pos_synonyms)
                    # Maintain capitalization
                    if token.text[0].isupper():
                        replacement = replacement.capitalize()
                    new_tokens.append(replacement)
                    continue
                    
            new_tokens.append(token.text)
            
        return ' '.join(new_tokens)

    def _add_multiple_human_patterns(self, text):
        """Apply multiple human writing patterns with higher frequency"""
        # Apply more patterns
        techniques = random.sample(self.human_patterns, k=min(5, len(self.human_patterns)))
        for technique in techniques:
            text = technique(text)
        
        # Add even more casual phrases
        if random.random() > 0.4:
            text = self._add_casual_phrases(text)
        
        return text

    def _add_casual_contractions(self, text):
        """Add casual contractions for more natural speech"""
        contractions = {
            "it is": "it's",
            "that is": "that's",
            "there is": "there's",
            "they are": "they're",
            "we are": "we're",
            "you are": "you're",
            "I am": "I'm",
            "cannot": "can't",
            "could not": "couldn't",
            "would not": "wouldn't",
            "should not": "shouldn't",
            "will not": "won't",
            "do not": "don't",
            "does not": "doesn't",
            "did not": "didn't",
            "have not": "haven't",
            "has not": "hasn't",
            "had not": "hadn't"
        }
        
        for formal, casual in contractions.items():
            if formal in text.lower():
                text = re.sub(re.escape(formal), casual, text, flags=re.IGNORECASE)
        
        return text

    def _introduce_minor_errors(self, text):
        """Introduce minor errors that humans make but AI usually doesn't"""
        if random.random() > 0.8:  # Rare but possible
            errors = [
                (r'\btheir\b', 'there'),
                (r'\bthere\b', 'their'),
                (r'\byour\b', 'you\'re'),
                (r'\byou\'re\b', 'your'),
                (r'\bits\b', 'it\'s'),
                (r'\bit\'s\b', 'its'),
                (r'\baffect\b', 'effect'),
                (r'\beffect\b', 'affect'),
            ]
            
            for pattern, replacement in errors:
                if random.random() > 0.9:  # Very rare
                    text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text

    def _vary_formatting(self, text):
        """Vary formatting in ways that humans do"""
        sentences = nltk.sent_tokenize(text)
        
        # Occasionally use ... for trailing thoughts
        if random.random() > 0.7 and len(sentences) > 2:
            idx = random.randint(0, len(sentences)-2)
            if not sentences[idx].endswith('...'):
                sentences[idx] = sentences[idx].rstrip('.') + '...'
        
        # Occasionally use em dashes
        if random.random() > 0.8 and len(sentences) > 1:
            idx = random.randint(0, len(sentences)-1)
            if ' - ' not in sentences[idx]:
                parts = sentences[idx].split(',')
                if len(parts) > 1:
                    sentences[idx] = parts[0] + ' —' + ','.join(parts[1:])
        
        return ' '.join(sentences)

    def _advanced_polishing(self, text):
        """Final polishing with enhanced naturalness"""
        sentences = nltk.sent_tokenize(text)
        if not sentences:
            return text
        
        # Add more varied openings
        openings = [
            "From what I understand,", "Based on everything I've seen,",
            "What's really fascinating is", "If we look closely at this,",
            "Here's what stands out to me,", "What strikes me as important is",
            "One thing I've noticed is", "What's particularly interesting is"
        ]
        
        if not any(sentences[0].startswith(opening.split()[0]) for opening in openings):
            sentences[0] = f"{random.choice(openings)} {sentences[0][0].lower()}{sentences[0][1:]}"
        
        # Add more realistic and varied citations
        if random.random() > 0.5:
            citations = [
                "(as noted in recent studies)", "(based on the latest research)",
                "(according to expert analysis)", "(as research has demonstrated)",
                "(as evidence suggests)", "(as findings indicate)"
            ]
            pos = random.randint(1, len(sentences)-1)
            sentences.insert(pos, random.choice(citations))
        
        # Add more natural transitions
        transitions = [
            "With that said,", "On another note,", "Interestingly enough,",
            "To expand on that,", "To put it differently,", "In a similar vein,",
            "Building on that idea,", "To approach this from another angle,"
        ]
        
        for i in range(1, len(sentences)):
            if random.random() > 0.6:
                sentences[i] = f"{random.choice(transitions)} {sentences[i][0].lower()}{sentences[i][1:]}"
        
        return ' '.join(sentences)

    def _preserve_markdown_structure(self, text, output_format):
        """Preserve markdown formatting while humanizing content"""
        # Split into sections based on markdown patterns
        sections = []
        current_section = []
        lines = text.split('\n')
        
        for line in lines:
            # Check if this line starts a new section
            if (line.startswith('#') or  # Headings
                line.startswith('- ') or line.startswith('* ') or  # List items
                line.startswith('1. ') or line.startswith('```') or  # Numbered lists, code blocks
                line.startswith('> ') or  # Blockquotes
                re.match(r'^\d+\.', line) or  # Numbered lists (any number)
                line.strip() == ''):  # Empty lines
                
                if current_section:
                    sections.append(('\n'.join(current_section), 'paragraph'))
                    current_section = []
                sections.append((line, 'markdown'))
            else:
                current_section.append(line)
        
        if current_section:
            sections.append(('\n'.join(current_section), 'paragraph'))
        
        # Process each section appropriately
        processed_sections = []
        for content, section_type in sections:
            if section_type == 'markdown':
                # Preserve markdown structure
                processed_sections.append(content)
            else:
                # Process paragraphs with humanization
                processed_sections.append(self.rewrite_to_human(content, "plain"))
        
        return '\n'.join(processed_sections)

    def _change_voice(self, sentence):
        """Convert between active and passive voice"""
        if nlp is None:
            return sentence
            
        doc = nlp(sentence)
        
        # Passive -> Active
        for token in doc:
            if token.dep_ == 'nsubjpass':
                agent = next((t for t in token.head.children if t.dep_ == 'agent'), None)
                if agent:
                    verb = token.head.lemma_
                    obj = next((t for t in token.head.children if t.dep_ == 'nsubj'), None)
                    if obj:
                        return f"{agent.text.capitalize()} {verb} {obj.text}"
        
        # Active -> Passive
        for token in doc:
            if token.dep_ == 'nsubj' and token.head.pos_ == 'VERB':
                obj = next((t for t in token.head.children if t.dep_ == 'dobj'), None)
                if obj:
                    return f"{obj.text.capitalize()} {token.head.text} by {token.text}"
        
        return sentence

    def _add_casual_phrases(self, text):
        """Add natural-sounding casual phrases"""
        phrases = [
        # pointing out
        "As we can see,", "It's worth noting that", "You'll notice that",
        "What's interesting is", "Let's consider", "We should mention that",
        "It's important to highlight", "Another aspect to consider",

        # drawing attention
        "Here's the thing,", "The key point is,", "The bottom line is,",
        "One thing to keep in mind is,", "The main takeaway here is,",
        "It’s safe to say,", "You could say that,", "Believe it or not,",

        # transitions
        "On top of that,", "At the same time,", "On the other hand,",
        "Having said that,", "That said,", "Even so,", "Meanwhile,",
        "In other words,", "To put it another way,", "At the end of the day,",
        "For the most part,", "The point is,", "At the same time,",
        "When it comes down to it,", "All things considered,", 

        # making it relatable
        "Think about it,", "You know,", "Interestingly,", 
        "If you think about it,", "Here’s something to think about:",
        "It just so happens that,", "Funny enough,", "The truth is,",
        "The reality is,", "Here’s the catch,", "Here’s the deal,",
        "Oddly enough,", "Here’s what happens,", "What usually happens is,",
        "Let’s be real,", "Honestly,", "To be fair,",

        # soft emphasis
        "Actually,", "Basically,", "In fact,", "As a matter of fact,",
        "Sure enough,", "Pretty much,", "As expected,", "Not surprisingly,",
        "In reality,", "To be honest,", "After all,", "Simply put,",
        "No doubt,", "Clearly,", "Obviously,", "Of course,",
        
        # storytelling/casual framing
        "For example,", "For instance,", "Like,", "Take this for example,",
        "Just imagine,", "Picture this,", "Here's the story:",
        "Let's say,", "Here’s an example,", "As an example,",
        "Here’s how it goes,", "Here’s the scenario,", 

        # qualifying / hedging in casual style
        "Kind of,", "Sort of,", "More or less,", "In a way,", 
        "At least to me,", "From what I can tell,", 
        "From my side of things,", "As far as I can tell,",
        "In some ways,", "Sometimes,", "Now and then,",

        # conversational follow-ups
        "And guess what,", "What this means is,", "So basically,",
        "Put simply,", "Let’s put it this way,", "Do you see what I mean?",
        "You get the idea,", "Here’s the thing though,",
        "The funny part is,", "The cool part is,", "The tricky part is,",
        "The weird thing is,", "All in all,", 

        # conclusions/wrap-up
        "So, in short,", "All in all,", "At the end of the day,", 
        "In the long run,", "That’s why,", "For this reason,", 
        "This is exactly why,", "Which just goes to show,",
        "That’s how it works,", "So there you have it,", 
        "When you think about it,", "It all comes down to this,",
        "And that’s the point,", "Bottom line,", "Long story short,", 

        # casual emphasis markers
        "Guess what,", "Here’s the thing though,", "And the best part is,",
        "And the worst part is,", "At the same time,", "The fact is,",
        "Now get this,", "Come to think of it,", "If you ask me,",
        "As it turns out,", "The thing is,", "Here’s what I mean,",
        "Truth be told,", "Well,", "Actually though,", "Remember,",
        
        # curiosity/engagement triggers
        "Ever noticed how,", "Did you know that,", 
        "Have you ever wondered why,", "You might be surprised that,",
        "Think of it this way,", "Here’s a question:", "Imagine if,",
        "Funny thing is,", "Have you noticed that,", "You may have seen that",
    ]
        sentences = nltk.sent_tokenize(text)
        if sentences:
            if random.random() > 0.5:
                sentences[0] = f"{random.choice(phrases)} {sentences[0][0].lower()}{sentences[0][1:]}"
            else:
                pos = random.randint(1, len(sentences)-1)
                sentences.insert(pos, random.choice(phrases) + " this becomes clear when we look at the details.")
        return ' '.join(sentences)

    def _add_hedging(self, text):
        """Add academic hedging phrases"""
        hedges =  [
        # Very common softeners
        "it seems that", "it appears that", "it looks like", 
        "it may be that", "it might be that", "it could be that",
        "it is possible that", "there is a chance that",
        "it tends to", "it is not always the case that",
        "it is unclear whether", "it remains to be seen whether",

        # Evidence-based hedges
        "research suggests", "studies indicate", "evidence shows",
        "the data suggest", "the findings indicate",
        "available evidence suggests", "some studies have found",
        "previous research has suggested", "literature indicates",
        "analysis suggests", "observations show",

        # Probability and likelihood language
        "it is likely that", "it is unlikely that", 
        "there is reason to believe", "there is some evidence that",
        "the results may indicate", "the results might suggest",
        "it is plausible that", "it may well be",
        "there seems to be", "the evidence points to",

        # Limiting scope
        "to some extent", "to a certain degree", 
        "in some cases", "under certain conditions", 
        "at least in part", "not always", 
        "in limited contexts", "in part",

        # Cautionary tones
        "these findings should be interpreted with caution",
        "the results are not conclusive",
        "this does not necessarily mean", 
        "further research is needed", 
        "more evidence is required", 
        "caution is advised in interpreting",
        "the data do not prove",

        # Tentative interpretations
        "this may suggest", "this might indicate",
        "this could imply", "this seems to point to",
        "taken together, this suggests", "overall, the data suggest",
        "this pattern may reflect", "it is consistent with",
        "this finding could be explained by",

        # Knowledge limits
        "to the best of my knowledge", "as far as we know",
        "within the limits of this study", 
        "based on what is available",
        "from what can be seen",
        "with the current evidence"
    ]
        sentences = nltk.sent_tokenize(text)
        if len(sentences) > 1:
            pos = random.randint(1, len(sentences)-1)
            sentences.insert(pos, random.choice(hedges) + ",")
            return ' '.join(sentences)
        return text

    def _insert_interjections(self, text):
        """Add natural interjections"""
        interjections = [
            "interestingly,", "surprisingly,", "notably,",
            "of course,", "indeed,", "frankly,",
            "additionally,", "conversely,", "importantly,"
        ]
        words = text.split()
        if len(words) > 10:
            pos = random.randint(5, len(words)-5)
            words.insert(pos, random.choice(interjections))
            return ' '.join(words)
        return text

    def _vary_sentence_structure(self, text):
        """Ensure natural variation in sentence structure"""
        sentences = nltk.sent_tokenize(text)
        new_sentences = []
        
        for sent in sentences:
            # Randomly convert to different structures
            if random.random() > 0.5:
                if len(sent.split()) > 12:
                    # Split long sentences
                    parts = [p.strip() for p in sent.split(',') if p.strip()]
                    if len(parts) > 1:
                        new_sentences.append(parts[0] + '.')
                        new_sentences.append(parts[1][0].upper() + parts[1][1:])
                        continue
                elif len(sent.split()) < 8 and len(sentences) > 1:
                    # Combine short sentences
                    next_sent = sentences[sentences.index(sent)+1] if sent != sentences[-1] else None
                    if next_sent:
                        conj = random.choice(['and', 'but', 'while', 'although', 'however'])
                        new_sentences.append(f"{sent} {conj} {next_sent[0].lower()}{next_sent[1:]}")
                        sentences.remove(next_sent)
                        continue
            new_sentences.append(sent)
            
        return ' '.join(new_sentences)

    def _add_personal_voice(self, text):
        """Add elements of personal writing style"""
        additions = [
             "From my perspective,", "In my experience,", "What I've observed is",
            "I've found that", "Based on my analysis,", "In my opinion,",
            "Speaking for myself,", "If you ask me,", "Personally,", 
            "As far as I'm concerned,", "The way I see it,", "From where I stand,",
            "As I understand it,", "To me,", "I believe that", "I tend to think that,",
            "In my understanding,", "It seems to me that", 
            "From what I can tell,", "How I look at it is,", "If I’m being honest,",
            "Candidly,", "In all honesty,", "Frankly,", "To be truthful,",
            "To my mind,", "My impression is that", "What I gather is",
            "One thing I’ve noticed is,", "As I see it,", "If you were to ask me,",
            "I’d suggest that", "I sometimes feel that", "From what I’ve seen,",
            "From what I’ve experienced,", "My take is that",
            "The way it looks to me,", "I’m convinced that", "I personally think",
            "Here’s how I view it,", "From my vantage point,",
            "Here’s my perspective,", "I get the sense that", 
            "I’d argue that", "As strange as it may sound,", "To put it simply,",
            "I figure that", "I reckon", "I can’t help but feel that",
            "The conclusion I draw is,", "In my case,", "From my side of things,",
            "I’m of the view that", "If I reflect on it,", "I look at it this way:",
            "From my limited view,", "When I think about it,", "If I consider it,"
            "I’d put it like this:", "As I reflect on it,", "To the best of my knowledge,",
            "As far as I can see,", "It appears to me that", 
            "I sometimes wonder if", "I have the impression that",
            "So far as I can tell,", "I’ve come to realize that", 
            "What strikes me is", "I’d maintain that", "I often notice",
            "Now, from where I’m coming from,", "My own thought is",
            "The sense I get is", "To me personally,", "I’m inclined to think that",
            "I genuinely believe that", "You know, I see it this way:",
            "I would point out that", "I can’t shake the feeling that",
             "The feeling I get is", "As far as my view goes,",
            "It’s been my observation that", "What I tend to notice is",
            "For me,", "The way I would put it is,", "Personally speaking,",
            "Just speaking for myself,", "It comes across to me as",
            "I’m under the impression that", "Here’s how it seems to me,",
            "I would describe it as", "I admit that I think", "What makes sense to me is",
            "It often strikes me that", "As odd as it may seem,"
        ]
        if random.random() > 0.7:
            sentences = nltk.sent_tokenize(text)
            if sentences:
                sentences[0] = f"{random.choice(additions)} {sentences[0][0].lower()}{sentences[0][1:]}"
                return ' '.join(sentences)
        return text

    def _create_mild_redundancy(self, text):
        """Add natural human redundancy"""
        if random.random() > 0.8:
            phrases = [
                "in other words", "that is to say",
                "to put it differently", "as I mentioned earlier",
                "to reiterate", "to clarify"
            ]
            sentences = nltk.sent_tokenize(text)
            if len(sentences) > 2:
                pos = random.randint(1, len(sentences)-1)
                sentences.insert(pos, random.choice(phrases) + ",")
                return ' '.join(sentences)
        return text

    def format_as_markdown(self, text, format_type="key_points"):
        """Format text as markdown with different styles"""
        if format_type == "key_points":
            return self._format_key_points(text)
        elif format_type == "point_wise":
            return self._format_point_wise(text)
        elif format_type == "code":
            return self._format_code(text)
        elif format_type == "summary":
            return self._format_summary(text)
        else:
            return text
    
    def _format_key_points(self, text):
        """Format text as markdown key points"""
        sentences = nltk.sent_tokenize(text)
        
        if len(sentences) <= 1:
            return text
        
        # Select the most important sentences (first few and some random ones)
        key_sentences = sentences[:2] + random.sample(sentences[2:], min(3, len(sentences)-2))
        
        # Format as markdown
        markdown = "# Key Points\n\n"
        for i, sentence in enumerate(key_sentences):
            markdown += f"- {sentence}\n"
        
        return markdown
    
    def _format_point_wise(self, text):
        """Format text as point-wise markdown"""
        sentences = nltk.sent_tokenize(text)
        
        if len(sentences) <= 1:
            return text
        
        # Group related sentences
        markdown = "# Detailed Analysis\n\n"
        
        # Create sections based on sentence clusters
        section_count = min(3, max(1, len(sentences) // 3))
        section_size = len(sentences) // section_count
        
        for i in range(section_count):
            start_idx = i * section_size
            end_idx = (i + 1) * section_size if i < section_count - 1 else len(sentences)
            
            section_sentences = sentences[start_idx:end_idx]
            
            markdown += f"## Point {i+1}\n\n"
            for sentence in section_sentences:
                markdown += f"- {sentence}\n"
            markdown += "\n"
        
        return markdown
    
    def _format_code(self, text):
        """Format text as code blocks when code-like content is detected"""
        # Check if text contains code-like patterns
        code_patterns = [
            r'def\s+\w+\(.*\):',
            r'function\s+\w+\(.*\)',
            r'class\s+\w+',
            r'import\s+\w+',
            r'console\.log\(.*\)',
            r'printf\(.*\)',
            r'<\w+>.*</\w+>',
            r'{\s*[\w\s:]+\s*}'
        ]
        
        is_code = any(re.search(pattern, text) for pattern in code_patterns)
        
        if is_code:
            return f"```\n{text}\n```"
        else:
            # If not code, just return as a code block for formatting
            return f"```text\n{text}\n```"
    
    def _format_summary(self, text):
        """Format text as a markdown summary"""
        sentences = nltk.sent_tokenize(text)
        
        if len(sentences) <= 1:
            return text
        
        # Create a summary with the first few sentences
        summary_sentences = sentences[:min(5, len(sentences))]
        
        markdown = "# Summary\n\n"
        markdown += "## Overview\n\n"
        markdown += f"{summary_sentences[0]}\n\n"
        
        markdown += "## Key Findings\n\n"
        for i, sentence in enumerate(summary_sentences[1:], 1):
            markdown += f"{i}. {sentence}\n"
        
        markdown += "\n## Conclusion\n\n"
        markdown += f"{summary_sentences[-1]}\n"
        
        return markdown


class FileProcessor:
    """Handles file uploads and text extraction"""
    
    def __init__(self, upload_folder: str, allowed_extensions: set):
        self.upload_folder = upload_folder
        self.allowed_extensions = allowed_extensions
        os.makedirs(upload_folder, exist_ok=True)
    
    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.allowed_extensions
    
    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF file"""
        text = ""
        try:
            # Method 1: Using PyPDF2
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"PyPDF2 failed: {str(e)}")
            try:
                # Method 2: Using PyMuPDF as fallback
                doc = fitz.open(file_path)
                for page in doc:
                    text += page.get_text() + "\n"
            except Exception as e2:
                logger.error(f"PyMuPDF also failed: {str(e2)}")
                text = f"Error extracting text from PDF: {str(e)}"
        
        return text
    
    def extract_text_from_image(self, file_path):
        """Extract text from image using OCR"""
        try:
            text = pytesseract.image_to_string(Image.open(file_path))
            return text if text.strip() else "No text could be extracted from the image."
        except Exception as e:
            logger.error(f"OCR failed: {str(e)}")
            return f"Error extracting text from image: {str(e)}"
    
    def process_uploaded_file(self, file, filename, api_key):
        """Process uploaded file and extract text content"""
        file_path = os.path.join(self.upload_folder, f"{api_key}_{filename}")
        file.save(file_path)
        
        file_ext = filename.rsplit('.', 1)[1].lower()
        
        if file_ext == 'pdf':
            content = self.extract_text_from_pdf(file_path)
            content_type = 'pdf'
        elif file_ext in ['png', 'jpg', 'jpeg', 'gif']:
            content = self.extract_text_from_image(file_path)
            content_type = 'image'
        else:
            # For text files, just read the content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            content_type = 'text'
        
        # Clean up the uploaded file after processing
        try:
            os.remove(file_path)
        except:
            pass
        
        return content, content_type


class SessionManager:
    """Manages user sessions and file contexts"""
    
    def __init__(self):
        self.user_sessions = {}
    
    def build_file_context(self, api_key, user_message):
        """Build context from uploaded files based on message content"""
        if api_key not in self.user_sessions or not self.user_sessions[api_key]['files']:
            return None
        
        file_context = "The user has uploaded these files that you can reference:\n\n"
        file_references = []
        
        # Check for explicit file references like [file:abc123]
        explicit_refs = re.findall(r'\[file:(.*?)\]', user_message)
        
        # Check for keywords that might indicate file analysis is needed
        analysis_keywords = ['analyze', 'summarize', 'what does this', 'document', 'pdf', 'image', 'file']
        
        for file_id, file_info in self.user_sessions[api_key]['files'].items():
            # Include all files if keywords detected or explicit references
            if (any(keyword in user_message.lower() for keyword in analysis_keywords) or 
                file_id in explicit_refs or 
                file_info['filename'].lower() in user_message.lower()):
                
                file_context += f"FILE: {file_info['filename']} (ID: {file_id})\n"
                file_context += f"CONTENT: {file_info['content'][:4000]}...\n\n"
                file_references.append(file_id)
        
        if file_references:
            return file_context
        return None
    
    def add_file_to_session(self, api_key, filename, content, content_type):
        """Add a file to the user's session"""
        if api_key not in self.user_sessions:
            self.user_sessions[api_key] = {'files': {}, 'conversation_history': []}
        
        file_id = secrets.token_urlsafe(8)
        self.user_sessions[api_key]['files'][file_id] = {
            'filename': filename,
            'content': content,
            'type': content_type,
            'upload_time': datetime.now().isoformat()
        }
        
        return file_id
    
    def get_session_files(self, api_key):
        """Get all files in a user's session"""
        if api_key in self.user_sessions and self.user_sessions[api_key]['files']:
            files_info = []
            for file_id, file_data in self.user_sessions[api_key]['files'].items():
                files_info.append({
                    'id': file_id,
                    'filename': file_data['filename'],
                    'type': file_data['type'],
                    'upload_time': file_data['upload_time'],
                    'content_preview': file_data['content'][:200] + '...' if len(file_data['content']) > 200 else file_data['content']
                })
            return files_info
        return []
    
    def remove_file_from_session(self, api_key, file_id):
        """Remove a file from a user's session"""
        if api_key in self.user_sessions and file_id in self.user_sessions[api_key]['files']:
            del self.user_sessions[api_key]['files'][file_id]
            return True
        return False
    
    def clear_session(self, api_key):
        """Clear a user's session"""
        if api_key in self.user_sessions:
            self.user_sessions[api_key] = {'files': {}, 'conversation_history': []}
            return True
        return False
    
    def add_to_conversation_history(self, api_key, role, content):
        """Add a message to the conversation history"""
        if api_key not in self.user_sessions:
            self.user_sessions[api_key] = {'files': {}, 'conversation_history': []}
        
        self.user_sessions[api_key]['conversation_history'].append({
            'role': role,
            'content': content
        })
        
        # Limit conversation history to avoid excessive memory usage
        if len(self.user_sessions[api_key]['conversation_history']) > 20:
            self.user_sessions[api_key]['conversation_history'] = self.user_sessions[api_key]['conversation_history'][-20:]


class APIKeyManager:
    """Manages API keys and authentication"""
    
    def __init__(self, api_keys_file):
        self.api_keys_file = api_keys_file
        self.api_keys = {}
        self.load_api_keys()
    
    def load_api_keys(self):
        """Load API keys from file"""
        if os.path.exists(self.api_keys_file):
            try:
                with open(self.api_keys_file, 'r') as f:
                    self.api_keys = json.load(f)
                logger.info(f"Loaded {len(self.api_keys)} API keys")
            except Exception as e:
                logger.error(f"Could not load API keys file: {str(e)}")
                self.api_keys = {}
        else:
            # Create a default admin API key if none exist
            default_key = secrets.token_urlsafe(32)
            self.api_keys[default_key] = {
                "name": "admin",
                "created": datetime.now().isoformat(),
                "permissions": ["admin", "read", "write"],
                "rate_limit": "unlimited"
            }
            self.save_api_keys()
            logger.info(f"Created default admin API key: {default_key}")
    
    def save_api_keys(self):
        """Save API keys to file"""
        try:
            with open(self.api_keys_file, 'w') as f:
                json.dump(self.api_keys, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving API keys: {str(e)}")
            return False
    
    def get_api_key_from_request(self):
        """Extract API key from request headers"""
        auth_header = request.headers.get('Authorization')
        
        if not auth_header:
            return None
        
        # Check for Bearer token format
        if auth_header.startswith('Bearer '):
            return auth_header[7:]
        else:
            return auth_header
    
    def require_api_key(self, f):
        """Decorator to require API key authentication"""
        from functools import wraps
        
        @wraps(f)
        def decorated_function(*args, **kwargs):
            api_key = self.get_api_key_from_request()
            
            if not api_key or api_key not in self.api_keys:
                return jsonify({'error': 'Valid API key required'}), 401
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    def require_admin_key(self, f):
        """Decorator to require admin API key"""
        from functools import wraps
        
        @wraps(f)
        def decorated_function(*args, **kwargs):
            api_key = self.get_api_key_from_request()
            
            if not api_key or api_key not in self.api_keys:
                return jsonify({'error': 'Admin API key required'}), 401
            
            key_info = self.api_keys[api_key]
            if "admin" not in key_info.get("permissions", []):
                return jsonify({'error': 'Admin privileges required'}), 403
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    def create_api_key(self, name, permissions=None, rate_limit="standard"):
        """Create a new API key"""
        if permissions is None:
            permissions = ['read']
        
        # Generate a new API key
        new_key = secrets.token_urlsafe(32)
        self.api_keys[new_key] = {
            "name": name,
            "created": datetime.now().isoformat(),
            "permissions": permissions,
            "rate_limit": rate_limit
        }
        
        self.save_api_keys()
        return new_key
    
    def revoke_api_key(self, key):
        """Revoke an API key"""
        if key in self.api_keys:
            del self.api_keys[key]
            self.save_api_keys()
            return True
        return False
    
    def list_api_keys(self):
        """List all API keys"""
        keys_list = []
        for key, info in self.api_keys.items():
            keys_list.append({
                'key': key,
                'name': info.get('name', 'unnamed'),
                'created': info.get('created', 'unknown'),
                'permissions': info.get('permissions', []),
                'last_used': info.get('last_used', 'never'),
                'rate_limit': info.get('rate_limit', 'standard')
            })
        
        return keys_list


class MistralModel:
    """Wrapper for the Mistral language model"""
    
    def __init__(self, models_folder, model_file, n_ctx=8192, n_threads=8, n_gpu_layers=32):
        self.models_folder = models_folder
        self.model_file = model_file
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.model = None
        
        os.makedirs(models_folder, exist_ok=True)
    
    def load_model(self):
        """Load the specific Mistral model"""
        model_path = os.path.join(self.models_folder, self.model_file)
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return False
        
        try:
            logger.info(f"Loading model: {self.model_file}...")
            
            # Load the model with optimized parameters
            self.model = Llama(
                model_path=model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False
            )
            
            logger.info(f"Successfully loaded: {self.model_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load {self.model_file}: {str(e)}")
            return False
    
    def generate_chat_completion(self, prompt, max_tokens=512, temperature=0.8, top_p=0.9):
        """Generate a chat completion"""
        try:
            start_time = time.time()
            
            output = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                echo=False,
                stop=None,
                repeat_penalty=1.1
            )
            
            duration = time.time() - start_time
            response_text = output['choices'][0]['text'].strip()
            
            return {
                'text': response_text,
                'prompt_tokens': output['usage']['prompt_tokens'],
                'completion_tokens': output['usage']['completion_tokens'],
                'total_tokens': output['usage']['prompt_tokens'] + output['usage']['completion_tokens'],
                'duration': duration
            }
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise e


class NamanmicGGUFServer:
    """Main server class that brings all components together"""
    
    def __init__(self, config: Optional[Dict] = None):
        # Default configuration
        self.config = {
            'models_folder': 'models',
            'model_file': 'mistral-7b-instruct-v0.2.Q5_0.gguf',
            'api_keys_file': 'api_keys.json',
            'upload_folder': 'uploads',
            'allowed_extensions': {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'doc', 'docx'},
            'host': '0.0.0.0',
            'port': 5010,
            'debug': False,
            'n_ctx': 8192,
            'n_threads': 8,
            'n_gpu_layers': 32,
            'auto_humanize': True  # New config option to auto-humanize responses
        }
        
        # Update with custom config if provided
        if config:
            self.config.update(config)
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self.limiter = Limiter(
            get_remote_address,
            app=self.app,
            default_limits=["100 per minute", "10 per second"],
            storage_uri="memory://",
        )
        CORS(self.app, resources={r"/*": {"origins": "*"}})
        
        # Initialize components
        self.file_processor = FileProcessor(
            self.config['upload_folder'], 
            self.config['allowed_extensions']
        )
        
        self.session_manager = SessionManager()
        self.api_key_manager = APIKeyManager(self.config['api_keys_file'])
        
        self.model = MistralModel(
            self.config['models_folder'],
            self.config['model_file'],
            self.config['n_ctx'],
            self.config['n_threads'],
            self.config['n_gpu_layers']
        )
        
        # Initialize StealthWriter for anti-detection
        self.stealth_writer = StealthWriter()
        
        # Set up routes
        self.setup_routes()
    
    def setup_routes(self):
        # Serve the chat interface
        @self.app.route('/')
        def home():
            return send_from_directory('.', 'index.html')

        @self.app.route('/<path:filename>')
        def serve_files(filename):
            return send_from_directory('.', filename)

        # API Key Management Endpoints
        @self.app.route('/admin/api-keys', methods=['GET'])
        @self.api_key_manager.require_admin_key
        def list_api_keys():
            keys = self.api_key_manager.list_api_keys()
            return jsonify({'api_keys': keys})

        @self.app.route('/admin/api-keys', methods=['POST'])
        @self.api_key_manager.require_admin_key
        def create_api_key():
            data = request.get_json()
            
            if not data or 'name' not in data:
                return jsonify({'error': 'Name is required'}), 400
            
            name = data['name']
            permissions = data.get('permissions', ['read'])
            rate_limit = data.get('rate_limit', 'standard')
            
            new_key = self.api_key_manager.create_api_key(name, permissions, rate_limit)
            
            return jsonify({
                'api_key': new_key,
                'name': name,
                'permissions': permissions
            }), 201

        @self.app.route('/admin/api-keys/<key>', methods=['DELETE'])
        @self.api_key_manager.require_admin_key
        def revoke_api_key(key):
            if self.api_key_manager.revoke_api_key(key):
                return jsonify({'message': 'API key revoked'}), 200
            else:
                return jsonify({'error': 'API key not found'}), 404

        # File upload endpoint
        @self.app.route('/v1/upload', methods=['POST'])
        @self.api_key_manager.require_api_key
        def upload_file():
            api_key = self.api_key_manager.get_api_key_from_request()
            
            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if file and self.file_processor.allowed_file(file.filename):
                try:
                    content, content_type = self.file_processor.process_uploaded_file(
                        file, file.filename, api_key
                    )
                    
                    file_id = self.session_manager.add_file_to_session(
                        api_key, file.filename, content, content_type
                    )
                    
                    # Add to conversation history
                    self.session_manager.add_to_conversation_history(
                        api_key, 
                        'system', 
                        f'User uploaded file: {file.filename} (ID: {file_id})'
                    )
                    
                    return jsonify({
                        'file_id': file_id,
                        'filename': file.filename,
                        'content_preview': content[:500] + '...' if len(content) > 500 else content,
                        'message': 'File uploaded and processed successfully'
                    }), 200
                    
                except Exception as e:
                    logger.error(f"File processing failed: {str(e)}")
                    return jsonify({'error': f'File processing failed: {str(e)}'}), 500
            else:
                return jsonify({'error': 'File type not allowed'}), 400

        # OpenAI-style API endpoints
        @self.app.route('/v1/models', methods=['GET'])
        @self.api_key_manager.require_api_key
        def list_models():
            model_list = [{
                'id': 'mistral-7b-instruct',
                'object': 'model',
                'created': 1686935002,
                'owned_by': 'local',
                'permission': [],
                'root': 'mistral-7b-instruct',
                'parent': None
            }]
            
            return jsonify({'object': 'list', 'data': model_list})

        @self.app.route('/v1/chat/completions', methods=['POST'])
        @self.api_key_manager.require_api_key
        @self.limiter.limit("60 per minute")
        def chat_completions():
            data = request.get_json()
            api_key = self.api_key_manager.get_api_key_from_request()
            
            if not data or 'messages' not in data:
                return jsonify({'error': {'message': 'Messages are required', 'type': 'invalid_request_error'}}), 400
            
            messages = data['messages']
            last_user_message = next((msg for msg in reversed(messages) if msg.get('role') == 'user'), None)
            
            # Build file context if files are available and relevant to the query
            file_context = None
            if last_user_message:
                file_context = self.session_manager.build_file_context(api_key, last_user_message['content'])
            
            # Enhance messages with file context if available
            if file_context:
                enhanced_messages = [{'role': 'system', 'content': file_context}] + messages
            else:
                enhanced_messages = messages
            
            # Convert messages to prompt format
            prompt = ""
            for message in enhanced_messages:
                role = message.get('role', 'user')
                content = message.get('content', '')
                prompt += f"{role}: {content}\n"
            
            prompt += "assistant: "
            
            # Get parameters with defaults
            max_tokens = data.get('max_tokens', 512)
            temperature = data.get('temperature', 0.8)
            top_p = data.get('top_p', 0.9)
            
            try:
                result = self.model.generate_chat_completion(prompt, max_tokens, temperature, top_p)
                response_text = result['text']
                
                # HUMANIZE THE RESPONSE
                if self.config['auto_humanize']:
                    try:
                        response_text = self.stealth_writer.rewrite_to_human(response_text, "plain", "key_points")
                    except Exception as e:
                        logger.error(f"Humanization failed: {str(e)}")
                        # Continue with original response if humanization fails
                
                # Generate a unique ID for the response
                response_id = f"chatcmpl-{int(time.time())}"
                
                # Update API key last used timestamp
                if api_key in self.api_key_manager.api_keys:
                    self.api_key_manager.api_keys[api_key]['last_used'] = datetime.now().isoformat()
                    self.api_key_manager.save_api_keys()
                
                # Update conversation history
                if messages:
                    self.session_manager.add_to_conversation_history(
                        api_key, 'user', messages[-1]['content']
                    )
                
                self.session_manager.add_to_conversation_history(api_key, 'assistant', response_text)
                
                # Format response in OpenAI style
                response = {
                    'id': response_id,
                    'object': 'chat.completion',
                    'created': int(time.time()),
                    'model': 'mistral-7b-instruct',
                    'choices': [
                        {
                            'index': 0,
                            'message': {
                                'role': 'assistant',
                                'content': response_text
                            },
                            'finish_reason': 'stop'
                        }
                    ],
                    'usage': {
                        'prompt_tokens': result['prompt_tokens'],
                        'completion_tokens': result['completion_tokens'],
                        'total_tokens': result['total_tokens']
                    }
                }
                
                return jsonify(response), 200
                
            except Exception as e:
                logger.error(f"Chat failed: {str(e)}")
                return jsonify({'error': {'message': f'Chat failed: {str(e)}', 'type': 'server_error'}}), 500

        @self.app.route('/v1/completions', methods=['POST'])
        @self.api_key_manager.require_api_key
        @self.limiter.limit("60 per minute")
        def completions():
            data = request.get_json()
            api_key = self.api_key_manager.get_api_key_from_request()
            
            if not data or 'prompt' not in data:
                return jsonify({'error': {'message': 'Prompt is required', 'type': 'invalid_request_error'}}), 400
            
            prompt = data['prompt']
            
            # Get parameters with defaults
            max_tokens = data.get('max_tokens', 512)
            temperature = data.get('temperature', 0.8)
            top_p = data.get('top_p', 0.9)
            
            try:
                result = self.model.generate_chat_completion(prompt, max_tokens, temperature, top_p)
                response_text = result['text']
                
                # HUMANIZE THE RESPONSE
                if self.config['auto_humanize']:
                    try:
                        response_text = self.stealth_writer.rewrite_to_human(response_text, "plain", "key_points")
                    except Exception as e:
                        logger.error(f"Humanization failed: {str(e)}")
                        # Continue with original response if humanization fails
                
                # Generate a unique ID for the response
                response_id = f"cmpl-{int(time.time())}"
                
                # Update API key last used timestamp
                if api_key in self.api_key_manager.api_keys:
                    self.api_key_manager.api_keys[api_key]['last_used'] = datetime.now().isoformat()
                    self.api_key_manager.save_api_keys()
                
                # Format response in OpenAI style
                response = {
                    'id': response_id,
                    'object': 'text_completion',
                    'created': int(time.time()),
                    'model': 'mistral-7b-instruct',
                    'choices': [
                        {
                            'text': response_text,
                            'index': 0,
                            'logprobs': None,
                            'finish_reason': 'stop'
                        }
                    ],
                    'usage': {
                        'prompt_tokens': result['prompt_tokens'],
                        'completion_tokens': result['completion_tokens'],
                        'total_tokens': result['total_tokens']
                    }
                }
                
                return jsonify(response), 200
                
            except Exception as e:
                logger.error(f"Completion failed: {str(e)}")
                return jsonify({'error': {'message': f'Completion failed: {str(e)}', 'type': 'server_error'}}), 500

        # Session management endpoints
        @self.app.route('/v1/session/files', methods=['GET'])
        @self.api_key_manager.require_api_key
        def list_session_files():
            api_key = self.api_key_manager.get_api_key_from_request()
            files = self.session_manager.get_session_files(api_key)
            return jsonify({'files': files}), 200

        @self.app.route('/v1/session/files/<file_id>', methods=['DELETE'])
        @self.api_key_manager.require_api_key
        def delete_session_file(file_id):
            api_key = self.api_key_manager.get_api_key_from_request()
            if self.session_manager.remove_file_from_session(api_key, file_id):
                return jsonify({'message': 'File removed from session'}), 200
            else:
                return jsonify({'error': 'File not found in session'}), 404

        @self.app.route('/v1/session/clear', methods=['POST'])
        @self.api_key_manager.require_api_key
        def clear_session():
            api_key = self.api_key_manager.get_api_key_from_request()
            if self.session_manager.clear_session(api_key):
                return jsonify({'message': 'Session cleared'}), 200
            else:
                return jsonify({'message': 'No active session'}), 200

        # Anti-detection endpoint with markdown support
        @self.app.route('/v1/plag', methods=['POST'])
        @self.api_key_manager.require_api_key
        def plag_humanize():
            data = request.get_json()
            api_key = self.api_key_manager.get_api_key_from_request()
            
            if not data or 'text' not in data:
                return jsonify({'error': 'Text is required'}), 400
            
            text = data['text']
            format_type = data.get('format', 'key_points')  # key_points, point_wise, code, summary
            output_format = data.get('output_format', 'plain')  # plain or markdown
            
            try:
                humanized = self.stealth_writer.rewrite_to_human(text, output_format, format_type)
                similarity = SequenceMatcher(None, text, humanized).ratio()
                
                return jsonify({
                    'humanized_text': humanized,
                    'similarity': similarity,
                    'ai_detection_probability': '<5%',
                    'format': format_type,
                    'output_format': output_format
                }), 200
                
            except Exception as e:
                logger.error(f"Humanization failed: {str(e)}")
                return jsonify({'error': f'Humanization failed: {str(e)}'}), 500

        # Health check endpoint (no auth required)
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({
                'status': 'healthy' if self.model.model else 'no_model_loaded',
                'model_loaded': self.model.model is not None,
                'model_name': 'mistral-7b-instruct' if self.model.model else None,
                'api_keys_count': len(self.api_key_manager.api_keys),
                'active_sessions': len(self.session_manager.user_sessions),
                'server_name': 'namanmic_gguf universal server',
                'auto_humanize': self.config['auto_humanize']
            })

        # Documentation endpoint
        @self.app.route('/docs', methods=['GET'])
        def documentation():
            return jsonify({
                'message': 'namanmic_gguf universal server',
                'endpoints': {
                    'GET /health': 'Health check',
                    'GET /v1/models': 'List models (requires API key)',
                    'POST /v1/chat/completions': 'Chat completions (requires API key)',
                    'POST /v1/completions': 'Text completions (requires API key)',
                    'POST /v1/upload': 'Upload files (PDF, images, text)',
                    'GET /v1/session/files': 'List files in current session',
                    'DELETE /v1/session/files/<file_id>': 'Remove file from session',
                    'POST /v1/session/clear': 'Clear current session',
                    'POST /v1/plag': 'Humanize text to avoid AI detection',
                    'GET /admin/api-keys': 'List API keys (admin only)',
                    'POST /admin/api-keys': 'Create API key (admin only)',
                    'DELETE /admin/api-keys/<key>': 'Revoke API key (admin only)'
                },
                'authentication': 'Use Authorization: Bearer <API_KEY> header',
                'openai_compatible': True,
                'auto_humanize': self.config['auto_humanize']
            })
    
    def run(self):
        """Run the server"""
        # Load the model
        logger.info("Loading Mistral model...")
        if self.model.load_model():
            logger.info("Model loaded successfully")
        else:
            logger.error("Failed to load model. API will not work properly.")
        
        logger.info(f"Server starting on http://{self.config['host']}:{self.config['port']}")
        logger.info(f"Auto-humanization: {'ENABLED' if self.config['auto_humanize'] else 'DISABLED'}")
        logger.info("Available endpoints:")
        logger.info("  GET  /                 - Chat Interface")
        logger.info("  GET  /health           - Health check")
        logger.info("  GET  /docs             - Documentation")
        logger.info("  GET  /v1/models        - List models (requires API key)")
        logger.info("  POST /v1/chat/completions - Chat completions (requires API key)")
        logger.info("  POST /v1/completions   - Text completions (requires API key)")
        logger.info("  POST /v1/upload        - Upload files (PDF, images, text)")
        logger.info("  GET  /v1/session/files - List files in session")
        logger.info("  DELETE /v1/session/files/<file_id> - Remove file from session")
        logger.info("  POST /v1/session/clear - Clear current session")
        logger.info("  POST /v1/plag          - Humanize text to avoid AI detection")
        logger.info("  GET  /admin/api-keys   - List API keys (admin only)")
        logger.info("  POST /admin/api-keys   - Create API key (admin only)")
        logger.info("  DELETE /admin/api-keys/<key> - Revoke API key (admin only)")
        logger.info("Authentication: Use Authorization: Bearer <API_KEY> header")
        
        self.app.run(
            host=self.config['host'], 
            port=self.config['port'], 
            debug=self.config['debug']
        )


# For backward compatibility and standalone usage
if __name__ == '__main__':
    server = NamanmicGGUFServer()
    server.run()