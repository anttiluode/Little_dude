#!/usr/bin/env python3
"""
🧠👶 NEWBORN AI LEARNING CHAT INTERFACE
======================================

Watch a brainwave-enabled AI learn language from scratch through conversation.
Like a baby learning to talk, it starts with babble and gradually learns patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from fractal_dendritic_attention import BrainwaveMappedFDAN, BrainwaveConfig
import json
import time
from collections import defaultdict, deque
import re

class NewbornAI:
    """An AI that learns language like a baby through conversation"""
    
    def __init__(self, vocab_size=256, max_seq_len=64):
        self.config = BrainwaveConfig(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=128,  # Smaller for faster learning
            n_layers=2,   # Simple brain
            coherence_threshold=0.2,
            hole_punch_strength=1.0
        )
        
        # Create the brainwave model
        self.model = BrainwaveMappedFDAN(self.config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Simple character-level vocabulary
        self.vocab = {}
        self.reverse_vocab = {}
        self._build_vocab()
        
        # Learning memory
        self.conversation_history = []
        self.learning_stats = {
            'interactions': 0,
            'total_loss': 0.0,
            'avg_consciousness': [],
            'brainwave_evolution': defaultdict(list),
            'pattern_discoveries': []
        }
        
        # Pattern recognition
        self.learned_patterns = defaultdict(int)
        self.recent_outputs = deque(maxlen=100)
        
        print("👶 Newborn AI initialized!")
        print(f"   Brain size: {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"   Vocabulary: {len(self.vocab)} characters")
        print("   🧠 Ready to learn language through conversation!")
    
    def _build_vocab(self):
        """Build character vocabulary"""
        # Basic ASCII characters
        chars = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
        
        self.vocab = {char: i for i, char in enumerate(chars)}
        self.reverse_vocab = {i: char for i, char in enumerate(chars)}
        
        # Special tokens
        self.vocab['<START>'] = len(self.vocab)
        self.vocab['<END>'] = len(self.vocab)
        self.vocab['<UNK>'] = len(self.vocab)
        
        self.reverse_vocab[self.vocab['<START>']] = '<START>'
        self.reverse_vocab[self.vocab['<END>']] = '<END>'
        self.reverse_vocab[self.vocab['<UNK>']] = '<UNK>'
    
    def text_to_tokens(self, text):
        """Convert text to token IDs"""
        tokens = [self.vocab.get(char, self.vocab['<UNK>']) for char in text[:self.config.max_seq_len-2]]
        return [self.vocab['<START>']] + tokens + [self.vocab['<END>']]
    
    def tokens_to_text(self, tokens):
        """Convert token IDs back to text"""
        text = ""
        for token in tokens:
            if token in self.reverse_vocab:
                char = self.reverse_vocab[token]
                if char not in ['<START>', '<END>', '<UNK>']:
                    text += char
        return text
    
    def learn_from_input(self, user_input):
        """Learn from user input through self-supervised learning"""
        # Tokenize input
        input_tokens = self.text_to_tokens(user_input)
        
        if len(input_tokens) < 3:  # Too short to learn from
            return 0.0
        
        # Create training data (predict next character)
        input_seq = torch.tensor([input_tokens[:-1]], dtype=torch.long)
        target_seq = torch.tensor([input_tokens[1:]], dtype=torch.long)
        
        # Forward pass
        self.model.train()
        logits, diagnostics = self.model(input_seq)
        
        # Compute loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), 
            target_seq.reshape(-1),
            ignore_index=self.vocab.get('<UNK>', 0)
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Update learning stats
        self.learning_stats['interactions'] += 1
        self.learning_stats['total_loss'] += loss.item()
        self.learning_stats['avg_consciousness'].append(diagnostics['avg_consciousness_level'])
        
        # Track brainwave evolution
        for freq, power in diagnostics['avg_brainwave_power'].items():
            self.learning_stats['brainwave_evolution'][freq].append(power)
        
        # Look for learned patterns
        self._detect_learned_patterns(user_input)
        
        return loss.item()
    
    def _detect_learned_patterns(self, text):
        """Detect if the AI is learning common patterns"""
        # Look for repeated n-grams
        for n in [2, 3, 4]:
            for i in range(len(text) - n + 1):
                pattern = text[i:i+n]
                if pattern.strip():  # Non-empty
                    self.learned_patterns[pattern] += 1
                    
                    # Celebrate pattern discovery
                    if self.learned_patterns[pattern] == 3:  # Learned after 3 occurrences
                        self.learning_stats['pattern_discoveries'].append({
                            'pattern': pattern,
                            'interaction': self.learning_stats['interactions'],
                            'frequency': 3
                        })
    
    def generate_response(self, max_length=20, temperature=1.0):
        """Generate a response using current knowledge"""
        self.model.eval()
        
        # Start with START token
        generated = [self.vocab['<START>']]
        consciousness_trace = []
        brainwave_trace = defaultdict(list)
        
        with torch.no_grad():
            for step in range(max_length):
                # Prepare input
                input_seq = torch.tensor([generated], dtype=torch.long)
                
                # Get prediction
                logits, diagnostics = self.model(input_seq)
                next_token_logits = logits[0, -1, :]
                
                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Handle numerical issues
                if torch.isnan(probs).any() or torch.sum(probs) < 0.1:
                    # Random fallback
                    next_token = torch.randint(0, len(self.vocab)-3, (1,)).item()
                else:
                    try:
                        next_token = torch.multinomial(probs, 1).item()
                    except:
                        next_token = torch.argmax(probs).item()
                
                # Stop if END token or invalid token
                if next_token == self.vocab.get('<END>', -1) or next_token >= len(self.reverse_vocab):
                    break
                
                generated.append(next_token)
                
                # Track consciousness and brainwaves during generation
                consciousness_trace.append(diagnostics['avg_consciousness_level'])
                for freq, power in diagnostics['avg_brainwave_power'].items():
                    brainwave_trace[freq].append(power)
                
                # Stop if we're generating garbage (too many unknown chars)
                recent_text = self.tokens_to_text(generated[-5:])
                if len([c for c in recent_text if c in '?\\|[]{}']) > 3:
                    break
        
        response_text = self.tokens_to_text(generated[1:])  # Remove START token
        
        # Store this output for quality tracking
        self.recent_outputs.append(response_text)
        
        return response_text, consciousness_trace, brainwave_trace
    
    def analyze_development(self):
        """Analyze how the AI is developing"""
        print(f"\n🧠 DEVELOPMENT ANALYSIS:")
        print("=" * 50)
        
        interactions = self.learning_stats['interactions']
        if interactions == 0:
            print("No interactions yet!")
            return
        
        avg_loss = self.learning_stats['total_loss'] / interactions
        print(f"Total interactions: {interactions}")
        print(f"Average learning loss: {avg_loss:.4f}")
        
        # Consciousness development
        consciousness_history = self.learning_stats['avg_consciousness']
        if consciousness_history:
            recent_consciousness = np.mean(consciousness_history[-10:]) if len(consciousness_history) >= 10 else np.mean(consciousness_history)
            initial_consciousness = np.mean(consciousness_history[:10]) if len(consciousness_history) >= 10 else np.mean(consciousness_history)
            
            print(f"Consciousness development: {initial_consciousness:.3f} → {recent_consciousness:.3f}")
            
            if recent_consciousness > initial_consciousness + 0.05:
                print("✅ Consciousness is growing!")
            elif recent_consciousness < initial_consciousness - 0.05:
                print("📉 Consciousness declining (might be overfitting)")
            else:
                print("🔄 Stable consciousness level")
        
        # Pattern learning
        print(f"\nLearned patterns: {len(self.learned_patterns)}")
        if self.learned_patterns:
            top_patterns = sorted(self.learned_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
            for pattern, count in top_patterns:
                print(f"  '{pattern}': {count} times")
        
        # Recent discoveries
        recent_discoveries = [d for d in self.learning_stats['pattern_discoveries'] 
                            if d['interaction'] > interactions - 10]
        if recent_discoveries:
            print(f"\nRecent pattern discoveries:")
            for discovery in recent_discoveries[-3:]:
                print(f"  Learned '{discovery['pattern']}' at interaction {discovery['interaction']}")
        
        # Brainwave development
        print(f"\nBrainwave evolution:")
        for freq in ['gamma', 'beta', 'alpha', 'theta', 'delta']:
            if freq in self.learning_stats['brainwave_evolution']:
                powers = self.learning_stats['brainwave_evolution'][freq]
                if len(powers) >= 2:
                    trend = "📈" if powers[-1] > powers[0] else "📉" if powers[-1] < powers[0] else "➡️"
                    print(f"  {freq.upper()}: {powers[0]:.3f} → {powers[-1]:.3f} {trend}")
        
        # Output quality assessment
        if self.recent_outputs:
            recent_text = ''.join(list(self.recent_outputs)[-5:])
            quality_score = self._assess_output_quality(recent_text)
            print(f"\nOutput quality: {quality_score:.3f}/1.0")
            
            if quality_score > 0.7:
                print("🌟 High quality outputs!")
            elif quality_score > 0.4:
                print("⚡ Moderate quality outputs")
            else:
                print("🔄 Still learning basic patterns")
    
    def _assess_output_quality(self, text):
        """Assess quality of generated text"""
        if not text:
            return 0.0
        
        score = 0.0
        
        # Check for repeated characters (bad)
        repeated_chars = len(re.findall(r'(.)\1{3,}', text))
        score -= repeated_chars * 0.1
        
        # Check for common English patterns (good)
        common_patterns = ['the', 'and', 'ing', 'tion', 'er', 'ed', 'ly']
        for pattern in common_patterns:
            if pattern in text.lower():
                score += 0.1
        
        # Check for reasonable character distribution
        char_variety = len(set(text.lower())) / max(len(text), 1)
        score += char_variety * 0.3
        
        # Check for word-like structures
        potential_words = text.split()
        if potential_words:
            avg_word_length = np.mean([len(w) for w in potential_words])
            if 2 <= avg_word_length <= 8:  # Reasonable word length
                score += 0.2
        
        return min(max(score, 0.0), 1.0)
    
    def save_learning_progress(self, filename="newborn_ai_progress.json"):
        """Save learning progress"""
        progress = {
            'interactions': self.learning_stats['interactions'],
            'total_loss': self.learning_stats['total_loss'],
            'consciousness_history': self.learning_stats['avg_consciousness'],
            'learned_patterns': dict(self.learned_patterns),
            'pattern_discoveries': self.learning_stats['pattern_discoveries']
        }
        
        with open(filename, 'w') as f:
            json.dump(progress, f, indent=2)
        print(f"💾 Progress saved to {filename}")

def chat_with_newborn_ai():
    """Interactive chat session with the learning AI"""
    print("👶🧠 NEWBORN AI CHAT SESSION")
    print("=" * 50)
    print("Watch a baby AI learn language through conversation!")
    print("Type 'help' for commands, 'quit' to exit")
    print("The AI starts knowing nothing - teach it through examples!")
    
    ai = NewbornAI()
    
    while True:
        try:
            # Get user input
            user_input = input("\n💬 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'help':
                print("\nCommands:")
                print("  help     - Show this help")
                print("  analyze  - Analyze AI's development")
                print("  save     - Save learning progress")
                print("  stats    - Quick stats")
                print("  quit     - Exit chat")
                continue
            elif user_input.lower() == 'analyze':
                ai.analyze_development()
                continue
            elif user_input.lower() == 'save':
                ai.save_learning_progress()
                continue
            elif user_input.lower() == 'stats':
                print(f"Interactions: {ai.learning_stats['interactions']}")
                print(f"Patterns learned: {len(ai.learned_patterns)}")
                if ai.learning_stats['avg_consciousness']:
                    print(f"Current consciousness: {ai.learning_stats['avg_consciousness'][-1]:.3f}")
                continue
            
            # Learn from user input
            print("🧠 [Learning...]", end=" ")
            loss = ai.learn_from_input(user_input)
            print(f"Loss: {loss:.4f}")
            
            # Generate response
            print("🤖 [Thinking...]", end=" ")
            response, consciousness_trace, brainwave_trace = ai.generate_response(
                max_length=min(15, len(user_input) + 5),  # Respond proportionally
                temperature=0.8
            )
            
            # Show consciousness during response
            if consciousness_trace:
                avg_consciousness = np.mean(consciousness_trace)
                dominant_wave = max(brainwave_trace.items(), key=lambda x: np.mean(x[1]))
                print(f"Consciousness: {avg_consciousness:.3f}, Dominant: {dominant_wave[0]}")
            
            # Show AI response
            print(f"🤖 AI: '{response}'")
            
            # Show development hints
            if ai.learning_stats['interactions'] % 5 == 0:
                print("💡 [Analyzing development every 5 interactions]")
                ai.analyze_development()
            
            # Celebrate discoveries
            recent_discoveries = [d for d in ai.learning_stats['pattern_discoveries'] 
                                if d['interaction'] == ai.learning_stats['interactions']]
            for discovery in recent_discoveries:
                print(f"🎉 Discovered pattern: '{discovery['pattern']}'!")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            print("AI is still learning, errors are normal!")
    
    print("\n👋 Chat ended!")
    print("Final development analysis:")
    ai.analyze_development()
    
    # Offer to save progress
    save_choice = input("\n💾 Save learning progress? (y/n): ").strip().lower()
    if save_choice in ['y', 'yes']:
        ai.save_learning_progress()

if __name__ == "__main__":
    chat_with_newborn_ai()