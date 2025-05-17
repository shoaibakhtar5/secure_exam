from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

class PlagiarismDetector:
    def __init__(self):
        self.prime = 101
        self.window_size = 4

    def rabin_karp(self, text1, text2):
        """
        Implements Rabin-Karp algorithm for string matching
        Returns similarity percentage
        """
        if not text1 or not text2:
            return 0

        def calculate_hash(text, start, end):
            result = 0
            for i in range(start, end):
                result += ord(text[i]) * (self.prime ** (i - start))
            return result

        n, m = len(text1), len(text2)
        if n < self.window_size or m < self.window_size:
            return self.levenshtein_distance(text1, text2)

        matches = 0
        total_windows = max(1, n - self.window_size + 1)

        pattern_hash = calculate_hash(text1, 0, self.window_size)
        text_hash = calculate_hash(text2, 0, self.window_size)

        for i in range(n - self.window_size + 1):
            if pattern_hash == text_hash:
                if text1[i:i+self.window_size] == text2[i:i+self.window_size]:
                    matches += 1
            if i < n - self.window_size:
                pattern_hash = (pattern_hash - ord(text1[i])) // self.prime
                pattern_hash += ord(text1[i + self.window_size]) * (self.prime ** (self.window_size - 1))

        return (matches / total_windows) * 100

    def kmp_search(self, pattern, text):
        """
        Implements KMP (Knuth-Morris-Pratt) pattern searching algorithm
        Returns the number of occurrences of pattern in text
        """
        if not pattern or not text:
            return 0

        def compute_lps(pattern):
            lps = [0] * len(pattern)
            length = 0
            i = 1

            while i < len(pattern):
                if pattern[i] == pattern[length]:
                    length += 1
                    lps[i] = length
                    i += 1
                else:
                    if length != 0:
                        length = lps[length - 1]
                    else:
                        lps[i] = 0
                        i += 1
            return lps

        M = len(pattern)
        N = len(text)
        lps = compute_lps(pattern)
        
        occurrences = 0
        i = j = 0
        
        while i < N:
            if pattern[j] == text[i]:
                i += 1
                j += 1
            
            if j == M:
                occurrences += 1
                j = lps[j-1]
            
            elif i < N and pattern[j] != text[i]:
                if j != 0:
                    j = lps[j-1]
                else:
                    i += 1
                    
        return occurrences

    def levenshtein_distance(self, s1, s2):
        """
        Calculates the Levenshtein distance between two strings
        Returns similarity percentage
        """
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return 0

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        max_length = max(len(s1), len(s2))
        similarity = (1 - previous_row[-1] / max_length) * 100
        return similarity

    def check_plagiarism(self, submission1, submission2):
        """
        Combines multiple algorithms to generate a comprehensive similarity score
        """
        rabin_karp_score = self.rabin_karp(submission1, submission2)
        levenshtein_score = self.levenshtein_distance(submission1, submission2)
        
        # Calculate pattern matches using KMP
        words1 = set(submission1.split())
        words2 = set(submission2.split())
        common_words = words1.intersection(words2)
        
        pattern_match_score = 0
        if common_words:
            matches = sum(self.kmp_search(word, submission2) for word in common_words)
            pattern_match_score = (matches / len(words2)) * 100

        # Weighted average of all scores
        final_score = (0.4 * rabin_karp_score + 
                      0.4 * levenshtein_score + 
                      0.2 * pattern_match_score)
        
        return {
            'similarity_score': round(final_score, 2),
            'rabin_karp_score': round(rabin_karp_score, 2),
            'levenshtein_score': round(levenshtein_score, 2),
            'pattern_match_score': round(pattern_match_score, 2)
        }

class AdvancedPlagiarismChecker:
    def __init__(self, stop_words=None):
        self.stop_words = set(stop_words) if stop_words else set()

    def preprocess(self, text):
        # Lowercase, remove punctuation, and stop words
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        words = text.split()
        if self.stop_words:
            words = [w for w in words if w not in self.stop_words]
        return ' '.join(words)

    def check_all(self, submissions):
        # submissions: list of (student_id, answer_text)
        ids = [s[0] for s in submissions]
        texts = [self.preprocess(s[1]) for s in submissions]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf_matrix)
        results = []
        n = len(submissions)
        for i in range(n):
            for j in range(i+1, n):
                score = sim_matrix[i, j] * 100
                results.append({
                    'student1': ids[i],
                    'student2': ids[j],
                    'similarity': round(score, 2)
                })
        return results 