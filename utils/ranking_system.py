import heapq
from datetime import datetime

class BSTNode:
    def __init__(self, student_id, score):
        self.student_id = student_id
        self.score = score
        self.left = None
        self.right = None
        self.timestamp = datetime.now()

class ScoreTracker:
    def __init__(self):
        self.root = None
        self.size = 0
        self.scores_heap = []  # For top K scores

    def insert_score(self, student_id, score):
        """Insert a new score into the BST"""
        if not self.root:
            self.root = BSTNode(student_id, score)
        else:
            self._insert_recursive(self.root, student_id, score)
        
        # Also add to heap for quick top-K retrieval
        heapq.heappush(self.scores_heap, (-score, student_id))
        self.size += 1

    def _insert_recursive(self, node, student_id, score):
        """Helper method for recursive insertion"""
        if score < node.score:
            if node.left is None:
                node.left = BSTNode(student_id, score)
            else:
                self._insert_recursive(node.left, student_id, score)
        else:
            if node.right is None:
                node.right = BSTNode(student_id, score)
            else:
                self._insert_recursive(node.right, student_id, score)

    def get_rank(self, score):
        """Get the rank of a score (how many scores are higher)"""
        return self._count_higher_scores(self.root, score)

    def _count_higher_scores(self, node, score):
        """Helper method to count scores higher than given score"""
        if not node:
            return 0
        
        if score >= node.score:
            return self._count_higher_scores(node.right, score)
        
        return 1 + self._count_higher_scores(node.right, score) + self._count_higher_scores(node.left, score)

    def get_top_k_scores(self, k):
        """Get top K scores using heap"""
        result = []
        temp_heap = self.scores_heap.copy()
        
        for _ in range(min(k, len(temp_heap))):
            score, student_id = heapq.heappop(temp_heap)
            result.append({
                'student_id': student_id,
                'score': -score  # Convert back to positive
            })
        
        return result

    def get_percentile(self, score):
        """Calculate percentile for a given score"""
        higher_scores = self._count_higher_scores(self.root, score)
        if self.size == 0:
            return 100
        return ((self.size - higher_scores) / self.size) * 100

class RankingSystem:
    def __init__(self):
        self.exam_scores = {}  # Maps exam_id to ScoreTracker

    def add_score(self, exam_id, student_id, score):
        """Add a new score to the ranking system"""
        if exam_id not in self.exam_scores:
            self.exam_scores[exam_id] = ScoreTracker()
        
        self.exam_scores[exam_id].insert_score(student_id, score)

    def get_student_rank(self, exam_id, score):
        """Get a student's rank for a specific exam"""
        if exam_id not in self.exam_scores:
            return None
        
        tracker = self.exam_scores[exam_id]
        rank = tracker.get_rank(score)
        percentile = tracker.get_percentile(score)
        
        return {
            'rank': rank + 1,  # Convert to 1-based ranking
            'percentile': round(percentile, 2),
            'total_students': tracker.size
        }

    def get_top_performers(self, exam_id, k=10):
        """Get top K performers for an exam"""
        if exam_id not in self.exam_scores:
            return []
        
        return self.exam_scores[exam_id].get_top_k_scores(k) 