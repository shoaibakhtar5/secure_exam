from collections import deque
import json
import time
from datetime import datetime

class ExamQueue:
    def __init__(self):
        self.active_exams = deque()
        self.exam_sessions = {}  # HashMap for storing active sessions

    def add_exam_session(self, exam_id, start_time, duration):
        """Add a new exam session to the queue"""
        session = {
            'exam_id': exam_id,
            'start_time': start_time,
            'duration': duration,
            'participants': set()
        }
        self.active_exams.append(session)
        self.exam_sessions[exam_id] = session
        return session

    def get_active_exams(self):
        """Get all currently active exams"""
        current_time = datetime.now()
        active = []
        
        while self.active_exams:
            session = self.active_exams[0]
            start = datetime.fromisoformat(session['start_time'])
            end_time = start.timestamp() + (session['duration'] * 60)
            
            if current_time.timestamp() > end_time:
                self.active_exams.popleft()
                del self.exam_sessions[session['exam_id']]
            else:
                break
                
        return list(self.active_exams)

    def join_exam(self, exam_id, student_id):
        """Add a student to an exam session"""
        if exam_id in self.exam_sessions:
            self.exam_sessions[exam_id]['participants'].add(student_id)
            return True
        return False

class ExamHashMap:
    def __init__(self):
        self.size = 1024
        self.map = [[] for _ in range(self.size)]

    def _hash(self, key):
        """Custom hash function for exam/student IDs"""
        hash_value = 0
        for char in str(key):
            hash_value = (hash_value * 31 + ord(char)) % self.size
        return hash_value

    def store_response(self, student_id, exam_id, response):
        """Store a student's exam response"""
        key = f"{student_id}:{exam_id}"
        index = self._hash(key)
        
        # Check if entry exists
        for i, (k, v) in enumerate(self.map[index]):
            if k == key:
                self.map[index][i] = (key, response)
                return
        
        # Add new entry
        self.map[index].append((key, response))

    def get_response(self, student_id, exam_id):
        """Retrieve a student's exam response"""
        key = f"{student_id}:{exam_id}"
        index = self._hash(key)
        
        for k, v in self.map[index]:
            if k == key:
                return v
        return None

    def get_all_responses(self, exam_id):
        """Get all responses for a specific exam"""
        responses = []
        for bucket in self.map:
            for key, value in bucket:
                if key.split(':')[1] == str(exam_id):
                    responses.append({
                        'student_id': key.split(':')[0],
                        'response': value
                    })
        return responses

class ExamManager:
    def __init__(self):
        self.queue = ExamQueue()
        self.responses = ExamHashMap()
        
    def create_exam(self, exam_data):
        """Create a new exam and add it to the queue"""
        with open('data/exams.json', 'r') as f:
            exams = json.load(f)
        
        exam_id = str(len(exams) + 1)
        exam_data['id'] = exam_id
        exams.append(exam_data)
        
        with open('data/exams.json', 'w') as f:
            json.dump(exams, f)
            
        self.queue.add_exam_session(
            exam_id,
            exam_data['start_time'],
            exam_data['duration']
        )
        return exam_id
        
    def submit_response(self, student_id, exam_id, response):
        """Submit a student's exam response"""
        self.responses.store_response(student_id, exam_id, response)
        
        # Store in persistent storage
        with open('data/answers.json', 'r') as f:
            answers = json.load(f)
            
        if exam_id not in answers:
            answers[exam_id] = {}
        answers[exam_id][student_id] = response
        
        with open('data/answers.json', 'w') as f:
            json.dump(answers, f)
            
    def get_exam_responses(self, exam_id):
        """Get all responses for an exam"""
        return self.responses.get_all_responses(exam_id) 