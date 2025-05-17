from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import json
import os
from datetime import datetime, timedelta, timezone
from sqlalchemy import func
import difflib
import re
import subprocess
import tempfile
import os
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename
import mimetypes

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:Shabi6264%40@127.0.0.1:3306/secure_exam'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(512))
    role = db.Column(db.String(20), nullable=False, default='student')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    student_class = db.Column(db.String(50))
    section = db.Column(db.String(10))
    department = db.Column(db.String(100))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Exam(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    duration = db.Column(db.Integer, nullable=False)  # in minutes
    total_marks = db.Column(db.Integer, nullable=False)
    start_time = db.Column(db.DateTime, nullable=False)
    end_time = db.Column(db.DateTime, nullable=False)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    exam_type = db.Column(db.String(50))
    exam_format = db.Column(db.String(20), default='regular')  # 'regular' or 'coding'
    student_class = db.Column(db.String(50))
    section = db.Column(db.String(10))
    department = db.Column(db.String(100))
    programming_language = db.Column(db.String(50))  # For coding exams
    max_file_size = db.Column(db.Integer)  # In KB
    allowed_imports = db.Column(db.Text)  # JSON string of allowed imports

class Answer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    exam_id = db.Column(db.Integer, db.ForeignKey('exam.id'))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    answer_text = db.Column(db.Text, nullable=False)
    score = db.Column(db.Float)
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)
    similarity_score = db.Column(db.Float)
    comment = db.Column(db.Text)  # Teacher comment
    status = db.Column(db.String(20), default='pending')  # pending, graded, flagged
    code_output = db.Column(db.Text)  # For coding exams
    test_cases_passed = db.Column(db.Integer)  # For coding exams
    total_test_cases = db.Column(db.Integer)  # For coding exams

class Question(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    exam_id = db.Column(db.Integer, db.ForeignKey('exam.id'))
    text = db.Column(db.Text, nullable=False)
    type = db.Column(db.String(20), nullable=False)  # 'short_answer', 'long_answer', 'multiple_choice', 'coding'
    marks = db.Column(db.Integer, nullable=False)
    options = db.Column(db.Text)  # JSON string for MCQ options
    correct_option = db.Column(db.Integer)  # Index of correct option for MCQ
    test_cases = db.Column(db.Text)  # JSON string for coding questions
    expected_output = db.Column(db.Text)  # JSON string for coding questions

class Assignment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    due_date = db.Column(db.DateTime, nullable=False)
    created_by = db.Column(db.Integer, db.ForeignKey('user.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    allowed_file_types = db.Column(db.String(200))
    assignment_filename = db.Column(db.String(255))
    assignment_file_path = db.Column(db.String(255))

class AssignmentSubmission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    assignment_id = db.Column(db.Integer, db.ForeignKey('assignment.id'))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    filename = db.Column(db.String(255))
    file_path = db.Column(db.String(255))
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)
    marks = db.Column(db.Float)
    comment = db.Column(db.Text)
    similarity_score = db.Column(db.Float)
    status = db.Column(db.String(20), default='pending')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Initialize database
with app.app_context():
    db.create_all()
    # Create admin user if not exists
    if not User.query.filter_by(username='admin').first():
        admin = User(username='admin', email='admin@secureexam.com', role='admin')
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        role = data.get('role')
        
        user = User.query.filter_by(username=username, role=role).first()
        if user and user.check_password(password):
            login_user(user)
            return jsonify({'status': 'success', 'role': user.role})
        
        return jsonify({'status': 'error', 'message': 'Invalid credentials'})
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        email = data.get('enrolment_number')
        role = data.get('role', 'student')
        student_class = data.get('student_semester')
        section = data.get('section')
        department = data.get('department')
        
        if User.query.filter_by(username=username).first():
            return jsonify({'status': 'error', 'message': 'Username already exists'})
            
        if User.query.filter_by(email=email).first():
            return jsonify({'status': 'error', 'message': 'Enrolment number already registered'})
        
        user = User(username=username, email=email, role=role, student_class=student_class, section=section, department=department)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        return jsonify({'status': 'success'})
    
    return render_template('register.html')

def get_pakistan_time():
    # Return naive datetime in PKT (no tzinfo)
    return datetime.utcnow() + timedelta(hours=5)

@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.role == 'admin':
        return render_template('admin_dashboard.html')
    if current_user.role == 'teacher':
        exams = Exam.query.filter_by(created_by=current_user.id).all()
        
        # Get analytics data
        total_submissions = Answer.query.filter(Answer.exam_id.in_([e.id for e in exams])).count()
        plagiarized_count = Answer.query.filter(
            Answer.exam_id.in_([e.id for e in exams]),
            Answer.similarity_score > 70
        ).count()
        
        # Get top students with plagiarism
        top_students_query = db.session.query(
            User.username,
            func.avg(Answer.similarity_score).label('avg_similarity')
        ).join(Answer).filter(
            Answer.exam_id.in_([e.id for e in exams]),
            Answer.similarity_score > 70
        ).group_by(User.username).order_by(func.avg(Answer.similarity_score).desc()).limit(5)
        
        top_students = [r[0] for r in top_students_query]
        top_students_scores = [round(r[1], 1) for r in top_students_query]
        
        # Get subject-wise stats
        subjects = [e.title for e in exams]
        subject_submissions = [
            Answer.query.filter_by(exam_id=e.id).count() 
            for e in exams
        ]
        
        # Get timeline data
        timeline_dates = []
        timeline_submissions = []
        for i in range(7):
            date = datetime.now() - timedelta(days=i)
            count = Answer.query.filter(
                Answer.exam_id.in_([e.id for e in exams]),
                func.date(Answer.submitted_at) == date.date()
            ).count()
            timeline_dates.append(date.strftime('%Y-%m-%d'))
            timeline_submissions.append(count)
        
        # Get recent submissions
        recent_submissions = []
        for answer in Answer.query.filter(
            Answer.exam_id.in_([e.id for e in exams])
        ).order_by(Answer.submitted_at.desc()).limit(10):
            exam = Exam.query.get(answer.exam_id)
            student = User.query.get(answer.user_id)
            file_url = getattr(answer, 'file_path', '') or ''
            file_name = getattr(answer, 'filename', '') or ''
            recent_submissions.append({
                'id': answer.id,
                'student_name': student.username,
                'exam_title': exam.title,
                'exam_type': exam.exam_type,
                'submission_time': answer.submitted_at.strftime('%Y-%m-%d %H:%M'),
                'status': answer.status,
                'plagiarism_score': round(answer.similarity_score, 1) if answer.similarity_score else None,
                'exam_id': answer.exam_id,
                'file_url': file_url,
                'file_name': file_name,
                'marks': answer.score,
                'comment': answer.comment
            })
        
        return render_template('teacher_dashboard.html',
            exams=exams,
            total_submissions=total_submissions,
            plagiarized_count=plagiarized_count,
            top_students=top_students,
            top_students_scores=top_students_scores,
            subjects=subjects,
            subject_submissions=subject_submissions,
            timeline_dates=timeline_dates,
            timeline_submissions=timeline_submissions,
            recent_submissions=recent_submissions
        )
    if current_user.role == 'student':
        now = get_pakistan_time()
        upcoming_exams = Exam.query.filter(
            Exam.end_time > now,
            Exam.student_class == current_user.student_class,
            Exam.section == current_user.section,
            Exam.department == current_user.department
        ).all()
        completed_exams = Exam.query.filter(Exam.end_time <= now).all()
        answers = Answer.query.filter_by(user_id=current_user.id).all()
        completed_exam_ids = [a.exam_id for a in answers]
        completed_exams = Exam.query.filter(Exam.id.in_(completed_exam_ids)).all()
        scores = [a.score for a in answers if a.score is not None]
        average_score = round(sum(scores) / len(scores), 2) if scores else 0
        recent_results = []
        for a in sorted(answers, key=lambda x: x.submitted_at, reverse=True)[:5]:
            exam = Exam.query.get(a.exam_id)
            recent_results.append({
                'title': exam.title if exam else 'Unknown',
                'type': getattr(exam, 'type', 'N/A'),
                'score': a.score if a.score is not None else 0,
                'rank': 1,
                'total_students': 1,
                'similarity_score': a.similarity_score if a.similarity_score is not None else 0,
                'exam_id': a.exam_id
            })
        exam_dates = [Exam.query.get(a.exam_id).start_time.strftime('%Y-%m-%d') for a in answers if Exam.query.get(a.exam_id)]
        score_history = [a.score if a.score is not None else 0 for a in answers]
        exam_type_scores = [0, 0, 0]
        for exam in upcoming_exams:
            if exam.start_time <= now <= exam.end_time:
                exam.status = 'active'
            else:
                exam.status = 'upcoming'
        return render_template(
            'student_dashboard.html',
            upcoming_exams=upcoming_exams,
            completed_exams=completed_exams,
            average_score=average_score,
            recent_results=recent_results,
            exam_dates=exam_dates,
            score_history=score_history,
            exam_type_scores=exam_type_scores
        )

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/create_exam', methods=['GET', 'POST'])
@login_required
def create_exam():
    if current_user.role != 'teacher':
        flash('Only teachers can create exams.', 'error')
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        title = request.form.get('title')
        exam_type = request.form.get('type')
        exam_format = request.form.get('exam_format', 'regular')
        duration = request.form.get('duration')
        total_marks = request.form.get('total_marks')
        start_date = request.form.get('start_date')
        start_time = request.form.get('start_time')
        passing_score = request.form.get('passing_score')
        student_class = request.form.get('student_class')
        section = request.form.get('section')
        department = request.form.get('department')
        
        # Coding exam specific fields
        programming_language = request.form.get('programming_language') if exam_format == 'coding' else None
        max_file_size = request.form.get('max_file_size') if exam_format == 'coding' else None
        allowed_imports = request.form.get('allowed_imports') if exam_format == 'coding' else None
        
        start_datetime = datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M")
        end_datetime = start_datetime + timedelta(minutes=int(duration))
        
        exam = Exam(
            title=title,
            description='',
            duration=int(duration),
            total_marks=int(total_marks),
            start_time=start_datetime,
            end_time=end_datetime,
            created_by=current_user.id,
            exam_type=exam_type,
            exam_format=exam_format,
            student_class=student_class,
            section=section,
            department=department,
            programming_language=programming_language,
            max_file_size=max_file_size,
            allowed_imports=allowed_imports
        )
        
        db.session.add(exam)
        db.session.commit()
        
        # Save questions
        i = 0
        while True:
            q_text = request.form.get(f'questions[{i}][text]')
            q_type = request.form.get(f'questions[{i}][type]')
            q_marks = request.form.get(f'questions[{i}][marks]')
            
            if not q_text or not q_type or not q_marks:
                i += 1
                if not request.form.get(f'questions[{i}][text]') and not request.form.get(f'questions[{i}][type]') and not request.form.get(f'questions[{i}][marks]'):
                    break
                continue
            
            options = None
            correct_option = None
            test_cases = None
            expected_output = None
            
            if q_type == 'multiple_choice':
                options = []
                for j in range(4):  # Assuming 4 options
                    option = request.form.get(f'questions[{i}][options][]')
                    if option:
                        options.append(option)
                correct_option = int(request.form.get(f'questions[{i}][correct]', 0))
                options = json.dumps(options)
            elif q_type == 'coding':
                test_cases = request.form.get(f'questions[{i}][test_cases]')
                expected_output = request.form.get(f'questions[{i}][expected_output]')
            
            question = Question(
                exam_id=exam.id,
                text=q_text,
                type=q_type,
                marks=int(q_marks),
                options=options,
                correct_option=correct_option,
                test_cases=test_cases,
                expected_output=expected_output
            )
            
            db.session.add(question)
            i += 1
        
        db.session.commit()
        return redirect(url_for('dashboard'))
    
    return render_template('create_exam.html')

@app.route('/exams')
@login_required
def my_exams():
    if current_user.role != 'student':
        flash('Access denied. This page is for students only.', 'error')
        return redirect(url_for('dashboard'))
    now = get_pakistan_time()
    upcoming_exams = Exam.query.filter(
        Exam.end_time > now,
        Exam.student_class == current_user.student_class,
        Exam.section == current_user.section,
        Exam.department == current_user.department
    ).all()
    for exam in upcoming_exams:
        if exam.start_time <= now <= exam.end_time:
            exam.status = 'active'
        else:
            exam.status = 'upcoming'
    return render_template('my_exams.html', exams=upcoming_exams, now=now)

@app.route('/results')
@login_required
def my_results():
    if current_user.role != 'student':
        flash('Access denied. This page is for students only.', 'error')
        return redirect(url_for('dashboard'))
    
    # Get all answers for the current user
    answers = Answer.query.filter_by(user_id=current_user.id).all()
    
    # Get exam details for each answer
    results = []
    for answer in answers:
        exam = Exam.query.get(answer.exam_id)
        if exam:
            # Get ranking information
            rank_info = get_rank_info(answer.exam_id, answer.score)
            results.append({
                'exam': exam,
                'answer': answer,
                'rank': rank_info['rank'],
                'total_students': rank_info['total_students'],
                'percentile': rank_info['percentile']
            })
    
    return render_template('my_results.html', results=results)

def get_rank_info(exam_id, score):
    """Helper function to get ranking information for a score"""
    # Get all scores for this exam
    all_scores = [a.score for a in Answer.query.filter_by(exam_id=exam_id).all() if a.score is not None]
    all_scores.sort(reverse=True)
    
    if not all_scores:
        return {'rank': 1, 'total_students': 1, 'percentile': 100}
    
    # Calculate rank
    rank = all_scores.index(score) + 1 if score in all_scores else len(all_scores) + 1
    
    # Calculate percentile
    percentile = ((len(all_scores) - rank + 1) / len(all_scores)) * 100
    
    return {
        'rank': rank,
        'total_students': len(all_scores),
        'percentile': round(percentile, 2)
    }

@app.route('/take_exam/<int:exam_id>', methods=['GET', 'POST'])
@login_required
def take_exam(exam_id):
    if current_user.role != 'student':
        flash('Access denied. This page is for students only.', 'error')
        return redirect(url_for('dashboard'))
    exam = Exam.query.get_or_404(exam_id)
    questions = Question.query.filter_by(exam_id=exam_id).all()
    now = get_pakistan_time()
    if not (exam.start_time <= now <= exam.end_time):
        flash('This exam is not currently active.', 'error')
        return redirect(url_for('my_exams'))
    if (exam.student_class != current_user.student_class or
        exam.section != current_user.section or
        exam.department != current_user.department):
        flash('You are not eligible to take this exam.', 'error')
        return redirect(url_for('my_exams'))
    existing_answer = Answer.query.filter_by(
        exam_id=exam_id,
        user_id=current_user.id
    ).first()
    if existing_answer:
        flash('You have already taken this exam.', 'error')
        return redirect(url_for('my_results'))
    if request.method == 'POST':
        answers = request.form.getlist('answers')
        # Save all answers as a JSON string in Answer.answer_text
        answer_dict = {}
        for question in questions:
            ans = request.form.get(f'answers[{question.id}]')
            answer_dict[str(question.id)] = ans
        answer = Answer(
            exam_id=exam_id,
            user_id=current_user.id,
            answer_text=json.dumps(answer_dict),
            submitted_at=now
        )
        answer.similarity_score = 0.0
        db.session.add(answer)
        db.session.commit()
        flash('Exam submitted successfully!', 'success')
        return redirect(url_for('my_results'))
    return render_template('take_exam.html', exam=exam, questions=questions)

@app.route('/exam_submissions/<int:exam_id>', methods=['GET', 'POST'])
@login_required
def exam_submissions(exam_id):
    if current_user.role != 'teacher':
        flash('Access denied. This page is for teachers only.', 'error')
        return redirect(url_for('dashboard'))
    from utils.plagiarism_detector import AdvancedPlagiarismChecker
    exam = Exam.query.get_or_404(exam_id)
    # Handle marks and comment assignment
    if request.method == 'POST':
        answer_id = int(request.form.get('answer_id'))
        marks = request.form.get('marks')
        comment = request.form.get('comment')
        answer = Answer.query.get(answer_id)
        if answer:
            if marks is not None and marks != '':
                answer.score = float(marks)
            if comment is not None:
                answer.comment = comment
            db.session.commit()
            flash('Saved successfully!', 'success')
        return redirect(url_for('exam_submissions', exam_id=exam_id))
    # Filtering
    student_query = request.args.get('student', '').strip().lower()
    semester_query = request.args.get('semester', '').strip().lower()
    section_query = request.args.get('section', '').strip().lower()
    department_query = request.args.get('department', '').strip().lower()
    answers = Answer.query.filter_by(exam_id=exam_id).all()
    students = {u.id: u for u in User.query.filter(User.id.in_([a.user_id for a in answers])).all()}
    # Collect unique values for dropdowns
    all_semesters = sorted(set((s.student_class or '').strip() for s in students.values() if s.student_class))
    all_departments = sorted(set((s.department or '').strip() for s in students.values() if s.department))
    all_sections = sorted(set((s.section or '').strip() for s in students.values() if s.section))
    filtered_answers = []
    for a in answers:
        student = students[a.user_id]
        if student_query:
            # Match if any part of the username starts with the query
            name_parts = student.username.lower().split()
            if not any(part.startswith(student_query) for part in name_parts):
                continue
        if semester_query and (not student.student_class or semester_query != student.student_class.lower()):
            continue
        if section_query and (not student.section or section_query != student.section.lower()):
            continue
        if department_query and (not student.department or department_query != student.department.lower()):
            continue
        filtered_answers.append(a)
    # Plagiarism check for all answers
    submissions = [(a.id, a.answer_text) for a in answers]
    checker = AdvancedPlagiarismChecker()
    plagiarism_results = checker.check_all(submissions) if answers else []
    plagiarism_map = {a.id: 0 for a in answers}
    for r in plagiarism_results:
        plagiarism_map[r['student1']] = max(plagiarism_map[r['student1']], r['similarity'])
        plagiarism_map[r['student2']] = max(plagiarism_map[r['student2']], r['similarity'])
    # Update similarity_score in the database for each answer
    for a in answers:
        if a.similarity_score != plagiarism_map[a.id]:
            a.similarity_score = plagiarism_map[a.id]
    db.session.commit()
    num_submissions = len(filtered_answers)
    num_graded = sum(1 for a in filtered_answers if a.score is not None)
    num_flagged = sum(1 for a in filtered_answers if plagiarism_map[a.id] > 70)
    return render_template('exam_submissions.html', exam=exam, answers=filtered_answers, students=students, plagiarism_map=plagiarism_map, num_submissions=num_submissions, num_graded=num_graded, num_flagged=num_flagged, student_query=student_query, semester_query=semester_query, section_query=section_query, department_query=department_query, all_semesters=all_semesters, all_departments=all_departments, all_sections=all_sections)

def check_code_similarity(code1, code2):
    """Check similarity between two code snippets."""
    # Remove comments and normalize whitespace
    def normalize_code(code):
        # Remove comments
        code = re.sub(r'//.*?$|/\*.*?\*/', '', code, flags=re.MULTILINE)
        # Normalize whitespace
        code = re.sub(r'\s+', ' ', code)
        return code.strip()
    
    code1 = normalize_code(code1)
    code2 = normalize_code(code2)
    
    # Use difflib to calculate similarity
    matcher = difflib.SequenceMatcher(None, code1, code2)
    return matcher.ratio() * 100

@app.route('/api/check_plagiarism/<int:exam_id>/<int:answer_id>')
@login_required
def api_check_plagiarism(exam_id, answer_id):
    if current_user.role not in ['teacher', 'admin']:
        return jsonify({'error': 'Unauthorized'}), 403
    
    exam = Exam.query.get_or_404(exam_id)
    current_answer = Answer.query.get_or_404(answer_id)
    results = []
    if exam.exam_format == 'coding':
        # For coding exams, compare with all other submissions
        other_answers = Answer.query.filter(
            Answer.exam_id == exam_id,
            Answer.id != answer_id
        ).all()
        for other in other_answers:
            # Multi-algorithm comparison
            details = {}
            from utils.plagiarism_detector import PlagiarismDetector
            detector = PlagiarismDetector()
            rabin_karp_score = detector.rabin_karp(current_answer.answer_text, other.answer_text)
            levenshtein_score = detector.levenshtein_distance(current_answer.answer_text, other.answer_text)
            kmp_score = detector.kmp_search(current_answer.answer_text, other.answer_text)
            # AST-based comparison for Python
            ast_score = None
            if exam.programming_language and exam.programming_language.lower() == 'python':
                try:
                    ast_score = compare_python_ast(current_answer.answer_text, other.answer_text)
                except Exception:
                    ast_score = None
            # ML-based similarity (TF-IDF + cosine)
            ml_score = ml_similarity(current_answer.answer_text, other.answer_text)
            details = {
                'ml': round(ml_score, 2),
                'rabin_karp': round(rabin_karp_score, 2),
                'levenshtein': round(levenshtein_score, 2),
                'kmp': round(kmp_score, 2),
                'ast': round(ast_score, 2) if ast_score is not None else None
            }
            # Aggregate score (weighted)
            scores = [s for s in [ml_score, rabin_karp_score, levenshtein_score, kmp_score, ast_score] if s is not None]
            similarity = sum(scores) / len(scores) if scores else 0
            if similarity > 30:
                results.append({
                    'student1': current_answer.id,
                    'student2': other.id,
                    'student1_name': User.query.get(current_answer.user_id).username,
                    'student2_name': User.query.get(other.user_id).username,
                    'similarity': round(similarity, 1),
                    'details': details
                })
    else:
        # For regular exams, use AdvancedPlagiarismChecker
        from utils.plagiarism_detector import AdvancedPlagiarismChecker
        answers = Answer.query.filter(Answer.exam_id == exam_id).all()
        students = {a.id: User.query.get(a.user_id).username for a in answers}
        submissions = [(a.id, a.answer_text) for a in answers]
        checker = AdvancedPlagiarismChecker()
        plagiarism_results = checker.check_all(submissions)
        for r in plagiarism_results:
            if (r['student1'] == answer_id or r['student2'] == answer_id) and r['similarity'] > 30:
                ml_score = ml_similarity(
                    next((s[1] for s in submissions if s[0] == r['student1']), ''),
                    next((s[1] for s in submissions if s[0] == r['student2']), '')
                )
                details = {
                    'ml': round(ml_score, 2),
                    'tfidf': round(r['similarity'], 2)
                }
                results.append({
                    'student1': r['student1'],
                    'student2': r['student2'],
                    'student1_name': students.get(r['student1'], 'Unknown'),
                    'student2_name': students.get(r['student2'], 'Unknown'),
                    'similarity': r['similarity'],
                    'details': details
                })
    return jsonify({'results': sorted(results, key=lambda x: x['similarity'], reverse=True)})

def compare_python_ast(code1, code2):
    try:
        tree1 = ast.dump(ast.parse(code1))
        tree2 = ast.dump(ast.parse(code2))
        matcher = difflib.SequenceMatcher(None, tree1, tree2)
        return matcher.ratio() * 100
    except Exception:
        return 0

def ml_similarity(text1, text2):
    try:
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([text1, text2])
        score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return score * 100
    except Exception:
        return 0

@app.route('/api/run_tests', methods=['POST'])
@login_required
def run_tests():
    if not request.is_json:
        return jsonify({'error': 'Invalid request format'}), 400
    
    data = request.get_json()
    question_id = data.get('question_id')
    code = data.get('code')
    test_input = data.get('test_input')
    expected_output = data.get('expected_output')
    language = data.get('language')
    
    if not all([question_id, code, test_input, expected_output, language]):
        return jsonify({'error': 'Missing required parameters'}), 400
    
    # Get question details
    question = Question.query.get_or_404(question_id)
    exam = Exam.query.get_or_404(question.exam_id)
    
    # Verify user has access to this exam
    if current_user.role == 'student':
        if not (exam.student_class == current_user.student_class and 
                exam.section == current_user.section and 
                exam.department == current_user.department):
            return jsonify({'error': 'Unauthorized'}), 403
    
    # Create temporary file for code
    with tempfile.NamedTemporaryFile(mode='w', suffix=get_file_extension(language), delete=False) as f:
        f.write(code)
        temp_file = f.name
    
    try:
        # Run the code with test cases
        test_cases = test_input.split('\n')
        expected_outputs = expected_output.split('\n')
        results = []
        
        for test_input, expected in zip(test_cases, expected_outputs):
            if not test_input.strip():
                continue
                
            try:
                # Execute the code with the test input
                process = subprocess.Popen(
                    get_run_command(language, temp_file),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                stdout, stderr = process.communicate(input=test_input)
                
                if stderr:
                    results.append({
                        'passed': False,
                        'expected': expected.strip(),
                        'actual': f'Error: {stderr.strip()}'
                    })
                else:
                    actual = stdout.strip()
                    results.append({
                        'passed': actual == expected.strip(),
                        'expected': expected.strip(),
                        'actual': actual
                    })
            except Exception as e:
                results.append({
                    'passed': False,
                    'expected': expected.strip(),
                    'actual': f'Error: {str(e)}'
                })
        
        return jsonify({
            'status': 'success',
            'test_cases': results
        })
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_file)
        except:
            pass

def get_file_extension(language):
    """Get the appropriate file extension for the programming language."""
    extensions = {
        'python': '.py',
        'java': '.java',
        'cpp': '.cpp',
        'javascript': '.js'
    }
    return extensions.get(language.lower(), '.txt')

def get_run_command(language, file_path):
    """Get the appropriate command to run the code based on the language."""
    commands = {
        'python': ['python', file_path],
        'java': ['java', file_path],
        'cpp': [file_path],  # Assuming it's already compiled
        'javascript': ['node', file_path]
    }
    return commands.get(language.lower(), ['echo', 'Unsupported language'])

@app.route('/view_submission/<int:submission_id>')
@login_required
def view_submission(submission_id):
    answer = Answer.query.get_or_404(submission_id)
    exam = Exam.query.get(answer.exam_id)
    student = User.query.get(answer.user_id)
    return render_template('view_submission.html', answer=answer, exam=exam, student=student)

@app.route('/view_result/<int:exam_id>')
@login_required
def view_result(exam_id):
    if current_user.role != 'student':
        flash('Access denied. This page is for students only.', 'error')
        return redirect(url_for('dashboard'))
    exam = Exam.query.get_or_404(exam_id)
    answer = Answer.query.filter_by(exam_id=exam_id, user_id=current_user.id).first()
    if not answer:
        flash('No result found for this exam.', 'error')
        return redirect(url_for('my_results'))
    # Get ranking information
    rank_info = get_rank_info(exam_id, answer.score)
    return render_template('view_result.html', exam=exam, answer=answer, rank=rank_info['rank'], total_students=rank_info['total_students'], percentile=rank_info['percentile'])

@app.route('/delete_exam/<int:exam_id>', methods=['POST'])
@login_required
def delete_exam(exam_id):
    if current_user.role != 'teacher':
        flash('Access denied. Only teachers can delete exams.', 'error')
        return redirect(url_for('dashboard'))
    exam = Exam.query.get_or_404(exam_id)
    if exam.created_by != current_user.id:
        flash('You can only delete exams you created.', 'error')
        return redirect(url_for('dashboard'))
    # Delete all answers and questions for this exam
    Answer.query.filter_by(exam_id=exam_id).delete()
    Question.query.filter_by(exam_id=exam_id).delete()
    db.session.delete(exam)
    db.session.commit()
    flash('Exam and all related data deleted successfully.', 'success')
    return redirect(url_for('dashboard'))

ASSIGNMENT_UPLOAD_FOLDER = os.path.join(os.getcwd(), 'assignment_files')
os.makedirs(ASSIGNMENT_UPLOAD_FOLDER, exist_ok=True)

@app.route('/create_assignment', methods=['GET', 'POST'])
@login_required
def create_assignment():
    if current_user.role != 'teacher':
        flash('Only teachers can create assignments.', 'error')
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        title = request.form.get('title')
        description = request.form.get('description')
        due_date = request.form.get('due_date')
        allowed_file_types = request.form.get('allowed_file_types')
        file = request.files.get('assignment_file')
        assignment_filename = None
        assignment_file_path = None
        if file and file.filename:
            assignment_filename = secure_filename(file.filename)
            assignment_file_path = os.path.join(ASSIGNMENT_UPLOAD_FOLDER, assignment_filename)
            file.save(assignment_file_path)
        due_date_dt = datetime.strptime(due_date, '%Y-%m-%dT%H:%M')
        assignment = Assignment(
            title=title,
            description=description,
            due_date=due_date_dt,
            created_by=current_user.id,
            allowed_file_types=allowed_file_types,
            assignment_filename=assignment_filename,
            assignment_file_path=assignment_file_path
        )
        db.session.add(assignment)
        db.session.commit()
        flash('Assignment created successfully!', 'success')
        return redirect(url_for('teacher_assignments'))
    return render_template('create_assignment.html')

@app.route('/teacher_assignments')
@login_required
def teacher_assignments():
    if current_user.role != 'teacher':
        flash('Access denied.', 'error')
        return redirect(url_for('dashboard'))
    assignments = Assignment.query.filter_by(created_by=current_user.id).all()
    return render_template('teacher_assignments.html', assignments=assignments)

@app.route('/student_assignments')
@login_required
def student_assignments():
    if current_user.role != 'student':
        flash('Access denied.', 'error')
        return redirect(url_for('dashboard'))
    assignments = Assignment.query.all()
    submissions = {s.assignment_id: s for s in AssignmentSubmission.query.filter_by(user_id=current_user.id).all()}
    return render_template('student_assignments.html', assignments=assignments, submissions=submissions)

ALLOWED_EXTENSIONS = set(['pdf', 'docx', 'txt', 'py', 'java', 'cpp', 'c', 'js'])
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'assignment_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename, allowed_types):
    ext = filename.rsplit('.', 1)[-1].lower()
    allowed = [t.strip().lower() for t in allowed_types.split(',')] if allowed_types else list(ALLOWED_EXTENSIONS)
    return '.' in filename and ext in allowed

@app.route('/api/check_assignment_plagiarism/<int:assignment_id>/<int:submission_id>')
@login_required
def api_check_assignment_plagiarism(assignment_id, submission_id):
    if current_user.role not in ['teacher', 'admin']:
        return jsonify({'error': 'Unauthorized'}), 403
    assignment = Assignment.query.get_or_404(assignment_id)
    current_submission = AssignmentSubmission.query.get_or_404(submission_id)
    all_submissions = AssignmentSubmission.query.filter_by(assignment_id=assignment_id).all()
    results = []
    # Read all files as text (if possible)
    def read_file_text(path):
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception:
            return ''
    current_text = read_file_text(current_submission.file_path)
    for other in all_submissions:
        if other.id == submission_id:
            continue
        other_text = read_file_text(other.file_path)
        # Use all algorithms/models
        from utils.plagiarism_detector import PlagiarismDetector
        detector = PlagiarismDetector()
        rabin_karp_score = detector.rabin_karp(current_text, other_text)
        levenshtein_score = detector.levenshtein_distance(current_text, other_text)
        kmp_score = detector.kmp_search(current_text, other_text)
        ast_score = None
        if current_submission.filename.endswith('.py') and other.filename.endswith('.py'):
            try:
                ast_score = compare_python_ast(current_text, other_text)
            except Exception:
                ast_score = None
        ml_score = ml_similarity(current_text, other_text)
        details = {
            'ml': round(ml_score, 2),
            'rabin_karp': round(rabin_karp_score, 2),
            'levenshtein': round(levenshtein_score, 2),
            'kmp': round(kmp_score, 2),
            'ast': round(ast_score, 2) if ast_score is not None else None
        }
        scores = [s for s in [ml_score, rabin_karp_score, levenshtein_score, kmp_score, ast_score] if s is not None]
        similarity = sum(scores) / len(scores) if scores else 0
        if similarity > 30:
            results.append({
                'submission1': current_submission.id,
                'submission2': other.id,
                'student1_name': User.query.get(current_submission.user_id).username,
                'student2_name': User.query.get(other.user_id).username,
                'similarity': round(similarity, 1),
                'details': details
            })
    return jsonify({'results': sorted(results, key=lambda x: x['similarity'], reverse=True)})

@app.route('/submit_assignment/<int:assignment_id>', methods=['GET', 'POST'])
@login_required
def submit_assignment(assignment_id):
    if current_user.role != 'student':
        flash('Access denied.', 'error')
        return redirect(url_for('dashboard'))
    assignment = Assignment.query.get_or_404(assignment_id)
    if request.method == 'POST':
        file = request.files['file']
        if not file:
            flash('No file uploaded.', 'error')
            return redirect(request.url)
        filename = secure_filename(file.filename)
        if not allowed_file(filename, assignment.allowed_file_types):
            flash('File type not allowed.', 'error')
            return redirect(request.url)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        submission = AssignmentSubmission(
            assignment_id=assignment_id,
            user_id=current_user.id,
            filename=filename,
            file_path=file_path
        )
        db.session.add(submission)
        db.session.commit()
        flash('Assignment submitted successfully!', 'success')
        return redirect(url_for('student_assignments'))
    return render_template('submit_assignment.html', assignment=assignment)

@app.route('/review_assignment/<int:assignment_id>', methods=['GET', 'POST'])
@login_required
def review_assignment(assignment_id):
    if current_user.role != 'teacher':
        flash('Access denied.', 'error')
        return redirect(url_for('dashboard'))
    assignment = Assignment.query.get_or_404(assignment_id)
    submissions = AssignmentSubmission.query.filter_by(assignment_id=assignment_id).all()
    students = {s.user_id: User.query.get(s.user_id) for s in submissions}

    if request.method == 'POST':
        submission_id = request.form.get('submission_id')
        marks = request.form.get('marks')
        comment = request.form.get('comment')
        submission = AssignmentSubmission.query.get(int(submission_id))
        if submission:
            if marks is not None and marks != '':
                submission.marks = float(marks)
            if comment is not None:
                submission.comment = comment
            db.session.commit()
            flash('Saved successfully!', 'success')
        return redirect(url_for('review_assignment', assignment_id=assignment_id))

    return render_template('review_assignment.html', assignment=assignment, submissions=submissions, students=students)

@app.route('/view_answer/<int:answer_id>')
@login_required
def view_answer(answer_id):
    if current_user.role != 'teacher':
        flash('Access denied.', 'error')
        return redirect(url_for('dashboard'))
    answer = Answer.query.get_or_404(answer_id)
    student = User.query.get(answer.user_id)
    exam = Exam.query.get(answer.exam_id)
    return render_template('view_answer.html', answer=answer, student=student, exam=exam)

# For assignment files (teacher uploads)
@app.route('/assignment_files/<filename>')
@login_required
def download_assignment_file(filename):
    return send_from_directory(ASSIGNMENT_UPLOAD_FOLDER, filename, as_attachment=True)

# For student submissions (student uploads)
@app.route('/assignment_uploads/<filename>')
@login_required
def download_submission_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True) 