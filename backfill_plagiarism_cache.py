from app import app, db, Exam, Answer, PlagiarismResult, Assignment, AssignmentSubmission, AssignmentPlagiarismResult
from utils.plagiarism_detector import AdvancedPlagiarismChecker, PlagiarismDetector
import json
from app import compare_python_ast, ml_similarity, read_file_text_and_images

def backfill_exams():
    with app.app_context():
        print('Backfilling exams...')
        for exam in Exam.query.all():
            answers = Answer.query.filter_by(exam_id=exam.id).all()
            submissions = [(a.id, a.answer_text) for a in answers]
            if not answers:
                continue
            if exam.exam_format == 'coding':
                for i, a1 in enumerate(answers):
                    for j, a2 in enumerate(answers):
                        if i >= j:
                            continue
                        detector = PlagiarismDetector()
                        rabin_karp_score = detector.rabin_karp(a1.answer_text, a2.answer_text)
                        levenshtein_score = detector.levenshtein_distance(a1.answer_text, a2.answer_text)
                        kmp_score = detector.kmp_search(a1.answer_text, a2.answer_text)
                        ast_score = None
                        if exam.programming_language and exam.programming_language.lower() == 'python':
                            try:
                                ast_score = compare_python_ast(a1.answer_text, a2.answer_text)
                            except Exception:
                                ast_score = None
                        ml_score = ml_similarity(a1.answer_text, a2.answer_text)
                        details = json.dumps({
                            'ml': round(ml_score, 2),
                            'rabin_karp': round(rabin_karp_score, 2),
                            'levenshtein': round(levenshtein_score, 2),
                            'kmp': round(kmp_score, 2),
                            'ast': round(ast_score, 2) if ast_score is not None else None
                        })
                        similarity = sum([s for s in [ml_score, rabin_karp_score, levenshtein_score, kmp_score, ast_score] if s is not None]) / len([s for s in [ml_score, rabin_karp_score, levenshtein_score, kmp_score, ast_score] if s is not None])
                        for pair in [(a1.id, a2.id), (a2.id, a1.id)]:
                            pr = PlagiarismResult.query.filter_by(exam_id=exam.id, answer1_id=pair[0], answer2_id=pair[1]).first()
                            if not pr:
                                pr = PlagiarismResult(exam_id=exam.id, answer1_id=pair[0], answer2_id=pair[1], similarity=similarity, details=details)
                                db.session.add(pr)
                            else:
                                pr.similarity = similarity
                                pr.details = details
                db.session.commit()
            else:
                checker = AdvancedPlagiarismChecker()
                plagiarism_results = checker.check_all(submissions)
                for r in plagiarism_results:
                    details = json.dumps({'tfidf': round(r['similarity'], 2)})
                    for pair in [(r['student1'], r['student2']), (r['student2'], r['student1'])]:
                        pr = PlagiarismResult.query.filter_by(exam_id=exam.id, answer1_id=pair[0], answer2_id=pair[1]).first()
                        if not pr:
                            pr = PlagiarismResult(exam_id=exam.id, answer1_id=pair[0], answer2_id=pair[1], similarity=r['similarity'], details=details)
                            db.session.add(pr)
                        else:
                            pr.similarity = r['similarity']
                            pr.details = details
                db.session.commit()
        print('Exam plagiarism cache backfilled.')

def backfill_assignments():
    with app.app_context():
        print('Backfilling assignments...')
        for assignment in Assignment.query.all():
            submissions = AssignmentSubmission.query.filter_by(assignment_id=assignment.id).all()
            if not submissions:
                continue
            for i, s1 in enumerate(submissions):
                text1 = read_file_text_and_images(s1.file_path, s1.filename)
                for j, s2 in enumerate(submissions):
                    if i >= j:
                        continue
                    text2 = read_file_text_and_images(s2.file_path, s2.filename)
                    detector = PlagiarismDetector()
                    rabin_karp_score = detector.rabin_karp(text1, text2)
                    levenshtein_score = detector.levenshtein_distance(text1, text2)
                    kmp_score = detector.kmp_search(text1, text2)
                    ast_score = None
                    if s1.filename.endswith('.py') and s2.filename.endswith('.py'):
                        try:
                            ast_score = compare_python_ast(text1, text2)
                        except Exception:
                            ast_score = None
                    ml_score = ml_similarity(text1, text2)
                    details = json.dumps({
                        'ml': {'score': round(ml_score, 2)},
                        'rabin_karp': {'score': round(rabin_karp_score, 2)},
                        'levenshtein': {'score': round(levenshtein_score, 2)},
                        'kmp': {'score': round(kmp_score, 2)},
                        'ast': {'score': round(ast_score, 2) if ast_score is not None else None}
                    })
                    similarity = sum([s for s in [ml_score, rabin_karp_score, levenshtein_score, kmp_score, ast_score] if s is not None]) / len([s for s in [ml_score, rabin_karp_score, levenshtein_score, kmp_score, ast_score] if s is not None])
                    for pair in [(s1.id, s2.id), (s2.id, s1.id)]:
                        pr = AssignmentPlagiarismResult.query.filter_by(assignment_id=assignment.id, submission1_id=pair[0], submission2_id=pair[1]).first()
                        if not pr:
                            pr = AssignmentPlagiarismResult(assignment_id=assignment.id, submission1_id=pair[0], submission2_id=pair[1], similarity=similarity, details=details)
                            db.session.add(pr)
                        else:
                            pr.similarity = similarity
                            pr.details = details
            db.session.commit()
        print('Assignment plagiarism cache backfilled.')

if __name__ == '__main__':
    backfill_exams()
    backfill_assignments()
    print('All done!') 