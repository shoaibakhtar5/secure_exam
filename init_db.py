# This file should be moved to the project root (C:\Users\DEll\Desktop\secure_exam) for correct imports.
from app import app, db

with app.app_context():
    db.create_all()
    print('Tables created successfully!')