# Secure Exam System - Project Documentation

## Overview
The Secure Exam System is a comprehensive web-based platform designed to facilitate secure online examinations and assignments with advanced plagiarism detection capabilities. This document outlines the key Data Structures and Algorithms (DSA) concepts implemented in the system.

## Core DSA Concepts Implemented

### 1. Data Structures

#### 1.1 Relational Database Models
- **User Management**
  - Hash Tables for user authentication
  - B-tree indexing for efficient user lookups
  - Implementation: `User` model with username, email, and password_hash

#### 1.2 Graph-based Relationships
- **Exam-Question Relationship**
  - Directed graph structure
  - One-to-many relationship between exams and questions
  - Implementation: `Exam` and `Question` models with foreign key relationships

#### 1.3 Tree Structures
- **Code Analysis**
  - Abstract Syntax Tree (AST) for code similarity detection
  - Implementation: `compare_python_ast()` function for code comparison

### 2. Algorithms

#### 2.1 Text Processing and Analysis
- **Text Normalization**
  - String manipulation algorithms
  - Regular expression processing
  - Implementation: `normalize_text()` and `normalize_code()` functions

#### 2.2 Plagiarism Detection
- **Multiple Algorithm Approach**
  1. Cosine Similarity
     - TF-IDF Vectorization
     - Implementation: `ml_similarity()` using sklearn's TfidfVectorizer
  2. Code Similarity
     - AST-based comparison
     - Implementation: `check_code_similarity()`
  3. Text Similarity
     - Sequence matching
     - Implementation: Using difflib for text comparison

#### 2.3 File Processing
- **Document Parsing**
  - PDF text extraction
  - DOCX parsing
  - Image processing (OCR)
  - Implementation: `extract_text_and_images_from_pdf()` and `extract_text_and_images_from_docx()`

### 3. Search and Sort Algorithms

#### 3.1 Database Queries
- **Efficient Retrieval**
  - Indexed searches
  - Sorting algorithms for result ranking
  - Implementation: SQLAlchemy queries with proper indexing

#### 3.2 Result Processing
- **Ranking System**
  - Score-based sorting
  - Implementation: `get_rank_info()` function

### 4. Security Algorithms

#### 4.1 Authentication
- **Password Hashing**
  - Secure hash algorithms
  - Implementation: Werkzeug's password hashing

#### 4.2 Session Management
- **Secure Session Handling**
  - Token-based authentication
  - Implementation: Flask-Login integration

## Technical Implementation Details

### 1. Database Schema
- Relational database using MySQL
- Optimized table structures with proper indexing
- Foreign key constraints for data integrity

### 2. API Endpoints
- RESTful API design
- Secure route handling
- Role-based access control

### 3. File Processing Pipeline
1. File upload validation
2. Content extraction
3. Similarity analysis
4. Result storage

### 4. Plagiarism Detection Pipeline
1. Text normalization
2. Feature extraction
3. Similarity computation
4. Result aggregation

## Performance Considerations

### 1. Time Complexity
- Database queries: O(log n) with proper indexing
- Text similarity: O(n) for basic comparisons
- Code similarity: O(n log n) for AST comparison

### 2. Space Complexity
- In-memory processing for small files
- Disk-based storage for large files
- Efficient caching of similarity results

## Future Improvements

1. **Algorithm Optimization**
   - Implement more efficient similarity algorithms
   - Add support for more programming languages
   - Enhance code analysis capabilities

2. **Performance Enhancements**
   - Implement caching mechanisms
   - Optimize database queries
   - Add batch processing capabilities

3. **Feature Additions**
   - Real-time plagiarism detection
   - Advanced code analysis
   - Machine learning-based similarity detection

## Conclusion
The Secure Exam System implements various DSA concepts to provide efficient and secure examination capabilities. The system's architecture allows for easy extension and modification of existing algorithms while maintaining performance and security standards. 