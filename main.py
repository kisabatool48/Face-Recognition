# main.py - Flask Web Application for Face Attendance System
from flask import Flask, render_template, request, jsonify, Response
import cv2
import pickle
import mysql.connector
import numpy as np
from datetime import datetime
import bcrypt
import base64
import os
import urllib.request

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Global variables
camera = None
recognizer = None
face_cascade = None

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'your_sql_password',
    'database': 'face_attendance'
}


def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)


def download_haar_cascade():
    """Download Haar cascade file if not present"""
    cascade_path = 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        print("Downloading Haar cascade file...")
        url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'
        urllib.request.urlretrieve(url, cascade_path)
        print("Haar cascade file downloaded successfully!")
    return cascade_path


def initialize_face_detection():
    """Initialize face detection"""
    global face_cascade
    try:
        cascade_path = download_haar_cascade()
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            raise Exception("Could not load cascade classifier")
        print("Face detection initialized successfully!")
        return True
    except Exception as e:
        print(f"Error initializing face detection: {e}")
        return False


def load_faces_from_db():
    """Load registered faces from database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, name, roll_number, department, face_image FROM students WHERE face_image IS NOT NULL")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        faces = []
        labels = []
        label_map = {}

        for row in rows:
            student_id, name, roll, dept, face_blob = row
            if face_blob:
                try:
                    face = pickle.loads(face_blob)
                    faces.append(face)
                    labels.append(student_id)
                    label_map[student_id] = {'name': name, 'roll': roll, 'dept': dept}
                except:
                    continue

        return faces, labels, label_map
    except Exception as e:
        print(f"Error loading faces: {e}")
        return [], [], {}


def train_recognizer():
    global recognizer
    try:
        faces, labels, label_map = load_faces_from_db()
        print(f"📦 Loaded {len(faces)} faces from DB")
        if len(faces) > 0:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.train(faces, np.array(labels))
            print(f"✅ Recognizer trained with labels: {labels}")
            return True, label_map
        else:
            print("⚠️ No faces to train recognizer.")
            return False, {}
    except Exception as e:
        print(f"❌ Recognizer training error: {e}")
        return False, {}



# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/register', methods=['POST'])
def register_student():
    try:
        data = request.json
        name = data.get('name')
        roll = data.get('roll')
        dept = data.get('department')

        if not all([name, roll, dept]):
            return jsonify({'success': False, 'message': 'All fields are required'})

        # Check if roll number already exists
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM students WHERE roll_number = %s", (roll,))
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Roll number already exists'})

        # Insert student without face image first
        cursor.execute(
            "INSERT INTO students (name, roll_number, department) VALUES (%s, %s, %s)",
            (name, roll, dept)
        )
        student_id = cursor.lastrowid
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({
            'success': True,
            'message': 'Student registered successfully',
            'student_id': student_id
        })

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/capture_face', methods=['POST'])
def capture_face():
    try:
        data = request.json
        student_id = data.get('student_id')
        image_data = data.get('image_data')

        if not student_id or not image_data:
            return jsonify({'success': False, 'message': 'Missing data'})

        # Decode base64 image
        image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64,
        image_bytes = base64.b64decode(image_data)

        # Convert to OpenCV image
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)


        if len(faces) == 0:
            return jsonify({'success': False, 'message': 'No face detected'})

        # Take the largest face
        (x, y, w, h) = max(faces, key=lambda face: face[2] * face[3])
        face_crop = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face_crop, (200, 200))
        face_resized = np.array(face_resized, dtype=np.uint8)  # ✅ Yeh line add karo


        # Save face to database
        face_blob = pickle.dumps(face_resized)
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE students SET face_image = %s WHERE id = %s", (face_blob, student_id))
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({'success': True, 'message': 'Face captured successfully'})

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/mark_attendance', methods=['POST'])
def mark_attendance():
    try:
        data = request.json
        roll_number = data.get('roll_number')
        image_data = data.get('image_data')

        if not roll_number or not image_data:
            return jsonify({'success': False, 'message': 'Missing data'})

        # Train recognizer if needed
        success, label_map = train_recognizer()
        success, label_map = train_recognizer()
        print(f"DEBUG: train_recognizer success = {success}")
        print(f"DEBUG: label_map keys = {list(label_map.keys())}")

        if not success:
            return jsonify({'success': False, 'message': 'No registered faces found'})

        # Decode image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print("DEBUG: Frame decoded successfully")

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        print(f"DEBUG: Detected {len(faces)} faces")

        if len(faces) == 0:
            return jsonify({'success': False, 'message': 'No face detected'})

        # Find student by roll number
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, department FROM students WHERE roll_number = %s", (roll_number,))
        student_data = cursor.fetchone()

        if not student_data:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Student not found'})

        student_id, name, dept = student_data

        # Check if already marked today
        cursor.execute(
            "SELECT id FROM attendance_log WHERE roll_number = %s AND DATE(timestamp) = CURDATE()",
            (roll_number,)
        )
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'message': 'Attendance already marked today'})

        # Recognize face
            for (x, y, w, h) in faces:
                print(f"DEBUG: Starting recognition for {roll_number}")
                face_crop = gray[y:y + h, x:x + w]
                face_resized = cv2.resize(face_crop, (200, 200))
                predicted_id, confidence = recognizer.predict(face_resized)
                print(f"🔍 Predicted ID: {predicted_id} | Actual ID: {student_id} | Confidence: {confidence}")
                

            if predicted_id == student_id and confidence < 70:
                # Mark attendance
                cursor.execute(
                    "INSERT INTO attendance_log (roll_number, name, department) VALUES (%s, %s, %s)",
                    (roll_number, name, dept)
                )
                conn.commit()
                cursor.close()
                conn.close()

                return jsonify({
                    'success': True,
                    'message': f'Attendance marked for {name}',
                    'student_name': name,
                    'confidence': float(confidence)
                })

        cursor.close()
        conn.close()
        return jsonify({'success': False, 'message': 'Face not recognized'})

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/admin_login', methods=['POST'])
def admin_login():
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash FROM admins WHERE username = %s", (username,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        if result and bcrypt.checkpw(password.encode(), result[0].encode()):
            return jsonify({'success': True, 'message': 'Login successful'})
        else:
            return jsonify({'success': False, 'message': 'Invalid credentials'})

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/students')
def get_students():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, roll_number, department FROM students ORDER BY name")
        students = cursor.fetchall()
        cursor.close()
        conn.close()

        students_list = []
        for student in students:
            students_list.append({
                'id': student[0],
                'name': student[1],
                'roll': student[2],
                'department': student[3]
            })

        return jsonify({'success': True, 'students': students_list})

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/attendance_log')
def get_attendance_log():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT roll_number, name, department, timestamp FROM attendance_log "
            "WHERE DATE(timestamp) = CURDATE() ORDER BY timestamp DESC"
        )
        attendance = cursor.fetchall()
        cursor.close()
        conn.close()

        attendance_list = []
        for record in attendance:
            attendance_list.append({
                'roll': record[0],
                'name': record[1],
                'department': record[2],
                'time': record[3].strftime('%H:%M:%S')
            })

        return jsonify({'success': True, 'attendance': attendance_list})

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/stats')
def get_stats():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Total students
        cursor.execute("SELECT COUNT(*) FROM students")
        total_students = cursor.fetchone()[0]

        # Today's attendance
        cursor.execute("SELECT COUNT(*) FROM attendance_log WHERE DATE(timestamp) = CURDATE()")
        today_attendance = cursor.fetchone()[0]

        # Total departments
        cursor.execute("SELECT COUNT(DISTINCT department) FROM students")
        total_departments = cursor.fetchone()[0]

        cursor.close()
        conn.close()

        return jsonify({
            'success': True,
            'stats': {
                'total_students': total_students,
                'today_attendance': today_attendance,
                'total_departments': total_departments
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


if __name__ == '__main__':
    # Initialize face detection
    if initialize_face_detection():
        print("Starting Flask application...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize face detection. Please check OpenCV installation.")