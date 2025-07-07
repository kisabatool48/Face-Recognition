import cv2
import pickle
import mysql.connector
import numpy as np

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='your_sql_password',  # Change for your setup
        database='face_attendance'
    )

def add_student():
    name = input("Enter name: ")
    roll = input("Enter roll number: ")
    dept = input("Enter department: ")

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO students (name, roll_number, department) VALUES (%s, %s, %s)", (name, roll, dept))
    conn.commit()
    cursor.close()
    conn.close()
    print("✅ Student added successfully.")

def view_students():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, roll_number, department FROM students")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        print("⚠️ No students found.")
    else:
        print("\n📋 Students List:")
        for row in rows:
            student_id, name, roll, dept = row
            print(f"ID: {student_id} | Name: {name} | Roll: {roll} | Dept: {dept}")
        print("")  # spacing

def update_student():
    student_id = input("Enter the ID of the student to update: ")
    name = input("New name: ")
    roll = input("New roll number: ")
    dept = input("New department: ")

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE students SET name = %s, roll_number = %s, department = %s WHERE id = %s",
                   (name, roll, dept, student_id))
    conn.commit()
    cursor.close()
    conn.close()
    print("✅ Student updated successfully.")

def delete_student():
    student_id = input("Enter the ID of the student to delete: ")

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM students WHERE id = %s", (student_id,))
    conn.commit()
    cursor.close()
    conn.close()
    print("🗑️ Student deleted successfully.")

def update_student_face(student_id, face_array):
    face_blob = pickle.dumps(face_array)  # Serialize numpy array

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE students SET face_image = %s WHERE id = %s", (face_blob, student_id))
    conn.commit()
    cursor.close()
    conn.close()
    print(f"✅ Face image updated for student ID {student_id}")

def capture_face_image(student_id):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    print("🎥 Position student's face and press SPACE to capture. ESC to cancel.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to capture frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Capture Face - Press SPACE", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("❌ Capture cancelled.")
            break
        elif key == 32:  # SPACE
            if len(faces) == 0:
                print("⚠️ No face detected. Try again.")
                continue

            (x, y, w, h) = faces[0]
            face_img = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_img, (200, 200))

            update_student_face(student_id, face_resized)
            print("📸 Face captured and saved!")
            break

    cap.release()
    cv2.destroyAllWindows()

def admin_menu():
    while True:
        print("\n🔧 Admin Panel - Choose an option:")
        print("1. Add Student")
        print("2. View Students")
        print("3. Update Student")
        print("4. Delete Student")
        print("5. Capture Student Face Image")
        print("6. Exit")

        choice = input("Enter choice: ")

        if choice == '1':
            add_student()
        elif choice == '2':
            view_students()
        elif choice == '3':
            update_student()
        elif choice == '4':
            delete_student()
        elif choice == '5':
            student_id = input("Enter student ID to capture face for: ")
            if student_id.isdigit():
                capture_face_image(int(student_id))
            else:
                print("❌ Invalid student ID. Must be a number.")
        elif choice == '6':
            print("👋 Exiting Admin Panel.")
            break
        else:
            print("❌ Invalid choice. Try again.")

if __name__ == "__main__":
    admin_menu()
