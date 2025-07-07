import cv2
import pickle
import mysql.connector
import sys
import os

# Set UTF-8 encoding for Windows
if sys.platform.startswith('win'):
    import codecs

    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# Database configuration - UPDATE THESE VALUES
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'your_sql_password',  # Change this to your actual MySQL password
    'database': 'face_attendance',
    'charset': 'utf8mb4',
    'collation': 'utf8mb4_unicode_ci'
}


def get_cascade_path():
    """Get the correct path for haarcascade file"""
    # Try different possible locations
    possible_paths = [
        'haarcascade_frontalface_default.xml',
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
        'venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml',
        os.path.join(cv2.__path__[0], 'data', 'haarcascade_frontalface_default.xml')
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found cascade file at: {path}")
            return path

    print("❌ Could not find haarcascade_frontalface_default.xml")
    print("💡 Possible solutions:")
    print(
        "   1. Download it from: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml")
    print("   2. Save it in your project directory")
    return None


def test_database_connection():
    """Test database connection"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        conn.close()
        print("✅ Database connection successful")
        return True
    except mysql.connector.Error as err:
        print(f"❌ Database connection failed: {err}")
        print("💡 Please check:")
        print("   - MySQL server is running")
        print("   - Username and password are correct")
        print("   - Database 'face_attendance' exists")
        return False


def init_database():
    """Initialize database and tables"""
    try:
        # Connect without specifying database first
        temp_config = DB_CONFIG.copy()
        temp_config.pop('database', None)

        conn = mysql.connector.connect(**temp_config)
        cursor = conn.cursor()

        # Create database if not exists
        cursor.execute("CREATE DATABASE IF NOT EXISTS face_attendance CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
        cursor.execute("USE face_attendance")

        # Create students table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                roll_number VARCHAR(50) NOT NULL UNIQUE,
                face_image LONGBLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """)

        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Database initialized successfully")
        return True
    except mysql.connector.Error as err:
        print(f"❌ Database initialization failed: {err}")
        return False


def register_student(name, roll_number, face_cascade):
    """Register a student with face detection"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        print("💡 Please check:")
        print("   - Camera is connected and working")
        print("   - No other applications are using the camera")
        return False

    print(f"📸 {name}, please look at the camera...")
    print("   - Press SPACE to capture when face is detected")
    print("   - Press 'q' to cancel")

    face_captured = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame")
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Face Detected - Press SPACE', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show status
        if len(faces) == 0:
            cv2.putText(frame, 'No face detected', (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif len(faces) > 1:
            cv2.putText(frame, 'Multiple faces - Show only one', (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Register Face - Press SPACE to capture, Q to quit', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("❌ Registration cancelled by user")
            break
        elif key == ord(' ') and len(faces) == 1:  # Space key and exactly one face
            try:
                # Capture the face
                x, y, w, h = faces[0]
                face_crop = gray[y:y + h, x:x + w]
                face_blob = pickle.dumps(face_crop)

                # Save to database
                conn = mysql.connector.connect(**DB_CONFIG)
                cursor = conn.cursor()

                # Check if roll number already exists
                cursor.execute("SELECT id FROM students WHERE roll_number = %s", (roll_number,))
                if cursor.fetchone():
                    print(f"❌ Roll number {roll_number} already exists")
                    cursor.close()
                    conn.close()
                    break

                # Insert new student
                cursor.execute(
                    "INSERT INTO students (name, roll_number, face_image) VALUES (%s, %s, %s)",
                    (name, roll_number, face_blob)
                )
                conn.commit()
                cursor.close()
                conn.close()

                print(f"✅ {name} (Roll: {roll_number}) registered successfully!")
                face_captured = True
                break

            except mysql.connector.Error as err:
                print(f"❌ Database error: {err}")
                break
            except Exception as e:
                print(f"❌ Error during registration: {e}")
                break

    cap.release()
    cv2.destroyAllWindows()
    return face_captured


def start_registration():
    """Main registration function"""
    print("🎓 IST Face Registration System")
    print("=" * 40)

    # Test database connection
    if not test_database_connection():
        return

    # Initialize database
    if not init_database():
        return

    # Check for cascade file
    cascade_path = get_cascade_path()
    if not cascade_path:
        return

    try:
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            print("❌ Failed to load face cascade classifier")
            return
    except Exception as e:
        print(f"❌ Error loading cascade: {e}")
        return

    print("\n📋 Face Registration Started")
    print("=" * 30)

    while True:
        print("\n" + "=" * 50)
        name = input("Enter student name (or type 'exit' to quit): ").strip()

        if name.lower() in ['exit', 'quit', '']:
            print("👋 Exiting registration system")
            break

        if len(name) < 2:
            print("❌ Please enter a valid name (at least 2 characters)")
            continue

        roll_number = input("Enter roll number: ").strip()

        if len(roll_number) < 1:
            print("❌ Please enter a valid roll number")
            continue

        print(f"\n📸 Starting camera for {name} (Roll: {roll_number})")
        success = register_student(name, roll_number, face_cascade)

        if success:
            print("✅ Registration completed successfully!")
        else:
            print("❌ Registration failed. Please try again.")

        # Ask if user wants to continue
        continue_reg = input("\nRegister another student? (y/n): ").strip().lower()
        if continue_reg not in ['y', 'yes']:
            break

    print("\n🎉 Registration session completed!")
    print("You can now use the web interface for attendance marking.")


if __name__ == "__main__":
    try:
        start_registration()
    except KeyboardInterrupt:
        print("\n\n👋 Registration interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your setup and try again.")