"""
database.py
------------
Module 1: Student Data Management

Provides a lightweight SQLite-backed data access layer for CRUD operations
on student records.

Features:
- Create student
- Read student(s)
- Update student
- Delete student
- Input validation
- Safe parameterized SQL queries
- Bulk loading without destroying the database schema
"""

import sqlite3
import logging
from pathlib import Path
from contextlib import contextmanager


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# DATABASE PATH
# ============================================================

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "students.db"


# ============================================================
# DATABASE SCHEMA
# ============================================================

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    student_id TEXT PRIMARY KEY,
    branch TEXT NOT NULL,
    gender TEXT NOT NULL,

    attendance_pct REAL NOT NULL
        CHECK (attendance_pct BETWEEN 0 AND 100),

    cgpa REAL NOT NULL
        CHECK (cgpa BETWEEN 0 AND 10),

    backlogs INTEGER NOT NULL
        CHECK (backlogs >= 0),

    internships INTEGER NOT NULL
        CHECK (internships >= 0),

    projects_completed INTEGER NOT NULL
        CHECK (projects_completed >= 0),

    certifications INTEGER NOT NULL
        CHECK (certifications >= 0),

    coding_score REAL NOT NULL
        CHECK (coding_score BETWEEN 0 AND 100),

    communication_score REAL NOT NULL
        CHECK (communication_score BETWEEN 0 AND 100),

    extracurricular_score REAL NOT NULL
        CHECK (extracurricular_score BETWEEN 0 AND 100),

    aptitude_score REAL NOT NULL
        CHECK (aptitude_score BETWEEN 0 AND 100),

    performance_score REAL,

    placed INTEGER,

    package_lpa REAL,

    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


# ============================================================
# REQUIRED FIELDS
# ============================================================

REQUIRED_FIELDS = [
    "student_id",
    "branch",
    "gender",
    "attendance_pct",
    "cgpa",
    "backlogs",
    "internships",
    "projects_completed",
    "certifications",
    "coding_score",
    "communication_score",
    "extracurricular_score",
    "aptitude_score",
]


# ============================================================
# ALLOWED UPDATE FIELDS
# ============================================================
# Only these fields are allowed to be modified through
# update_student().
#
# This prevents arbitrary SQL column names from being inserted
# into the UPDATE query.

ALLOWED_UPDATE_FIELDS = {
    "branch",
    "gender",
    "attendance_pct",
    "cgpa",
    "backlogs",
    "internships",
    "projects_completed",
    "certifications",
    "coding_score",
    "communication_score",
    "extracurricular_score",
    "aptitude_score",
}


# ============================================================
# CUSTOM VALIDATION ERROR
# ============================================================

class ValidationError(Exception):
    """Raised when a student record fails input validation."""
    pass


# ============================================================
# DATABASE CONNECTION
# ============================================================

@contextmanager
def get_connection(db_path: Path = DB_PATH):
    """
    Create a safe SQLite connection.

    Automatically:
    - creates the parent directory
    - commits successful transactions
    - rolls back failed transactions
    - closes the connection
    """

    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))

    conn.row_factory = sqlite3.Row

    try:
        yield conn

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def init_db(db_path: Path = DB_PATH):
    """
    Initialize the students table if it does not already exist.
    """

    with get_connection(db_path) as conn:
        conn.execute(SCHEMA)

    logger.info("Database initialised at %s", db_path)


# ============================================================
# VALIDATE STUDENT RECORD
# ============================================================

def _validate(record: dict):
    """
    Validate required fields and important value ranges.
    """

    # --------------------------------------------------------
    # Required field validation
    # --------------------------------------------------------

    missing = [
        field
        for field in REQUIRED_FIELDS
        if field not in record
    ]

    if missing:
        raise ValidationError(
            f"Missing required fields: {missing}"
        )

    # --------------------------------------------------------
    # CGPA validation
    # --------------------------------------------------------

    if not (0 <= float(record["cgpa"]) <= 10):
        raise ValidationError(
            "cgpa must be between 0 and 10"
        )

    # --------------------------------------------------------
    # Attendance validation
    # --------------------------------------------------------

    if not (0 <= float(record["attendance_pct"]) <= 100):
        raise ValidationError(
            "attendance_pct must be between 0 and 100"
        )

    # --------------------------------------------------------
    # Backlog validation
    # --------------------------------------------------------

    if int(record["backlogs"]) < 0:
        raise ValidationError(
            "backlogs cannot be negative"
        )

    # --------------------------------------------------------
    # Non-negative count fields
    # --------------------------------------------------------

    count_fields = [
        "internships",
        "projects_completed",
        "certifications",
    ]

    for field in count_fields:
        if int(record[field]) < 0:
            raise ValidationError(
                f"{field} cannot be negative"
            )

    # --------------------------------------------------------
    # Score validation
    # --------------------------------------------------------

    score_fields = [
        "coding_score",
        "communication_score",
        "extracurricular_score",
        "aptitude_score",
    ]

    for field in score_fields:

        value = float(record[field])

        if not (0 <= value <= 100):
            raise ValidationError(
                f"{field} must be between 0 and 100"
            )


# ============================================================
# CREATE STUDENT
# ============================================================

def add_student(
    record: dict,
    db_path: Path = DB_PATH
):
    """
    Create:
    Insert a new student record.

    Uses parameterized SQL values to protect against
    SQL injection.
    """

    _validate(record)

    # Only allow known database columns.
    allowed_columns = set(REQUIRED_FIELDS) | {
        "performance_score",
        "placed",
        "package_lpa",
    }

    invalid_columns = [
        column
        for column in record.keys()
        if column not in allowed_columns
    ]

    if invalid_columns:
        raise ValidationError(
            f"Invalid fields: {invalid_columns}"
        )

    cols = list(record.keys())

    placeholders = ", ".join(
        ["?"] * len(cols)
    )

    col_str = ", ".join(cols)

    values = [
        record[column]
        for column in cols
    ]

    with get_connection(db_path) as conn:

        conn.execute(
            f"""
            INSERT INTO students ({col_str})
            VALUES ({placeholders})
            """,
            values
        )

    logger.info(
        "Inserted student %s",
        record.get("student_id")
    )


# ============================================================
# READ SINGLE STUDENT
# ============================================================

def get_student(
    student_id: str,
    db_path: Path = DB_PATH
):
    """
    Read:
    Fetch a single student record by ID.
    """

    with get_connection(db_path) as conn:

        row = conn.execute(
            """
            SELECT *
            FROM students
            WHERE student_id = ?
            """,
            (student_id,)
        ).fetchone()

    return dict(row) if row else None


# ============================================================
# READ ALL STUDENTS
# ============================================================

def list_students(
    db_path: Path = DB_PATH,
    limit: int = 1000
):
    """
    Read:
    Return student records from the database.
    """

    if limit <= 0:
        raise ValidationError(
            "limit must be greater than 0"
        )

    with get_connection(db_path) as conn:

        rows = conn.execute(
            """
            SELECT *
            FROM students
            ORDER BY student_id
            LIMIT ?
            """,
            (limit,)
        ).fetchall()

    return [
        dict(row)
        for row in rows
    ]


# ============================================================
# UPDATE STUDENT
# ============================================================

def update_student(
    student_id: str,
    updates: dict,
    db_path: Path = DB_PATH
):
    """
    Update:
    Modify one or more allowed fields of an existing student.

    Only fields present in ALLOWED_UPDATE_FIELDS can be changed.
    """

    # --------------------------------------------------------
    # Empty update protection
    # --------------------------------------------------------

    if not updates:
        raise ValidationError(
            "No fields provided for update."
        )

    # --------------------------------------------------------
    # Validate field names
    # --------------------------------------------------------

    invalid_fields = [
        field
        for field in updates.keys()
        if field not in ALLOWED_UPDATE_FIELDS
    ]

    if invalid_fields:
        raise ValidationError(
            f"Invalid update field(s): {invalid_fields}"
        )

    # --------------------------------------------------------
    # Validate updated values
    # --------------------------------------------------------

    validation_record = {
        field: updates[field]
        for field in REQUIRED_FIELDS
        if field in updates
    }

    # Validate only fields that are being updated.
    if "cgpa" in updates:
        if not (0 <= float(updates["cgpa"]) <= 10):
            raise ValidationError(
                "cgpa must be between 0 and 10"
            )

    if "attendance_pct" in updates:
        if not (0 <= float(updates["attendance_pct"]) <= 100):
            raise ValidationError(
                "attendance_pct must be between 0 and 100"
            )

    if "backlogs" in updates:
        if int(updates["backlogs"]) < 0:
            raise ValidationError(
                "backlogs cannot be negative"
            )

    count_fields = [
        "internships",
        "projects_completed",
        "certifications",
    ]

    for field in count_fields:

        if field in updates:

            if int(updates[field]) < 0:
                raise ValidationError(
                    f"{field} cannot be negative"
                )

    score_fields = [
        "coding_score",
        "communication_score",
        "extracurricular_score",
        "aptitude_score",
    ]

    for field in score_fields:

        if field in updates:

            value = float(updates[field])

            if not (0 <= value <= 100):
                raise ValidationError(
                    f"{field} must be between 0 and 100"
                )

    # --------------------------------------------------------
    # Create SQL UPDATE statement
    # --------------------------------------------------------

    set_clause = ", ".join(
        [
            f"{field} = ?"
            for field in updates.keys()
        ]
    )

    values = list(
        updates.values()
    )

    values.append(student_id)

    # --------------------------------------------------------
    # Execute update
    # --------------------------------------------------------

    with get_connection(db_path) as conn:

        cur = conn.execute(
            f"""
            UPDATE students
            SET
                {set_clause},
                updated_at = CURRENT_TIMESTAMP
            WHERE student_id = ?
            """,
            values
        )

        if cur.rowcount == 0:

            raise ValidationError(
                f"No student found with id {student_id}"
            )

    logger.info(
        "Updated student %s",
        student_id
    )


# ============================================================
# DELETE STUDENT
# ============================================================

def delete_student(
    student_id: str,
    db_path: Path = DB_PATH
):
    """
    Delete:
    Remove a student record.
    """

    with get_connection(db_path) as conn:

        cur = conn.execute(
            """
            DELETE FROM students
            WHERE student_id = ?
            """,
            (student_id,)
        )

        if cur.rowcount == 0:

            raise ValidationError(
                f"No student found with id {student_id}"
            )

    logger.info(
        "Deleted student %s",
        student_id
    )


# ============================================================
# BULK LOAD DATAFRAME
# ============================================================

def bulk_load_from_dataframe(
    df,
    db_path: Path = DB_PATH
):
    """
    Bulk-load a pandas DataFrame into the students table.

    IMPORTANT:
    This version does NOT use pandas to_sql(if_exists='replace')
    because that would destroy the original database schema,
    constraints and table structure.

    Existing records with the same student_id are replaced safely.
    """

    # Initialize database/table.
    init_db(db_path)

    required_columns = set(REQUIRED_FIELDS) | {
        "performance_score",
        "placed",
        "package_lpa",
    }

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValidationError(
            f"DataFrame is missing columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # Insert records
    # --------------------------------------------------------

    with get_connection(db_path) as conn:

        for _, row in df.iterrows():

            record = row.to_dict()

            conn.execute(
                """
                INSERT OR REPLACE INTO students (
                    student_id,
                    branch,
                    gender,
                    attendance_pct,
                    cgpa,
                    backlogs,
                    internships,
                    projects_completed,
                    certifications,
                    coding_score,
                    communication_score,
                    extracurricular_score,
                    aptitude_score,
                    performance_score,
                    placed,
                    package_lpa
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["student_id"],
                    record["branch"],
                    record["gender"],
                    record["attendance_pct"],
                    record["cgpa"],
                    record["backlogs"],
                    record["internships"],
                    record["projects_completed"],
                    record["certifications"],
                    record["coding_score"],
                    record["communication_score"],
                    record["extracurricular_score"],
                    record["aptitude_score"],
                    record["performance_score"],
                    record["placed"],
                    record["package_lpa"],
                )
            )

    logger.info(
        "Bulk-loaded %d records into %s",
        len(df),
        db_path
    )


# ============================================================
# DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    import pandas as pd

    # Initialize database.
    init_db()

    # Dataset location.
    csv_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "students_dataset.csv"
    )

    # Check dataset exists.
    if not csv_path.exists():

        raise FileNotFoundError(
            f"Dataset not found: {csv_path}"
        )

    # Read dataset.
    df = pd.read_csv(csv_path)

    # Load into database.
    bulk_load_from_dataframe(df)

    # Display total records.
    print(
        f"Loaded {len(list_students())} "
        f"students into the database."
    )