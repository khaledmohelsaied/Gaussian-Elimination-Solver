# Gaussian Elimination: Step-by-Step Teaching Tool

This is a Python application built with CustomTkinter that not only solves systems of linear equations but also provides a detailed, step-by-step log that *teaches* the user how the algorithm works.

This was built as a college project to demonstrate the core textbook algorithm for Gaussian Elimination, as taught in a first-year university Linear Algebra course.

## Core Features

* **Two Solving Modes:**
    * **Gaussian Elimination (REF):** The default mode. It creates the "staircase" (Row Echelon Form) and then solves the system using **Back Substitution**.
    * **Gauss-Jordan (RREF):** An optional mode that creates the "identity matrix" (Reduced Row Echelon Form) and reads the solution directly.

* **Detailed "Teacher" Log:**
    * The log is not just a list of operations; it's a "tutor" that explains the *why* behind each step, such as finding the "Leading Entry" and "neglecting" the finished rows/columns.
    * It follows the exact 5-step algorithm from the textbook.

* **Full Solution Logic (Mathematically Correct):**
    * **Unique Solution:** Correctly finds the unique answer for a consistent system.
    * **No Solution (Inconsistent):** Correctly identifies `[0 0 | 1]` (contradiction) and stops the calculation.
    * **Infinite Solutions (Dependent):** This is the most advanced feature. The program correctly:
        1.  Identifies a `[0 0 | 0]` row.
        2.  Declares which variables are "free variables."
        3.  Assigns a parameter (e.g., `t`).
        4.  Solves for the *other* variables **in terms of `t`** to present the true **General Solution** (e.g., `x1 = 9 + 5t`).

* **Numerical Stability Handling:**
    * The program correctly identifies "ill-conditioned" matrices (e.g., `1e-7`) that *appear* solvable but are numerically unstable.
    * It correctly identifies "dangerously small" leading entries (e.g., `1e-15`) and logs a **`NUMERICAL STABILITY WARNING`** to the user, explaining *why* the resulting answer will be wrong.

## How to Run

1.  Ensure you have Python 3 installed.
2.  Install the required libraries:
    ```sh
    pip install customtkinter numpy
    ```
3.  Run the program:
    ```sh
    python gaussian_solver_FINAL_TEACHER.py
    ```# Gaussian-Elimination-Solver
