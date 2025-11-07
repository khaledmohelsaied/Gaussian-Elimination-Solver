import customtkinter as ctk
import numpy as np

# --- Helper Function for Smart Formatting ---
def format_num(val, precision=7):
    """Formats a number as int if possible, otherwise as float."""
    if np.isclose(val, np.round(val), atol=1e-15):
        return str(int(np.round(val))) # It's an integer
    else:
        return f"{val:.{precision}f}" # It's a float

# --- Helper function to build solution strings like "9 - 4t" ---
def format_expression(constant, terms, free_var_map):
    """Formats a symbolic solution (constant + terms) into a string."""
    expr = ""
    if not np.isclose(constant, 0) or len(terms) == 0:
        expr += f"{format_num(constant)}"
    
    # Sort terms to be in a consistent order (t, s, r...)
    sorted_vars = sorted(terms.keys(), key=lambda v: free_var_map.get(v, 0))

    for var in sorted_vars:
        coeff = terms[var]
        if np.isclose(coeff, 0):
            continue
        
        # Format the coefficient
        coeff_str = format_num(abs(coeff))
        sign = "-" if coeff < 0 else "+"
        
        # Don't show 1 as a coefficient
        if coeff_str == "1":
            term_str = f" {sign} {var}"
        else:
            term_str = f" {sign} {coeff_str}{var}"
            
        expr += term_str

    # Clean up leading "+ " if constant was 0
    if expr.startswith(" + "):
        expr = expr[3:]
    # Clean up leading " "
    return expr.strip()


# --- Main Application Class ---
class GaussianSolverApp(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="transparent") 

        # --- Constants ---
        self.ZERO_TOLERANCE = 1e-10 # NEW: Treat anything smaller than this as zero
        self.PIVOT_WARNING_THRESHOLD = 1e-9 # Threshold for "dangerously small" entry
        self.FREE_VAR_NAMES = ['z', 's', 'r', 'q', 'p'] # Names for free variables

        # --- Window Setup ---
        # --- FIX: Renamed title ---
        self.title_label_main = ctk.CTkLabel(self, text="Gaussian Elimination (Step-by-Step)", font=("Arial", 24, "bold"))
        self.title_label_main.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="n")

        self.n = 3 # Default matrix size
        self.matrix_entries = [] # To store the input widgets

        # --- Define Fonts ---
        self.TITLE_FONT = ("Arial", 24, "bold")
        self.SECTION_FONT = ("Arial", 18, "bold")
        self.BUTTON_FONT = ("Arial", 16, "bold")
        self.NORMAL_FONT = ("Arial", 16)
        self.ENTRY_FONT = ("Arial", 18)
        self.LOG_FONT = ("Courier New", 14) 
        self.CHECKBOX_FONT = ("Arial", 16, "bold")

        # --- Main Layout ---
        self.grid_rowconfigure(0, weight=0) # Title
        self.grid_rowconfigure(1, weight=1) # Main Content
        self.grid_columnconfigure(0, weight=1)

        # --- Main Content Frame (holds left and right panels) ---
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1) # Input Panel (smaller)
        self.main_frame.grid_columnconfigure(1, weight=2) # Output Panel (larger)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # --- 1. LEFT PANEL (INPUT) ---
        self.input_panel = ctk.CTkFrame(self.main_frame)
        self.input_panel.grid(row=0, column=0, padx=(0, 10), pady=10, sticky="nsew")
        
        self.input_panel.grid_rowconfigure(0, weight=0) # Title
        self.input_panel.grid_rowconfigure(1, weight=0) # Setup
        self.input_panel.grid_rowconfigure(2, weight=0) # RREF Checkbox
        self.input_panel.grid_rowconfigure(3, weight=1) # Matrix
        self.input_panel.grid_rowconfigure(4, weight=0) # Solve Button
        self.input_panel.grid_columnconfigure(0, weight=1)

        # 1a. Input Title
        self.input_title = ctk.CTkLabel(self.input_panel, text="1. Configuration & Matrix", font=self.SECTION_FONT)
        self.input_title.grid(row=0, column=0, padx=20, pady=(15, 10), columnspan=2)

        # 1b. Setup Frame
        self.setup_frame = ctk.CTkFrame(self.input_panel, fg_color="transparent")
        self.setup_frame.grid(row=1, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
        self.setup_frame.grid_columnconfigure(1, weight=1) # Make entry expand
        
        self.size_label = ctk.CTkLabel(self.setup_frame, text="Equations (n):", font=self.NORMAL_FONT)
        self.size_label.grid(row=0, column=0, padx=(0, 10), pady=10)

        self.size_entry = ctk.CTkEntry(self.setup_frame, width=70, font=self.ENTRY_FONT)
        self.size_entry.insert(0, "3")
        self.size_entry.grid(row=0, column=1, padx=5, pady=10, sticky="w")

        self.create_matrix_btn = ctk.CTkButton(self.setup_frame, text="Create Matrix", command=self.create_matrix_inputs, font=self.BUTTON_FONT)
        self.create_matrix_btn.grid(row=0, column=2, padx=(10, 0), pady=10, sticky="e")

        # 1c. Method Checkbox (RREF)
        self.use_gauss_jordan_var = ctk.BooleanVar(value=False) # Default to Gaussian (REF)
        self.method_checkbox = ctk.CTkCheckBox(
            self.input_panel, 
            text="Use Gauss-Jordan (RREF)",
            variable=self.use_gauss_jordan_var,
            font=self.CHECKBOX_FONT,
            onvalue=True,
            offvalue=False
        )
        self.method_checkbox.grid(row=2, column=0, columnspan=2, padx=20, pady=10, sticky="w")
        
        # 1e. Matrix Input Frame
        self.matrix_frame = ctk.CTkScrollableFrame(self.input_panel)
        self.matrix_frame.grid(row=3, column=0, columnspan=2, padx=20, pady=10, sticky="nsew")

        # 1f. Solve Button Frame
        self.solve_btn = ctk.CTkButton(self.input_panel, text="Solve System", command=self.solve_system, font=self.BUTTON_FONT, fg_color="#008000", hover_color="#006400",
                                       height=50)
        self.solve_btn.grid(row=4, column=0, columnspan=2, padx=20, pady=(10, 20), sticky="ew")

        # --- 2. RIGHT PANEL (OUTPUT) ---
        self.output_panel = ctk.CTkFrame(self.main_frame)
        self.output_panel.grid(row=0, column=1, padx=(10, 0), pady=10, sticky="nsew")
        self.output_panel.grid_rowconfigure(0, weight=0) # Title
        self.output_panel.grid_rowconfigure(1, weight=1) # Textbox
        self.output_panel.grid_columnconfigure(0, weight=1)
        
        # 2a. Output Title
        self.output_title = ctk.CTkLabel(self.output_panel, text="2. Solution & Step-by-Step Log", font=self.SECTION_FONT)
        self.output_title.grid(row=0, column=0, padx=20, pady=(15, 10))

        # 2b. Output Textbox
        self.output_textbox = ctk.CTkTextbox(self.output_panel, state="disabled", font=self.LOG_FONT, wrap="none")
        self.output_textbox.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="nsew")

        # Initialize the first matrix grid
        self.create_matrix_inputs()
        
        # --- FIX: Renamed title ---
        master.title("Gaussian Elimination (Step-by-Step)")


    # --- GUI Methods ---

    def create_matrix_inputs(self):
        try:
            n = int(self.size_entry.get())
            if not (2 <= n <= 10):
                self.log_raw("Error: Invalid input size. Size must be between 2 and 10.")
                self.size_entry.configure(border_color="#9B0000", border_width=2) # Highlight
                return
            self.n = n
            self.size_entry.configure(border_width=1, border_color="gray50") # Clear error
        except ValueError:
            self.output_textbox.configure(state="normal")
            self.output_textbox.delete("1.0", "end")
            self.log_raw("Error: Invalid input size. Size must be between 2 and 10.")
            self.size_entry.configure(border_color="#9B0000", border_width=2) # Highlight
            return

        for widget in self.matrix_frame.winfo_children():
            widget.destroy()
        self.matrix_entries = []
        self.matrix_frame.configure(fg_color="transparent")
        
        # Center the grid
        self.matrix_frame.grid_columnconfigure(list(range(self.n + 2)), weight=1)

        # Create column labels
        for j in range(self.n):
            label = ctk.CTkLabel(self.matrix_frame, text=f"x{j+1}", font=self.NORMAL_FONT)
            label.grid(row=0, column=j, padx=5, pady=10)
        
        sep_label = ctk.CTkLabel(self.matrix_frame, text="|", font=self.ENTRY_FONT)
        sep_label.grid(row=0, column=self.n, padx=5, pady=10)
        
        b_label = ctk.CTkLabel(self.matrix_frame, text="b", font=self.NORMAL_FONT)
        b_label.grid(row=0, column=self.n + 1, padx=5, pady=10)

        # Create entry grid
        for i in range(self.n):
            row_entries = []
            for j in range(self.n + 1):
                if j == self.n:
                    sep = ctk.CTkFrame(self.matrix_frame, width=2, fg_color="gray50")
                    sep.grid(row=i+1, column=j, padx=5, pady=5, sticky="ns")

                entry = ctk.CTkEntry(self.matrix_frame, width=80, font=self.ENTRY_FONT, justify="left")
                entry.grid(row=i+1, column=j + (1 if j >= self.n else 0), padx=5, pady=8)
                row_entries.append(entry)
            self.matrix_entries.append(row_entries)
    
    # --- Logging Methods ---

    def log_message(self, message, **kwargs):
        """Logs a formatted 'teacher' string."""
        formatted_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, (float, np.float64)):
                formatted_kwargs[k] = format_num(v)
            elif isinstance(v, (int, np.int_)):
                 formatted_kwargs[k] = str(v)
            else:
                formatted_kwargs[k] = v
        
        try:
            formatted_message = message.format(**formatted_kwargs)
        except Exception as e:
            formatted_message = f"LOGGING_ERROR: {e} | Original: {message}"
        self.log_raw(formatted_message)
        
    def log_raw(self, text):
        """Logs a raw string (like a matrix) directly to the textbox."""
        self.output_textbox.configure(state="normal")
        self.output_textbox.insert("end", text + "\n")
        self.output_textbox.configure(state="disabled")
        self.output_textbox.see("end")
        self.update_idletasks()

    def get_header_string(self, n_cols):
        """Gets the formatted header string for the matrix log."""
        headers = []
        for j in range(n_cols - 1): # n_cols-1 == n
            headers.append(f"x{j+1:<10}") # 10-char wide column
        
        header_str = "   ".join(headers)
        header_str = f"      {header_str}   |   {'b':<10}"
        
        s = f"{header_str}\n"
        s += "    " + "-" * (len(header_str) - 4) + "\n"
        return s
    
    def format_matrix_as_string(self, matrix):
        """Formats a numpy matrix ROWS into a clean string for logging (no header)."""
        n_rows, n_cols = matrix.shape
        s = "" 
        
        for i in range(n_rows):
            row_label = f"R{i+1}"
            row_str = f"{row_label:<3} | ["
            
            for j in range(n_cols):
                if j == n_cols - 1:
                    row_str += " |"
                
                val = matrix[i, j]
                # --- FIX: Use new ZERO_TOLERANCE for display ---
                if abs(val) < self.ZERO_TOLERANCE: val = 0.0 
                
                val_str = format_num(val, precision=7)
                row_str += f"{val_str:>10}" 
                
            row_str += " ]"
            s += row_str + "\n"
        return s

    def get_matrix_from_inputs(self):
        """Validates and parses the matrix from the GUI. Returns a numpy array."""
        matrix = np.zeros((self.n, self.n + 1), dtype=np.float64) 
        has_error = False 

        for i in range(self.n):
            for j in range(self.n + 1):
                if i < len(self.matrix_entries) and j < len(self.matrix_entries[i]):
                    self.matrix_entries[i][j].configure(border_width=1, border_color="gray50")
        
        self.size_entry.configure(border_width=1, border_color="gray50")

        for i in range(self.n):
            for j in range(self.n + 1):
                if i < len(self.matrix_entries) and j < len(self.matrix_entries[i]):
                    val_str = self.matrix_entries[i][j].get()
                    if not val_str:
                        self.matrix_entries[i][j].configure(border_color="#E07A5F", border_width=2)
                        self.log_raw(f"ERROR: Empty cell at (Row {i+1}, Col {j+1}).")
                        has_error = True
                    else:
                        try:
                            matrix[i, j] = float(val_str)
                        except ValueError:
                            self.matrix_entries[i][j].configure(border_color="#9B0000", border_width=2)
                            self.log_raw(f"ERROR: Invalid input at (Row {i+1}, Col {j+1}). Please enter numbers only.")
                            has_error = True
        
        if has_error:
            return None 
        else:
            return matrix 

    # --- Main Solve Function ---
    
    def is_zero(self, val):
        """Helper function to check if a value is effectively zero."""
        return np.isclose(val, 0, atol=self.ZERO_TOLERANCE)

    def solve_system(self):
        """
        Solves the system using the Doctor's 5-step textbook algorithm.
        This is a column-driven approach.
        """
        
        # --- 1. Setup ---
        self.output_textbox.configure(state="normal")
        self.output_textbox.delete("1.0", "end") # Clear log
        self.output_textbox.configure(state="disabled")

        self.solve_btn.configure(text="Solving...", state="disabled")
        self.create_matrix_btn.configure(state="disabled")
        self.update_idletasks() 
        
        try:
            # --- 2. Get & Validate Matrix ---
            matrix = self.get_matrix_from_inputs()
            if matrix is None:
                raise ValueError("Invalid input. Please check highlighted cells.")

            n = self.n
            is_rref = self.use_gauss_jordan_var.get()
            
            # Track which columns are pivot columns
            pivot_cols_found = [False] * n
            free_var_map = {} 
            
            self.log_raw("Starting with the Augmented Matrix:")
            self.log_raw(self.get_header_string(n + 1))
            self.log_raw(self.format_matrix_as_string(matrix))
            
            if is_rref:
                # --- FIX: Renamed title ---
                self.master.title("Gauss-Jordan (RREF) Solver")
                self.title_label_main.configure(text="Gauss-Jordan (RREF) Solver")
                self.log_raw("\nPHASE 1: Elimination to Reduced Row Echelon Form (RREF)")
                self.log_raw("GOAL: Transform the left side into an 'identity matrix' (1s on the diagonal, 0s everywhere else).")
            else:
                # --- FIX: Renamed title ---
                self.master.title("Gaussian Elimination (REF) Solver")
                self.title_label_main.configure(text="Gaussian Elimination (REF) Solver")
                self.log_raw("\nPHASE 1: Forward Elimination (to Row Echelon Form)")
                self.log_raw("GOAL: Transform the matrix into a 'staircase' shape (1s on the diagonal, 0s below).")
            
            self.log_raw("------------------------------------------------------------------")

            # --- 3. Phase 1: Elimination (Doctor's Algorithm) ---
            
            pivot_row = 0
            pivot_col = 0
            
            while pivot_row < n and pivot_col < n:
                k = pivot_row + 1
                k_th = f"{k}{'st' if k == 1 else 'nd' if k == 2 else 'rd' if k == 3 else 'th'}"
                self.log_message("\n== Finding Leading Entry {k} (The {k_th} 'staircase' step) ==", k=k, k_th=k_th)
                self.log_message("   (Starting search at or below R{r}, C{c})", r=pivot_row+1, c=pivot_col+1)
                
                # --- Step 1: Find Pivot Column ---
                self.log_message("   Step 1: Find leftmost non-zero column (at or below R{r}).", r=pivot_row+1)
                
                i = pivot_row
                # --- FIX: Use new is_zero() function ---
                while i < n and self.is_zero(matrix[i, pivot_col]):
                    i += 1
                
                if i == n:
                    self.log_message("   Column {c} is all zeros at or below R{r}. Skipping to next column.", c=pivot_col+1, r=pivot_row+1)
                    pivot_col += 1 
                    continue 
                
                self.log_message("   Found column for leading entry: C{c}.", c=pivot_col+1)
                pivot_cols_found[pivot_col] = True 
                
                # --- Step 2: Simple Swap ---
                self.log_message("   Step 2: Get a non-zero entry to the leading position (R{r}).", r=pivot_row+1)
                if i != pivot_row:
                    self.log_message("   Entry at (R{r}, C{c}) is 0. Found non-zero entry in R{i_row}.", r=pivot_row+1, c=pivot_col+1, i_row=i+1)
                    self.log_message("   Operation: Swap R{r} with R{i_row}.", r=pivot_row+1, i_row=i+1)
                    matrix[[pivot_row, i]] = matrix[[i, pivot_row]] 
                    self.log_raw(self.format_matrix_as_string(matrix))
                else:
                    self.log_message("   Entry at (R{r}, C{c}) is already non-zero. No swap needed.", r=pivot_row+1, c=pivot_col+1)
                
                # --- Step 3: Normalize (Make Leading 1) ---
                pivot_val = matrix[pivot_row, pivot_col]
                self.log_message("   Step 3: Make the leading entry at (R{r}, C{c}) a 'Leading 1'.", r=pivot_row+1, c=pivot_col+1)
                
                if abs(pivot_val) < self.PIVOT_WARNING_THRESHOLD:
                    self.output_textbox.tag_add("warning", "end-2l", "end-1l")
                    self.output_textbox.tag_config("warning", foreground="yellow")
                    self.log_message("   NUMERICAL STABILITY WARNING:\n   The entry's value {val_str} is dangerously small!", val_str=f"{pivot_val:.7e}")

                if not np.isclose(pivot_val, 1.0):
                    self.log_message("   Operation: R{row} = R{row} / {val}", row=pivot_row+1, val=pivot_val)
                    matrix[pivot_row] = matrix[pivot_row] / pivot_val
                    matrix[pivot_row, pivot_col] = 1.0 
                    self.log_raw(self.format_matrix_as_string(matrix))
                else:
                    self.log_raw("   Entry is already 1. No normalization needed.")

                # --- Step 4: Elimination (Clear the column) ---
                if is_rref:
                    self.log_message("   Step 4: Clear all other entries in Column {col} to 0.", col=pivot_col+1)
                else:
                    self.log_message("   Step 4: Clear all entries *below* the Leading 1 in Column {col} to 0.", col=pivot_col+1)
                    
                for i in range(n):
                    if i == pivot_row:
                        continue 
                    
                    if not is_rref and i < pivot_row:
                        continue
                        
                    factor = matrix[i, pivot_col]
                    
                    # --- FIX: Use new is_zero() function ---
                    if self.is_zero(factor):
                        self.log_message("   Element at (R{row}, C{col}) is already 0. Skipping.", row=i+1, col=pivot_col+1)
                    else:
                        self.log_message("   Operation: R{row_i} = R{row_i} - ({val_str}) * R{row_p}", row_i=i+1, val_str=format_num(factor), row_p=pivot_row+1)
                        matrix[i] = matrix[i] - (factor * matrix[pivot_row])
                        matrix[i, pivot_col] = 0.0 
                
                self.log_raw(self.format_matrix_as_string(matrix))
                
                # --- Step 5: Repeat ---
                self.log_raw("   Step 5: Process complete for this leading entry.")
                self.log_raw("   WHY WE REPEAT: The algorithm is now finished with")
                self.log_message("   Row {r} and this leading entry's column (C{c}).", r=pivot_row+1, c=pivot_col+1)
                self.log_raw("   We now 'hide' this row and repeat all steps (1-4)")
                self.log_raw("   to find the *next* leading entry, which must be down")
                self.log_raw("   and to the right of the one we just found.")
                self.log_raw("------------------------------------------------------------------")
                
                pivot_row += 1
                pivot_col += 1

            # --- 4. Phase 1 Complete ---
            if is_rref:
                self.log_raw("Elimination Complete. Matrix is in Reduced Row Echelon Form (RREF).")
            else:
                self.log_raw("PHASE 1 Complete: Matrix is in Row Echelon Form (REF).")

            # --- 5. Phase 2: Check for Errors and Solve ---
            
            has_infinite_solutions = False
            # Check for "No Solution" or "Infinite Solutions"
            for i in range(n - 1, -1, -1): # Check from the bottom up
                row = matrix[i, :n]
                b_val = matrix[i, n]
                
                # --- FIX: Use new is_zero() function ---
                if np.all(self.is_zero(row)): # Row is all zeros [0 0 0 | b]
                    if self.is_zero(b_val):
                        self.log_message("\nFound row [0 0 ... | 0] at R{row}.\nThis indicates a dependent system with INFINITELY MANY SOLUTIONS.", row=i+1)
                        has_infinite_solutions = True
                    else:
                        self.log_message("\nFound row [0 0 ... | {b_val}] at R{row}.\nThis is a contradiction (0 = {b_val}).\nThe system has NO SOLUTION.", b_val=b_val, row=i+1)
                        raise ValueError("Inconsistent system (e.g., 0 = 1). No solution exists.")
            
            # --- FIX: Add "Consistent System" log ---
            if not has_infinite_solutions:
                self.log_raw("\nThis is a 'consistent' system. A unique solution was found.")
            else:
                self.log_raw("\nThis is a 'consistent' system.")


            solution = np.empty(n, dtype=object)
            solution_was_found = [False] * n 
            
            # --- Identify free variables ---
            free_var_count = 0
            for c in range(n):
                if not pivot_cols_found[c]:
                    var_name = self.FREE_VAR_NAMES[free_var_count % len(self.FREE_VAR_NAMES)]
                    free_var_map[c] = var_name # e.g., {2: 't'}
                    free_var_count += 1
            
            if has_infinite_solutions:
                self.log_raw("\nNOTE: This system has free variables.")
                self.log_raw("To find the 'General Solution', we will express the")
                self.log_raw("other variables in terms of these free variables.")
                for col_index, var_name in free_var_map.items():
                    self.log_message("   Let x{i} = {var} (free variable)", i=col_index+1, var=var_name)
                self.log_raw("------------------------------------------------------------------")
            
            
            if is_rref:
                # --- RREF: Read the Symbolic Solution ---
                self.log_raw("\nPHASE 2: Read the Solution")
                self.log_raw("GOAL: The left side is in RREF. We can now express")
                self.log_raw("each 'leading' variable in terms of the constants")
                self.log_raw("and any 'free' variables.")
                self.log_raw("------------------------------------------------------------------")
                
                self.log_raw("Final General Solution:")
                
                # Initialize all solutions
                for i in range(n):
                    if i in free_var_map:
                        solution[i] = (0.0, {free_var_map[i]: 1.0}) # e.g., x3 = 0 + 1t
                        solution_was_found[i] = True
                    else:
                        solution[i] = (0.0, {}) # Placeholder
                
                # Read the 'leading' variables from the RREF
                for r in range(n): # For each row
                    pivot_col = -1
                    for c in range(n):
                        # --- FIX: Use new is_zero() function ---
                        if np.isclose(matrix[r, c], 1.0) and pivot_cols_found[c]:
                            pivot_col = c
                            break
                    
                    if pivot_col != -1:
                        # This row defines a leading variable
                        constant = matrix[r, n]
                        terms = {}
                        # Look for free variables *in this row*
                        for c_free in free_var_map.keys():
                            # --- FIX: Use new is_zero() function ---
                            if not self.is_zero(matrix[r, c_free]):
                                terms[free_var_map[c_free]] = -matrix[r, c_free]
                        
                        solution[pivot_col] = (constant, terms)
                        solution_was_found[pivot_col] = True

                # Print the final formatted solution
                for i in range(n):
                    (const, terms) = solution[i]
                    expr = format_expression(const, terms, free_var_map)
                    self.log_message("x{i} = {expr}", i=i+1, expr=expr)
            
            else:
                # --- REF: Symbolic Back Substitution ---
                self.log_raw("\nPHASE 2: Back Substitution")
                self.log_raw("  The GOAL is to solve the simplified 'triangular' system.")
                self.log_raw("  We start from the last equation and work our way back up,")
                self.log_raw("  expressing each variable in terms of constants and free variables.")
                self.log_raw("------------------------------------------------------------------")
                
                # Initialize free variables
                for i in range(n):
                    if i in free_var_map:
                        solution[i] = (0.0, {free_var_map[i]: 1.0})
                        solution_was_found[i] = True

                for i in range(n - 1, -1, -1): # From n-1 down to 0
                    
                    pivot_col = -1
                    for j in range(n):
                        # --- FIX: Use new is_zero() function ---
                        if np.isclose(matrix[i, j], 1.0) and pivot_cols_found[j]:
                            pivot_col = j
                            break
                    
                    if pivot_col == -1: 
                        continue # This is a zero row or a free var row

                    self.log_message("Solving for x{i} using Row {r}:", i=pivot_col+1, r=i+1)
                    
                    # --- NEW Symbolic Sum ---
                    sum_knowns_const = 0.0
                    sum_knowns_terms = {}
                    sum_knowns_str = ""

                    for j in range(pivot_col + 1, n):
                        matrix_val = matrix[i, j]
                        # --- FIX: Use new is_zero() function ---
                        if self.is_zero(matrix_val):
                            continue
                        
                        if solution_was_found[j]:
                            (const, terms) = solution[j]
                            
                            sum_knowns_const += matrix_val * const
                            
                            for var, coeff in terms.items():
                                sum_knowns_terms[var] = sum_knowns_terms.get(var, 0) + matrix_val * coeff
                            
                            expr = format_expression(const, terms, free_var_map)
                            sum_knowns_str += f" + ({format_num(matrix_val)} * ({expr}))"

                    if sum_knowns_str == "":
                        sum_knowns_str = "0"
                    
                    b_val = matrix[i, n]
                    entry_val = matrix[i, pivot_col] 
                    
                    final_const = (b_val - sum_knowns_const) / entry_val
                    final_terms = {}
                    for var, coeff in sum_knowns_terms.items():
                        final_terms[var] = -coeff / entry_val

                    solution[pivot_col] = (final_const, final_terms)
                    solution_was_found[pivot_col] = True
                    
                    self.log_message("  x{i} = (b - (sum of knowns)) / (leading entry)", i=pivot_col+1)
                    self.log_message("  x{i} = ({b_val_str} - ({sum_knowns_str})) / {entry_val_str} = {solution_str}", 
                                   i=pivot_col+1, 
                                   b_val_str=format_num(b_val), 
                                   sum_knowns_str=sum_knowns_str.lstrip(" + "), 
                                   entry_val_str=format_num(entry_val),
                                   solution_str=format_expression(final_const, final_terms, free_var_map))
                
                self.log_raw("------------------------------------------------------------------")
                self.output_textbox.configure(font=self.LOG_FONT)
                if has_infinite_solutions:
                    self.log_raw("Final General Solution (from Back Substitution):")
                else:
                    self.log_raw("Final Solution (from Back Substitution):")
                
                for i in range(n):
                    if solution_was_found[i]:
                        (const, terms) = solution[i]
                        expr = format_expression(const, terms, free_var_map)
                        self.log_message(f"x{i+1} = {expr}")
                    else:
                         # This should not happen if logic is correct, but as a fallback:
                         self.log_raw(f"x{i+1} = ??? (Unsolved)")

        except ValueError as e:
            # Handle "Matrix is singular" or "No solution"
            self.log_message("\n!!! COMPUTATION STOPPED: {e} !!!", e=str(e))
        except Exception as e:
            # Handle any other unexpected error
            self.output_textbox.configure(state="normal")
            self.output_textbox.delete("1.0", "end") # Clear log
            self.log_message("\n!!! AN UNEXPECTED ERROR OCCURRED: {e} !!!", e=e)
        finally:
            # --- 7. Re-enable Buttons ---
            self.solve_btn.configure(text="Solve Again", state="normal")
            self.create_matrix_btn.configure(state="normal")


# --- Main Execution ---
if __name__ == "__main__":
    app = ctk.CTk()
    app.grid_rowconfigure(0, weight=1)
    app.grid_columnconfigure(0, weight=1)
    
    # Create the main app frame
    main_app_frame = GaussianSolverApp(master=app)
    # Place the main app frame into the root window
    main_app_frame.grid(row=0, column=0, sticky="nsew") 

    app.mainloop()

