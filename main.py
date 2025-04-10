import os
import re
from typing import Dict, List, Tuple, Counter
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class ATETestLookup:
    def __init__(self, testprogram_dir: str):
        """
        Initialize the ATE Test Lookup tool with the directory containing testprogram files.
        
        Args:
            testprogram_dir: Path to the directory containing testprogram source files
        """
        self.testprogram_dir = testprogram_dir
        # Try to load with specific version compatibility
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"Error initializing SentenceTransformer: {e}")
            print("Please make sure you have installed the correct versions:")
            print("pip install sentence-transformers==2.2.2 transformers==4.30.0 torch==2.0.1")
            raise e
            
        self.file_info = []  # Will store (file_path, file_content)
        self.file_embeddings = None
        
        # Store info about code blocks for more specific lookup
        self.code_blocks = []  # Will store (file_path, block_type, block_name, block_content, start_line, end_line)
        self.code_block_embeddings = None
        
    def scan_files(self, file_extensions: List[str] = ['.c', '.cpp', '.h', '.py']):
        """
        Scan all files with given extensions in the testprogram directory.
        
        Args:
            file_extensions: List of file extensions to scan
        """
        print(f"Scanning files in {self.testprogram_dir}...")
        
        for root, _, files in os.walk(self.testprogram_dir):
            for file in files:
                if any(file.endswith(ext) for ext in file_extensions):
                    file_path = os.path.join(root, file)
                    self._process_file(file_path)
        
        print(f"Scanned {len(self.file_info)} files")
        print(f"Extracted {len(self.code_blocks)} code blocks")
        
    def _process_file(self, file_path: str):
        """
        Process a file by reading its content and extracting code blocks.
        
        Args:
            file_path: Path to the source file
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Store the whole file content for full-text search
            self.file_info.append((file_path, content))
            
            # Extract code blocks for more specific lookup
            self._extract_code_blocks(file_path, content)
                
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    def _extract_code_blocks(self, file_path: str, content: str):
        """
        Extract different types of code blocks from the file.
        
        Args:
            file_path: Path to the source file
            content: Content of the file
        """
        # Extract functions
        self._extract_functions(file_path, content)
        
        # Extract classes
        self._extract_classes(file_path, content)
        
        # Extract variables
        self._extract_variables(file_path, content)
        
        # Extract comments
        self._extract_comments(file_path, content)
        
        # Extract string literals
        self._extract_string_literals(file_path, content)
    
    def _extract_functions(self, file_path: str, content: str):
        """
        Extract function definitions from file content.
        """
        # Get line numbers for better context
        lines = content.split('\n')
        
        if file_path.endswith('.py'):
            # Python function pattern
            pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)(?:\s*->(?:[^:]+))?\s*:((?:\n\s+[^\n]+)*)'
            
            for match in re.finditer(pattern, content):
                func_name = match.group(1)
                func_content = match.group(0)
                
                # Calculate line numbers
                start_line = content[:match.start()].count('\n') + 1
                end_line = start_line + func_content.count('\n')
                
                self.code_blocks.append((file_path, 'function', func_name, func_content, start_line, end_line))
        
        elif file_path.endswith(('.c', '.cpp', '.h')):
            # C/C++ function pattern - simplified for brevity
            pattern = r'(?:\w+(?:\s*[*&])?)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*\{(?:[^{}]|(?R))*\}'
            
            for match in re.finditer(pattern, content):
                func_name = match.group(1)
                if func_name not in ['if', 'for', 'while', 'switch']:
                    func_content = match.group(0)
                    
                    # Calculate line numbers
                    start_line = content[:match.start()].count('\n') + 1
                    end_line = start_line + func_content.count('\n')
                    
                    self.code_blocks.append((file_path, 'function', func_name, func_content, start_line, end_line))
    
    def _extract_classes(self, file_path: str, content: str):
        """
        Extract class definitions from file content.
        """
        if file_path.endswith('.py'):
            # Python class pattern
            pattern = r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)[^\{:]*:(?:\n\s+[^\n]+)*'
            
            for match in re.finditer(pattern, content):
                class_name = match.group(1)
                class_content = match.group(0)
                
                # Calculate line numbers
                start_line = content[:match.start()].count('\n') + 1
                end_line = start_line + class_content.count('\n')
                
                self.code_blocks.append((file_path, 'class', class_name, class_content, start_line, end_line))
        
        elif file_path.endswith(('.cpp', '.h')):
            # C++ class pattern
            pattern = r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)[^{]*\{(?:[^{}]|(?R))*\}'
            
            for match in re.finditer(pattern, content):
                class_name = match.group(1)
                class_content = match.group(0)
                
                # Calculate line numbers
                start_line = content[:match.start()].count('\n') + 1
                end_line = start_line + class_content.count('\n')
                
                self.code_blocks.append((file_path, 'class', class_name, class_content, start_line, end_line))
    
    def _extract_variables(self, file_path: str, content: str):
        """
        Extract variable definitions from file content.
        """
        # Common variable declaration patterns
        patterns = [
            # C/C++ variable declarations
            r'(?:\w+(?:\s*[*&])?)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:=|;)',
            # Python variable assignments
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s*='
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, content):
                var_name = match.group(1)
                var_content = match.group(0)
                
                # Skip keywords
                if var_name in ['if', 'for', 'while', 'return', 'switch', 'case']:
                    continue
                
                # Calculate line numbers
                line_num = content[:match.start()].count('\n') + 1
                
                self.code_blocks.append((file_path, 'variable', var_name, var_content, line_num, line_num))
    
    def _extract_comments(self, file_path: str, content: str):
        """
        Extract comments from file content.
        """
        # C/C++ style comments
        if file_path.endswith(('.c', '.cpp', '.h')):
            # Single line comments
            for match in re.finditer(r'//(.+)$', content, re.MULTILINE):
                comment = match.group(0)
                line_num = content[:match.start()].count('\n') + 1
                self.code_blocks.append((file_path, 'comment', '', comment, line_num, line_num))
            
            # Multi-line comments
            for match in re.finditer(r'/\*(.+?)\*/', content, re.DOTALL):
                comment = match.group(0)
                start_line = content[:match.start()].count('\n') + 1
                end_line = start_line + comment.count('\n')
                self.code_blocks.append((file_path, 'comment', '', comment, start_line, end_line))
        
        # Python comments
        if file_path.endswith('.py'):
            # Single line comments
            for match in re.finditer(r'#(.+)$', content, re.MULTILINE):
                comment = match.group(0)
                line_num = content[:match.start()].count('\n') + 1
                self.code_blocks.append((file_path, 'comment', '', comment, line_num, line_num))
    
    def _extract_string_literals(self, file_path: str, content: str):
        """
        Extract string literals from file content.
        """
        # Match different types of string literals
        patterns = [
            # Double quoted strings
            r'"((?:\\.|[^"\\])*)"',
            # Single quoted strings
            r"'((?:\\.|[^'\\])*)'",
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, content):
                string_content = match.group(0)
                line_num = content[:match.start()].count('\n') + 1
                self.code_blocks.append((file_path, 'string', '', string_content, line_num, line_num))
    
    def create_embeddings(self):
        """
        Create embeddings for files and code blocks.
        """
        if not self.file_info:
            print("No files found. Please scan files first.")
            return
        
        print("Creating embeddings for files...")
        file_texts = [os.path.basename(path) + " " + content for path, content in self.file_info]
        self.file_embeddings = self.model.encode(file_texts)
        
        if self.code_blocks:
            print("Creating embeddings for code blocks...")
            block_texts = [f"{block_type} {block_name} {content}" for _, block_type, block_name, content, _, _ in self.code_blocks]
            self.code_block_embeddings = self.model.encode(block_texts)
        
        print("Embeddings created successfully")
    
    def find_test(self, test_name: str, top_n: int = 5) -> Dict:
        """
        Comprehensive search for the test name across all files and code blocks.
        
        Args:
            test_name: The ATE test name to look up
            top_n: Number of top matches to return
            
        Returns:
            Dictionary with search results organized by type
        """
        results = {
            'file_semantic_matches': [],
            'code_block_semantic_matches': [],
            'exact_matches': [],
            'regex_matches': []
        }
        
        # 1. Semantic search on whole files
        if self.file_embeddings is not None:
            test_embedding = self.model.encode([test_name])
            similarities = cosine_similarity(test_embedding, self.file_embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:top_n]
            
            for idx in top_indices:
                file_path, _ = self.file_info[idx]
                score = similarities[idx]
                results['file_semantic_matches'].append((file_path, score))
        
        # 2. Semantic search on code blocks
        if self.code_block_embeddings is not None:
            test_embedding = self.model.encode([test_name])
            similarities = cosine_similarity(test_embedding, self.code_block_embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:top_n]
            
            for idx in top_indices:
                file_path, block_type, block_name, _, start_line, end_line = self.code_blocks[idx]
                score = similarities[idx]
                results['code_block_semantic_matches'].append((file_path, block_type, block_name, start_line, end_line, score))
        
        # 3. Exact string matching
        for file_path, content in self.file_info:
            if test_name in content:
                # Count occurrences
                count = content.count(test_name)
                results['exact_matches'].append((file_path, count))
        
        # 4. Regex pattern matching for partial matches or variations
        # This can find test names that appear as part of other words or with different separators
        pattern = r'[\w_]*' + re.escape(test_name).replace('_', '[\W_]?') + r'[\w_]*'
        for file_path, content in self.file_info:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                results['regex_matches'].append((file_path, len(matches), matches[:5]))  # Show first 5 matches as examples
        
        # Sort results by relevance
        results['file_semantic_matches'].sort(key=lambda x: x[1], reverse=True)
        results['code_block_semantic_matches'].sort(key=lambda x: x[5], reverse=True)
        results['exact_matches'].sort(key=lambda x: x[1], reverse=True)
        results['regex_matches'].sort(key=lambda x: x[1], reverse=True)
        
        # Calculate overall most relevant files
        file_scores = {}
        
        # Add scores from semantic file matches
        for file_path, score in results['file_semantic_matches']:
            file_scores[file_path] = file_scores.get(file_path, 0) + score * 2  # Weight semantic matches higher
        
        # Add scores from exact matches
        for file_path, count in results['exact_matches']:
            file_scores[file_path] = file_scores.get(file_path, 0) + count * 1.5  # Exact matches are important
        
        # Add scores from regex matches
        for file_path, count, _ in results['regex_matches']:
            file_scores[file_path] = file_scores.get(file_path, 0) + count * 0.5  # Partial matches less important
        
        # Add scores from code block matches
        for file_path, _, _, _, _, score in results['code_block_semantic_matches']:
            file_scores[file_path] = file_scores.get(file_path, 0) + score
        
        # Sort files by overall relevance score
        most_relevant_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        results['most_relevant_files'] = most_relevant_files[:top_n]
        
        return results
    
    def extract_context(self, file_path: str, line_number: int, context_lines: int = 5) -> str:
        """
        Extract lines around the specified line number for context.
        
        Args:
            file_path: Path to the file
            line_number: Line number to focus on
            context_lines: Number of lines to include before and after
            
        Returns:
            String with the lines of context
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            start = max(0, line_number - context_lines - 1)
            end = min(len(lines), line_number + context_lines)
            
            context = []
            for i in range(start, end):
                # Highlight the target line
                if i == line_number - 1:
                    context.append(f">>> {i+1}: {lines[i].rstrip()}")
                else:
                    context.append(f"    {i+1}: {lines[i].rstrip()}")
            
            return '\n'.join(context)
        except Exception as e:
            return f"Error extracting context: {e}"
    
    def find_test_locations(self, file_path: str, test_name: str) -> List[int]:
        """
        Find all line numbers where the test name appears in a file.
        
        Args:
            file_path: Path to the file
            test_name: Test name to search for
            
        Returns:
            List of line numbers where the test name appears
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            line_numbers = []
            for i, line in enumerate(lines):
                if test_name in line:
                    line_numbers.append(i + 1)
            
            return line_numbers
        except Exception as e:
            print(f"Error searching file {file_path}: {e}")
            return []

def main():
    # Replace with your actual testprogram directory
    testprogram_dir = "C:/Users/rejav/Documents/SilTest-side-projects/GitHub/ATE_Debug_Assisntant-Semantic_Mapper/sample_testprogram_elmos_v52241b"
    
    # Create a sample test directory and file for demonstration if they don't exist
    if not os.path.exists(testprogram_dir):
        os.makedirs(testprogram_dir)
        
        # Create a C file with the test name in various contexts
        with open(os.path.join(testprogram_dir, "smps_test.c"), "w") as f:
            f.write("""
            #include <stdio.h>
            
            // Constants for SMPS testing
            #define SMPS_PS_ERR_THRESHOLD 0.05
            #define MAX_SMPS_RETRIES 3
            
            // Global variables
            float g_smps_ps_err_value = 0.0;
            
            // Function that measures SMPS power supply error
            int measure_smps_ps_err(int channel) {
                // Initialize measurement equipment
                int result = 0;
                
                printf("Starting SMPS PS error measurement on channel %d\\n", channel);
                
                // Perform smps_ps_err test
                result = perform_measurement(channel, "smps_ps_err");
                
                // Store the measurement for later reference
                g_smps_ps_err_value = result * 0.001;
                
                return result;
            }
            
            // Supporting function for power supply testing
            int perform_measurement(int channel, char* test_name) {
                // This is just a simulated function
                printf("Performing measurement: %s\\n", test_name);
                return 42;  // Simulated measurement value
            }
            
            // Test handler function
            void run_power_supply_tests() {
                int smps_results[3];
                
                // Run SMPS tests
                smps_results[0] = measure_smps_ps_err(1);
                
                // Process results
                if (g_smps_ps_err_value > SMPS_PS_ERR_THRESHOLD) {
                    printf("SMPS PS Error exceeded threshold!\\n");
                }
            }
            """)
        
        # Create another C file with fewer references to the test
        with open(os.path.join(testprogram_dir, "test_utils.c"), "w") as f:
            f.write("""
            #include <stdio.h>
            
            // Test utility functions
            
            // Log test results to file
            void log_test_result(char* test_name, int result) {
                printf("Logging result for %s: %d\\n", test_name, result);
                // In a real implementation, this would write to a log file
            }
            
            // Get test limits
            typedef struct {
                float low_limit;
                float high_limit;
            } test_limits_t;
            
            test_limits_t get_test_limits(char* test_name) {
                test_limits_t limits;
                
                // Set default limits
                limits.low_limit = -1.0;
                limits.high_limit = 1.0;
                
                // Test-specific limits
                if (strcmp(test_name, "smps_ps_err") == 0) {
                    limits.low_limit = -0.05;
                    limits.high_limit = 0.05;
                }
                
                return limits;
            }
            """)
        
        # Create a Python file with the test name in various contexts
        with open(os.path.join(testprogram_dir, "smps_analyzer.py"), "w") as f:
            f.write("""
            # SMPS Power Supply Analysis Module
            
            class PowerSupplyAnalyzer:
                def __init__(self):
                    self.test_results = {}
                    self.smps_ps_err_threshold = 0.05
                
                def analyze_smps_ps_err(self, measurement_data):
                    \"\"\"
                    Analyze SMPS power supply error measurements
                    \"\"\"
                    print("Analyzing SMPS PS error data")
                    
                    # Process the measurement data
                    error_value = max(measurement_data)
                    
                    # Store result
                    self.test_results['smps_ps_err'] = error_value
                    
                    # Return pass/fail status
                    return error_value <= self.smps_ps_err_threshold
                
                def get_test_result(self, test_name):
                    \"\"\"
                    Get result for a specific test
                    \"\"\"
                    return self.test_results.get(test_name, None)
            
            # Utility functions
            def process_data_file(filename):
                # In a real implementation, this would read from a data file
                test_data = {
                    'smps_ps_err': [0.01, 0.02, 0.015, 0.03]
                }
                return test_data
            """)
    
    try:
        # Create the ATE test lookup tool
        lookup = ATETestLookup(testprogram_dir)
        
        # Scan files and create embeddings
        lookup.scan_files()
        lookup.create_embeddings()
        
        # Search for a specific test name
        test_name = "smps_ps_err"
        print(f"\nComprehensive search for test: {test_name}")
        
        results = lookup.find_test(test_name)
        
        # Display the most relevant files first
        print("\n--- MOST RELEVANT FILES (OVERALL RANKING) ---")
        for file_path, score in results['most_relevant_files']:
            print(f"File: {file_path}")
            print(f"Relevance Score: {score:.4f}")
            
            # Show where the test name appears in this file
            line_numbers = lookup.find_test_locations(file_path, test_name)
            if line_numbers:
                print(f"Test name appears on lines: {', '.join(map(str, line_numbers))}")
                
                # Show context for the first occurrence
                if line_numbers:
                    print("\nContext for first occurrence:")
                    context = lookup.extract_context(file_path, line_numbers[0])
                    print(context)
                    
            print("-" * 50)
        
        # Display detailed semantic matches for code blocks
        print("\n--- TOP CODE BLOCK MATCHES ---")
        for file_path, block_type, block_name, start_line, end_line, score in results['code_block_semantic_matches'][:5]:
            print(f"File: {file_path}")
            print(f"Block Type: {block_type}")
            print(f"Block Name: {block_name}")
            print(f"Lines: {start_line}-{end_line}")
            print(f"Similarity: {score:.4f}")
            
            # Show context
            print("\nContext:")
            context = lookup.extract_context(file_path, start_line)
            print(context)
            print("-" * 50)
        
        # Display exact matches
        print("\n--- EXACT STRING MATCHES ---")
        for file_path, count in results['exact_matches']:
            print(f"File: {file_path}")
            print(f"Occurrences: {count}")
            print("-" * 50)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()