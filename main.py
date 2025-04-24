import os
import re
import pickle
from typing import Dict, List, Tuple, Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ATETestLookup:
    def __init__(self, testprogram_dir: str):
        """
        Initialize the ATE Test Lookup tool with the directory containing testprogram files.
        
        Args:
            testprogram_dir: Path to the directory containing testprogram source files
        """
        self.testprogram_dir = testprogram_dir
        self.model = self._get_cached_model()
            
        self.file_info = []  # Will store (file_path, file_content)
        self.file_embeddings = None
        
        # Store info about code blocks for more specific lookup
        self.code_blocks = []  # Will store (file_path, block_type, block_name, block_content, start_line, end_line)
        self.code_block_embeddings = None
        
        # Element type weights
        self.element_weights = {
            'function': 3.0,      # Higher weight for function declarations
            'output_var': 2.5,    # Higher weight for output variables
            'comment': 2.0,       # Higher weight for comments
            'string': 2.0,        # Higher weight for string literals
            'class': 1.5,         # Medium weight for class declarations
            'variable': 0.8,      # Lower weight for general variables
            'operand_var': 0.5    # Lowest weight for operand variables
        }
    
    def _get_cached_model(self, model_name='all-MiniLM-L6-v2', cache_path='sentence_transformer_cache.pkl'):
        """
        Load the SentenceTransformer model from cache if available, otherwise initialize and cache it.
        
        Args:
            model_name: Name of the sentence transformer model to use
            cache_path: Path to store the cached model
            
        Returns:
            The loaded SentenceTransformer model
        """
        if os.path.exists(cache_path):
            print(f"Loading SenteceTransformer model from cache: '{cache_path}'")
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cached model: {e}")
                print("Will initialize a new model instead.")
        
        # If cache doesn't exist or couldn't be loaded, initialize the model
        print(f"Initializing SentenceTransformer model: '{model_name}'")
        try:
            # Import only when needed
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
            
            # Save the model to cache
            print(f"Saving model to cache: {cache_path}")
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(model, f)
            except Exception as e:
                print(f"Error caching model: {e}")
            
            return model
        except Exception as e:
            print(f"Error initializing SentenceTransformer: {e}")
            print("Please make sure you have installed the correct versions:")
            print("pip install sentence-transformers==2.2.2 transformers==4.30.0 torch==2.0.1")
            raise e
        
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
        
        # Extract variables (with role detection)
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
            # C/C++ function pattern - simplified, without recursive pattern matching
            # Look for function declarations with opening brace
            pattern = r'(?:\w+(?:\s*[*&])?)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*\{'
            
            for match in re.finditer(pattern, content):
                func_name = match.group(1)
                start_pos = match.start()
                
                # Skip if the function name is a C/C++ keyword
                if func_name in ['if', 'for', 'while', 'switch']:
                    continue
                
                # Find the matching closing brace
                # This is a simple approach that might not handle all nested braces correctly
                open_braces = 1
                pos = match.end()
                while open_braces > 0 and pos < len(content):
                    if content[pos] == '{':
                        open_braces += 1
                    elif content[pos] == '}':
                        open_braces -= 1
                    pos += 1
                
                if open_braces == 0:
                    func_content = content[start_pos:pos]
                    
                    # Calculate line numbers
                    start_line = content[:start_pos].count('\n') + 1
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
            pattern = r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)[^{]*\{'
            
            for match in re.finditer(pattern, content):
                class_name = match.group(1)
                start_pos = match.start()
                
                # Find the matching closing brace
                open_braces = 1
                pos = match.end()
                while open_braces > 0 and pos < len(content):
                    if content[pos] == '{':
                        open_braces += 1
                    elif content[pos] == '}':
                        open_braces -= 1
                    pos += 1
                
                if open_braces == 0:
                    class_content = content[start_pos:pos]
                    
                    # Calculate line numbers
                    start_line = content[:start_pos].count('\n') + 1
                    end_line = start_line + class_content.count('\n')
                    
                    self.code_blocks.append((file_path, 'class', class_name, class_content, start_line, end_line))
    
    def _extract_variables(self, file_path: str, content: str):
        """
        Extract variable definitions from file content with role detection (output vs operand).
        """
        # Process line by line to better determine context
        lines = content.split('\n')
        
        # C/C++ output variable patterns (typically on the left side of assignments)
        c_output_pattern = r'(?:\w+(?:\s*[*&])?)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*'
        c_return_pattern = r'return\s+([a-zA-Z_][a-zA-Z0-9_]*);'
        
        # Python output variable patterns
        py_output_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*'
        py_return_pattern = r'return\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        
        # Operand patterns (variables used on right side of expressions)
        operand_pattern = r'=\s*[^=]*?\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        
        # Common skip keywords
        skip_keywords = ['if', 'for', 'while', 'return', 'switch', 'case', 
                         'break', 'continue', 'else', 'elif', 'def', 'class', 
                         'import', 'from', 'True', 'False', 'None']
        
        for i, line in enumerate(lines):
            # Skip comments
            if line.strip().startswith('//') or line.strip().startswith('#'):
                continue
                
            # Extract output variables in C/C++
            if file_path.endswith(('.c', '.cpp', '.h')):
                # Find variable declarations with assignments
                for match in re.finditer(c_output_pattern, line):
                    var_name = match.group(1)
                    if var_name not in skip_keywords:
                        var_content = match.group(0) + line[match.end():].split(';')[0]
                        self.code_blocks.append((file_path, 'output_var', var_name, var_content, i+1, i+1))
                
                # Find return variables
                for match in re.finditer(c_return_pattern, line):
                    var_name = match.group(1)
                    if var_name not in skip_keywords:
                        var_content = match.group(0)
                        self.code_blocks.append((file_path, 'output_var', var_name, var_content, i+1, i+1))
                
            # Extract output variables in Python
            if file_path.endswith('.py'):
                # Find variable assignments
                for match in re.finditer(py_output_pattern, line):
                    var_name = match.group(1)
                    if var_name not in skip_keywords:
                        var_content = match.group(0) + line[match.end():]
                        self.code_blocks.append((file_path, 'output_var', var_name, var_content, i+1, i+1))
                
                # Find return variables
                for match in re.finditer(py_return_pattern, line):
                    var_name = match.group(1)
                    if var_name not in skip_keywords:
                        var_content = match.group(0)
                        self.code_blocks.append((file_path, 'output_var', var_name, var_content, i+1, i+1))
            
            # Extract operand variables (with lower weight)
            for match in re.finditer(operand_pattern, line):
                var_name = match.group(1)
                if var_name not in skip_keywords:
                    var_content = line
                    self.code_blocks.append((file_path, 'operand_var', var_name, var_content, i+1, i+1))
            
            # Extract defined variables that might not be assigned yet
            if file_path.endswith(('.c', '.cpp', '.h')):
                var_pattern = r'(?:\w+(?:\s*[*&])?)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*;'
                for match in re.finditer(var_pattern, line):
                    var_name = match.group(1)
                    if var_name not in skip_keywords:
                        var_content = match.group(0)
                        self.code_blocks.append((file_path, 'variable', var_name, var_content, i+1, i+1))
    
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
            
            # Python docstrings
            for match in re.finditer(r'"""(.+?)"""', content, re.DOTALL):
                comment = match.group(0)
                start_line = content[:match.start()].count('\n') + 1
                end_line = start_line + comment.count('\n')
                self.code_blocks.append((file_path, 'comment', '', comment, start_line, end_line))
            
            for match in re.finditer(r"'''(.+?)'''", content, re.DOTALL):
                comment = match.group(0)
                start_line = content[:match.start()].count('\n') + 1
                end_line = start_line + comment.count('\n')
                self.code_blocks.append((file_path, 'comment', '', comment, start_line, end_line))
    
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
                
                # Skip very short strings
                if len(string_content) < 3:  # Just quotes with maybe one character
                    continue
                    
                line_num = content[:match.start()].count('\n') + 1
                self.code_blocks.append((file_path, 'string', '', string_content, line_num, line_num))
    
    def create_embeddings(self):
        """
        Create embeddings for files and code blocks, with different weighting for different code elements.
        """
        if not self.file_info:
            print("No files found. Please scan files first.")
            return
        
        print("Creating embeddings for files...")
        file_texts = [os.path.basename(path) + " " + content for path, content in self.file_info]
        self.file_embeddings = self.model.encode(file_texts)
        
        if self.code_blocks:
            print("Creating embeddings for code blocks...")
            # Create a weighted representation for code blocks based on their type
            block_texts = []
            for _, block_type, block_name, content, _, _ in self.code_blocks:
                # Apply different weights based on element type by repeating key elements
                weight = self.element_weights.get(block_type, 1.0)
                
                # For higher weighted elements, repeat their name to emphasize importance
                repeat_count = int(weight * 2)
                emphasized_name = " ".join([block_name] * repeat_count) if block_name else ""
                
                # Create a weighted text representation
                weighted_text = f"{block_type} {emphasized_name} {content}"
                block_texts.append(weighted_text)
                
            self.code_block_embeddings = self.model.encode(block_texts)
        
        print("Embeddings created successfully")
    
    def find_test(self, test_name: str, test_number: str = None, top_n: int = 5) -> Dict:
        """
        Comprehensive search for the test name across all files and code blocks.
        
        Args:
            test_name: The ATE test name to look up
            test_number: The test number to look for (optional)
            top_n: Number of top matches to return
            
        Returns:
            Dictionary with search results organized by type
        """
        results = {
            'file_semantic_matches': [],
            'code_block_semantic_matches': [],
            'exact_matches': [],
            'regex_matches': [],
            'test_number_matches': []
        }
        
        # Clean test name to focus on core identifier
        cleaned_test_name = self._clean_test_name(test_name)
        
        # 1. Semantic search on whole files
        if self.file_embeddings is not None:
            test_embedding = self.model.encode([test_name])
            similarities = cosine_similarity(test_embedding, self.file_embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:top_n]
            
            for idx in top_indices:
                file_path, _ = self.file_info[idx]
                score = similarities[idx]
                results['file_semantic_matches'].append((file_path, score))
        
        # 2. Semantic search on code blocks with type-based weighting
        if self.code_block_embeddings is not None:
            test_embedding = self.model.encode([test_name])
            similarities = cosine_similarity(test_embedding, self.code_block_embeddings)[0]
            
            # Apply post-scoring weight adjustment based on block type
            weighted_similarities = []
            for i, (_, block_type, _, _, _, _) in enumerate(self.code_blocks):
                weight = self.element_weights.get(block_type, 1.0)
                weighted_similarities.append((i, similarities[i] * weight))
            
            # Sort by weighted similarity
            top_weighted = sorted(weighted_similarities, key=lambda x: x[1], reverse=True)[:top_n]
            top_indices = [idx for idx, _ in top_weighted]
            
            for idx in top_indices:
                file_path, block_type, block_name, _, start_line, end_line = self.code_blocks[idx]
                score = similarities[idx]
                weighted_score = score * self.element_weights.get(block_type, 1.0)
                results['code_block_semantic_matches'].append(
                    (file_path, block_type, block_name, start_line, end_line, weighted_score)
                )
        
        # 3. Exact string matching
        for file_path, content in self.file_info:
            occurrences = []
            
            # Count exact matches of the test name
            if test_name in content:
                count = content.count(test_name)
                # Also locate line numbers for the matches
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if test_name in line:
                        occurrences.append(i + 1)
                
                results['exact_matches'].append((file_path, count, occurrences))
        
        # 4. Regex pattern matching for partial matches or variations
        escaped_test_name = re.escape(test_name)
        pattern = r'[\w_]*' + escaped_test_name.replace('_', r'[\W_]?') + r'[\w_]*'
        
        for file_path, content in self.file_info:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                # Find line numbers for matches
                lines = content.split('\n')
                match_lines = []
                for i, line in enumerate(lines):
                    for match in matches:
                        if match in line:
                            match_lines.append(i + 1)
                            break
                
                results['regex_matches'].append((file_path, len(matches), match_lines, matches[:5]))
        
        # 5. Test number matching (new functionality)
        if test_number:
            for file_path, content in self.file_info:
                if test_number in content:
                    # Count occurrences and get line numbers
                    count = content.count(test_number)
                    lines = content.split('\n')
                    line_numbers = [i+1 for i, line in enumerate(lines) if test_number in line]
                    results['test_number_matches'].append((file_path, count, line_numbers))
        
        # Sort results by relevance
        results['file_semantic_matches'].sort(key=lambda x: x[1], reverse=True)
        results['code_block_semantic_matches'].sort(key=lambda x: x[5], reverse=True)
        results['exact_matches'].sort(key=lambda x: x[1], reverse=True)
        results['regex_matches'].sort(key=lambda x: x[1], reverse=True)
        results['test_number_matches'].sort(key=lambda x: x[1], reverse=True)
        
        # Calculate overall most relevant files with improved weighting
        file_scores = {}
        
        # Add scores from semantic file matches
        for file_path, score in results['file_semantic_matches']:
            file_scores[file_path] = file_scores.get(file_path, 0) + score * 1.5
        
        # Add scores from exact matches with higher weight
        for file_path, count, _ in results['exact_matches']:
            file_scores[file_path] = file_scores.get(file_path, 0) + count * 2.5
        
        # Add scores from regex matches
        for file_path, count, _, _ in results['regex_matches']:
            file_scores[file_path] = file_scores.get(file_path, 0) + count * 0.5
        
        # Add scores from code block matches with element type weights
        for file_path, block_type, block_name, _, _, score in results['code_block_semantic_matches']:
            element_boost = 1.0
            
            if block_name and (block_name == test_name or block_name == cleaned_test_name):
                element_boost = 3.0
            
            file_scores[file_path] = file_scores.get(file_path, 0) + score * element_boost
        
        # NEW: Significant boost for files that contain both test name and test number
        if test_number:
            test_number_files = {file_path for file_path, _, _ in results['test_number_matches']}
            test_name_files = {file_path for file_path, _, _ in results['exact_matches']}
            
            # Files that contain both get a major boost
            files_with_both = test_number_files & test_name_files
            for file_path in files_with_both:
                file_scores[file_path] = file_scores.get(file_path, 0) + 10.0  # Large boost
            
            # Files with just the test number get a moderate boost
            files_with_number_only = test_number_files - test_name_files
            for file_path in files_with_number_only:
                file_scores[file_path] = file_scores.get(file_path, 0) + 3.0
        
        # Sort files by overall relevance score
        most_relevant_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        results['most_relevant_files'] = most_relevant_files[:top_n]
        
        return results
    
    def _clean_test_name(self, test_name: str) -> str:
        """
        Clean the test name to focus on core identifier by removing common prefixes/suffixes.
        
        Args:
            test_name: Original test name
            
        Returns:
            Cleaned test name
        """
        # Remove common ATE test prefixes and suffixes
        prefixes = ['test_', 'v_', 'f_']
        suffixes = ['_test', '_val', '_pdp']
        
        cleaned = test_name
        for prefix in prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):]
                
        for suffix in suffixes:
            if cleaned.endswith(suffix):
                cleaned = cleaned[:-len(suffix)]
                
        return cleaned
    
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
    
    def analyze_test_usage(self, file_path: str, test_name: str) -> Dict:
        """
        Analyze how the test is used in the file - as function name, variable, argument, etc.
        
        Args:
            file_path: Path to the file
            test_name: Test name to search for
            
        Returns:
            Dictionary with analysis results
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            analysis = {
                'is_function': False,
                'is_output_var': False,
                'is_operand': False,
                'in_comment': False,
                'in_string': False,
                'usage_count': content.count(test_name),
                'important_lines': []
            }
            
            lines = content.split('\n')
            
            # Extended regex patterns to identify usage context
            func_pattern = fr'(?:def|int|void|float|double|char)\s+{re.escape(test_name)}\s*\('
            output_pattern = fr'{re.escape(test_name)}\s*='
            operand_pattern = fr'[^=]=[^=]*{re.escape(test_name)}[^=]*;'
            comment_pattern = fr'(?://|#|/\*)[^\n]*{re.escape(test_name)}'
            string_pattern = fr'["\']{re.escape(test_name)}["\']'
            
            for i, line in enumerate(lines):
                if re.search(func_pattern, line):
                    analysis['is_function'] = True
                    analysis['important_lines'].append((i+1, 'function', line))
                    
                if re.search(output_pattern, line):
                    analysis['is_output_var'] = True
                    analysis['important_lines'].append((i+1, 'output_var', line))
                    
                if re.search(operand_pattern, line):
                    analysis['is_operand'] = True
                    analysis['important_lines'].append((i+1, 'operand', line))
                    
                if re.search(comment_pattern, line):
                    analysis['in_comment'] = True
                    analysis['important_lines'].append((i+1, 'comment', line))
                    
                if re.search(string_pattern, line):
                    analysis['in_string'] = True
                    analysis['important_lines'].append((i+1, 'string', line))
                    
            return analysis
        
        except Exception as e:
            print(f"Error analyzing file {file_path}: {e}")
            return {
                'error': str(e),
                'usage_count': 0
            }

def main():
    # Replace with your actual testprogram directory
    testprogram_dir = "C:/Users/rejav/Documents/SilTest-side-projects/GitHub/ATE_Debug_Assisntant-Semantic_Mapper/sample_testprogram_elmos_v52241b"
    
    if not os.path.exists(testprogram_dir):
        print("Specified testprogram directory does not exist.")
    
    try:
        # Search for a specific test name and number
        test_name = "v_source_off_hys"
        test_number = "8865"  # Example test number
        print(f"\nComprehensive search for test: '{test_name}' (number: {test_number})")

        # Create the ATE test lookup tool
        lookup = ATETestLookup(testprogram_dir)
        
        # Scan files and create embeddings
        lookup.scan_files()
        lookup.create_embeddings()
        
        results = lookup.find_test(test_name, test_number)
        
        # Display the most relevant files first
        print("\n--- MOST RELEVANT FILES (OVERALL RANKING) ---")
        for file_path, score in results['most_relevant_files']:
            print(f"File: {file_path}")
            print(f"Relevance Score: {score:.4f}")
            
            # Show where the test name appears in this file
            line_numbers = lookup.find_test_locations(file_path, test_name)
            if line_numbers:
                print(f"Test name appears on lines: {', '.join(map(str, line_numbers))}")
            
            # Show where the test number appears in this file
            if test_number:
                number_lines = lookup.find_test_locations(file_path, test_number)
                if number_lines:
                    print(f"Test number appears on lines: {', '.join(map(str, number_lines))}")
                
            # Analyze how the test is used in this file
            usage_analysis = lookup.analyze_test_usage(file_path, test_name)
            
            # Print detailed usage information
            usage_types = []
            if usage_analysis['is_function']:
                usage_types.append("FUNCTION NAME")
            if usage_analysis['is_output_var']:
                usage_types.append("OUTPUT VARIABLE")
            if usage_analysis['in_comment']:
                usage_types.append("IN COMMENTS")
            if usage_analysis['in_string']:
                usage_types.append("IN STRINGS")
            if usage_analysis['is_operand']:
                usage_types.append("AS OPERAND")
                
            if usage_types:
                print(f"Used as: {', '.join(usage_types)}")
            
            # Show context for the most important usage
            if usage_analysis['important_lines']:
                priority_order = {'function': 1, 'output_var': 2, 'comment': 3, 'string': 4, 'operand': 5}
                important_line = sorted(usage_analysis['important_lines'], 
                                       key=lambda x: priority_order.get(x[1], 99))[0]
                
                line_num, usage_type, _ = important_line
                print(f"\nContext for {usage_type.upper()} usage on line {line_num}:")
                context = lookup.extract_context(file_path, line_num)
                print(context)
            else:
                # If no detailed analysis is available, show context for the first occurrence
                if line_numbers:
                    print("\nContext for first occurrence:")
                    context = lookup.extract_context(file_path, line_numbers[0])
                    print(context)
                    
            print("-" * 50)
        
        # Display test number matches if provided
        if test_number and results['test_number_matches']:
            print("\n--- TEST NUMBER MATCHES ---")
            for file_path, count, line_numbers in results['test_number_matches'][:5]:
                print(f"File: {file_path}")
                print(f"Occurrences: {count}")
                print(f"Line numbers: {', '.join(map(str, line_numbers[:10]))}" + 
                     (", ..." if len(line_numbers) > 10 else ""))
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
        for file_path, count, line_numbers in results['exact_matches']:
            print(f"File: {file_path}")
            print(f"Occurrences: {count}")
            print(f"Line numbers: {', '.join(map(str, line_numbers[:10]))}" + 
                 (", ..." if len(line_numbers) > 10 else ""))
            print("-" * 50)
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()