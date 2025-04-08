import os
import re
from typing import Dict, List, Tuple
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
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.function_info = []  # Will store (file_path, function_name, function_content)
        self.function_embeddings = None
        
    def scan_files(self, file_extensions: List[str] = ['.c', '.cpp', '.h', '.py']):
        """
        Scan all files with given extensions in the testprogram directory and extract functions.
        
        Args:
            file_extensions: List of file extensions to scan
        """
        print(f"Scanning files in {self.testprogram_dir}...")
        
        for root, _, files in os.walk(self.testprogram_dir):
            for file in files:
                if any(file.endswith(ext) for ext in file_extensions):
                    file_path = os.path.join(root, file)
                    self._extract_functions_from_file(file_path)
        
        print(f"Found {len(self.function_info)} functions across all files")
        
    def _extract_functions_from_file(self, file_path: str):
        """
        Extract function definitions from a source file.
        
        Args:
            file_path: Path to the source file
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Simple regex patterns for function detection - may need refinement for specific languages
            # For C/C++ style functions
            c_pattern = r'(\w+)\s+(\w+)\s*\([^)]*\)\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
            # For Python functions
            py_pattern = r'def\s+(\w+)\s*\([^)]*\):\s*((?:.+\n)+?(?:\s+.+\n)*)'
            
            if file_path.endswith(('.py')):
                matches = re.finditer(py_pattern, content)
                for match in matches:
                    function_name = match.group(1)
                    function_content = match.group(0)
                    self.function_info.append((file_path, function_name, function_content))
            else:
                matches = re.finditer(c_pattern, content)
                for match in matches:
                    function_name = match.group(2)
                    function_content = match.group(0)
                    self.function_info.append((file_path, function_name, function_content))
                
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    def create_embeddings(self):
        """
        Create embeddings for all extracted functions.
        """
        if not self.function_info:
            print("No functions found. Please scan files first.")
            return
        
        print("Creating embeddings for functions...")
        texts = [f"{func_name} {func_content}" for _, func_name, func_content in self.function_info]
        self.function_embeddings = self.model.encode(texts)
        print("Embeddings created successfully")
    
    def find_test(self, test_name: str, top_n: int = 5) -> List[Tuple[str, str, float]]:
        """
        Find functions likely to contain the given test name.
        
        Args:
            test_name: The ATE test name to look up
            top_n: Number of top matches to return
            
        Returns:
            List of tuples (file_path, function_name, similarity_score)
        """
        if self.function_embeddings is None:
            print("No embeddings available. Please create embeddings first.")
            return []
        
        # Create embedding for the test name
        test_embedding = self.model.encode([test_name])
        
        # Calculate similarities
        similarities = cosine_similarity(test_embedding, self.function_embeddings)[0]
        
        # Get indices of top N similarities
        top_indices = np.argsort(similarities)[::-1][:top_n]
        
        # Return file paths, function names and similarity scores
        results = []
        for idx in top_indices:
            file_path, function_name, _ = self.function_info[idx]
            score = similarities[idx]
            results.append((file_path, function_name, score))
            
        return results
    
    def search_by_regex(self, test_name: str) -> List[Tuple[str, str]]:
        """
        Alternative search method using direct regex pattern matching.
        
        Args:
            test_name: The ATE test name to look up
            
        Returns:
            List of tuples (file_path, function_name) where test_name appears
        """
        results = []
        for file_path, function_name, function_content in self.function_info:
            if re.search(re.escape(test_name), function_content):
                results.append((file_path, function_name))
        return results

# Usage example
def main():
    # Replace with your actual testprogram directory
    testprogram_dir = "./testprogram"
    
    # Create the ATE test lookup tool
    lookup = ATETestLookup(testprogram_dir)
    
    # Scan files and create embeddings
    lookup.scan_files()
    lookup.create_embeddings()
    
    # Search for a specific test name
    test_name = "smps_ps_err"
    print(f"\nSearching for test: {test_name}")
    
    # Using embeddings (semantic search)
    results = lookup.find_test(test_name)
    print("\nTop matches (using embeddings):")
    for file_path, function_name, score in results:
        print(f"File: {file_path}")
        print(f"Function: {function_name}")
        print(f"Similarity: {score:.4f}")
        print("-" * 50)
    
    # Using regex (direct pattern matching)
    print("\nDirect matches (using regex):")
    direct_matches = lookup.search_by_regex(test_name)
    for file_path, function_name in direct_matches:
        print(f"File: {file_path}")
        print(f"Function: {function_name}")
        print("-" * 50)

if __name__ == "__main__":
    main()