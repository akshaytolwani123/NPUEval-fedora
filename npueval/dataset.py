# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator

class NPUEvalDataset:
    """Dataset class for NPUEval benchmark data."""
    
    def __init__(self, jsonl_path: str = None):
        """
        Initialize the dataset.
        
        Args:
            jsonl_path: Path to the JSONL dataset file
        """
        if jsonl_path is None:
            # Get path relative to this module's location
            package_dir = Path(__file__).parent.parent
            self.jsonl_path = package_dir / "dataset" / "npueval.jsonl"
        else:
            self.jsonl_path = Path(jsonl_path)
        
        self._tests: Optional[List[Dict[str, Any]]] = None
    
    def load(self) -> List[Dict[str, Any]]:
        """
        Load the dataset from file.
        
        Returns:
            List of test cases as dictionaries
        """
        if not os.path.exists(self.jsonl_path):
            raise FileNotFoundError(f"Dataset file not found: {self.jsonl_path}")
        
        with open(self.jsonl_path, 'r') as f:
            self._tests = [json.loads(line) for line in f]
        
        return self._tests
    
    @property
    def tests(self) -> List[Dict[str, Any]]:
        """Get all test cases, loading if necessary."""
        if self._tests is None:
            self.load()
        return self._tests
    
    def __len__(self) -> int:
        """Return number of test cases."""
        return len(self.tests)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a specific test case by index."""
        return self.tests[index]
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Make the dataset iterable."""
        return iter(self.tests)
    
    def get_by_name(self, kernel_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a test case by its name.
        
        Args:
            kernel_name: The name of the test case to retrieve
            
        Returns:
            Test case dictionary or None if not found
        """
        for test in self.tests:
            if test.get('kernel_name') == kernel_name:
                return test
        return None
    
    def reload(self) -> List[Dict[str, Any]]:
        """Force reload the dataset from file."""
        self._tests = None
        return self.load()


# Create a default instance
dataset = NPUEvalDataset()