#!/usr/bin/env python3
"""
Performance-Focused Commit Analyzer

Analyzes commit extractions to identify performance optimizations and extract
quantitative performance claims for test generation.

This replaces the broken classification approach with performance-first analysis.
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PerformanceClaimExtraction:
    """Extracted performance claims from a commit"""
    # Basic commit info
    commit_hash: str
    message: str
    
    # Performance analysis
    has_performance_intent: bool
    performance_claims: List[str]  # e.g., ["2.8x speedup", "reduce memory by 30%"]
    optimization_type: str  # "kernel", "algorithm", "memory", "parallelization", "other"
    optimization_keywords: List[str]  # ["triton", "cuda", "block_size", etc.]
    
    # Quantitative extractions
    expected_speedup: Optional[float]  # e.g., 2.8 from "2.8x speedup"
    performance_metric: Optional[str]  # "speedup", "memory_reduction", "latency"
    
    # Context for test generation
    affected_functions: List[str]  # Functions that were optimized
    optimization_domain: str  # "matrix_ops", "kernels", "memory", "general"
    test_complexity: str  # "simple", "medium", "complex"
    
    # Technical details
    file_types: List[str]  # [".cu", ".py", ".cpp"] to understand tech stack
    has_cuda_gpu_code: bool
    has_benchmarks: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'commit_hash': self.commit_hash,
            'message': self.message,
            'has_performance_intent': self.has_performance_intent,
            'performance_claims': self.performance_claims,
            'optimization_type': self.optimization_type,
            'optimization_keywords': self.optimization_keywords,
            'expected_speedup': self.expected_speedup,
            'performance_metric': self.performance_metric,
            'affected_functions': self.affected_functions,
            'optimization_domain': self.optimization_domain,
            'test_complexity': self.test_complexity,
            'file_types': self.file_types,
            'has_cuda_gpu_code': self.has_cuda_gpu_code,
            'has_benchmarks': self.has_benchmarks,
        }


class PerformanceCommitAnalyzer:
    """Analyzes commits to extract performance optimization information"""
    
    def __init__(self):
        # Performance-indicating keywords in commit messages
        self.performance_keywords = {
            'explicit': [
                'speedup', 'faster', 'optimize', 'optimization', 'performance',
                'efficient', 'efficiency', 'accelerate', 'improve', 'boost'
            ],
            'kernel': [
                'kernel', 'cuda', 'triton', 'gpu', 'block', 'grid', 'thread',
                'warp', 'shared', 'memory', 'bandwidth', 'throughput'
            ],
            'algorithm': [
                'algorithm', 'complexity', 'cache', 'parallel', 'vectorize',
                'batch', 'fusion', 'inline', 'unroll'
            ],
            'memory': [
                'memory', 'alloc', 'pool', 'buffer', 'copy', 'bandwidth',
                'cache', 'prefetch', 'streaming'
            ]
        }
        
        # Patterns for extracting quantitative claims
        self.speedup_patterns = [
            r'(\d+(?:\.\d+)?)\s*[x×]\s*(?:speedup|faster|improvement)',
            r'avg\s+(\d+(?:\.\d+)?)\s*[x×]\s*speedup',
            r'up\s+to\s+(\d+(?:\.\d+)?)\s*[x×]\s*faster',
            r'(\d+(?:\.\d+)?)\s*[x×]\s*performance\s+(?:gain|improvement)',
        ]
        
        self.percentage_patterns = [
            r'(\d+)%\s*(?:faster|speedup|improvement|reduction)',
            r'reduce.*?by\s+(\d+)%',
            r'improve.*?by\s+(\d+)%',
        ]
    
    def analyze_commit(self, commit_data: Dict[str, Any]) -> PerformanceClaimExtraction:
        """Analyze a single commit for performance optimizations"""
        
        commit_hash = commit_data.get('commit_hash', '')
        message = commit_data.get('message', '')
        files_changed = commit_data.get('files_changed', [])
        csv_metadata = commit_data.get('csv_metadata', {})
        
        # Extract basic performance intent
        has_performance_intent = self._detect_performance_intent(message, csv_metadata)
        performance_claims = self._extract_performance_claims(message)
        optimization_type = self._classify_optimization_type(message, files_changed)
        optimization_keywords = self._extract_optimization_keywords(message, files_changed)
        
        # Extract quantitative claims
        expected_speedup, performance_metric = self._extract_quantitative_claims(message)
        
        # Analyze technical context
        affected_functions = self._extract_affected_functions(files_changed)
        optimization_domain = self._determine_optimization_domain(message, files_changed)
        test_complexity = self._assess_test_complexity(files_changed, optimization_type)
        
        # Technical details
        file_types = self._get_file_types(files_changed)
        has_cuda_gpu_code = self._detect_cuda_gpu_code(files_changed)
        has_benchmarks = csv_metadata.get('json_has_benchmarks', 'FALSE') == 'TRUE'
        
        return PerformanceClaimExtraction(
            commit_hash=commit_hash,
            message=message,
            has_performance_intent=has_performance_intent,
            performance_claims=performance_claims,
            optimization_type=optimization_type,
            optimization_keywords=optimization_keywords,
            expected_speedup=expected_speedup,
            performance_metric=performance_metric,
            affected_functions=affected_functions,
            optimization_domain=optimization_domain,
            test_complexity=test_complexity,
            file_types=file_types,
            has_cuda_gpu_code=has_cuda_gpu_code,
            has_benchmarks=has_benchmarks,
        )
    
    def _detect_performance_intent(self, message: str, csv_metadata: Dict) -> bool:
        """Detect if commit has performance optimization intent"""
        message_lower = message.lower()
        
        # Check explicit performance keywords
        for category, keywords in self.performance_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return True
        
        # Check CSV metadata category
        category = csv_metadata.get('category', '').lower()
        if 'kernel' in category:
            return True
            
        # Check for quantitative claims
        if self._extract_quantitative_claims(message)[0] is not None:
            return True
            
        return False
    
    def _extract_performance_claims(self, message: str) -> List[str]:
        """Extract explicit performance claims from commit message"""
        claims = []
        
        # Look for speedup claims
        for pattern in self.speedup_patterns:
            matches = re.finditer(pattern, message, re.IGNORECASE)
            for match in matches:
                claims.append(f"{match.group(1)}x speedup")
        
        # Look for percentage improvements
        for pattern in self.percentage_patterns:
            matches = re.finditer(pattern, message, re.IGNORECASE)
            for match in matches:
                claims.append(f"{match.group(1)}% improvement")
        
        # Look for general optimization claims
        optimization_phrases = [
            r'optimize.*?(?:for|beam search|attention|memory)',
            r'improve.*?performance',
            r'reduce.*?(?:latency|memory|time)',
            r'faster.*?(?:execution|inference|training)',
        ]
        
        for pattern in optimization_phrases:
            matches = re.finditer(pattern, message, re.IGNORECASE)
            for match in matches:
                claims.append(match.group(0))
        
        return claims
    
    def _classify_optimization_type(self, message: str, files_changed: List[Dict]) -> str:
        """Classify the type of optimization"""
        message_lower = message.lower()
        
        # Check for kernel optimizations
        kernel_indicators = ['kernel', 'cuda', 'triton', 'gpu', 'block size']
        if any(indicator in message_lower for indicator in kernel_indicators):
            return 'kernel'
        
        # Check file extensions for kernel code
        cuda_extensions = ['.cu', '.cuh', '.hip']
        for file_change in files_changed:
            file_path = file_change.get('file_path', '')
            if any(file_path.endswith(ext) for ext in cuda_extensions):
                return 'kernel'
        
        # Check for algorithm optimizations
        algo_indicators = ['algorithm', 'complexity', 'cache', 'parallel']
        if any(indicator in message_lower for indicator in algo_indicators):
            return 'algorithm'
        
        # Check for memory optimizations
        memory_indicators = ['memory', 'alloc', 'copy', 'bandwidth']
        if any(indicator in message_lower for indicator in memory_indicators):
            return 'memory'
        
        # Check for parallelization
        parallel_indicators = ['parallel', 'multi-gpu', 'distributed', 'thread']
        if any(indicator in message_lower for indicator in parallel_indicators):
            return 'parallelization'
        
        return 'other'
    
    def _extract_optimization_keywords(self, message: str, files_changed: List[Dict]) -> List[str]:
        """Extract optimization-related keywords from message and code"""
        keywords = set()
        message_lower = message.lower()
        
        # Extract from message
        all_keywords = []
        for category, kw_list in self.performance_keywords.items():
            all_keywords.extend(kw_list)
        
        for keyword in all_keywords:
            if keyword in message_lower:
                keywords.add(keyword)
        
        # Extract from file paths and diffs
        for file_change in files_changed:
            file_path = file_change.get('file_path', '').lower()
            diff = file_change.get('diff', '').lower()
            
            # Look for technical keywords in paths
            tech_keywords = ['triton', 'cuda', 'kernel', 'cache', 'attention', 'layer']
            for keyword in tech_keywords:
                if keyword in file_path or keyword in diff:
                    keywords.add(keyword)
        
        return sorted(list(keywords))
    
    def _extract_quantitative_claims(self, message: str) -> Tuple[Optional[float], Optional[str]]:
        """Extract quantitative performance claims"""
        
        # Try to extract speedup values
        for pattern in self.speedup_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                try:
                    speedup = float(match.group(1))
                    return speedup, 'speedup'
                except ValueError:
                    continue
        
        # Try to extract percentage improvements
        for pattern in self.percentage_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                try:
                    percentage = float(match.group(1))
                    # Convert percentage to speedup multiplier
                    speedup = 1.0 + (percentage / 100.0)
                    return speedup, 'percentage_improvement'
                except ValueError:
                    continue
        
        return None, None
    
    def _extract_affected_functions(self, files_changed: List[Dict]) -> List[str]:
        """Extract function names that were modified"""
        functions = set()
        
        for file_change in files_changed:
            diff = file_change.get('diff', '')
            
            # Look for function definitions in diffs
            # Python functions
            py_functions = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', diff)
            functions.update(py_functions)
            
            # C/C++/CUDA functions
            c_functions = re.findall(r'^\s*(?:__global__|__device__|__host__)?\s*\w+\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', diff, re.MULTILINE)
            functions.update(c_functions)
            
            # Class methods
            methods = re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(self', diff)
            functions.update(methods)
        
        return sorted(list(functions))
    
    def _determine_optimization_domain(self, message: str, files_changed: List[Dict]) -> str:
        """Determine the domain of optimization"""
        message_lower = message.lower()
        
        # Check file paths for domain clues
        file_paths = [fc.get('file_path', '').lower() for fc in files_changed]
        all_paths = ' '.join(file_paths)
        
        # Matrix operations domain
        if any(term in message_lower or term in all_paths for term in 
               ['matmul', 'mm', 'scaled_mm', 'attention', 'linear']):
            return 'matrix_ops'
        
        # Kernel domain
        if any(term in message_lower or term in all_paths for term in 
               ['kernel', 'cuda', 'triton', 'gpu']):
            return 'kernels'
        
        # Memory domain
        if any(term in message_lower or term in all_paths for term in 
               ['memory', 'cache', 'alloc', 'copy']):
            return 'memory'
        
        # Model/inference domain
        if any(term in message_lower or term in all_paths for term in 
               ['model', 'inference', 'runner', 'engine']):
            return 'inference'
        
        return 'general'
    
    def _assess_test_complexity(self, files_changed: List[Dict], optimization_type: str) -> str:
        """Assess the complexity of test needed"""
        
        # Count lines changed
        total_lines_changed = sum(
            fc.get('lines_added', 0) + fc.get('lines_removed', 0) 
            for fc in files_changed
        )
        
        # Kernel optimizations are typically complex
        if optimization_type == 'kernel':
            return 'complex'
        
        # Large changes need complex tests
        if total_lines_changed > 100:
            return 'complex'
        elif total_lines_changed > 20:
            return 'medium'
        else:
            return 'simple'
    
    def _get_file_types(self, files_changed: List[Dict]) -> List[str]:
        """Get unique file extensions from changed files"""
        extensions = set()
        
        for file_change in files_changed:
            file_path = file_change.get('file_path', '')
            if '.' in file_path:
                ext = '.' + file_path.split('.')[-1]
                extensions.add(ext)
        
        return sorted(list(extensions))
    
    def _detect_cuda_gpu_code(self, files_changed: List[Dict]) -> bool:
        """Detect if changes involve CUDA/GPU code"""
        gpu_indicators = ['.cu', '.cuh', '.hip']
        
        for file_change in files_changed:
            file_path = file_change.get('file_path', '')
            diff = file_change.get('diff', '').lower()
            
            # Check file extensions
            if any(file_path.endswith(ext) for ext in gpu_indicators):
                return True
            
            # Check for GPU keywords in diff
            gpu_keywords = ['cuda', '__global__', '__device__', '__shared__', 'threadidx', 'blockidx']
            if any(keyword in diff for keyword in gpu_keywords):
                return True
        
        return False


def analyze_commit_extractions(extractions_dir: str, output_file: str = None) -> List[PerformanceClaimExtraction]:
    """Analyze all commit extractions for performance optimizations"""
    
    analyzer = PerformanceCommitAnalyzer()
    extractions_path = Path(extractions_dir)
    
    results = []
    
    # Process all JSON files except summary
    for json_file in extractions_path.glob("*.json"):
        if json_file.name == "extraction_summary.json":
            continue
        
        try:
            with open(json_file, 'r') as f:
                commit_data = json.load(f)
            
            analysis = analyzer.analyze_commit(commit_data)
            results.append(analysis)
            
            print(f"✓ Analyzed {json_file.name}: "
                  f"Performance intent: {analysis.has_performance_intent}, "
                  f"Type: {analysis.optimization_type}, "
                  f"Claims: {len(analysis.performance_claims)}")
            
        except Exception as e:
            print(f"✗ Failed to analyze {json_file.name}: {e}")
    
    # Save results if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    # Print summary
    performance_commits = [r for r in results if r.has_performance_intent]
    print(f"\n=== ANALYSIS SUMMARY ===")
    print(f"Total commits analyzed: {len(results)}")
    print(f"Performance-related commits: {len(performance_commits)}")
    print(f"Success rate: {len(performance_commits)/len(results)*100:.1f}%")
    
    # Breakdown by optimization type
    from collections import Counter
    opt_types = Counter(r.optimization_type for r in performance_commits)
    print(f"\nOptimization types:")
    for opt_type, count in opt_types.most_common():
        print(f"  {opt_type}: {count}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python performance_analyzer.py <extractions_dir>")
        sys.exit(1)
    
    extractions_dir = sys.argv[1]
    output_file = "performance_analysis_results.json"
    
    analyze_commit_extractions(extractions_dir, output_file)