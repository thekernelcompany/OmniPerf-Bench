#!/usr/bin/env python3
"""
Patch to use OpenRouter instead of OpenAI directly
"""
import os
import sys

# Set OpenRouter configuration
os.environ['OPENAI_API_KEY'] = os.environ.get('OPENROUTER_API_KEY', '')
os.environ['OPENAI_KEY'] = os.environ.get('OPENROUTER_API_KEY', '')
os.environ['OPENAI_API_BASE'] = 'https://openrouter.ai/api/v1'

# Monkey patch the OpenAI client
def patch_openai():
    import openai
    from openai import OpenAI
    
    original_init = OpenAI.__init__
    
    def patched_init(self, *args, **kwargs):
        # Override base_url for OpenRouter
        kwargs['base_url'] = 'https://openrouter.ai/api/v1'
        kwargs['api_key'] = os.environ.get('OPENROUTER_API_KEY', kwargs.get('api_key'))
        # Add default headers for OpenRouter
        kwargs['default_headers'] = kwargs.get('default_headers', {})
        kwargs['default_headers']['HTTP-Referer'] = 'https://github.com/gso-bench'
        kwargs['default_headers']['X-Title'] = 'GSO Benchmark'
        original_init(self, *args, **kwargs)
    
    OpenAI.__init__ = patched_init

# Apply patch
patch_openai()

# Run the original script
if __name__ == "__main__":
    # Remove this script from sys.argv
    script_args = sys.argv[1:]
    
    # Import and run the commits analysis
    from gso.collect.analysis.commits import main
    sys.argv = ['commits.py'] + script_args
    main()