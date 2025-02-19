import sys
import os
from datetime import datetime
from contextlib import contextmanager

class OutputCapture:
    """
    Utility class to capture output to both terminal and file.
    
    Usage:
        with OutputCapture('path/to/output/dir') as output:
            # Your analysis code here
            print("This will be saved to file and shown in terminal")
    """
    def __init__(self, output_dir, txt_file_name='analysis_results.txt'):
        self.terminal = sys.stdout
        os.makedirs(output_dir, exist_ok=True)
        self.logfile = open(os.path.join(output_dir, txt_file_name), 'w', encoding='utf-8')
        
        # Write header with timestamp
        self.logfile.write(f"Analysis Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.logfile.write("=" * 80 + "\n\n")
    
    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def close(self):
        self.logfile.close()
    
    def __enter__(self):
        sys.stdout = self
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self.terminal
        self.logfile.close()

def get_output_dir(model_name, type='overall'):
    """
    Creates and returns the output directory path for a specific model.
    
    Args:
        model_name (str): Name of the model folder (e.g., '1_stages_model')
        type (str): Type of output directory (default: 'overall')
    Returns:
        str: Path to the output directory
    """
    # Get the root directory (where utils folder is)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Create path for model's output
    output_dir = os.path.join(root_dir, model_name, 'output', type)
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir