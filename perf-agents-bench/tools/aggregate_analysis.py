
from pathlib import Path
import logging
from bench.analysis.output_writer import OutputWriter

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    root_dir = Path("./state/analysis")
    writer = OutputWriter(root_dir)
    
    print("Loading all analysis files...")
    analyses = writer.load_all_analyses()
    print(f"Found {len(analyses)} analysis files.")
    
    if analyses:
        print("Regenerating aggregate report...")
        report_path = writer.write_aggregate_report(analyses)
        print(f"Report written to: {report_path}")
    else:
        print("No analysis files found!")

if __name__ == "__main__":
    main()
