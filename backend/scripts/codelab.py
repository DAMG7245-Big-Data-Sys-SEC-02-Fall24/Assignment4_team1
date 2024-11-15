import os
import subprocess
import sys

def export_codelab(markdown_file):
    """Export a codelab from the provided Markdown file using claat."""
    print(f"Exporting codelab from {markdown_file}...")
    try:
    
        # Run the claat export command
        subprocess.run(["claat", "export", markdown_file], check=True)
        print(f"Codelab exported successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error exporting codelab: {e}")
        sys.exit(1)

def serve_codelab():
    """Serve the codelab locally."""
    print(f"Starting local codelab server...")
    try:
        # Run the claat serve command
        subprocess.run(["claat", "serve"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error serving codelab: {e}")
        sys.exit(1)

def main(md_path:'str'):
    # Path to your Markdown file
    markdown_file = md_path  # Update this path as needed

    # Check if the Markdown file exists
    if not os.path.isfile(markdown_file):
        sys.exit(f"Markdown file not found: {markdown_file}")

    # Export the codelab
    export_codelab(markdown_file)

    # Serve the codelab
    serve_codelab()


if __name__ == "__main__":
    main("/Users/udaykiran/Desktop/BigData/Assignments/Assignment4_team1/poc/abc/research_summary_20241115_012352.md")
