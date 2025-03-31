import re


def aggregate_files(input_files, output_file):
    """
    Combine the contents of multiple files into a single output file,
    moving all import statements and global variable declarations to the top.

    Args:
        input_files (list): List of file paths to read from
        output_file (str): Path to the output file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        all_imports = []
        all_variables = []
        file_contents = []

        # First pass: collect all imports, variable declarations, and file contents
        for file_path in input_files:
            try:
                with open(file_path, 'r') as infile:
                    content = infile.read()

                    # Extract import statements using regex
                    # This pattern matches both 'import x' and 'from x import y' statements
                    import_pattern = re.compile(r'^(?:from\s+\S+\s+import\s+\S+|import\s+\S+).*$', re.MULTILINE)
                    imports = import_pattern.findall(content)

                    # Filter out imports that start with "src."
                    filtered_imports = [imp for imp in imports if not imp.startswith('from src') and not imp.startswith('import src.')]

                    # Add found imports to our collection
                    all_imports.extend(filtered_imports)

                    # Extract global variable declarations
                    # This pattern looks for assignments at the module level (not indented)
                    var_pattern = re.compile(r'^([A-Z_][A-Z0-9_]*\s*=.*)$', re.MULTILINE)
                    variables = var_pattern.findall(content)

                    # Add found variables to our collection
                    all_variables.extend(variables)

                    # Remove import statements and global variables from the content
                    content_without_imports = import_pattern.sub('', content)
                    content_without_imports_vars = var_pattern.sub('', content_without_imports)

                    # Clean up any consecutive empty lines
                    content_cleaned = re.sub(r'\n\s*\n', '\n\n', content_without_imports_vars)

                    file_contents.append(content_cleaned)

            except FileNotFoundError:
                print(f"Warning: File not found: {file_path}")
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

        # Remove duplicate imports while preserving order
        unique_imports = []
        for imp in all_imports:
            if imp not in unique_imports:
                unique_imports.append(imp)

        # Remove duplicate variable declarations while preserving order
        unique_variables = []
        for var in all_variables:
            if var not in unique_variables:
                unique_variables.append(var)

        # Write to output file
        with open(output_file, 'w') as outfile:
            # First write all import statements
            outfile.write("# Consolidated imports\n")
            for imp in unique_imports:
                outfile.write(imp + '\n')

            outfile.write("\n# Consolidated global variables\n")
            # Then write all global variable declarations
            for var in unique_variables:
                outfile.write(var + '\n')

            outfile.write("\n# File contents\n")

            # Then write the file contents
            for content in file_contents:
                outfile.write(content.strip())
                outfile.write('\n\n')

        return True
    except Exception as e:
        print(f"Error writing to output file {output_file}: {e}")
        return False


# Example usage
if __name__ == "__main__":
    aggregate_files([
        "src/audio.py",
        "src/birdnet.py",
        "src/predict.py",
        "src/trim.py",
        "src/utils.py",
        "src/wavelet.py"
    ], "python_functions_workbook.py")
    print("Completed aggregation of files with organized imports and global variables.")