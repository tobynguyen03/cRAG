import json
import os

class SciqagData:
    def __init__(self, json_file) -> None:
        # Pass directory/file name of SciQAG data json file
        self.json_file = json_file
        
        # Data from SciQAG json file
        self.data = None

        self.txt_doi_pairs = []
    

    def load_data(self):
        """Read the JSON file and store its content in the data attribute."""
        with open(self.json_file, 'r') as file:
            self.data = json.load(file)

    def extract_papers_text(self, directory):
        """
        Extract 'txt' and 'doi' pairs from the JSON data and save each
        as a .txt file in the specified directory.
        
        Parameters:
        - directory (str): The directory to save the .txt files.
        """
        if self.data is None:
            print("No data loaded. Please load the data first using load_data().")
            return
        
        # Clear any existing pairs
        self.txt_doi_pairs = []

        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)

        def recursive_extract(data):
            # Search for "txt" and "doi" in dictionaries within the JSON data
            if isinstance(data, dict):
                if "txt" in data and "doi" in data:
                    self.txt_doi_pairs.append({"doi": data["doi"], "txt": data["txt"]})
                # Otherwise, continue searching deeper
                for value in data.values():
                    recursive_extract(value)
            # If it's a list, iterate over elements to continue searching
            elif isinstance(data, list):
                for item in data:
                    recursive_extract(item)

        # Start the recursive extraction
        recursive_extract(self.data)
        print(f"Extracted {len(self.txt_doi_pairs)} 'txt' and 'doi' pairs.")

        # Write each pair to a .txt file in the specified directory
        for entry in self.txt_doi_pairs:
            # Replace any invalid filename characters in "doi" and set the file path
            filename = f"{entry['doi'].replace('/', '_')}.txt"
            filepath = os.path.join(directory, filename)
            
            # Write the "txt" content to the file
            with open(filepath, 'w') as file:
                file.write(entry['txt'])
        
        print(f"Files successfully written to directory: {directory}")