import json
import os

class SciQAGData:
    def __init__(self, json_file) -> None:
        # Pass directory/file name of SciQAG data json file
        self.json_file = json_file
        
        # Data from SciQAG json file
        self.data = None

        self.txt_doi_pairs = []
        self.qna_pairs = []
    

    def load_data(self):
        """Read the JSON file and store its content in the data attribute."""
        with open(self.json_file, 'r') as file:
            self.data = json.load(file)


    def extract_papers_text_from_final(self, directory):
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
            filename = f"{entry['doi'].replace('/', '_').replace('.json', '')}.txt"
            filepath = os.path.join(directory, filename)
            
            # Write the "txt" content to the file
            with open(filepath, 'w') as file:
                file.write(entry['txt'])
        
        print(f"Files successfully written to directory: {directory}")


    def extract_qna_pairs_from_final(self):
        """
        Extract Q&A pairs from the JSON data and store them in the qna_pairs attribute.
        Used with .json with all SciQAG data ("final_all_select1000.json"), NOT train or test.
        """
        if self.data is None:
            print("No data loaded. Please load the data first using load_data().")
            return
        
        # Clear any existing pairs
        self.qna_pairs = []

        def recursive_extract(data):
            # Search for the "Q&A" key in dictionaries
            if isinstance(data, dict):
                if "Q&A" in data:
                    # Convert the Q&A pairs into a list of dictionaries with Q as key and A as value
                    self.qna_pairs.extend({qa["Q"]: qa["A"]} for qa in data["Q&A"])
                # Continue searching through nested values
                for value in data.values():
                    recursive_extract(value)
            elif isinstance(data, list):
                for item in data:
                    recursive_extract(item)

        # Start the recursive extraction
        recursive_extract(self.data)
        print(f"Extracted {len(self.qna_pairs)} Q&A pairs.")


    def extract_qna_pairs_from_train(self):
        """
        Extract Q&A pairs from the Train dataset ("train_qas_179511.json") and store them in qna_pairs.
        Each question (input) will be the key, and the answer (output) will be the value.
        """
        if self.data is None:
            print("No data loaded. Please load the data first using load_data().")
            return
        
        # Clear any existing pairs
        self.qna_pairs = []

        for entry in self.data:
            if "input" in entry and "output" in entry:
                self.qna_pairs.append({entry["input"]: entry["output"]})

        print(f"Extracted {len(self.qna_pairs)} Q&A pairs from Train dataset.")


    def extract_qna_pairs_from_test(self):
        """
        Extract Q&A pairs from the Test dataset ("test_qas_8531") and store them in qna_pairs.
        Each question (Q) will be the key, and the answer (A) will be the value.
        """
        if self.data is None:
            print("No data loaded. Please load the data first using load_data().")
            return
        
        # Clear any existing pairs
        self.qna_pairs = []

        for entry in self.data:
            if "Q" in entry and "A" in entry:
                self.qna_pairs.append({entry["Q"]: entry["A"]})

        print(f"Extracted {len(self.qna_pairs)} Q&A pairs from Test dataset.")
        



if __name__ == "__main__": 
    
    sciqag_data = SciQAGData("data/SciQAG/final_all_select1000.json")
    sciqag_data.load_data()
    sciqag_data.extract_papers_text_from_final("data/SciQAG/papers/")
    