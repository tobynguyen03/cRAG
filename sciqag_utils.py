import json
import os

class SciQAGData:
    def __init__(self, json_file) -> None:
        # Pass directory/file name of SciQAG data json file
        self.json_file = json_file
        
        # Data from SciQAG json file
        self.data = None

        self.txt_doi_pairs = []
        self.qa_pairs = []
    

    def load_data(self):
        """Read the JSON file and store its content in the data attribute."""
        with open(self.json_file, 'r') as file:
            self.data = json.load(file)


    def extract_papers_text_from_final(self, papers_directory):
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
        os.makedirs(papers_directory, exist_ok=True)

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
            filepath = os.path.join(papers_directory, filename)
            
            # Write the "txt" content to the file
            with open(filepath, 'w') as file:
                file.write(entry['txt'])
        
        print(f"Files successfully written to directory: {papers_directory}")


    def extract_qa_pairs_from_final(self):
        """
        Extract Q&A pairs from the JSON data and store them in the qna_pairs attribute.
        Used with .json with all SciQAG data ("final_all_select1000.json"), NOT train or test.
        """
        if self.data is None:
            print("No data loaded. Please load the data first using load_data().")
            return
        
        # Clear any existing pairs
        self.qa_pairs = []

        def recursive_extract(data):
            # Search for the "Q&A" key in dictionaries
            if isinstance(data, dict):
                if "Q&A" in data:
                    # Convert the Q&A pairs into a list of dictionaries with Q as key and A as value
                    self.qa_pairs.extend({qa["Q"]: qa["A"]} for qa in data["Q&A"])
                # Continue searching through nested values
                for value in data.values():
                    recursive_extract(value)
            elif isinstance(data, list):
                for item in data:
                    recursive_extract(item)

        # Start the recursive extraction
        recursive_extract(self.data)
        print(f"Extracted {len(self.qa_pairs)} Q&A pairs.")


    def extract_qa_pairs_from_train(self):
        """
        Extract Q&A pairs from the Train dataset ("train_qas_179511.json") and store them in qna_pairs.
        Each question (input) will be the key, and the answer (output) will be the value.
        """
        if self.data is None:
            print("No data loaded. Please load the data first using load_data().")
            return
        
        # Clear any existing pairs
        self.qa_pairs = []

        for entry in self.data:
            if "input" in entry and "output" in entry:
                self.qa_pairs.append({entry["input"]: entry["output"]})

        print(f"Extracted {len(self.qa_pairs)} Q&A pairs from Train dataset.")


    def extract_qa_pairs_from_test(self):
        """
        Extract Q&A pairs from the Test dataset ("test_qas_8531.json") and store them in qna_pairs.
        Each question (Q) will be the key, and the answer (A) will be the value.
        """
        if self.data is None:
            print("No data loaded. Please load the data first using load_data().")
            return
        
        # Clear any existing pairs
        self.qa_pairs = []

        for entry in self.data:
            if "Q" in entry and "A" in entry:
                self.qa_pairs.append({entry["Q"]: entry["A"]})

        print(f"Extracted {len(self.qa_pairs)} Q&A pairs from Test dataset.")


    def extract_qa_pairs_from_final_and_output(self, questions_path):
        """
        Extract Q&A pairs from the JSON data and print them individually to directory.
        Used with .json with all SciQAG data ("final_all_select1000_shortest.json"), NOT train or test.
        """
        if self.data is None:
            print("No data loaded. Please load the data first using load_data().")
            return
        
        # Clear any existing pairs
        self.qa_pairs = []

        def recursive_extract(data):
            # Search for the "Q&A" key in dictionaries
            if isinstance(data, dict):
                if "Q&A" in data:
                    # Convert the Q&A pairs into a list of dictionaries with Q as key and A as value
                    self.qa_pairs.extend({qa["Q"]: qa["A"]} for qa in data["Q&A"])
                # Continue searching through nested values
                for value in data.values():
                    recursive_extract(value)
            elif isinstance(data, list):
                for item in data:
                    recursive_extract(item)

        # Start the recursive extraction
        recursive_extract(self.data)
        print(f"Extracted {len(self.qa_pairs)} Q&A pairs.")

        # Ensure the directory exists
        os.makedirs(questions_path, exist_ok=True)

        # Iterate over the list of questions and save each as a JSON file
        for i, qa_pair in enumerate(self.qa_pairs, start=1):
            # Define the filename for the current question
            filename = os.path.join(questions_path, f"sciqag_question_{i}.json")

            # Write the current Q&A pair to the file
            with open(filename, 'w') as file:
                json.dump(qa_pair, file, indent=4)

        print(f"Successfully saved {len(self.qa_pairs)} questions to {questions_path}")


    def extract_papers_and_qa_pairs_from_final(self):
        """
        Extract 'doi', 'txt', and associated Q&A pairs from the JSON data.
        Combines both the paper text and Q&A pair extraction functionalities.
        Returns a list of dictionaries where each dictionary contains:
        - 'doi': The DOI of the paper
        - 'txt': The full text of the paper
        - 'qa_pairs': A list of dictionaries with questions as keys and answers as values

        Does NOT operate on class attribute like self.txt_doi_pairs or self.qa_pairs
        """
        if self.data is None:
            print("No data loaded. Please load the data first using load_data().")
            return []

        combined_data = []

        def recursive_extract(data):
            # If the data is a dictionary
            if isinstance(data, dict):
                if "txt" in data and "doi" in data and "Q&A" in data:
                    # Extract 'txt', 'doi', and 'Q&A' pairs
                    doi = data["doi"]
                    txt = data["txt"]
                    qa_pairs = [{qa["Q"]: qa["A"]} for qa in data["Q&A"]]

                    # Append the combined data entry
                    combined_data.append({
                        "doi": doi,
                        "txt": txt,
                        "qa_pairs": qa_pairs
                    })
                # Continue searching through nested values
                for value in data.values():
                    recursive_extract(value)
            # If the data is a list
            elif isinstance(data, list):
                for item in data:
                    recursive_extract(item)

        # Start the recursive extraction
        recursive_extract(self.data)

        print(f"Extracted {len(combined_data)} entries with DOI, text, and Q&A pairs.")
        return combined_data