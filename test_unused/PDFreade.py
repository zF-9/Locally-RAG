# importing required modules
from pypdf import PdfReader

# creating a pdf reader object
reader = PdfReader('Rust_for_Beginners_A_Guide_to_Application_Development_with_Tauri.pdf')

# printing number of pages in pdf file
print(len(reader.pages))

# getting a specific page from the pdf file
#page = reader.pages[11]

for x in range(0, len(reader.pages)):
    # extracting text from page
    page = reader.pages[x]
    text = page.extract_text()
    print(text)