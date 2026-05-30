import PyPDF2
try:
    reader = PyPDF2.PdfReader('GuardPoint AI – YC Strategic Proposal.pdf')
    text = '\n'.join([page.extract_text() for page in reader.pages])
    with open('guardpoint_template.txt', 'w', encoding='utf-8') as f:
        f.write(text)
    print("Success")
except Exception as e:
    print(f"Error: {e}")
