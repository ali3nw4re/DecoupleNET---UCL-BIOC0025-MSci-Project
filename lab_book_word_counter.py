import json

with open("Lab Book.ipynb") as json_file:
    data = json.load(json_file)

markdown_word_count = 0
code_word_count = 0

for cell in data["cells"]:
    cell_type = cell["cell_type"]
    content = cell["source"]
    if cell_type == "markdown":
        for line in content:
            words = [word for word in line.split() if not word.startswith("#")]
            markdown_word_count += len(words)
    elif cell_type == "code":
        for line in content:  
            words = line.split()
            code_word_count += len(words)

print("")
print("     Text word count:  " + str(markdown_word_count))
print("     Code word count:  " + str(code_word_count))
print("     Total word count: " + str(markdown_word_count+code_word_count))
print("")