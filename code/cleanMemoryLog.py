import re

writeFile = open(f'results-txt/{dataset_name}TR6-memory.txt', 'w')
data = r''''''

# The pattern to find all matches of the number before the specified string
pattern = r'\s(\d+\.\d+)\sMiB\s+1\s+return self\.standarisationFunction\(self\.model\.predict\(np\.array\(\[x\]\), verbose=5\)\[0\]\[0\]\)'

matches = re.findall(pattern, data, re.DOTALL)

# Find all matches before the string <keras.engine.sequential.Sequential object at
final_matches = []
for match in matches:
    index = data.find(match)
    if index != -1 and index < data.find('<keras.engine.sequential.Sequential object at'):
        final_matches.append(match)
        writeFile.write(match + '\n')

print(final_matches)
