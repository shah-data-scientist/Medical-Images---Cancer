import json

# Load notebook
with open('2_unsupervised_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 30 is the problematic one
cell_30 = nb['cells'][30]

# Get the content (it's all in one line)
content = ''.join(cell_30['source'])

# Split by ## headings and other natural break points
# Replace common markdown patterns with newline-separated versions
reformatted = content.replace('###', '\n\n###')
reformatted = reformatted.replace('##', '\n\n##')
reformatted = reformatted.replace('✅ **', '\n✅ **')
reformatted = reformatted.replace('⚠️ **', '\n⚠️ **')
reformatted = reformatted.replace('---', '\n\n---\n\n')
reformatted = reformatted.replace('1. **', '\n1. **')
reformatted = reformatted.replace('2. **', '\n2. **')
reformatted = reformatted.replace('3. **', '\n3. **')
reformatted = reformatted.replace('4. **', '\n4. **')
reformatted = reformatted.replace('5. **', '\n5. **')
reformatted = reformatted.replace('6. **', '\n6. **')
reformatted = reformatted.replace('**Discovery', '\n\n**Discovery')
reformatted = reformatted.replace('**Why', '\n\n**Why')
reformatted = reformatted.replace('**Key ', '\n\n**Key ')
reformatted = reformatted.replace('**Expected', '\n\n**Expected')
reformatted = reformatted.replace('**Cluster', '\n\n**Cluster')
reformatted = reformatted.replace('**What the', '\n\n**What the')
reformatted = reformatted.replace('**Why K-Means', '\n\n**Why K-Means')
reformatted = reformatted.replace('**Why not', '\n\n**Why not')
reformatted = reformatted.replace('**Critical', '\n\n**Critical')
reformatted = reformatted.replace('**Weak', '\n\n**Weak')
reformatted = reformatted.replace('**Strong', '\n\n**Strong')
reformatted = reformatted.replace('**Files', '\n\n**Files')
reformatted = reformatted.replace('**Ready', '\n\n**Ready')

# Clean up multiple newlines (max 2 in a row for paragraph breaks)
while '\n\n\n\n' in reformatted:
    reformatted = reformatted.replace('\n\n\n\n', '\n\n\n')

while '\n\n\n' in reformatted:
    reformatted = reformatted.replace('\n\n\n', '\n\n')

# Split into lines for the notebook format
lines = reformatted.split('\n')

# Convert to notebook format (each line as a separate string)
cell_30['source'] = [line + '\n' if i < len(lines)-1 else line for i, line in enumerate(lines)]

# Save the updated notebook
with open('2_unsupervised_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Cell 30 reformatted successfully!")
print(f"Original: 1 line")
print(f"Reformatted: {len(cell_30['source'])} lines")
