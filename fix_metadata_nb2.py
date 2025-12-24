import json
nb = json.load(open('2_unsupervised_analysis.ipynb','r',encoding='utf-8'))
for i, line in enumerate(nb['cells'][4]['source']):
    if 'metadata_df = pd.read_csv' in line:
        nb['cells'][4]['source'].insert(i+1, "\n# Add label_name if missing\n")
        nb['cells'][4]['source'].insert(i+2, "if 'label_name' not in metadata_df.columns:\n")
        nb['cells'][4]['source'].insert(i+3, "    metadata_df['label_name'] = pd.Series(labels).map({0: 'normal', 1: 'cancer', -1: 'unlabeled'})\n")
        break
json.dump(nb,open('2_unsupervised_analysis.ipynb','w',encoding='utf-8'),indent=1)
print("Fixed")
