# converts json to tsv, with review text and rating only

import json
import os
import ast

file_name = "Electronics_5.json"

rel_folder_path = '\Datasets' + '\\'
currentPath = str(os.path.dirname(os.path.realpath(__file__)))
file_path = currentPath + rel_folder_path + file_name
new_file_name = file_name[0: -5] + '.tsv'
new_file_path = currentPath + rel_folder_path + new_file_name

progress_count = 0
print(file_path)
with open(file_path, 'r') as f:
	with open(new_file_path, 'w') as g:
		line = f.readline()
		while line:
			parsed_line = ast.literal_eval(line)
			if progress_count == 0:
				column_titles = 'reviewText\toverall'
				g.write(column_titles + '\n')

			review = {x: str(parsed_line[x]) for x in parsed_line}
			line_data = list(review.values())
			line_data = [line_data[i] for i in range(len(line_data)) if i == 4 or i == 5]
			try:
				line_data[1] = str(round(float(line_data[1]), 0))
				line_data = '\t '.join(line_data)
				g.write(line_data + '\n')
			# omits reviews where score not given as number,
			# since there are not many of them
			except(ValueError):
				pass
			line = f.readline()
			if progress_count % 1000 == 0:
				print(progress_count)
			progress_count += 1
