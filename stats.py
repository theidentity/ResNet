import os

def get_count(path):
	total = 0
	folder_name = []
	count = []
	for root, dirs, files in os.walk(path):
		folder_name.append(root)
		count.append(len(files))
		total += len(files)
	return folder_name,count,total

def print_counts(path):
	folder_name,count,total = get_count(path)

	for name,num in zip(folder_name,count):
		print name,num

	print total

print_counts('data/train')
print_counts('data/validation')
