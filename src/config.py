def load_filenames(filename='params/filenames.txt'):
	"""
	This function loads the filenames used in the project
	It loads all lines from the file containing the filenames
	as a list and each line (=1 filename) is stripped and added
	to the returned list
	"""
	with open(filename, 'r') as file:
		return [line.strip() for line in file.readlines()]
