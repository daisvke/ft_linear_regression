import argparse

def main():
	# Create the parser
	parser = argparse.ArgumentParser(description="Process some integers.")

	# Add arguments
	parser.add_argument('args', metavar='N', type=str, nargs='+',
						help='an integer for the accumulator')

	# Parse the arguments
	args = parser.parse_args()

	# Access the arguments
	for arg in args.args:
		print(arg)

if __name__ == "__main__":
	main()
