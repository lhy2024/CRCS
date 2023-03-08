def gather_images(str):	
	from PIL import Image
	import os
	import sys

	"""
	用法：python -m IPython
	%run this.py input_dir
	"""

	dirname = str
	dirname = dirname  + "/"  

	files = os.listdir(dirname)	# TODO
	files.sort()

	right_bottom_file = files[-1]
	assert right_bottom_file.endswith(".jpg")
	right_bottom = right_bottom_file[:-4]
	n_row, n_col = right_bottom.split("_")
	n_row = int(n_row)
	n_col = int(n_col)

	with Image.open(dirname+right_bottom_file) as right_bottom_image:
		width = right_bottom_image.size[0]+n_col*256
		height = right_bottom_image.size[1]+n_row*256

	gathered_image = Image.new("RGB", (width, height))
	for file in files:
		with Image.open(dirname+file) as image:
			assert file.endswith(".jpg")
			basename = file[:-4]
			n_row, n_col = basename.split("_")
			n_row = int(n_row)
			n_col = int(n_col)
			x = n_col*256
			y = n_row*256
			gathered_image.paste(image, (x, y))

	gathered_image.save(dirname[:-1]+".jpg")
	gathered_image.close()
