import os
from ctypes import *
import itertools
	# from decimal import Decimal

def extract_image(input):
	

	#img_opt_lib = CDLL(os.path.abspath(r".\ImageOperationC.dll")) 
	img_opt_lib = CDLL(os.path.dirname(__file__)+r".\ImageOperationC.dll")  

	input_file = input[0]
	h_img = c_long()
	succeed = img_opt_lib.create_image_file(input_file.encode("gb2312"), byref(h_img))	
	assert succeed  

	width = c_int()
	height = c_int()
	scanScale = c_int()
	spendTime = c_float()
	scanTime = c_double()
	capRes = c_float()
	blockSize = c_int()

	succeed = img_opt_lib.get_header_info(h_img, byref(height), byref(width), byref(scanScale), byref(spendTime), byref(scanTime), byref(capRes), byref(blockSize))
	assert succeed
	assert blockSize.value == 256

	scanScale = scanScale.value
	width = width.value
	height = height.value

	# target_scale = Decimal(sys.argv[3])
	# assert scanScale / target_scale

	output_dir = input[1]
	target_scale = float(input[2])

	ptr_type = POINTER(c_ubyte)	
	data_len = c_int()
	data_ptr = ptr_type()



	for count_x in itertools.count():
		for count_y in itertools.count():
			x = count_x*blockSize.value
			y = count_y*blockSize.value
			img_opt_lib.get_image_block_stream(h_img, c_float(target_scale), x, y, byref(data_len), byref(data_ptr))
			if data_len.value == 0:
				break
			py_buffer = create_string_buffer(data_len.value)	
			memmove(pointer(py_buffer), data_ptr, data_len.value)
			succeed = img_opt_lib.delete_image_block_stream(data_ptr)
			assert succeed
			output_file_name = ''.join([output_dir, "\\", "%04d"%count_y, "_", "%04d"%count_x, ".jpg"])
			with open(output_file_name, "wb") as output_file:
				output_file.write(py_buffer.raw)
		if count_y == 0:
			break
			
