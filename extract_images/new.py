from glob import glob
import os
import extract_image_all
import gather_images_all

scales = [0.625,5,10]
types = ['ACTH','Tpit','HE']

input = ['','','']


def main():

	os.chdir(r"E:\1_task\2021kfb")
	
	for dir_name in glob("??_??????"):
		os.chdir(str(dir_name))
		file = "HE_10"
		if os.path.exists(file) == False:
			for type in types:
				# scale = 0.625
				for scale in scales:
					input[0] = str(dir_name) + '_' + type +'.kfb'
					
					if not os.path.isdir(type + '_' + str(scale)):
						os.mkdir(type + '_' + str(scale))
					
					input[1] = type + '_' + str(scale)
					input[2] = str(scale)
					
					extract_image_all.extract_image(input)
					if scale < 10:
						gather_images_all.gather_images(input[1])
				
		os.chdir("..//")
		
	for dir_name in glob("??_??????_?"):
		os.chdir(str(dir_name))
		file = "HE_10" # 判断文件夹
		if os.path.exists(file) == False: 
			for type in types:
				# scale = 0.625
				for scale in scales:
					input[0] = str(dir_name) + '_' + type +'.kfb'
					
					if not os.path.isdir(type + '_' + str(scale)):
						os.mkdir(type + '_' + str(scale))
					
					input[1] = type + '_' + str(scale)
					input[2] = str(scale)
					
					extract_image_all.extract_image(input)
					if scale < 10:
						gather_images_all.gather_images(input[1])
				
		os.chdir("..//")
		
		
	# os.chdir(r"E:\1_task\2021kfb")
	
main()

