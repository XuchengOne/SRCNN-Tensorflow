from scipy import misc
from skimage.measure import compare_psnr
import argparse

def compute_psnr(is_grayscale):
	for i in range(5):
		if is_grayscale:
			# read image in grayscale
			ori_image = misc.imread(str(i)+"-test_image(original).png")
			bic_image = misc.imread(str(i)+"-test_image(bicubic).png")
			res_image = misc.imread(str(i)+"-test_image.png")
		else:
			# read image in RGB
			ori_image = misc.imread(str(i)+"-test_image(original).png", mode='RGB')
			bic_image = misc.imread(str(i)+"-test_image(bicubic).png", mode='RGB')
			res_image = misc.imread(str(i)+"-test_image.png", mode='RGB')

		# compute psnr and print
		print "For Image %d: " % (i)
		print "PSNR between Original and Bicubic Interploted: %f" % (compare_psnr(ori_image, bic_image))
		print "PSNR between Original and Result Image: %f" % (compare_psnr(ori_image, res_image))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Compute the PSNR between original images and bicubic interpolated images, as well as between original images and test images.')
	parser.add_argument('--is_grayscale', default=False, type=bool)
	args = parser.parse_args()
	compute_psnr(args.is_grayscale)
