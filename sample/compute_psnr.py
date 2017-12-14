from scipy import misc
from skimage.measure import compare_psnr

def compute_psnr():
	for i in range(5):
		# read image
		ori_image = misc.imread(str(i)+"-test_image(original).png")
		bic_image = misc.imread(str(i)+"-test_image(bicubic).png")
		res_image = misc.imread(str(i)+"-test_image.png")

		# crop original image
		ori_h, ori_w = ori_image.shape
		res_h, res_w = res_image.shape
		margin_h = (ori_h - res_h) // 2
		margin_w = (ori_w - res_w) // 2
		ori_image = ori_image[margin_h:res_h+margin_h, margin_w:res_w+margin_w]
		bic_image = bic_image[margin_h:res_h+margin_h, margin_w:res_w+margin_w]

		# compute psnr and print
		print "For Image %d: " % (i)
		print "PSNR between Original and Bicubic Interploted: %f" % (compare_psnr(ori_image, bic_image))
		print "PSNR between Original and Result Image: %f" % (compare_psnr(ori_image, res_image))

if __name__ == '__main__':
	compute_psnr()