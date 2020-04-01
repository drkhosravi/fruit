import os
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import glob
import shutil
import random
from tqdm import tqdm

def main():
	#plt.axis([-50,50,0,10000])
	fig1, ax1 = plt.subplots()
	#fig2, ax2 = plt.subplots()

	ax1.set_xlim(-50, 50)
	ax1.set_ylim(0, 10000)
	plt.ion()
	plt.show()

	#plt.close(fig2)
	
	x = np.arange(-50, 51)
	for pow in range(1,5):   # plot x^1, x^2, ..., x^4
		y = [Xi**pow for Xi in x]
		ax1.plot(x, y)
		#plt.draw()
		plt.pause(2)
		#input("Press [enter] to continue.")

if __name__ == '__main__':
    main()


class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)
  
def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
                    if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))
  
    return dataset

def get_image_paths(facedir):
    image_paths = []
    #facedir = facedir + '\\aug'
    if os.path.isdir(facedir):
        #images = os.listdir(facedir)
        #images = [f for f in os.listdir(facedir) if (f != 'aug' and f.find('rc') == -1)]#only load original images
        images = [f for f in os.listdir(facedir) if os.path.isfile(os.path.join(facedir, f))] #all images including augmented
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths
  
def create_validation_data(trn_dir, val_dir, split=0.1, ext='png'):
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
        
    train_ds = glob.glob(trn_dir + f'/*/*.{ext}')
    print(len(train_ds))
    
    valid_sz = int(split * len(train_ds)) if split < 1.0 else split 
    
    valid_ds = random.sample(train_ds, valid_sz)
    print(len(valid_ds))
    
    for fname in tqdm(valid_ds):
        basename = os.path.basename(fname)
        label = fname.split('\\')[-2]
        src_folder = os.path.join(trn_dir, label)
        tgt_folder = os.path.join(val_dir, label)
        if not os.path.exists(tgt_folder):
            os.mkdir(tgt_folder)
        shutil.move(os.path.join(src_folder, basename), os.path.join(tgt_folder, basename))
		
		  
def split_dataset(dataset, split_ratio, min_nrof_images_per_class, mode):
    if mode=='SPLIT_CLASSES':
        nrof_classes = len(dataset)
        class_indices = np.arange(nrof_classes)
        np.random.shuffle(class_indices)
        split = int(round(nrof_classes*(1-split_ratio)))
        train_set = [dataset[i] for i in class_indices[0:split]]
        test_set = [dataset[i] for i in class_indices[split:-1]]
    elif mode=='SPLIT_IMAGES':
        train_set = []
        test_set = []
        for cls in dataset:
            paths = cls.image_paths
            np.random.shuffle(paths)
            nrof_images_in_class = len(paths)
            split = int(math.floor(nrof_images_in_class*(1-split_ratio)))
            if split==nrof_images_in_class:
                split = nrof_images_in_class-1
            if split>=min_nrof_images_per_class and nrof_images_in_class-split>=1:
                train_set.append(ImageClass(cls.name, paths[:split]))
                test_set.append(ImageClass(cls.name, paths[split:]))
    else:
        raise ValueError('Invalid train/test split mode "%s"' % mode)
    return train_set, test_set


def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths) # [i] * n creates n-dim array with values eqaul to i
    return image_paths_flat, labels_flat

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

def crop(image, random_crop, image_size):
    if (image.shape[0] >= image_size) and (image.shape[1] >= image_size):
        szw1 = int(image.shape[1]//2)
        szh1 = int(image.shape[0]//2)
        sz2 = int(image_size//2)
        if random_crop:
            difw = szw1-sz2
            difh = szh1-sz2
            (y, x) = (np.random.randint(-difh, difh+1), np.random.randint(-difw, difw+1))
        else:
            (y, x) = (0,0)
        cropped = image[(szh1-sz2+y):(szh1+sz2+y),(szw1-sz2+x):(szw1+sz2+x),:]
        return cropped
    return image
  
def rand_rotate(img, rot_range=[-10, 10]):
	rows,cols = img.shape[0], img.shape[1]
	angle = np.random.randint(rot_range[0], rot_range[1])
	M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
	dst = cv2.warpAffine(img,M,(cols,rows), borderMode=cv2.BORDER_REFLECT101)
	return dst	

def rand_bright(img, b_range=[-20, 20], c_range=[1.0, 2.0]):
	a = np.random.uniform(c_range[0], c_range[1])
	b  = np.random.randint(b_range[0], b_range[1])
	dst = cv2.convertScaleAbs(img, alpha=a, beta=b) #dst = a*img + b
	return dst

def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image

#gray to rgb
def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

# W: Resize width (before crop)
# crop_size: square crop size
def load_images(image_paths, do_random_crop, do_random_flip, resize_w, crop_size, do_prewhiten=True):
	nrof_samples = len(image_paths)
	images = np.zeros((nrof_samples, crop_size, crop_size, 3), dtype = np.float32)
	for i in range(nrof_samples):
		img = cv2.imread(image_paths[i], cv2.IMREAD_COLOR)
	
		#if(img.shape[1] <= 512):#generated images using augment_images
			#img = cv2.resize(img, (round(img.shape[0]*crop_size/img.shape[1]), crop_size))
		#else:
			#img = cv2.resize(img, (round(img.shape[1]*resize_w/img.shape[0]), resize_w))
		if(img.shape[0] > img.shape[1]):
			img = cv2.resize(img, (round(img.shape[1]*resize_w/img.shape[0]), resize_w))
		else:
			img = cv2.resize(img, (resize_w, round(img.shape[0]*resize_w/img.shape[1])))

		if img.ndim == 2:
		    img = to_rgb(img)

		img = crop(img, do_random_crop, crop_size)
		img = flip(img, do_random_flip)

		if i % 10 == 0:
		    cv2.imshow('img', img)
		    cv2.waitKey(1)
		    print('image # %d'%(i))
		if do_prewhiten:
		    img = prewhiten(img)
		images[i,:,:,:] = img

	print(str(nrof_samples) + ' images loaded successfully')
	cv2.destroyAllWindows()
	return images

def save_images(image_paths, new_path):
	nrof_samples = len(image_paths)
	for i in range(nrof_samples):

		s = image_paths[i]
		n = s.rfind('\\')
		fn = s[n+1:]
		n2 = s.rfind('\\', 0, n)
		cls = s[n2:n+1]
		os.rename(s, new_path + cls + fn)			
		
		if i % 10 == 0:
		    print('image # %d'%(i))

	print(str(nrof_samples) + ' images saved successfully')

# crop_size: square crop size
# resize_w: Resize width (after crop)
def augment_images(image_paths, crop_range = 4, rotate_range = [-5, 5], br_range = [-15, 0], contrast_range = [0.8, 1.2], do_flip = True, crop_size = 720, resize_w = 512):
	nrof_samples = len(image_paths)
	for i in range(nrof_samples):
		s = image_paths[i]
		img = cv2.imread(s, cv2.IMREAD_COLOR)
		if img.ndim == 2:
		    img = to_rgb(img)
		n = s.rfind('\\')
		path = s[0:n]
		fn = s[n+1:s.rfind('.')]
		for j in range(crop_range):
			#for k in range(2):
			img2 = rand_rotate(img, rotate_range)
			img2 = crop(img2, True, crop_size)
			img2 = cv2.resize(img2, (round(img2.shape[0]*resize_w/img2.shape[1]), resize_w))
			cv2.imwrite(path + '/aug/' + fn + '_' + str(j+1) + '_rc' + '.jpg', img2) #rc: rotate-crop
				
			cv2.imwrite(path + '/aug/' + fn + '_' + str(j+1) + '_rcf' + '.jpg', np.fliplr(img2)) #f: flip

			img3 = rand_bright(img2, br_range, contrast_range)
			cv2.imwrite(path + '/aug/' + fn + '_' + str(j+1) + '_rcb' + '.jpg', img3)
				
			cv2.imwrite(path + '/aug/' + fn + '_' + str(j+1) + '_rcbf' + '.jpg', np.fliplr(img3))

			#img = flip(img, do_random_flip)
			cv2.imshow('img', img2)
			cv2.waitKey(1)
		if i % 10 == 0:
		    print('image # %d'%(i))
	print('augmentation completed successfully')


def CreateHeatMap(model, img_path):
	from keras import backend as K
	img = cv2.imread(img_path, cv2.IMREAD_COLOR)
	img = cv2.resize(img, (256, 256))

	x = prewhiten(img)

	# We add a dimension to transform our array into a "batch"
	# of size (1, 256, 256, 3)
	x = np.expand_dims(x, axis=0)
	preds = model.predict(x)
	name = ['Albaloo', 'Aloo', 'Holoo', 'Shalil', 'Sib', 'Zardaloo']
	class_label = np.argmax(preds[0])
	print('Predicted:', name[class_label])

	"""To visualize which parts of our image were the most "class_label"-like, let's set up the Grad-CAM process:"""

	# This is the "winner" entry in the prediction vector
	winner = model.output[:, class_label]

	# the last convolutional layer in model
	last_conv_layer = model.get_layer(index=4)

	# This is the gradient of the "winner" class with regard to
	# the output feature map of `last conv layer`
	grads = K.gradients(winner, last_conv_layer.output)[0]

	# This is a vector of shape (64,), where each entry
	# is the mean intensity of the gradient over a specific feature map channel
	pooled_grads = K.mean(grads, axis=(0, 1, 2))

	# This function allows us to access the values of the quantities we just defined:
	# `pooled_grads` and the output feature map of `last conv layer`,
	# given a sample image
	iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

	# These are the values of these two quantities, as Numpy arrays,
	# given our sample image of two elephants
	pooled_grads_value, conv_layer_output_value = iterate([x])

	# We multiply each channel in the feature map array
	# by "how important this channel is" with regard to the elephant class
	for i in range(64):
		conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

	# The channel-wise mean of the resulting feature map
	# is our heatmap of class activation
	heatmap = np.mean(conv_layer_output_value, axis=-1)

	"""For visualization purpose, we will also normalize the heatmap between 0 and 1:"""

	heatmap = np.maximum(heatmap, 0)
	heatmap /= np.max(heatmap)

	plt.matshow(heatmap)
	plt.show()

	"""Finally, we will use OpenCV to generate an image that superimposes the original image with the heatmap we just obtained:"""

	# We use cv2 to load the original image
	#img = cv2.imread(img_path)
	cv2.imshow('input', img)
	cv2.imwrite('input.jpg', img)
	# We resize the heatmap to have the same size as the original image
	heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

	# We convert the heatmap to RGB
	heatmap = np.uint8(255 * heatmap)

	# We apply the heatmap to the original image
	heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

	# 0.4 here is a heatmap intensity factor
	superimposed_img = heatmap * 0.7 + img

	# Save the image to disk
	#cv2.imwrite('D:/Shahrood Univ/_DNN/Samples/DNN06-Visualization/elephant_cam.jpg', superimposed_img)
	#rescale to 0, 255
	minVal = np.amin(superimposed_img)
	maxVal = np.amax(superimposed_img)
	draw = cv2.convertScaleAbs(superimposed_img, alpha=255.0/(maxVal - minVal), beta=-minVal * 255.0/(maxVal - minVal))
	cv2.imshow('output', draw)
	cv2.imwrite('output.jpg', draw)
	cv2.waitKey()


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
	
#plt.figure(figsize=(20, 20))
fig1, ax1 = plt.subplots(figsize=(15, 8))
fig2, ax2 = plt.subplots(figsize=(15, 8))
fig1.tight_layout()
fig2.tight_layout()
plt.grid(linestyle='--', color='gray') #{'-', '--', '-.', ':', '', (offset, on-off-seq), ...}
ax1.set_xlabel('Epochs') 
ax1.set_ylabel('Loss')
#ax1.set_xlim(0, 50)
ax1.set_title('Training and validation loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
#ax2.set_xlim(0, 50)
ax2.set_title('Training and validation accuracy')
def plot_graphs(loss, val_loss, acc, val_acc, wait_for_user_action = False, save_fig_path = None):
	plt.ion() #Interactive ON
	plt.show()

	if(len(loss) == 1):
		ax1.clear()
		ax2.clear()
	#plt.clf()   # clear figure

	ax1.plot(loss, 'blue', label='Training loss', antialiased=True)
	ax1.plot(val_loss, 'red', label='Validation loss', antialiased=True)
	if(ax1.get_legend() == None):
		ax1.legend()

	if(wait_for_user_action):
		plt.pause(100)
	else:
		plt.pause(0.001)

	ax2.plot(acc, 'green', label='Training acc', antialiased=True)
	ax2.plot(val_acc, 'orange', label='Validation acc', antialiased=True)
	if(ax2.get_legend() == None):
		ax2.legend()
	if(wait_for_user_action):
		plt.pause(100)
	else:
		plt.pause(0.001)
	if(save_fig_path is not None):		
		fig1.savefig(save_fig_path + '\\loss.png')
		fig2.savefig(save_fig_path + '\\acc.png')