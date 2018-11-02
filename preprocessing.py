# import os,glob
# import cv2
# import copy


import sys,time,logging
if sys.version_info.major >= 3:
   import threading as threadmod
else:
   import thread as threadmod

# from scipy import sparse
# import tensorflow as tf

import numpy as np
# import imgaug as ia
# from imgaug import augmenters as iaa
from keras.utils import Sequence
# import xml.etree.ElementTree as ET
from utils import BoundBox  # , bbox_iou
logger = logging.getLogger(__name__)
# Right now the truth are format as:
# bool objectFound   # if truth particle is in the hard scattering process
# bbox x                    # globe eta
# bbox y                    # globe phi
# bbox width             # Gaussian sigma (required to be 3*sigma<pi)
# bbox height            # Gaussian sigma (required to be 3*sigma<pi)
# bool class1             # truth u/d
# bool class2             # truth s
# bool class3             # truth c
# bool class4             # truth b
# bool class_other        # truth g

BBOX_CENTER_X = 1
BBOX_CENTER_Y = 2
BBOX_WIDTH = 3
BBOX_HEIGHT = 4


class BatchGenerator(Sequence):
    def __init__(self,filelist,
               config,
               evts_per_file,
               shuffle=True,
               jitter=True,
               norm=None,
               sparse=False,
               name='',
               rank=0,
               nranks=1):
        self.config          = config
        self.filelist        = filelist
        self.evts_per_file   = evts_per_file
        self.batch_size      = self.config['BATCH_SIZE']
        self.nevts           = len(filelist) * self.evts_per_file
        # self.nbatches        = int(self.nevts * (1. / self.batch_size))
        self.num_classes     = len(config['LABELS'])
        self.img_c           = config['IMAGE_C']
        self.img_w           = config['IMAGE_W']
        self.grid_w          = config['GRID_W']
        self.img_h           = config['IMAGE_H']
        self.pix_per_grid_x  = int(self.img_w / self.grid_w) + 1
        self.grid_h          = config['GRID_H']
        self.pix_per_grid_y  = int(self.img_h / self.grid_h) + 1
        self.name            = name
        self.rank            = rank
        self.nranks          = nranks
        self.num_batches     = int(np.floor(float(self.nevts) / self.batch_size / self.nranks))

        self.sparse  = sparse
        self.shuffle = shuffle
        self.jitter  = jitter
        self.norm    = norm

        logger.info('%s: len(filelist)    = %s',self.name,len(self.filelist))
        logger.info('%s: nevts            = %s',self.name,self.nevts)
        logger.info('%s: batch_size       = %s',self.name,self.batch_size)
        logger.info('%s: num_batches      = %s',self.name,self.num_batches)
        logger.info('%s: evts_per_file    = %s',self.name,self.evts_per_file)
        logger.info('%s: sparse           = %s',self.name,self.sparse)
        logger.info('%s: norm             = %s',self.name,self.norm)
        logger.info('%s: grid_w           = %s',self.name,self.grid_w)
        logger.info('%s: grid_h           = %s',self.name,self.grid_h)
        logger.info('%s: pix_per_grid_x   = %s',self.name,self.pix_per_grid_x)
        logger.info('%s: pix_per_grid_y   = %s',self.name,self.pix_per_grid_y)


        np.random.seed(threadmod.get_ident() // 2**32)

        self.anchors = [BoundBox(0, 0, config['ANCHORS'][2 * i], config['ANCHORS'][2 * i + 1]) for i in range(int(len(config['ANCHORS']) // 2))]


        '''
        ### augmentors by https://github.com/aleju/imgaug
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        self.aug_pipe = iaa.Sequential(
          [
              # apply the following augmenters to most images
              #iaa.Fliplr(0.5), # horizontally flip 50% of all images
              #iaa.Flipud(0.2), # vertically flip 20% of all images
              #sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
              sometimes(iaa.Affine(
                  #scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                  #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                  #rotate=(-5, 5), # rotate by -45 to +45 degrees
                  #shear=(-5, 5), # shear by -16 to +16 degrees
                  #order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                  #cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                  #mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
              )),
              # execute 0 to 5 of the following (less important) augmenters per image
              # don't execute all of them, as that would often be way too strong
              iaa.SomeOf((0, 5),
                  [
                      #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                      iaa.OneOf([
                          iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                          iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                          iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                      ]),
                      iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                      #iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                      # search either for all edges or for directed edges
                      #sometimes(iaa.OneOf([
                      #    iaa.EdgeDetect(alpha=(0, 0.7)),
                      #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                      #])),
                      iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                      iaa.OneOf([
                          iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                          #iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                      ]),
                      #iaa.Invert(0.05, per_channel=True), # invert color channels
                      iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                      iaa.Multiply((0.5, 1.5), per_channel=0.5), # change brightness of images (50-150% of original value)
                      iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                      #iaa.Grayscale(alpha=(0.0, 1.0)),
                      #sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                      #sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
                  ],
                  random_order=True
              )
          ],
          random_order=True
        )
        '''

        # self.batches_processed = []

        if shuffle:
            np.random.shuffle(self.filelist)

    def __len__(self):
        return self.num_batches

    def get_num_classes(self):
        return self.num_classes

    def size(self):
        return self.nevts

    def num_images(self):
        return self.nevts

    # Convert a list of scipy sparse arrays in csr format to a 3D sparse tensorflow Tensor
    # sparse_matrices: list of sparse scipy arrays
    def importSparse2DenseTensor(self, sparse_matrices, shape):
        dense_mats = []
        for i, s_mat in enumerate(sparse_matrices):
            dns = np.array(s_mat.todense())
            dense_mats.append(dns)
        return np.stack(dense_mats)  # tf.sparse_to_dense(new_indices, shape, np.concatenate(new_data)).eval()

    def load_annotation(self, i):
        if self.sparse:
            file_content = np.load(self.filelist[i])
            objs = file_content[2]
            # logger.debug('load_annotation objs = %s',objs)
        else:
            file_index = int(i / self.evts_per_file)
            image_index = i % self.evts_per_file
            file_content = np.load(self.filelist[file_index])
            objs = file_content['truth'][image_index]
        
        annots = []
        for obj in objs:
            annot = [obj[BBOX_CENTER_X], obj[BBOX_CENTER_Y], obj[BBOX_WIDTH], obj[BBOX_HEIGHT], np.argmax(obj[5:])]
            annots.append(annot)
        # logger.debug('load_annotation annots = %s',annots)
        return np.array(annots)

    def load_image(self, i):
        if self.sparse:
            if i < len(self.filelist):
                file_content = np.load(self.filelist[i])
                return self.importSparse2DenseTensor(file_content[0], (self.img_c,self.img_h,self.img_w))
            else:
                raise Exception('%s: tried to read file index %s but filelist is length %s' % (self.name,i,len(self.filelist)))
        else:
            file_index = int(i / self.evts_per_file)
            image_index = i % self.evts_per_file
            file_content = np.load(self.filelist[file_index])
            return file_content['raw'][image_index]

    # return a batch of images starting at the given index
    def __getitem__(self, idx):
        start = time.time()
        logger.debug('%s: starting get batch of size %s',self.name,self.batch_size)
        # self.batches_processed.append(idx)

        # convert idx to batch index based on rank ID
        batch_index = self.rank + idx * self.nranks

        instance_count = 0

        # Initialize x_batch based on data input format. Represents the input images
        img_shape = (self.img_c,self.img_h,self.img_w)
        if self.sparse:
            x_batch = []
        else:
            x_batch = np.zeros((self.batch_size,) + img_shape)

        # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
        b_batch = np.zeros((self.batch_size, 1, 1, 1, self.config['TRUE_BOX_BUFFER'], 4))           # turam - removed space for anchor boxes
        # y_batch = np.zeros((r_bound - l_bound, self.config['GRID_H'],  self.config['GRID_W'], self.config['BOX'], 4+1+len(self.config['LABELS'])))         # desired network output
        y_batch = np.zeros((self.batch_size,
                            self.grid_h,
                            self.grid_w,
                            self.config['BOX'],
                            4 + 1 + len(self.config['LABELS'])))

        global_image_index = self.batch_size * batch_index
        image_index = global_image_index % self.evts_per_file
        file_index = global_image_index // self.evts_per_file

        logger.debug('%s: thread %s opening file with idx %s batch_index %s file_index %s image_index %s',
            self.name,threadmod.get_ident(), idx, batch_index, file_index,image_index)

        if file_index < len(self.filelist):
            file_content = np.load(self.filelist[file_index])
            if self.sparse:
                file_index += 1
        else:
            raise Exception('%s: file_index %s is outside range for filelist %s' % (self.name,file_index,len(self.filelist)))

        for i in range(self.batch_size):
            logger.debug('[%s] %s: loop %s start',time.time() - start,self.name,i)

            if not self.sparse and image_index >= self.evts_per_file:
                file_index += 1
                file_content = np.load(self.filelist[file_index])
                image_index = 0
            
            if self.sparse:
                x_batch.append(self.importSparse2DenseTensor(file_content[0], img_shape))
                all_objs = file_content[2]
            else:
                x_batch[instance_count] = file_content['raw'][image_index]
                all_objs = file_content['truth'][image_index]

            logger.debug('%s: loop %s file loaded',self.name,i)

            # augment input image and fix object's position and size
            # img, all_objs = self.aug_image(img, all_objs, jitter=self.jitter)

            # construct output from object's x, y, w, h
            true_box_index = 0
            
            for obj in all_objs:
                try:
                    center_x = obj[BBOX_CENTER_X]
                    center_x = center_x / self.pix_per_grid_x
                    center_y = obj[BBOX_CENTER_Y]
                    center_y = center_y / self.pix_per_grid_y

                    grid_x = int(center_x)
                    grid_y = int(center_y)

                    center_w = (obj[BBOX_WIDTH]) / self.pix_per_grid_x  # unit: grid cell
                    center_h = (obj[BBOX_HEIGHT]) / self.pix_per_grid_y  # unit: grid cell
                  
                    box = [center_x, center_y, center_w, center_h]

                    # turam - note: removed anchor box handling

                    # assign ground truth x, y, w, h, confidence and class probs to y_batch
                    y_batch[instance_count, grid_y, grid_x, 0, 0:4] = box
                    y_batch[instance_count, grid_y, grid_x, 0, 4] = 1.
                    y_batch[instance_count, grid_y, grid_x, 0, 5:5 + self.num_classes] = obj[5:5 + self.num_classes]
                  
                    # assign the true box to b_batch
                    b_batch[instance_count, 0, 0, 0, true_box_index] = box
                  
                    true_box_index += 1
                    logger.debug('%s: loop %s b_batch = %s ',self.name,i,b_batch[instance_count])
                    logger.debug('%s: loop %s y_batch = %s ',self.name,i,y_batch[instance_count][grid_y][grid_x])
                    logger.debug('%s: loop %s obj     = %s ',self.name,i,obj)
                    # true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']
                except Exception:
                    if self.sparse:
                        logger.exception('%s: recieved exception while processing truth object %s from file[%s] %s',
                            self.name,obj,file_index-1,self.filelist[file_index-1])
                        raise
                    else:
                        logger.exception('%s: recieved exception while processing truth object %s from file %s',
                            self.name,obj,file_content.fid.name)
                        raise

            logger.debug('%s: loop %s images converted',self.name,i)
            
            # assign input image to x_batch
            if self.norm:
                x_batch[instance_count] = x_batch[instance_count] / np.amax(x_batch[instance_count])
            
            # turam - disable plotting
            
            """
            else:
                # plot image and bounding boxes for sanity check
                for obj in all_objs:
                    if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
                        cv2.rectangle(img[:,:,::-1], (obj['xmin'],obj['ymin']), (obj['xmax'],obj['ymax']), (255,0,0), 3)
                        cv2.putText(img[:,:,::-1], obj['name'],
                                    (obj['xmin']+2, obj['ymin']+12),
                                    0, 1.2e-3 * img.shape[0],
                                    (0,255,0), 2)
                        
                x_batch[instance_count] = img
            """

            # increase instance counter in current batch
            instance_count += 1
            image_index += 1
       

        # print(' new batch created', idx)
        if self.sparse:
            x_batch = np.stack(x_batch)

        end = time.time()
        average_read_time = (end - start) / self.batch_size

        logger.debug('%s: x_batch.shape = %s',self.name,x_batch.shape)
        logger.debug('%s: b_batch.shape = %s',self.name,b_batch.shape)
        logger.debug('%s: y_batch.shape = %s',self.name,y_batch.shape)
        logger.debug('%s: exiting after reading seconds per image: %10.4f',self.name,average_read_time)

        return [x_batch, b_batch], y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.filelist)
        # logger.warning('%s: batches processed(%s): %s',self.name,len(self.batches_processed),self.batches_processed)
        # self.batches_processed = []

    """
    def aug_image(self, image, all_objs, jitter):
        h, w, c = image.shape

        if jitter:
            ### scale the image
            scale = np.random.uniform() / 10. + 1.
            image = cv2.resize(image, (0,0), fx = scale, fy = scale)

            ### translate the image
            max_offx = (scale-1.) * w
            max_offy = (scale-1.) * h
            offx = int(np.random.uniform() * max_offx)
            offy = int(np.random.uniform() * max_offy)
            
            image = image[offy : (offy + h), offx : (offx + w)]

            ### flip the image
            flip = np.random.binomial(1, .5)
            if flip > 0.5: image = cv2.flip(image, 1)
                
            image = self.aug_pipe.augment_image(image)
            
        # resize the image to standard size
        image = cv2.resize(image, (self.config['IMAGE_H'], self.config['IMAGE_W']))
        image = image[:,:,::-1]

        # fix object's position and size
        for obj in all_objs:
            for attr in ['xmin', 'xmax']:
                if jitter: obj[attr] = int(obj[attr] * scale - offx)
                    
                obj[attr] = int(obj[attr] * float(self.config['IMAGE_W']) / w)
                obj[attr] = max(min(obj[attr], self.config['IMAGE_W']), 0)
                
            for attr in ['ymin', 'ymax']:
                if jitter: obj[attr] = int(obj[attr] * scale - offy)
                    
                obj[attr] = int(obj[attr] * float(self.config['IMAGE_H']) / h)
                obj[attr] = max(min(obj[attr], self.config['IMAGE_H']), 0)

            if jitter and flip > 0.5:
                xmin = obj['xmin']
                obj['xmin'] = self.config['IMAGE_W'] - obj['xmax']
                obj['xmax'] = self.config['IMAGE_W'] - xmin
                
        return image, all_objs
        """


def global_to_grid(x,y,w,h,num_grid_x,num_grid_y):
   ''' convert global bounding box coords to
   grid coords. x,y = box center in relative coords
   going from 0 to 1. w,h are the box width and
   height in relative coords going from 0 to 1.
   num_grid_x,num_grid_y define the number of bins
   in x and y for the grid.
   '''
   global_coords  = [x,y]
   global_sizes   = [w,h]
   num_grid_bins  = [num_grid_x,num_grid_y]
   grid_coords    = [0.,0.]
   grid_sizes     = [0.,0.]
   for i in range(len(global_coords)):
      grid_bin_size = 1. / num_grid_bins[i]
      grid_bin = int(global_coords[i] / grid_bin_size)
      grid_coords[i] = (global_coords[i] - grid_bin_size * grid_bin) / grid_bin_size
      grid_sizes[i] = (global_sizes[i] / grid_bin_size)

   return grid_coords,grid_sizes


def grid_to_global(grid_bin_x, grid_bin_y,grid_x,grid_y,grid_w,grid_h,num_grid_x,num_grid_y):
   ''' inverse of global_to_grd '''
   grid_bins      = [grid_bin_x,grid_bin_y]
   grid_coords    = [grid_x,grid_y]
   grid_sizes     = [grid_w,grid_h]
   num_grid_bins  = [num_grid_x,num_grid_y]
   global_coords  = [0.,0.]
   global_sizes   = [0.,0.]
   for i in range(len(global_coords)):
      grid_bin_size     = 1. / num_grid_bins[i]
      global_coords[i]  = (grid_bins[i] + grid_coords[i]) * grid_bin_size
      global_sizes[i]   = grid_sizes[i] * grid_bin_size

   return global_coords,global_sizes


def main():
    import argparse,json,glob
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Atlas Training')
    parser.add_argument('--config_file', '-c',
                        help='configuration in standard json format.',required=True)
    parser.add_argument('--sparse', action='store_true',
                       help="Indicate that the input data is in sparse format")
    args = parser.parse_args()

    config = json.load(open(args.config_file))

    glob_str = config['train']['train_image_folder']  # + '/*.npz'
    filelist = glob.glob(glob_str)

    generator_config = {
        'IMAGE_C': config['model']['input_shape'][0],
        'IMAGE_H': config['model']['input_shape'][1],
        'IMAGE_W': config['model']['input_shape'][2],
        'GRID_H': 8,
        'GRID_W': 180,
        'BOX': 1,
        'LABELS': config['model']['labels'],
        'CLASS': len(config['model']['labels']),
        'ANCHORS': config['model']['anchors'],
        'BATCH_SIZE': config['train']['batch_size'],
        'TRUE_BOX_BUFFER': config['model']['max_box_per_image'],
    }


    gen = BatchGenerator(filelist,generator_config,
                         config['train']['evts_per_file'],
                         sparse=args.sparse)


    length = len(gen)
    one_percent = int(0.001*length)
    for i in range(length):
        if i % one_percent == 0:
            logger.info('on %s of %s',i,length)
        gen[i]






if __name__ == '__main__':
    main()



