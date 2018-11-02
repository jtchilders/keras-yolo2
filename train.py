#! /usr/bin/env python

import argparse,sys
import glob,logging
import numpy as np
# from preprocessing import parse_annotation
from frontend import YOLO
import json
logger = logging.getLogger(__name__)


def _main_():

   parser = argparse.ArgumentParser(description='Atlas Training')
   parser.add_argument('--config_file', '-c',
                       help='configuration in standard json format.',required=True)
   parser.add_argument('--tb_logdir', '-l',
                       help='tensorboard logdir for this job.',default=None)
   parser.add_argument('--horovod', default=False,
                       help='use Horovod',action='store_true')
   parser.add_argument('--ml_comm', default=False,
                       help='use Cray PE ML Plugin',action='store_true')
   parser.add_argument('--num_files','-n', default=-1, type=int,
                       help='limit the number of files to process. default is all')
   parser.add_argument('--lr', default=0.01, type=int,
                       help='learning rate')
   parser.add_argument('--num_intra', type=int,
                       help='num_intra',default=62)
   parser.add_argument('--num_inter', type=int,
                       help='num_inter',default=2)
   parser.add_argument('--kmp_blocktime', type=int, default=10,
                       help='KMP BLOCKTIME')
   parser.add_argument('--kmp_affinity', default='granularity=fine,verbose,compact,1,0',
                       help='KMP AFFINITY')
   parser.add_argument('--sparse', action='store_true',
                       help="Indicate that the input data is in sparse format")
   parser.add_argument('--random-seed', type=int,default=0,dest='random_seed',
                       help="Set the random number seed. This needs to be the same for all ranks to ensure the batch generator serves data properly.")
   args = parser.parse_args()


   with open(args.config_file) as config_buffer:
      config = json.loads(config_buffer.read())

   if args.tb_logdir is not None:
      config['tensorboard']['log_dir'] = args.tb_logdir

   glob_str = config['train']['train_image_folder']  # + '/*.npz'
   filelist = glob.glob(glob_str)
   if args.num_files > 0:
    filelist = filelist[:args.num_files]

   logger.info('config = %s',config)
   logger.info('train_image_folder  =   %s',glob_str)
   logger.info('tensorboard log_dir =   %s',args.tb_logdir)
   logger.info('random-seed         =   %s',args.random_seed)
   np.random.seed(args.random_seed)


   ###############################
   #   Parse the annotations
   ###############################
   '''
   # parse annotations of the training set
   train_imgs, train_labels = parse_annotation(config['train']['train_annot_folder'],
                                             config['train']['train_image_folder'],
                                             config['model']['labels'])

   # parse annotations of the validation set, if any, otherwise split the training set
   if os.path.exists(config['valid']['valid_annot_folder']):
     valid_imgs, valid_labels = parse_annotation(config['valid']['valid_annot_folder'],
                                                 config['valid']['valid_image_folder'],
                                                 config['model']['labels'])
   else:'''
   train_valid_split = int(config['train']['fraction_train'] * len(filelist))
   np.random.shuffle(filelist)
   logger.info('First file in list  =   %s',filelist[0])

   valid_imgs = filelist[train_valid_split:]
   train_imgs = filelist[:train_valid_split]
   
   logger.info('Length of filelist:      %s',len(filelist))
   logger.info('Length of train_images:  %s',len(train_imgs))
   logger.info('Length of valid_imgs:    %s',len(valid_imgs))

   ###############################
   #   Construct the model
   ###############################

   yolo = YOLO(config,args)


   ###############################
   #   Start the training process
   ###############################

   yolo.train(train_imgs       = train_imgs,
            valid_imgs         = valid_imgs)

   logger.info('exiting')


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
    _main_()
    logger.info('exited')
    sys.exit(0)
