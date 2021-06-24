#!/usr/bin/env python3
'''
script including
class object with global settings
'''

import sys

class CONFIG:
  
  #------------------#
  # select or define #
  #------------------#
  
  img_types   = ['kitti', 'mot']
  model_names = ['mask_rcnn', 'yolact'] 
  
  IMG_TYPE    = img_types[0]
  MODEL_NAME  = model_names[0] 
  
  #---------------------#
  # set necessary path  #
  #---------------------#
  
  my_io_path  = '/home/user/object_tracking_io/false_negatives/' + IMG_TYPE + '/'
  
  #--------------------------------------------------------------------#
  # select tasks to be executed by setting boolean variable True/False #
  #--------------------------------------------------------------------#
  
  DETECT_FN         = False
  PLOT_FN_INSTANCES = False
  ANALYZE_TRACKING  = False
  COMPUTE_METRICS   = False
  VISUALIZE_METRICS = False
  ANALYZE_METRICS   = False
  COMPUTE_MAP       = False 
  COMPUTE_FN_FP     = False
  PLOT_FN_FP        = False
  
  #-----------#
  # optionals #
  #-----------#
  
  SCORE_THRESHOLD   = '00' 
  NUM_CORES         = 1
  NUM_PREV_FRAMES   = 10
  NUM_RESAMPLING    = 10
  FLAG_OBJ_SEG      = 0 # 0: object detection, 1: segmentation
  IOU_THRESHOLD     = 0.5
  FLAG_DOUBLE_PRED  = 0 # 0: one pred - one gt, 1: one pred - more gt -> reduces fn
  FLAG_DOUBLE_GT    = 0 # 0: one gt - one pred, 1: one gt - more pred -> reduces fp
  MC_THRESHOLD      = 0.5
  
  if IMG_TYPE == 'kitti':
    NUM_IMAGES = 2981
    CLASSES = [1,2]
  elif IMG_TYPE == 'mot':
    NUM_IMAGES = 2382
    NUM_RESAMPLING = min(NUM_RESAMPLING, 4)
    CLASSES = [2]
  EPS_MATCHING = 100
  NUM_REG_MATCHING = 5
  
  IMG_DIR               = my_io_path + 'inputimages/val/'
  GT_DIR                = my_io_path + 'groundtruth/val/'
  TIME_SERIES_INST_DIR  = my_io_path + 'time_series_instances/' + MODEL_NAME + str(SCORE_THRESHOLD) + '/'
  SCORE_DIR             = my_io_path + 'score/'                 + MODEL_NAME + str(SCORE_THRESHOLD) + '/'
  DEPTH_DIR             = my_io_path + 'depth/'
  
  HELPER_DIR            = my_io_path + 'helpers/'               + MODEL_NAME + str(SCORE_THRESHOLD) + '/'
  ANALYZE_INSTANCES_DIR = my_io_path + 'results_instances/'     + MODEL_NAME + str(SCORE_THRESHOLD) + '/'
  IMG_FN_INSTANCES_DIR  = my_io_path + 'img_fn_instances/'      + MODEL_NAME + str(SCORE_THRESHOLD) + '/'
  METRICS_DIR           = my_io_path + 'metrics/'               + MODEL_NAME + str(SCORE_THRESHOLD) + '/'
  IMG_METRICS_DIR       = my_io_path + 'img_metrics/'           + MODEL_NAME + str(SCORE_THRESHOLD) + '_os' + str(FLAG_OBJ_SEG) + '/'
  ANALYZE_DIR           = my_io_path + 'results_analyze/'       + MODEL_NAME + str(SCORE_THRESHOLD) + '_os' + str(FLAG_OBJ_SEG) + '/npf' + str(NUM_PREV_FRAMES) + '_runs' + str(NUM_RESAMPLING) + '/'
  RESULTS_MAP_DIR       = my_io_path + 'results_map/'           + MODEL_NAME + str(SCORE_THRESHOLD) + '/'
  RESULTS_FN_FP_DIR     = my_io_path + 'results_fn_fp/'         + MODEL_NAME + str(SCORE_THRESHOLD) + '/'
  IMG_MAP_FN_FP_DIR     = my_io_path + 'img_map_fn_fp/'         + MODEL_NAME + str(SCORE_THRESHOLD) + '/'
  


'''
In case of problems, feel free to contact
  Kira Maag, kmaag@uni-wuppertal.de
'''
