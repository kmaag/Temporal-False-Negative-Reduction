#!/usr/bin/env python3
'''
script including
functions for handling input/output like loading/saving
'''

import os
import pickle
import numpy as np
from PIL import Image

from global_defs import CONFIG


def get_save_path_image_i( vid, i ):
  if CONFIG.IMG_TYPE == 'kitti':
    return CONFIG.IMG_DIR + vid + '/' + str(i).zfill(6) +'.png'
  elif CONFIG.IMG_TYPE == 'mot':
    return CONFIG.IMG_DIR + vid + '/' + str(i+1).zfill(6) +'.jpg'


def get_save_path_gt_i( vid, i ):
  if CONFIG.IMG_TYPE == 'kitti':
    return CONFIG.GT_DIR + vid + '/' + str(i).zfill(6) +'.png'
  elif CONFIG.IMG_TYPE == 'mot':
    return CONFIG.GT_DIR + vid + '/' + str(i+1).zfill(6) +'.png'
  
  
def get_save_path_depth_i( vid, i ):
  if CONFIG.IMG_TYPE == 'kitti':
    depth_name = os.listdir( CONFIG.DEPTH_DIR + vid + '/' )[0]
    depth_name = depth_name.split('sync_')[0]
    return CONFIG.DEPTH_DIR + vid + '/' + depth_name + 'sync_' + str(i).zfill(10) +'.png'
  elif CONFIG.IMG_TYPE == 'mot':
    return CONFIG.DEPTH_DIR + vid + '/' + vid + '_' + str(i+1).zfill(6) +'.png'


def get_save_path_time_series_instances_i( vid, i, eps, num_reg ):
  return CONFIG.TIME_SERIES_INST_DIR + vid + '/time_series_instances' + str(i).zfill(6) + '_eps' + str(eps) + '_num_reg' + str(num_reg) + '.p'


def get_save_path_metrics_i( vid, i, eps, num_reg, flag_3d=0 ):
  if flag_3d == 0:
    if vid == 'all':
      return CONFIG.METRICS_DIR + 'metrics' + str(i).zfill(6) + '_eps' + str(eps) + '_num_reg' + str(num_reg) + '.p'
    else:
      return CONFIG.METRICS_DIR + vid + '/metrics' + str(i).zfill(6) + '_eps' + str(eps) + '_num_reg' + str(num_reg) + '.p'
  elif flag_3d == 1:
    return CONFIG.METRICS_DIR + vid + '_metrics_eps' + str(eps) + '_num_reg' + str(num_reg) + '.p'
    

def ground_truth_load( vid, i ):
  read_path = get_save_path_gt_i( vid, i )
  gt = np.asarray( Image.open(read_path) )
  return gt


def depth_load( vid, i ):
  read_path = get_save_path_depth_i( vid, i )
  depth = np.asarray(Image.open(read_path))
  gt = ground_truth_load(vid, i)
  y_diff = int((gt.shape[1]-depth.shape[1])/2)
  depth_rescaled = np.zeros(gt.shape)
  depth_rescaled[gt.shape[0]-depth.shape[0]:, y_diff:y_diff+depth.shape[1]] = depth
  depth_rescaled /= 256.0
  return depth_rescaled


def score_small_load( vid, i ):
  read_path = CONFIG.SCORE_DIR + vid + '/score_small' + str(i).zfill(6) + '.p'
  score = pickle.load( open( read_path, 'rb' ) )
  return score
  
  
def time_series_instances_load( vid, i, eps, num_reg ):
  read_path = get_save_path_time_series_instances_i( vid, i, eps, num_reg )
  instances = pickle.load( open( read_path, 'rb' ) )
  return instances


def metrics_dump( time_series_metrics, vid, i, eps, num_reg, flag_3d=0 ):
  dump_path = get_save_path_metrics_i( vid, i, eps, num_reg, flag_3d )
  pickle.dump( time_series_metrics, open( dump_path, "wb" ) )


def metrics_load( vid, i, eps, num_reg, flag_3d=0 ):
  read_path = get_save_path_metrics_i( vid, i, eps, num_reg, flag_3d )
  time_series_metrics = pickle.load( open( read_path, 'rb' ) )
  return time_series_metrics


def helpers_p_save(name, data):
  if not os.path.exists( CONFIG.HELPER_DIR ):
    os.makedirs( CONFIG.HELPER_DIR )
  pickle.dump( data, open( CONFIG.HELPER_DIR + name + '.p', 'wb' ) )
  
  
def helpers_p_load(name):  
  data = pickle.load( open( CONFIG.HELPER_DIR + name + '.p', 'rb' ) ) 
  return data
  
  
def write_instace_pred( ):
  
  result_path = os.path.join(CONFIG.ANALYZE_INSTANCES_DIR, 'instance_results_table.txt')
  with open(result_path, 'wt') as fi:
    
    tm = sorted(os.listdir( CONFIG.ANALYZE_INSTANCES_DIR ))
    for t in tm:
      if '.p' in t:
        
        tracking_metrics  = pickle.load( open( CONFIG.ANALYZE_INSTANCES_DIR + t, 'rb' ) )        
        print(t, ':', file=fi )
        print("Recall & Precision & FAR & F  \\\\ ", file=fi )
        print( "{:.4f}".format(tracking_metrics['recall'][0]), "& {:.4f}".format(tracking_metrics['precision'][0]), "& {:.2f}".format(tracking_metrics['far'][0]), "& {:.4f} \\\\ ".format(tracking_metrics['f_measure'][0]), file=fi )
        print("FP & FN & IDsw & FM \\\\ ", file=fi )
        print( "{:.0f}".format(tracking_metrics['fp'][0]), "& {:.0f}".format(tracking_metrics['misses'][0]), "& {:.0f}".format(tracking_metrics['switch_id'][0]), "({:.4f})".format(tracking_metrics['switch_id'][0] / tracking_metrics['gt_obj'][0]), "& {:.0f} \\\\ ".format(tracking_metrics['switch_tracked'][0]), file=fi )
        print("TP & MotA & MotP BB & MotB geo \\\\ ", file=fi )
        print( "{:.0f}".format(tracking_metrics['matches'][0]), "& {:.4f}".format(tracking_metrics['mot_a'][0]), "& {:.2f}".format(tracking_metrics['mot_p_bb'][0]), "& {:.2f} \\\\ ".format(tracking_metrics['mot_p_geo'][0]), file=fi )
        print('GT & TT & MT & PT & ML & TL', file=fi )
        print( '{:.0f}'.format(tracking_metrics['num_gt_ids'][0]), '& {:.0f}'.format(tracking_metrics['totally_tracked'][0]), '& {:.0f}'.format(tracking_metrics['mostly_tracked'][0]), '& {:.0f}'.format(tracking_metrics['partially_tracked'][0]), '& {:.0f} '.format(tracking_metrics['mostly_lost'][0]), '& {:.0f}'.format(tracking_metrics['totally_lost'][0]), file=fi )
        print(' ', file=fi)
      
        
def write_instances_info( metrics, mean_stats, std_stats ):
  
  num_prev_frames = CONFIG.NUM_PREV_FRAMES
  max_inst = int( np.load(CONFIG.HELPER_DIR + 'max_inst.npy') )
  
  if CONFIG.FLAG_OBJ_SEG == 0:
    short_name = '_o'
    th_value = 0.5
  else:
    short_name = '_s'
    th_value = 1e-100
  iou_name = 'iou' + short_name
  
  with open(CONFIG.ANALYZE_DIR + 'instances_info.txt', 'wt') as fi:
    
    num_iou_0 = 0
    num_iou_b0 = 0
    for i in range(len(metrics['S'])):
      if metrics['S'][i] > 0:
        if metrics[iou_name][i] >= th_value:
          num_iou_b0 += 1
        elif metrics[iou_name][i] < th_value:
          num_iou_0 += 1
    
    print( 'total number of instances greater score threshold (in the dataset): ', num_iou_0+num_iou_b0, file=fi )
    print( 'IoU < ' + str(th_value) + ': ', num_iou_0, file=fi )
    print( 'IoU >= ' + str(th_value) + ': ', num_iou_b0, file=fi )
    print( ' ', file=fi)
    
    num_iou_0 = 0
    num_iou_b0 = 0
    counter = 0
    list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR ))
    for vid in list_videos:
      images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + '/' ))
      for i in range(len(images_all)):
        if i >= num_prev_frames:
          for j in range(max_inst):
            if metrics['S'][counter] > 0:
              if metrics[iou_name][counter] >= th_value:
                num_iou_b0 += 1
              elif metrics[iou_name][counter] < th_value:
                num_iou_0 += 1
            counter += 1
        else:
          counter += max_inst

    print( 'number of instances: ', num_iou_0+num_iou_b0, file=fi )
    print( 'IoU < ' + str(th_value) + ': ', num_iou_0, file=fi )
    print( 'IoU >= ' + str(th_value) + ': ', num_iou_b0, file=fi )
    print( ' ', file=fi)
    
    M = sorted([ s for s in mean_stats if iou_name in s ])    
    for i in range(CONFIG.NUM_PREV_FRAMES+1):
      print( 'number of considered frames: ', i+1,  file=fi)
      for s in M: print( s, ': {:.0f}'.format(mean_stats[s][i])+'($\pm${:.0f})'.format(std_stats[s][i]), file=fi )
      print( ' ', file=fi)
    
    
    
    
    
