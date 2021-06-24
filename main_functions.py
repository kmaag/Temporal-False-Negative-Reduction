#!/usr/bin/env python3
'''
script including
class objects and functions called in main
'''

import os
import gc
import time
import pickle
import numpy as np
import matplotlib.colors as colors
from multiprocessing import Pool
from sklearn.metrics import r2_score, mean_squared_error, roc_curve, auc

from global_defs import CONFIG
from calculate   import compute_metrics_2d, compute_3d_gt_metrics, compute_ya_y0a, classification_fit_and_predict, regression_fit_and_predict, comp_mean_average_precision, compute_detected_fn, compute_num_fn, get_occlusion_level
from helper      import compute_max_inst, comp_inst_in_bd, time_series_metrics_to_nparray, split_tvs_and_concatenate, concatenate_val_for_visualization
from in_out      import ground_truth_load, depth_load, write_instace_pred, time_series_instances_load, score_small_load, helpers_p_save, helpers_p_load, get_save_path_time_series_instances_i, get_save_path_metrics_i, metrics_dump, metrics_load, write_instances_info
from metrics     import analyze_instances_vid, compute_missed, compute_consequetive_instances
from plot        import plot_fn_inst, plot_time_series_per_gt, plot_scatter_metric_iou, plot_regression_scatter, plot_coef_timeline, plot_train_val_test_timeline, plot_fn_vs_fp
                        
                      

#----------------------------#
class detect_fn(object):
#----------------------------#
  
  def __init__(self, num_cores=1):
    '''
    object initialization
    :param epsilon: (int) used in matching algorithm
    :param num_reg: (int) used in matching algorithm
    ''' 
    self.epsilon = CONFIG.EPS_MATCHING
    self.num_reg = CONFIG.NUM_REG_MATCHING
    
    
  def detect_fn_instances(self):
    '''
    detection of false negative instances 
    '''
    print('detection of false negative instances')
    
    if os.path.isfile( CONFIG.HELPER_DIR + 'inst_time_series_tested.p' ):
      inst_time_series_tested = helpers_p_load('inst_time_series_tested')
    else:
      inst_time_series = self.detect_by_time_series()
      helpers_p_save('inst_time_series', inst_time_series)
      
      inst_time_series_tested = self.test_covered(inst_time_series)
      helpers_p_save('inst_time_series_tested', inst_time_series_tested)
    
  
  def detect_by_time_series(self):
    
    print('start detection')
    
    max_inst = compute_max_inst()
    
    new_instances = { 'vid_no': list([]), 'frame_no': list([]), 'inst_array': list([]), 'score_val': list([]) }
    
    list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR ))
    
    for vid in list_videos: 
      print('vid', vid)
      images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + '/' ))
      
      # size, mean x, mean y, score
      values = np.ones((max_inst, len(images_all), 4)) *-1
      
      for n in range(len(images_all)):
        
        inst_image  = time_series_instances_load( vid, n, self.epsilon, self.num_reg )
        inst_image[inst_image<0] *= -1
        scores = score_small_load(vid, n)
        
        for i in range(inst_image.shape[0]):
          
          id_num = inst_image[i].max() % 10000
          tmp_ind = np.where(inst_image[i]>0)
          
          values[id_num-1, n, 0] = np.sum(inst_image[i]>0)
          values[id_num-1, n, 1] = np.sum(tmp_ind[0]) / values[id_num-1, n, 0]
          values[id_num-1, n, 2] = np.sum(tmp_ind[1]) / values[id_num-1, n, 0]
          values[id_num-1, n, 3] = scores[i]
          print(vid, n, i, inst_image[i].max(), id_num, values[id_num-1, n, 0], values[id_num-1, n, 1], values[id_num-1, n, 2], values[id_num-1, n, 3] )
      
      for m in range(max_inst):
        if np.sum(values[m,:,0]>-1) >= 2:
          frame_ind = np.where(values[m,:,0]>-1)
          print('info:', m, max_inst, np.sum(values[m,:,0]>-1), frame_ind, np.squeeze(np.asarray(frame_ind))[1], np.shape(np.squeeze(values[m,frame_ind,:])))
          
          shifted_inst, num_frame, score_value = compute_consequetive_instances(vid, len(images_all), m+1, np.squeeze(np.asarray(frame_ind)), np.squeeze(values[m,frame_ind,:]))

          for k in range(len(num_frame)): 
            new_instances['vid_no'].append( vid )
            new_instances['frame_no'].append( num_frame[k] )
            new_instances['inst_array'].append( shifted_inst[k] ) 
            new_instances['score_val'].append( score_value[k] )
    return new_instances  
  
  
  def test_covered(self, instances_test):
    
    print('test if possible instances are covered by other instances')
    
    new_instances = { 'vid_no': list([]), 'frame_no': list([]), 'inst_array': list([]), 'score_val': list([]) }
    print('num instance before testing', len(instances_test['vid_no']))
    
    list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR ))
    
    for vid in list_videos: 
      print('vid', vid)
      images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + '/' ))
      
      for n in range(len(images_all)):
      
        for j in range(len(instances_test['frame_no'])):
          if vid==instances_test['vid_no'][j] and n==instances_test['frame_no'][j]:
            #print(vid, n, instances_test['vid_no'][j], instances_test['frame_no'][j])
            
            flag_covered = 0
            
            gt_image = ground_truth_load(vid, n)
            intersection = np.sum( np.logical_and(instances_test['inst_array'][j]>0,gt_image==10000) )
            pixel_inst_j = np.sum(instances_test['inst_array'][j]>0)
            if pixel_inst_j > 0 and intersection/pixel_inst_j >= 0.8:
              flag_covered = 1

            inst_image  = time_series_instances_load( vid, n, self.epsilon, self.num_reg )
            inst_image[inst_image<0] *= -1
            for i in range(inst_image.shape[0]):
              intersection = np.sum( np.logical_and(instances_test['inst_array'][j]>0,inst_image[i]>0) )
              union = np.sum(instances_test['inst_array'][j]>0) + np.sum(inst_image[i]>0) - intersection
              if union > 0 and intersection / union > 0.95:
                flag_covered = 1
                
            if flag_covered == 0:
              new_instances['vid_no'].append( instances_test['vid_no'][j] )
              new_instances['frame_no'].append( instances_test['frame_no'][j] ) 
              new_instances['inst_array'].append( comp_inst_in_bd(instances_test['inst_array'][j]) ) 
              new_instances['score_val'].append( instances_test['score_val'][j] ) 
                
    print('num instance after testing', len(new_instances['vid_no']))
    return new_instances



#----------------------------#
class plot_fn_instances(object):
#----------------------------#
  
  def __init__(self, num_cores=1):
    '''
    object initialization
    :param num_cores: (int)   number of cores used for parallelization
    :param iou_th:    (float) iou threshold
    :param flag_dp:   (int)   used for false negative calculation
    :param flag_dg:   (int)   used for false negative calculation
    '''
    self.num_cores = num_cores if not hasattr(CONFIG, 'NUM_CORES') else CONFIG.NUM_CORES  
    self.iou_th    = CONFIG.IOU_THRESHOLD
    self.flag_dp   = CONFIG.FLAG_DOUBLE_PRED
    self.flag_dg   = CONFIG.FLAG_DOUBLE_GT
    
    
  def plot_fn_instances_per_image(self):
    '''
    plot false negative instances 
    '''
    print('plot false negative instances')
    flag_new = 1
    
    save_path_fix = '_' + str(self.iou_th) + '_' + str(self.flag_dp) + str(self.flag_dg)
    
    if flag_new:
      instances_new = helpers_p_load('inst_time_series_tested')
      array_vid = np.asarray(instances_new['vid_no'])
      array_frame = np.asarray(instances_new['frame_no'])
      if CONFIG.IMG_TYPE == 'kitti':
        array_inst = np.stack(instances_new['inst_array'])
      array_score = np.asarray(instances_new['score_val'])
      save_path_fix = save_path_fix + '_time'
    
    colors_list_tmp = list(colors._colors_full_map.values())  # 1163 colors
    colors_list = []
    for color in colors_list_tmp:
      if len(color) == 7:
        colors_list.append(color)
    
    list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR ))
    for vid in list_videos: 
      
      save_path = CONFIG.IMG_FN_INSTANCES_DIR + vid + save_path_fix
      if not os.path.exists( save_path ):
        os.makedirs( save_path )
    
      images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + '/' ))
      
      if CONFIG.IMG_TYPE == 'mot':
        array_inst_tmp = []
        gt = ground_truth_load(vid, 0)
        for i in range(array_vid.shape[0]):
          if array_vid[i] == vid:
            array_inst_tmp.append(instances_new['inst_array'][i])
          else:
            array_inst_tmp.append(np.zeros(gt.shape,dtype='int16'))
        array_inst = np.stack(array_inst_tmp)

      p = Pool(self.num_cores)
      if flag_new:
        p_args = [ (vid,k,save_path,colors_list,array_inst[np.logical_and(array_vid==vid, array_frame==k)],array_score[np.logical_and(array_vid==vid, array_frame==k)]) for k in range(len(images_all)) ]
      else:
        p_args = [ (vid,k,save_path,colors_list) for k in range(len(images_all)) ]
      p.starmap( plot_fn_inst, p_args ) 
      p.close()
      
  
  def visualize_time_series_per_gt(self):
    '''
    plot time series of gt instances
    '''
    print('plot times series of instances')
    
    list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR ))   
    
    for vid in list_videos:
      plot_time_series_per_gt(vid)
      
      

#----------------------------#
class analyze_instance_prediction(object):
#----------------------------#

  def __init__(self, num_cores=1):
    '''
    object initialization
    :param num_cores: (int) number of cores used for parallelization
    :param classes:   (int) classes of the dataset
    '''
    self.num_cores = num_cores if not hasattr(CONFIG, 'NUM_CORES') else CONFIG.NUM_CORES
    self.classes   = CONFIG.CLASSES
    

  def analyze_instances(self):
    '''
    analyze results of the instance prediction
    '''
    
    print('calculating instance prediction evaluation metrics')
    start = time.time()
    
    flag_detect = 1
    
    if not os.path.exists( CONFIG.ANALYZE_INSTANCES_DIR ):
      os.makedirs( CONFIG.ANALYZE_INSTANCES_DIR )
    
    path_tmp = 'inst_pred_'
    if flag_detect == 1:
      path_tmp = 'inst_pred_detect_'

    for c in self.classes:
      print('class', c)
      
      list_gt_ids = []
      num_img_per_vid = []
      
      list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR ))   
      for vid in list_videos:
        images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + '/' ))
 
        num_img_per_vid.append( len(images_all) )
        
        list_gt_ids_tmp = []
        
        for n in range(len(images_all)):
          
          gt_image = ground_truth_load(vid, n)
          
          for k in np.unique(gt_image):
            if k != 0 and k!= 10000:
              
              gt_class = gt_image // 1000
              if c == gt_class[gt_image==k].max():
            
                list_gt_ids_tmp.append(k)
          
        list_gt_ids_tmp = np.unique(list_gt_ids_tmp)
        list_gt_ids.append(list_gt_ids_tmp) 

      p_args = [ (list_videos[k],num_img_per_vid[k],np.asarray(list_gt_ids[k]),c,flag_detect) for k in range(len(list_videos)) ]
      Pool(self.num_cores).starmap( analyze_instances_vid, p_args ) 
        
    print('Start concatenate')
    
    for c in self.classes:
      print('class', c)
      
      tracking_metrics = { 'num_frames': np.zeros((1)), 'gt_obj': np.zeros((1)), 'fp': np.zeros((1)), 'misses': np.zeros((1)), 'mot_a': np.zeros((1)), 'dist_bb': np.zeros((1)), 'dist_geo': np.zeros((1)), 'matches': np.zeros((1)), 'mot_p_bb': np.zeros((1)), 'mot_p_geo': np.zeros((1)), 'far': np.zeros((1)), 'f_measure': np.zeros((1)), 'precision': np.zeros((1)), 'recall': np.zeros((1)), 'switch_id': np.zeros((1)), 'num_gt_ids': np.zeros((1)), 'mostly_tracked': np.zeros((1)), 'partially_tracked': np.zeros((1)), 'mostly_lost': np.zeros((1)), 'switch_tracked': np.zeros((1)), 'totally_tracked': np.zeros((1)), 'totally_lost': np.zeros((1)) }
      
      list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR ))
      for vid in list_videos:
        
        tracking_metrics_i  = pickle.load( open( CONFIG.ANALYZE_INSTANCES_DIR + path_tmp + vid + '_class' + str(c) + '.p', 'rb' ) )
        
        for tm in tracking_metrics:
          if tm not in ['mot_a', 'mot_p_bb', 'mot_p_geo', 'far', 'f_measure', 'precision', 'recall']:
            tracking_metrics[tm] += tracking_metrics_i[tm]
      
      tracking_metrics = self.comp_pre_rec(tracking_metrics)
      
      if len(self.classes) == 1:
        pickle.dump( tracking_metrics, open( CONFIG.ANALYZE_INSTANCES_DIR + path_tmp + '.p', 'wb' ) )  
      else: 
        pickle.dump( tracking_metrics, open( CONFIG.ANALYZE_INSTANCES_DIR + path_tmp + 'class' + str(c) + '.p', 'wb' ) )  
        
    if len(self.classes) > 1:
      
      for tm in tracking_metrics:
          tracking_metrics[tm] = 0
            
      for c in self.classes:
        tracking_metrics_i  = pickle.load( open( CONFIG.ANALYZE_INSTANCES_DIR + path_tmp + 'class' + str(c) + '.p', 'rb' ) )
        
        for tm in tracking_metrics:
          if tm not in ['mot_a', 'mot_p_bb', 'mot_p_geo', 'far', 'f_measure', 'precision', 'recall']:
            tracking_metrics[tm] += tracking_metrics_i[tm]
            
      tracking_metrics = self.comp_pre_rec(tracking_metrics)
      pickle.dump( tracking_metrics, open( CONFIG.ANALYZE_INSTANCES_DIR + path_tmp + '.p', 'wb' ) )  
        
    print('Time needed:', time.time()-start)
    write_instace_pred( )
      
      
  def comp_pre_rec(self, tracking_metrics):
    '''
    helper function for metrics calculation
    ''' 

    tracking_metrics['mot_a'] = 1 - ((tracking_metrics['misses'] + tracking_metrics['fp'] + tracking_metrics['switch_id'])/tracking_metrics['gt_obj'])
    
    tracking_metrics['mot_p_bb'] = tracking_metrics['dist_bb'] / tracking_metrics['matches']
    
    tracking_metrics['mot_p_geo'] = tracking_metrics['dist_geo'] / tracking_metrics['matches']
    
    tracking_metrics['far'] = tracking_metrics['fp'] / tracking_metrics['num_frames'] * 100
    
    tracking_metrics['f_measure'] = (2 * tracking_metrics['matches']) / (2 * tracking_metrics['matches'] + tracking_metrics['misses'] + tracking_metrics['fp'])
    
    tracking_metrics['precision'] = tracking_metrics['matches'] / ( tracking_metrics['matches'] + tracking_metrics['fp'])
    
    tracking_metrics['recall'] = tracking_metrics['matches'] / ( tracking_metrics['matches'] + tracking_metrics['misses'])
    
    return tracking_metrics
    
      

#----------------------------#
class compute_time_series_metrics(object):
#----------------------------#

  def __init__(self, num_cores=1):
    """
    object initialization
    :param num_cores: (int) number of cores used for parallelization
    :param epsilon:   (int) used in matching algorithm
    :param num_reg:   (int) used in matching algorithm
    :param max_inst:  (int) maximum number of instances in all image sequences
    """
    self.num_cores = num_cores if not hasattr(CONFIG, 'NUM_CORES') else CONFIG.NUM_CORES
    self.epsilon   = CONFIG.EPS_MATCHING
    self.num_reg   = CONFIG.NUM_REG_MATCHING
    self.max_inst  = compute_max_inst()


  def compute_time_series_metrics_per_image(self):
    """
    perform time series metrics computation 
    """
    print('calculating time series metrics')
    start = time.time()
    
    if not os.path.isfile( get_save_path_metrics_i( 'all', '_2d', self.epsilon, self.num_reg ) ):
      
      instances_new = helpers_p_load('inst_time_series_tested')
      array_vid = np.asarray(instances_new['vid_no'])
      array_frame = np.asarray(instances_new['frame_no'])
      if CONFIG.IMG_TYPE == 'kitti':
        array_inst = np.stack(instances_new['inst_array'])
      array_score = np.asarray(instances_new['score_val'])
      
      list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR ))
      for vid in list_videos: 
        print(vid)
        gc.collect()
        
        if not os.path.exists( CONFIG.METRICS_DIR + vid + "/" ):
          os.makedirs( CONFIG.METRICS_DIR + vid + "/" )
        
        images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + '/' ))
        
        if CONFIG.IMG_TYPE == 'mot':
          array_inst_tmp = []
          gt = ground_truth_load(vid, 0)
          for i in range(array_vid.shape[0]):
            if array_vid[i] == vid:
              array_inst_tmp.append(instances_new['inst_array'][i])
            else:
              array_inst_tmp.append(np.zeros(gt.shape,dtype='int16'))
          array_inst = np.stack(array_inst_tmp)
          
        p = Pool(self.num_cores)
        p_args = [ (vid,k,array_inst[np.logical_and(array_vid==vid, array_frame==k)], array_score[np.logical_and(array_vid==vid, array_frame==k)]) for k in range(len(images_all)) ]
        p.starmap( self.compute_time_series_metrics_i, p_args ) 
        p.close()
      
      metrics = metrics_load( list_videos[0], 0, self.epsilon, self.num_reg )
      for vid in list_videos: 
        images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + '/' ))
        for i in range(len(images_all)):
          if vid != list_videos[0] or i != 0:
            m = metrics_load( vid, i, self.epsilon, self.num_reg )
            for j in metrics:
              metrics[j] += m[j]
      
      metrics_dump( metrics, 'all', '_2d', self.epsilon, self.num_reg )
      
    compute_3d_gt_metrics(self.max_inst)
    

  def compute_time_series_metrics_i( self, vid, i, new_instances, new_scores ):
    """
    compute time series metrics 
    """
    
    if os.path.isfile( get_save_path_time_series_instances_i( vid, i, self.epsilon, self.num_reg ) ) and not os.path.isfile( get_save_path_metrics_i( vid, i, self.epsilon, self.num_reg ) ):
      
      start = time.time()
      
      instances = time_series_instances_load( vid, i, self.epsilon, self.num_reg )
      instances = np.concatenate((instances, new_instances), axis=0)
      scores = score_small_load(vid, i)
      scores = np.concatenate((scores, new_scores), axis=0)
      gt = ground_truth_load(vid, i)
      depth = depth_load(vid, i)

      time_series_metrics = compute_metrics_2d( instances, gt, depth, scores, self.max_inst )
      metrics_dump( time_series_metrics, vid, i, self.epsilon, self.num_reg ) 
      print('image', i, 'processed in {}s\r'.format( round(time.time()-start) ) )           
      
      
      
#----------------------------#
class visualize_time_series_metrics(object):
#----------------------------#

  def __init__(self):
    '''
    object initialization
    '''


  def visualize_metrics_vs_iou(self):
    '''
    plot metrics vs iou as scatterplots
    '''
    print('visualize metrics vs iou')
    
    list_metrics = ['S', 'S_in', 'S_bd', 'S_rel', 'S_rel_in', 'D', 'D_in', 'D_bd', 'D_rel', 'D_rel_in', 'mean_x', 'mean_y', 'score', 'survival', 'ratio', 'deformation', 'occlusion', 'diff_mean', 'diff_size', 'diff_depth']
    plot_scatter_metric_iou(list_metrics)
    


#----------------------------#    
class analyze_metrics(object):
#----------------------------#

  def __init__(self, num_cores=1):
    '''
    object initialization
    :param num_cores:       (int) number of cores used for parallelization
    :param epsilon:         (int) used in matching algorithm
    :param num_reg:         (int) used in matching algorithm
    :param num_prev_frames: (int) number of previous considered frames
    :param runs:            (int) number of resamplings 
    :param flag_obj_seg:    (int) iou calculation as in object detection or semantic segmentation
    '''
    self.num_cores       = num_cores if not hasattr(CONFIG, 'NUM_CORES') else CONFIG.NUM_CORES
    self.epsilon         = CONFIG.EPS_MATCHING
    self.num_reg         = CONFIG.NUM_REG_MATCHING
    self.num_prev_frames = CONFIG.NUM_PREV_FRAMES
    self.runs            = CONFIG.NUM_RESAMPLING
    self.flag_obj_seg    = CONFIG.FLAG_OBJ_SEG
    
    
  def analyze_time_series_metrics( self ):
    '''
    analze time series metrics
    '''
    print('start analyzing')
    t= time.time()
    
    if not os.path.exists( CONFIG.ANALYZE_DIR+'stats/' ):
      os.makedirs( CONFIG.ANALYZE_DIR+'stats/' )
      os.makedirs( CONFIG.ANALYZE_DIR+'scatter/' )
      os.makedirs( CONFIG.ANALYZE_DIR+'train_val_test_timeline/' )
      os.makedirs( CONFIG.ANALYZE_DIR+'feature_importance/' )
    
    self.tvs = np.load(CONFIG.HELPER_DIR + 'tvs_runs' + str(self.runs ) + '.npy')
    
    metrics = metrics_load( self.tvs[0], 0, self.epsilon, self.num_reg, 1 ) 
    self.X_names = sorted([ m for m in metrics if m not in ['iou_s','iou0_s','iou_o','iou0_o']]) 
    print('Metrics: ', self.X_names)

    classification_stats = ['penalized_train_acc', 'penalized_train_auroc', 'penalized_val_acc', 'penalized_val_auroc', 'penalized_test_acc', 'penalized_test_auroc', 'entropy_train_acc','entropy_train_auroc', 'entropy_val_acc','entropy_val_auroc', 'entropy_test_acc','entropy_test_auroc', 'score_train_acc','score_train_auroc', 'score_val_acc','score_val_auroc', 'score_test_acc','score_test_auroc', 'iou0_found', 'iou0_not_found', 'not_iou0_found', 'not_iou0_not_found' ]
    regression_stats = ['regr_train_r2', 'regr_train_mse', 'regr_val_r2', 'regr_val_mse', 'regr_test_r2', 'regr_test_mse', 'entropy_train_r2', 'entropy_train_mse', 'entropy_val_r2', 'entropy_val_mse', 'entropy_test_r2', 'entropy_test_mse', 'score_train_r2', 'score_train_mse', 'score_val_r2', 'score_val_mse', 'score_test_r2', 'score_test_mse' ]
    
    stats = self.init_stats_frames( classification_stats, regression_stats )
    
    print('start runs')
    
    for run in range(self.runs):
      stats = self.fit_reg_cl_run_timeseries(stats, run)
    
    pickle.dump( stats, open( CONFIG.ANALYZE_DIR + 'stats/stats.p', 'wb' ) )
      
    print('regression (and classification) finished')
    
    mean_stats = dict({})
    std_stats = dict({})
    
    for s in classification_stats:
      mean_stats[s] = 0.5*np.ones((self.num_prev_frames+1))
      std_stats[s] = 0.5*np.ones((self.num_prev_frames+1))
    for s in regression_stats:
      mean_stats[s] = np.zeros((self.num_prev_frames+1,))
      std_stats[s] = np.zeros((self.num_prev_frames+1,))
    mean_stats['coef'] = np.zeros((self.num_prev_frames+1, len(self.X_names)*(self.num_prev_frames+1)))
    std_stats['coef'] = np.zeros((self.num_prev_frames+1, len(self.X_names)*(self.num_prev_frames+1)))
    
    for num_frames in range(self.num_prev_frames+1):
      for s in stats:
        if s not in [ 'metric_names']:
          mean_stats[s][num_frames] = np.mean(stats[s][num_frames], axis=0)
          std_stats[s][num_frames]  = np.std( stats[s][num_frames], axis=0)

    plot_coef_timeline(mean_stats, self.X_names)
    
    num_timeseries = np.arange(1, self.num_prev_frames+2)
      
    plot_train_val_test_timeline(num_timeseries, np.asarray(mean_stats['regr_train_r2']), np.asarray(std_stats['regr_train_r2']), np.asarray(mean_stats['regr_val_r2']), np.asarray(std_stats['regr_val_r2']), np.asarray(mean_stats['regr_test_r2']), np.asarray(std_stats['regr_test_r2']), 'r2')
          
    plot_train_val_test_timeline(num_timeseries, np.asarray(mean_stats['penalized_train_auroc']), np.asarray(std_stats['penalized_train_auroc']), np.asarray(mean_stats['penalized_val_auroc']), np.asarray(std_stats['penalized_val_auroc']), np.asarray(mean_stats['penalized_test_auroc']), np.asarray(std_stats['penalized_test_auroc']), 'auc')
      
    plot_train_val_test_timeline(num_timeseries, np.asarray(mean_stats['penalized_train_acc']), np.asarray(std_stats['penalized_train_acc']), np.asarray(mean_stats['penalized_val_acc']), np.asarray(std_stats['penalized_val_acc']), np.asarray(mean_stats['penalized_test_acc']), np.asarray(std_stats['penalized_test_acc']), 'acc')
      
    write_instances_info(metrics, mean_stats, std_stats)
    
    print('time needed ', time.time()-t)
    
    
  def init_stats_frames( self, classification_stats, regression_stats ):
  
    stats = dict({})
                    
    for s in classification_stats:
      stats[s] = 0.5*np.ones((self.num_prev_frames+1, self.runs))
      
    for s in regression_stats:
      stats[s] = np.zeros((self.num_prev_frames+1, self.runs))
    
    stats['coef'] = np.zeros((self.num_prev_frames+1, self.runs, len(self.X_names)*(self.num_prev_frames+1)))
    stats['metric_names'] = self.X_names
    
    return stats 
  
  
  def fit_reg_cl_run_timeseries( self, stats, run ): 
    
    num_metrics = len(self.X_names)
    
    if self.flag_obj_seg == 0:
      short_name = '_o'
    else:
      short_name = '_s'
    
    metrics = metrics_load( self.tvs[run], 0, self.epsilon, self.num_reg, 1 ) 
    Xa = time_series_metrics_to_nparray( metrics, self.X_names, normalize=True )
    ya = metrics['iou'+short_name]
    y0a = metrics['iou0'+short_name]
    
    Xa_train_all, Xa_val_all, Xa_test_all, ya_train, ya_val, ya_test, y0a_train, y0a_val, y0a_test = split_tvs_and_concatenate( Xa, ya, y0a, self.tvs[run], run )
    
    for num_frames in range(self.num_prev_frames+1):
      
      Xa_train = Xa_train_all[:,0:(num_metrics * (num_frames+1))]
      Xa_val= Xa_val_all[:,0:(num_metrics * (num_frames+1))]
      Xa_test = Xa_test_all[:,0:(num_metrics * (num_frames+1))]
      
      print('run', run, self.tvs[run], 'num frames', num_frames, 'shapes:', 'Xa train', np.shape(Xa_train), 'Xa val', np.shape(Xa_val), 'Xa test', np.shape(Xa_test), 'ya train', np.shape(ya_train), 'ya val', np.shape(ya_val), 'ya test', np.shape(ya_test))
      
      # classification
      y0a_train_pred, y0a_val_pred, y0a_test_pred, coefs_model = classification_fit_and_predict( Xa_train, y0a_train, Xa_val, y0a_val, Xa_test, num_frames )

      stats['penalized_train_acc'][num_frames,run] = np.mean( np.argmax(y0a_train_pred,axis=-1)==y0a_train )
      stats['penalized_val_acc'][num_frames,run] = np.mean( np.argmax(y0a_val_pred,axis=-1)==y0a_val )
      stats['penalized_test_acc'][num_frames,run] = np.mean( np.argmax(y0a_test_pred,axis=-1)==y0a_test )

      fpr, tpr, _ = roc_curve(y0a_train, y0a_train_pred[:,1])
      stats['penalized_train_auroc'][num_frames,run] = auc(fpr, tpr)
      fpr, tpr, _ = roc_curve(y0a_val, y0a_val_pred[:,1])
      stats['penalized_val_auroc'][num_frames,run] = auc(fpr, tpr)
      fpr, tpr, _ = roc_curve(y0a_test, y0a_test_pred[:,1])
      stats['penalized_test_auroc'][num_frames,run] = auc(fpr, tpr)
      
      stats['iou0_found'][num_frames,run] = np.sum( np.logical_and(np.argmax(y0a_test_pred,axis=-1)  == 1, y0a_test == 1) ) \
                                          + np.sum( np.logical_and(np.argmax(y0a_val_pred,axis=-1) == 1, y0a_val == 1) ) \
                                          + np.sum( np.logical_and(np.argmax(y0a_train_pred,axis=-1) == 1, y0a_train == 1) )
      stats['iou0_not_found'][num_frames,run] = np.sum( np.logical_and(np.argmax(y0a_test_pred,axis=-1) == 0, y0a_test == 1) ) \
                                              + np.sum( np.logical_and(np.argmax(y0a_val_pred,axis=-1) == 0, y0a_val == 1) ) \
                                              + np.sum( np.logical_and(np.argmax(y0a_train_pred,axis=-1) == 0, y0a_train == 1) )
      stats['not_iou0_found'][num_frames,run] = np.sum( np.logical_and(np.argmax(y0a_test_pred,axis=-1) == 0, y0a_test == 0) ) \
                                              + np.sum( np.logical_and(np.argmax(y0a_val_pred,axis=-1) == 0, y0a_val == 0) ) \
                                              + np.sum( np.logical_and(np.argmax(y0a_train_pred,axis=-1) == 0, y0a_train == 0) )
      stats['not_iou0_not_found'][num_frames,run] = np.sum( np.logical_and(np.argmax(y0a_test_pred,axis=-1) == 1, y0a_test == 0) ) \
                                                  + np.sum( np.logical_and(np.argmax(y0a_val_pred,axis=-1) == 1, y0a_val == 0) ) \
                                                  + np.sum( np.logical_and(np.argmax(y0a_train_pred,axis=-1) == 1, y0a_train == 0) )
        
      coefs_tmp = np.zeros(((self.num_prev_frames+1)*num_metrics))
      coefs_tmp[0:(num_frames+1)*num_metrics] = coefs_model
      stats['coef'][num_frames,run] = np.asarray(coefs_tmp)
      
      # regression
      ya_train_pred, ya_val_pred, ya_test_pred, _ = regression_fit_and_predict( Xa_train, ya_train, Xa_val, ya_val, Xa_test, num_frames )

      stats['regr_train_mse'][num_frames,run] = np.sqrt( mean_squared_error(ya_train, ya_train_pred) )
      stats['regr_val_mse'][num_frames,run] = np.sqrt( mean_squared_error(ya_val, ya_val_pred) )
      stats['regr_test_mse'][num_frames,run] = np.sqrt( mean_squared_error(ya_test, ya_test_pred) )
      
      stats['regr_train_r2'][num_frames,run]  = r2_score(ya_train, ya_train_pred)
      stats['regr_val_r2'][num_frames,run]  = r2_score(ya_val, ya_val_pred)
      stats['regr_test_r2'][num_frames,run]  = r2_score(ya_test, ya_test_pred)
      
      if run == 0:
        plot_regression_scatter( Xa_test, ya_test, ya_test_pred, self.X_names, num_frames )
      
    return stats



#----------------------------#
class compute_mean_ap(object):
#----------------------------#

  def __init__(self):
    '''
    object initialization
    '''

  def compute_map(self):
    '''
    calculate mean average precision with application of meta classification
    '''

    print('calculating mean average precision using meta classification')
    start = time.time()
    
    comp_mean_average_precision('')

    print('Time needed:', time.time()-start)


  
#----------------------------#
class compute_fn_fp(object):
#----------------------------#

  def __init__(self):
    '''
    object initialization
    :param epsilon:         (int)   used in matching algorithm
    :param num_reg:         (int)   used in matching algorithm
    :param runs:            (int)   number of resamplings 
    :param num_prev_frames: (int)   number of previous considered frames
    :param iou_th:          (float) iou threshold
    :param flag_dp:         (int)   used for false negative calculation
    :param flag_dg:         (int)   used for false negative calculation
    :param mc_threshold:    (float) threshold for meta classification
    :param max_inst:        (int)   maximum number of instances in all image sequences
    '''
    self.epsilon         = CONFIG.EPS_MATCHING
    self.num_reg         = CONFIG.NUM_REG_MATCHING
    self.runs            = CONFIG.NUM_RESAMPLING
    self.num_prev_frames = CONFIG.NUM_PREV_FRAMES
    self.iou_th          = CONFIG.IOU_THRESHOLD
    self.flag_dp         = CONFIG.FLAG_DOUBLE_PRED
    self.flag_dg         = CONFIG.FLAG_DOUBLE_GT
    self.mc_threshold    = CONFIG.MC_THRESHOLD
    self.max_inst        = compute_max_inst()


  def compute_fn_fp_vid(self):
    '''
    calculate mean average precision with application of meta classification
    '''

    print('calculating false negatives/positives instances using meta classification', self.iou_th, self.mc_threshold)
    start = time.time()
    
    if not os.path.isfile( CONFIG.HELPER_DIR + 'detected_fn' + str(self.iou_th) + str(self.flag_dp) + '.npy' ):
      detected_fn = compute_detected_fn(self.iou_th, self.flag_dp, self.max_inst)
      np.save(os.path.join(CONFIG.HELPER_DIR, 'detected_fn' + str(self.iou_th) + str(self.flag_dp)), detected_fn)
    else:
      detected_fn = np.load(CONFIG.HELPER_DIR + 'detected_fn' + str(self.iou_th) + str(self.flag_dp) + '.npy')
    
    compute_num_fn(''+str( np.round(1-self.mc_threshold,2) ), detected_fn, self.iou_th, self.flag_dp, self.flag_dg, self.num_prev_frames, np.round(1-self.mc_threshold,2))
    
    inst_time_series_tested = helpers_p_load('inst_time_series_tested')
    
    if not os.path.isfile( CONFIG.HELPER_DIR + 'y0a' + str(self.iou_th) + str(self.flag_dg) + '.npy' ):
      _, y0a = compute_ya_y0a(self.iou_th, self.flag_dg, self.max_inst)
      np.save(os.path.join(CONFIG.HELPER_DIR, 'y0a' + str(self.iou_th) + str(self.flag_dg)), y0a)
    else:
      y0a = np.load(CONFIG.HELPER_DIR + 'y0a' + str(self.iou_th) + str(self.flag_dg) + '.npy')
    
    tvs = np.load(CONFIG.HELPER_DIR + 'tvs_runs' + str(self.runs ) + '.npy')
    
    if CONFIG.IMG_TYPE == 'kitti':
      self.runs = 3 
    
    list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR ))
    y0a_pred_zero_val = np.zeros(( (CONFIG.NUM_IMAGES-(len(list_videos)*self.num_prev_frames)) * self.max_inst ))  
      
    for run in range(self.runs): 
      
      train_val_test_string = tvs[run]
      
      metrics = metrics_load( tvs[run], 0, self.epsilon, self.num_reg, 1 ) 
      X_names = sorted([ m for m in metrics if m not in ['iou_s','iou0_s','iou_o','iou0_o']]) 
      Xa = time_series_metrics_to_nparray( metrics, X_names, normalize=True )
      
      print(len(metrics['class']), np.shape(y0a), np.shape(y0a_pred_zero_val))

      Xa_train, _, _, _, _, _, y0a_train, _, _ = split_tvs_and_concatenate( Xa, y0a, y0a, tvs[run] )
      
      Xa_val, y0a_val, y0a_zero_val, not_del_rows_val, _ = concatenate_val_for_visualization( Xa, y0a )
      
      print('shapes:', 'Xa train', np.shape(Xa_train), 'y0a train', np.shape(y0a_train), 'Xa val',  np.shape(Xa_val), 'y0a val', np.shape(y0a_val))
          
      y_train_pred, y_val_pred, _, _ = classification_fit_and_predict(Xa_train, y0a_train, Xa_val, y0a_val, Xa_val) 
      
      print(run, np.sum(y0a_val), np.sum(np.argmax(y_val_pred,axis=-1)), np.sum((y0a_val+np.argmax(y_val_pred,axis=-1)==2)))
      
      fpr, tpr, _ = roc_curve(y0a_train, y_train_pred[:,1])
      print('time series model auroc score (train):', auc(fpr, tpr) )
      fpr, tpr, _ = roc_curve(y0a_val, y_val_pred[:,1])
      print('time series model auroc score (val):', auc(fpr, tpr) )
      print(' ')
          
      y0a_pred = np.ones((y0a_zero_val.shape[0]))*-1
      y0a_calc = np.ones((y0a_zero_val.shape[0]))*-1 
      
      y_val_pred_argmax = [1 if y_val_pred[i,1]>self.mc_threshold else 0 for i in range(y_val_pred.shape[0])]

      counter = 0
      for i in range(y0a_zero_val.shape[0]):
        if not_del_rows_val[i] == True:
          y0a_pred[i] = y_val_pred_argmax[counter]
          y0a_calc[i] = y0a_val[counter]
          counter += 1  

      counter = 0
      for vid,v in zip(list_videos, range(len(list_videos))):
        images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + '/' ))
        if train_val_test_string[v] == 'v' or train_val_test_string[v] == 's':
          
          y0a_pred_zero_val[counter*self.max_inst:(counter+len(images_all))*self.max_inst] = y0a_pred[counter*self.max_inst:(counter+len(images_all))*self.max_inst]
        counter += len(images_all)

    print('FP:', np.sum(y0a_pred_zero_val==1), 'TP:', np.sum(y0a_pred_zero_val==0))
    print('calculating false negatives/positives')
    
    compute_num_fn('_meta_classif'+str(self.mc_threshold), detected_fn, self.iou_th, self.flag_dp, self.flag_dg, self.num_prev_frames, 0, inst_time_series_tested, y0a_pred_zero_val)
    
    print('Time needed:', time.time()-start)


    
#----------------------------#
class plot_fn_fp(object):
#----------------------------#

  def __init__(self):
    '''
    object initialization
    :param num_prev_frames: (int)   number of previous considered frames
    :param flag_dp:         (int)   used for false negative calculation
    :param flag_dg:         (int)   used for false negative calculation
    :param iou_threshold:   (float) iou threshold
    '''
    self.num_prev_frames  = CONFIG.NUM_PREV_FRAMES
    self.flag_dp          = CONFIG.FLAG_DOUBLE_PRED
    self.flag_dg          = CONFIG.FLAG_DOUBLE_GT
    self.iou_threshold    = CONFIG.IOU_THRESHOLD


  def plot_fnfp(self):
    '''
    plot false negatives vs false positives
    '''
    print('plot false negatives vs false positives')
    
    flag_all = 0
    
    if flag_all == 0:
      name = '_separated_'
    elif flag_all == 1:
      name = '_all_'
    save_path1 = CONFIG.IMG_MAP_FN_FP_DIR[:-1] + name + str(self.flag_dp) + str(self.flag_dg) + '/'
    
    if not os.path.exists( save_path1 ):
      os.makedirs( save_path1 )
    
    # iou th constant, meta classification and score th runs
    var_part = 'th' + str(self.iou_threshold) + '_p' + str(self.flag_dp) + '_g' + str(self.flag_dg) + '_npf' + str(self.num_prev_frames) + '.'
    name1 = 'fnfp'
    name2 = 'meta_classif'
    
    counter_th = 0
    for data in sorted(os.listdir( CONFIG.RESULTS_FN_FP_DIR )):
      if name2 in data and var_part in data:
        counter_th += 1
    
    fp_fn_classic = np.zeros((counter_th, 4))
    fp_fn_meta = np.zeros((counter_th, 4))
    fp_fn_classic, fp_fn_meta = self.fill_array_small(fp_fn_classic, fp_fn_meta, var_part, name1, name2, flag_all)
    
    save_path = save_path1 + 'fn_fp_iou_' + var_part 
    plot_fn_vs_fp(save_path, fp_fn_classic, fp_fn_meta)
    
    #iou th constant, meta classification and score th runs & occlusion levels
    occlusions = get_occlusion_level()
    for i in range(occlusions.shape[0]):
      print(i)
      
      fp_fn_classic = np.zeros((counter_th, 4))
      fp_fn_meta = np.zeros((counter_th, 4))
      fp_fn_classic, fp_fn_meta = self.fill_array_small(fp_fn_classic, fp_fn_meta, var_part, name1, name2, flag_all, 1, i)
      
      save_path = save_path1 + 'fn_fp_iou_occ' + str(i) + '_' + var_part 
      plot_fn_vs_fp(save_path, fp_fn_classic, fp_fn_meta)


  def fill_array_small(self, fp_fn_classic, fp_fn_meta, var_part, name1, name2, flag_all=0, flag_occ=0, i=-1):
    '''
    fill arrays with values: FP, FN, threshold, TP
    '''
    
    counter_classic = 0
    counter_meta = 0
    for data in sorted(os.listdir( CONFIG.RESULTS_FN_FP_DIR )):

      if name1 in data and not name1 + '_' in data and var_part in data:
        with open(CONFIG.RESULTS_FN_FP_DIR + data , 'r') as f:
          
          dataset = f.read().strip().split('\n')
          fp_fn_classic[counter_classic, 2] = float( data.split('fnfp')[1].split('_')[0] )
          fp_fn_classic[counter_classic, 3] = int(float( (dataset[0].split(':')[1]).split('.0')[1] ))
          fp_fn_classic[counter_classic, 0] = int(float( dataset[1].split(':')[1] ))
          if flag_all == 1:
            fp_fn_classic[counter_classic, 1] = int(float( dataset[2].split(':')[1] ))
          else:
            array_tmp = ((dataset[3].split('[')[1]).split(']')[0]).split(' ')
            for k in range(len(array_tmp)):
              if '' in array_tmp:
                array_tmp.remove('')
            fp_fn_classic[counter_classic, 1] = int(float(array_tmp[1]))

          if flag_occ == 1:
            array_tmp = (((dataset[4].split(':')[1]).split('[')[1]).split(']')[0]).split(' ')
            array_tmp1 = (((dataset[5].split(':')[1]).split('[')[1]).split(']')[0]).split(' ')
            for k in range(max(len(array_tmp),len(array_tmp1))):
              if '' in array_tmp:
                array_tmp.remove('')
              if '' in array_tmp1:
                array_tmp1.remove('')
            if flag_all == 1:
              fp_fn_classic[counter_classic, 1] = int(float(array_tmp[i])) + int(float(array_tmp1[i]))
            else:
              fp_fn_classic[counter_classic, 1] = int(float(array_tmp[i]))
          
        counter_classic += 1
      
      if name2 in data and var_part in data:
        with open(CONFIG.RESULTS_FN_FP_DIR + data , 'r') as f:
          
          dataset = f.read().strip().split('\n')
          fp_fn_meta[counter_meta, 2] = float( data.split('meta_classif')[1].split('_')[0] )
          fp_fn_meta[counter_meta, 3] = int(float( (dataset[0].split(':')[1]).split('.0')[1] ))
          fp_fn_meta[counter_meta, 0] = int(float( dataset[1].split(':')[1] ))
          if flag_all == 1:
            fp_fn_meta[counter_meta, 1] = int(float( dataset[2].split(':')[1] ))
          else:
            array_tmp = ((dataset[3].split('[')[1]).split(']')[0]).split(' ')
            for k in range(len(array_tmp)):
              if '' in array_tmp:
                array_tmp.remove('')
            fp_fn_meta[counter_meta, 1] = int(float(array_tmp[1]))

          if flag_occ == 1:
            array_tmp = (((dataset[4].split(':')[1]).split('[')[1]).split(']')[0]).split(' ')
            array_tmp1 = (((dataset[5].split(':')[1]).split('[')[1]).split(']')[0]).split(' ')
            for k in range(max(len(array_tmp),len(array_tmp1))):
              if '' in array_tmp:
                array_tmp.remove('')
              if '' in array_tmp1:
                array_tmp1.remove('')
            if flag_all == 1:
              fp_fn_meta[counter_meta, 1] = int(float(array_tmp[i])) + int(float(array_tmp1[i]))
            else:
              fp_fn_meta[counter_meta, 1] = int(float(array_tmp[i]))
            
        counter_meta += 1
    return fp_fn_classic, fp_fn_meta  
  
  
  



    
