#!/usr/bin/env python3
'''
script including
functions that do calculations
'''

import os
import gc
import time
import pickle
import numpy as np
import xgboost as xgb
from multiprocessing import Pool
from scipy.stats import linregress
from sklearn.metrics import r2_score

from global_defs import CONFIG
from helper      import time_series_metrics_to_nparray
from in_out      import ground_truth_load, time_series_instances_load, metrics_load, get_save_path_metrics_i, metrics_dump, helpers_p_load, score_small_load
from metrics     import compute_matches_gt_pred, shifted_iou, test_occlusion


def create_bb(mask):
  '''
  create bounding boxes
  '''
  
  # shape boxes: [num_instances, (y1, x1, y2, x2)], mask: [height, width, num_instances] with 0/1
  boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
  for i in range(mask.shape[-1]):
    m_i = mask[:, :, i]
    horizontal_indicies = np.where(np.any(m_i, axis=0))[0]
    vertical_indicies = np.where(np.any(m_i, axis=1))[0]
    if horizontal_indicies.shape[0]:
      x1, x2 = horizontal_indicies[[0, -1]]
      y1, y2 = vertical_indicies[[0, -1]]
      # x2 and y2 should not be part of the box. Increment by 1.
      x2 += 1
      y2 += 1
    else:
      x1, x2, y1, y2 = 0, 0, 0, 0
    boxes[i] = np.array([y1, x1, y2, x2])
  return boxes.astype(np.int32)
    
    
def compute_metrics_2d( instances, gt, depth, scores, max_instances ): 
  
  instances_tmp = instances.copy()
  instances_tmp[instances_tmp<0] *= -1
  instances_id = instances_tmp  % 10000
  instances_id[instances<0] *= -1
  instances_class = instances_tmp // 10000
    
  pred_match_s, _ = compute_matches_gt_pred(np.array(gt, dtype='int16'), instances, scores, 0, 0, 1) # semantic segmentation
  pred_match_o, _ = compute_matches_gt_pred(np.array(gt, dtype='int16'), instances, scores, 0.5, 0, 0) # object detection
  
  metrics = { 'iou_s': list([]), 'iou0_s': list([]), 'iou_o': list([]), 'iou0_o': list([]), 'class': list([]), 'mean_x': list([]), 'mean_y': list([]), 'S': list([]), 'S_in': list([]), 'S_bd': list([]), 'S_rel': list([]), 'S_rel_in': list([]), 'D': list([]), 'D_in': list([]), 'D_bd': list([]), 'D_rel': list([]), 'D_rel_in': list([]), 'score': list([]) } 

  # all arrays have the same lenght and empty instances get the vaulue 0 forall metrics
  for i in range( 1, max_instances+1 ):
    
    for m in metrics:
      metrics[m].append( 0 )
      
    index = -1
    for m in range(instances.shape[0]):
      if instances_id[m,:,:].min() == -i:
        index = m
        break
    
    if index > -1:
      
      metrics['class'][-1] = instances_class[index].max()
      metrics['score'][-1] = scores[index]
      
      n_in = np.sum(instances_id[index]==i)
      n_bd = np.sum(instances_id[index]==-i)
      metrics['S'       ][-1] = n_in + n_bd
      metrics['S_in'    ][-1] = n_in
      metrics['S_bd'    ][-1] = n_bd
      metrics['S_rel'   ][-1] = float( n_in + n_bd ) / float(n_bd)
      metrics['S_rel_in'][-1] = float( n_in ) / float(n_bd)
      
      tmp_ind = np.where(instances_id[index]!=0)
      metrics['mean_x'][-1] = np.sum(tmp_ind[0]) / (n_in + n_bd)
      metrics['mean_y'][-1] = np.sum(tmp_ind[1]) / (n_in + n_bd)
      
      tmp_ind_in = np.where(instances_id[index]>0)
      tmp_ind_bd = np.where(instances_id[index]<0)
      metrics['D'       ][-1] = np.sum(depth[tmp_ind[0],tmp_ind[1]]) / (n_in + n_bd)
      if n_in > 0:
        metrics['D_in'    ][-1] = np.sum(depth[tmp_ind_in[0],tmp_ind_in[1]]) / n_in
      metrics['D_bd'    ][-1] = np.sum(depth[tmp_ind_bd[0],tmp_ind_bd[1]]) / n_bd
      metrics['D_rel'   ][-1] = metrics['D'   ][-1] * metrics['S_rel'   ][-1]
      metrics['D_rel_in'][-1] = metrics['D_in'][-1] * metrics['S_rel_in'][-1]
      
      I_s = np.sum( np.logical_and(instances[index]!=0,gt==pred_match_s[index]) )
      U_s = np.sum(instances[index]!=0) + np.sum(gt==pred_match_s[index]) - I_s
      if U_s > 0:
        metrics['iou_s' ][-1] = float(I_s) / float(U_s)
        metrics['iou0_s'][-1] = 0 if (float(I_s) / float(U_s))>0 else 1
      else:
        metrics['iou_s' ][-1] = 0
        metrics['iou0_s'][-1] = 1
        
      I_o = np.sum( np.logical_and(instances[index]!=0,gt==pred_match_o[index]) )
      U_o = np.sum(instances[index]!=0) + np.sum(gt==pred_match_o[index]) - I_o
      if U_o > 0:
        metrics['iou_o' ][-1] = float(I_o) / float(U_o)
        metrics['iou0_o'][-1] = 0 if (float(I_o) / float(U_o))>=0.5 else 1
      else:
        metrics['iou_o' ][-1] = 0
        metrics['iou0_o'][-1] = 1
  
  return metrics


def compute_ya_y0a_i(vid, n, instance_new, new_scores, iou_threshold, flag_double_gt, max_instances):
  
  epsilon = CONFIG.EPS_MATCHING
  num_reg = CONFIG.NUM_REG_MATCHING
  
  gc.collect()
  
  metrics = { 'iou': list([]), 'iou0': list([]) }
  
  gt = ground_truth_load(vid, n)
  instances  = time_series_instances_load( vid, n, epsilon, num_reg )
  instances = np.concatenate((instances, instance_new), axis=0)
  instances[instances<0] *= -1
  instances_id = instances  % 10000
  
  scores = score_small_load(vid, n)
  scores = np.concatenate((scores, new_scores), axis=0)
  
  pred_match, _ = compute_matches_gt_pred(np.array(gt, dtype='int16'), instances, scores, iou_threshold, 0, flag_double_gt)

  # all arrays have the same lenght and empty instances get the vaulue 0 forall metrics
  for i in range( 1, max_instances+1 ):
    
    for m in metrics:
      metrics[m].append( 0 )
      
    index = -1
    for m in range(instances.shape[0]):
      if instances_id[m,:,:].max() == i:
        index = m
        break
    
    if index > -1:
      
      I = np.sum( np.logical_and(instances[index]!=0,gt==pred_match[index]) )
      U = np.sum(instances[index]!=0) + np.sum(gt==pred_match[index]) - I
      if U > 0:
        metrics['iou' ][-1] = float(I) / float(U)
        metrics['iou0'][-1] = 0 if (float(I) / float(U))>=iou_threshold else 1
        if iou_threshold == 0:
          metrics['iou0'][-1] = int(I == 0)
      else:
        metrics['iou' ][-1] = 0
        metrics['iou0'][-1] = 1

  print('image', n, 'finished')
  return metrics


def compute_ya_y0a(iou_threshold, flag_double_gt, max_instances):
  '''
  compute iou values
  '''
  print('compute iou values with iou threshold', iou_threshold, 'and flag_double_gt', flag_double_gt)
  
  instances_new = helpers_p_load('inst_time_series_tested')
  array_vid = np.asarray(instances_new['vid_no'])
  array_frame = np.asarray(instances_new['frame_no'])
  if CONFIG.IMG_TYPE == 'kitti':
    array_inst = np.stack(instances_new['inst_array'])
  array_score = np.asarray(instances_new['score_val'])
  
  metrics = { 'iou': list([]), 'iou0': list([]) }
  metrics_i = { 'iou': list([]), 'iou0': list([]) }
  
  list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR ))
  for vid in list_videos: 
    print(vid, len(metrics['iou']))
    
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
    
    p = Pool(CONFIG.NUM_CORES)
    p_args = [ (vid,k,array_inst[np.logical_and(array_vid==vid, array_frame==k)],array_score[np.logical_and(array_vid==vid, array_frame==k)],iou_threshold,flag_double_gt,max_instances) for k in range(len(images_all)) ]
    metrics_i = p.starmap( compute_ya_y0a_i, p_args ) 
    p.close()
    
    for i in range(len(images_all)):  
      metrics['iou'].extend(metrics_i[i]['iou'])
      metrics['iou0'].extend(metrics_i[i]['iou0'])
    
    print('save', vid, len(metrics['iou']))
    pickle.dump( metrics, open( os.path.join(CONFIG.HELPER_DIR, 'y0a' + str(iou_threshold) + str(flag_double_gt) + vid + '.p'), "wb" ) )
            
  return metrics['iou'], metrics['iou0']
            

def compute_splitting( runs ):    
  '''
  compute train/val/test splitting
  '''
  print('compute splitting')
  
  if CONFIG.IMG_TYPE == 'kitti':
    train_val_test = [6,1,2]
  elif CONFIG.IMG_TYPE == 'mot':
    train_val_test = [3,1,0] 

  list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR ))
  tvs = np.zeros((len(list_videos)), dtype='int16')
  list_tvs = []
  
  if CONFIG.IMG_TYPE == 'kitti':
    list_tvs.append('ssvtttttt')
    list_tvs.append('tttvssttt')
    list_tvs.append('ttttttsvs')

  run = 0
  while True:
    
    np.random.seed( run )
    mask = np.random.rand(len(list_videos))
    sorted_mask = np.argsort(mask)
    
    counter = 0
    for t in range(train_val_test[0]):
      tvs[sorted_mask[counter]] = 0
      counter += 1
    for v in range(train_val_test[1]):
      tvs[sorted_mask[counter]] = 1
      counter += 1
    for s in range(train_val_test[2]):
      tvs[sorted_mask[counter]] = 2
      counter += 1
    
    tmp_name = ''
    for i in range(len(list_videos)):
      if tvs[i] == 0:
        tmp_name += 't'
      elif tvs[i] == 1:
        tmp_name += 'v'
      elif tvs[i] == 2:
        tmp_name += 's'
        
    if tmp_name not in list_tvs:
      list_tvs.append(tmp_name)
      
    if len(list_tvs) == runs:
      break
    
    run += 1

  print('train/val/test splitting', list_tvs)
  np.save(os.path.join(CONFIG.HELPER_DIR, 'tvs_runs' + str(runs)), list_tvs)


def survival_fit_and_predict( X_train, y_train, X_val, X_test ):
  '''
  survival analysis
  '''
  
  model = xgb.XGBRegressor(objective='survival:cox',
                        base_score=1,
                        n_estimators=50,
                        max_depth=4, learning_rate=0.1)
  model.fit(X_train, y_train)
  y_train_pred = model.predict(X_train)
  y_val_pred = model.predict(X_val, output_margin=True)
  y_test_pred = model.predict(X_test, output_margin=True)
  
  return y_train_pred, y_val_pred, y_test_pred
  

def compute_3d_gt_metrics( max_inst ):
  '''
  compute additional three-dim. time series metrics 
  '''
  print('calculating 3d time series metrics')
  start = time.time()
  
  epsilon = CONFIG.EPS_MATCHING
  num_reg = CONFIG.NUM_REG_MATCHING
  classes = CONFIG.CLASSES
  runs = CONFIG.NUM_RESAMPLING
  num_images = CONFIG.NUM_IMAGES
  
  instances_new = helpers_p_load('inst_time_series_tested')
  array_vid = np.asarray(instances_new['vid_no'])
  array_frame = np.asarray(instances_new['frame_no'])
  if CONFIG.IMG_TYPE == 'kitti':
    array_inst = np.stack(instances_new['inst_array'])
  array_score = np.asarray(instances_new['score_val'])
  
  metrics_2d = metrics_load( 'all', '_2d', epsilon, num_reg )
  print(len(metrics_2d['class']), len(metrics_2d['class'])/num_images )
  
  survival_prev_frames = 5
  frames_regression = 5
  
  list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR ))
  
  if not os.path.isfile( CONFIG.HELPER_DIR + 'matching_inst_gt.npy' ):
    
    matching_inst_gt = np.zeros((len(list_videos), num_images, max_inst), dtype='int16')
    gt_01 = np.zeros((len(list_videos), num_images, 3000), dtype='int16')
    
    # height / width
    size_ratio = np.zeros((len(list_videos), len(classes))) 
    counter_size_ratio = np.zeros((len(list_videos), len(classes))) 
    instances_size_ratio = np.zeros((len(list_videos), num_images, max_inst))
    
    # deformations
    instances_deformation = np.zeros((len(list_videos), num_images, max_inst))
    
    # occlusions
    instances_occlusion = np.zeros((len(list_videos), num_images, max_inst))
    
    # mean x, mean y, size, depth
    predicted_measures = np.zeros((len(list_videos), num_images, max_inst, 4))
    # distance mean predicted and calculated, difference size, difference depth
    diff_measures = np.zeros((len(list_videos), num_images, max_inst, 3))
    
    counter = 0
    
    for vid, v in zip(list_videos, range(len(list_videos))): 
      if v == 0 or v == 1 or v == 2:
        images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + '/' ))
        counter += len(images_all)
        continue
      print('video:', vid)
      
      if CONFIG.IMG_TYPE == 'mot':
        array_inst_tmp = []
        gt = ground_truth_load(vid, 0)
        print(array_vid.shape[0])
        for i in range(array_vid.shape[0]):
          if array_vid[i] == vid:
            array_inst_tmp.append(instances_new['inst_array'][i])
          else:
            array_inst_tmp.append(np.zeros(gt.shape,dtype='int16'))
        array_inst = np.stack(array_inst_tmp)
      print('stacked')
        
      images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + '/' ))
      for n in range(len(images_all)):
        print(n, counter)
      
        gt = np.array(ground_truth_load(vid, n), dtype='int16')
        instance  = time_series_instances_load( vid, n, epsilon, num_reg )
        instance = np.concatenate((instance, array_inst[np.logical_and(array_vid==vid, array_frame==n)]), axis=0)
        scores = score_small_load(vid, n)
        scores = np.concatenate((scores, array_score[np.logical_and(array_vid==vid, array_frame==n)]), axis=0)
        
        pred_match, _ = compute_matches_gt_pred(gt, instance, scores, 0.5, 0, 1)
        
        instance[instance<0] *= -1
        instance = instance % 10000
        
        ## survival, ratio
        for j in range(instance.shape[0]):
          
          matching_inst_gt[v, n, instance[j].max()-1] = pred_match[j]

          coord_indicies = np.where(instance[j] > 0)
          if (coord_indicies[1].max()-coord_indicies[1].min()) != 0:
            instances_size_ratio[v, n, instance[j].max()-1] = (coord_indicies[0].max()-coord_indicies[0].min()) / (coord_indicies[1].max()-coord_indicies[1].min())

        for j in np.unique(gt):
          if j != 0 and j != 10000:
            
            gt_01[v, n, j] = 1
            
            coord_indicies = np.where(gt == j)
            if (gt[gt==j].max() // 1000) == classes[0]:
              size_ratio[v, 0] += (coord_indicies[0].max()-coord_indicies[0].min()) / (coord_indicies[1].max()-coord_indicies[1].min())
              counter_size_ratio[v, 0] += 1
            else:
              size_ratio[v, 1] += (coord_indicies[0].max()-coord_indicies[0].min()) / (coord_indicies[1].max()-coord_indicies[1].min())
              counter_size_ratio[v, 1] += 1   
        
        ## deformation and occlusion
        if n >= 1:
          
          for j in range(instance.shape[0]):
            if instance[j].max() in np.unique(instance_t_1):
              
              instance_tmp = instance[j].copy()
              instance_tmp[instance_tmp > 0] = 1
              
              for k in range(instance_t_1.shape[0]):
                if instance_t_1[k].max() == instance[j].max():
                  break
              instance_t_1_tmp = instance_t_1[k].copy()
              instance_t_1_tmp[instance_t_1_tmp > 0] = 1
              
              instances_deformation[v, n, instance[j].max()-1] = shifted_iou(instance_tmp, instance_t_1_tmp)
              instances_occlusion[v, n, instance[j].max()-1] = test_occlusion(instance, j, instance_t_1_tmp) 
        instance_t_1 = instance.copy()
        
        ## diff_mean, diff_size, diff_depth
        reg_steps = min(frames_regression, n)
        for j in range(instance.shape[0]):
          
          id_instance = instance[j].max()-1
          mean_x_list = []
          mean_y_list = []
          size_list = []
          depth_list = []
          time_list = []

          for k in range(reg_steps):
            
            if metrics_2d['S'][max_inst*counter+id_instance - (max_inst*(reg_steps-k))] > 0:
              mean_x_list.append(metrics_2d['mean_x'][max_inst*counter+id_instance - (max_inst*(reg_steps-k))])
              mean_y_list.append(metrics_2d['mean_y'][max_inst*counter+id_instance - (max_inst*(reg_steps-k))])
              size_list.append(metrics_2d['S'][max_inst*counter+id_instance - (max_inst*(reg_steps-k))])
              depth_list.append(metrics_2d['D'][max_inst*counter+id_instance - (max_inst*(reg_steps-k))])
              time_list.append(k)
          
          if len(time_list) == 0:
            predicted_measures[v, n, id_instance, 0] = metrics_2d['mean_x'][max_inst*counter+id_instance]
            predicted_measures[v, n, id_instance, 1] = metrics_2d['mean_y'][max_inst*counter+id_instance]
            predicted_measures[v, n, id_instance, 2] = metrics_2d['S'][max_inst*counter+id_instance]
            predicted_measures[v, n, id_instance, 3] = metrics_2d['D'][max_inst*counter+id_instance]
          elif len(time_list) == 1:
            predicted_measures[v, n, id_instance, 0] = mean_x_list[0]
            predicted_measures[v, n, id_instance, 1] = mean_y_list[0]
            predicted_measures[v, n, id_instance, 2] = size_list[0] 
            predicted_measures[v, n, id_instance, 3] = depth_list[0] 
          else:
            b_x, a_x, _, _, _ = linregress(time_list, mean_x_list)
            b_y, a_y, _, _, _ = linregress(time_list, mean_y_list)
            b_s, a_s, _, _, _ = linregress(time_list, size_list)
            b_d, a_d, _, _, _ = linregress(time_list, depth_list)
            predicted_measures[v, n, id_instance, 0] = a_x + b_x * reg_steps
            predicted_measures[v, n, id_instance, 1] = a_y + b_y * reg_steps
            predicted_measures[v, n, id_instance, 2] = a_s + b_s * reg_steps
            predicted_measures[v, n, id_instance, 3] = a_d + b_d * reg_steps
          
          diff_measures[v, n, id_instance, 0] = ( (predicted_measures[v, n, id_instance, 0] - metrics_2d['mean_x'][max_inst*counter+id_instance])**2 + (predicted_measures[v, n, id_instance, 1] - metrics_2d['mean_y'][max_inst*counter+id_instance])**2 )**0.5
          diff_measures[v, n, id_instance, 1] = predicted_measures[v, n, id_instance, 2] - metrics_2d['S'][max_inst*counter+id_instance]
          diff_measures[v, n, id_instance, 2] = predicted_measures[v, n, id_instance, 3] - metrics_2d['D'][max_inst*counter+id_instance]
        counter += 1
        
    np.save(os.path.join(CONFIG.HELPER_DIR, 'matching_inst_gt'), matching_inst_gt)
    np.save(os.path.join(CONFIG.HELPER_DIR, 'gt_01'), gt_01) 
    np.save(os.path.join(CONFIG.HELPER_DIR, 'size_ratio'), size_ratio) 
    np.save(os.path.join(CONFIG.HELPER_DIR, 'counter_size_ratio'), counter_size_ratio) 
    np.save(os.path.join(CONFIG.HELPER_DIR, 'instances_size_ratio'), instances_size_ratio) 
    np.save(os.path.join(CONFIG.HELPER_DIR, 'instances_deformation'), instances_deformation) 
    np.save(os.path.join(CONFIG.HELPER_DIR, 'instances_occlusion'), instances_occlusion) 
    np.save(os.path.join(CONFIG.HELPER_DIR, 'diff_measures'), diff_measures) 
    
  else:

    matching_inst_gt = np.load(CONFIG.HELPER_DIR + 'matching_inst_gt.npy')
    gt_01 = np.load(CONFIG.HELPER_DIR + 'gt_01.npy')
    size_ratio = np.load(CONFIG.HELPER_DIR + 'size_ratio.npy')
    counter_size_ratio = np.load(CONFIG.HELPER_DIR + 'counter_size_ratio.npy')
    instances_size_ratio = np.load(CONFIG.HELPER_DIR + 'instances_size_ratio.npy')
    instances_deformation = np.load(CONFIG.HELPER_DIR + 'instances_deformation.npy')
    instances_occlusion = np.load(CONFIG.HELPER_DIR + 'instances_occlusion.npy')
    diff_measures = np.load(CONFIG.HELPER_DIR + 'diff_measures.npy')
    
  print('preparations done in {}s\r'.format( round(time.time()-start) ) )
  
  if not os.path.isfile( CONFIG.HELPER_DIR + 'tvs_runs' + str(runs) + '.npy' ):
    compute_splitting(runs)
  tvs = np.load(CONFIG.HELPER_DIR + 'tvs_runs' + str(runs) + '.npy') 
  
  new_metrics = ['survival', 'ratio', 'deformation', 'diff_mean', 'diff_size', 'diff_depth', 'occlusion']
  
  Xa_names = sorted([ m for m in metrics_2d if m not in ['iou_s','iou0_s','iou_o','iou0_o']]) 
  Xa = time_series_metrics_to_nparray( metrics_2d, Xa_names, normalize=True )
  
  for run in range(runs):
    
    if not os.path.isfile( get_save_path_metrics_i(tvs[run], 0, epsilon, num_reg, 1) ):
      
      print('start run', run)
      start = time.time()
      
      metrics_3d = metrics_2d.copy()
      for m in list(new_metrics):
        metrics_3d[m] = list([])
      
      # create survival model
      Xa_train_surv = np.zeros(( num_images * max_inst, Xa.shape[1] * (survival_prev_frames+1)))
      ya_train_surv = np.zeros(( num_images* max_inst ))
      Xa_train = np.zeros(( num_images * max_inst, Xa.shape[1] * (survival_prev_frames+1)))
      ya_train = np.zeros(( num_images * max_inst ))
      Xa_val = np.zeros(( num_images * max_inst, Xa.shape[1] * (survival_prev_frames+1)))
      ya_val = np.zeros(( num_images * max_inst ))

      counter = 0
      counter_train_surv = 0
      counter_train = 0
      counter_val = 0

      for vid,v in zip(list_videos, range(len(list_videos))):
        images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + '/' ))
        for i in range(len(images_all)):
          
          for j in range(max_inst):
            
            flag_same_gt = 1
              
            if ~(Xa[counter]==0).all():
              
              tmp = np.zeros(( Xa.shape[1] * (survival_prev_frames+1) ))
              for k in range(0,survival_prev_frames+1):
                if counter-(max_inst*k) >= 0:
                  tmp[Xa.shape[1]*k:Xa.shape[1]*(k+1)] = Xa[counter-(max_inst*k)] 
                
                if i >= survival_prev_frames:
                  if matching_inst_gt[v, i, j] == 0 or matching_inst_gt[v, i, j] != matching_inst_gt[v, i-k, j]:
                    flag_same_gt = 0
                else:
                  flag_same_gt = 0 
              
              if tvs[run][v] == 't' and flag_same_gt == 1:
                Xa_train_surv[counter_train_surv,:] = tmp
                ya_train_surv[counter_train_surv] = np.sum( gt_01[v, (i+1):, matching_inst_gt[v, i, j]] )
                counter_train_surv +=1
              if tvs[run][v] == 't':
                Xa_train[counter_train,:] = tmp
                ya_train[counter_train] = np.sum( gt_01[v, (i+1):, matching_inst_gt[v, i, j]] )
                counter_train +=1
              if tvs[run][v] == 'v' or tvs[run][v] == 's':
                Xa_val[counter_val,:] = tmp
                ya_val[counter_val] = np.sum( gt_01[v, (i+1):, matching_inst_gt[v, i, j]] )
                counter_val +=1
            counter += 1

      # delete rows with zeros
      Xa_train_surv = Xa_train_surv[:counter_train_surv,:] 
      ya_train_surv = ya_train_surv[:counter_train_surv] 
      Xa_train = Xa_train[:counter_train,:] 
      ya_train = ya_train[:counter_train] 
      Xa_val = Xa_val[:counter_val,:] 
      ya_val = ya_val[:counter_val] 
      
      ya_train = np.squeeze(ya_train)
      ya_val = np.squeeze(ya_val)
      
      print('Shapes train survival: ', np.shape(Xa_train_surv), 'train: ', np.shape(Xa_train), 'val: ', np.shape(Xa_val) )
      y_train_surv_pred, y_train_pred, y_val_pred = survival_fit_and_predict(Xa_train_surv, ya_train_surv, Xa_train, Xa_val)
      
      print('survival model r2 score (train survival ):', r2_score(ya_train_surv,y_train_surv_pred) )
      print('survival model r2 score (train):', r2_score(ya_train,y_train_pred) )
      print('survival r2 score (val):', r2_score(ya_val,y_val_pred) )

      size_ratio_train = np.zeros((len(classes))) 
      counter_size_ratio_train = np.zeros((len(classes))) 
      for v in range(len(list_videos)):
        if tvs[run][v] == 't':
          size_ratio_train += size_ratio[v]
          counter_size_ratio_train += counter_size_ratio[v] 
      size_ratio_train /= counter_size_ratio_train
      print('size ratio', size_ratio_train)
        
      # add new metric survival
      counter = 0
      counter_train = 0
      counter_val = 0
      
      for vid,v in zip(list_videos, range(len(list_videos))):
        images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + '/' ))
        for i in range(len(images_all)):
          
          for j in range(max_inst):
            
            for m in list(new_metrics):
              metrics_3d[m].append( 0 )
              
            if ~(Xa[counter]==0).all():
              
              if metrics_3d['class'][counter] == classes[0]:
                metrics_3d['ratio'][-1] = float(instances_size_ratio[v, i, j] / size_ratio_train[0])
              else:
                metrics_3d['ratio'][-1] = float(instances_size_ratio[v, i, j] / size_ratio_train[1])
              
              if tvs[run][v] == 't':
                metrics_3d['survival'][-1] = y_train_pred[counter_train]
                counter_train +=1
              if tvs[run][v] == 'v' or tvs[run][v] == 's':
                metrics_3d['survival'][-1] = y_val_pred[counter_val]
                counter_val +=1
              
              metrics_3d['deformation'][-1] = instances_deformation[v, i, j]
              metrics_3d['occlusion'][-1] = instances_occlusion[v, i, j]
              metrics_3d['diff_mean'][-1] = diff_measures[v, i, j, 0]
              metrics_3d['diff_size'][-1] = abs(diff_measures[v, i, j, 1])
              metrics_3d['diff_depth'][-1] = abs(diff_measures[v, i, j, 2])
            counter += 1
            
      print('len', len(metrics_3d['iou_s']), len(metrics_3d['survival']))
      metrics_dump( metrics_3d, tvs[run], 0, epsilon, num_reg, 1 ) 
      print('run', run, 'processed in {}s\r'.format( round(time.time()-start) ) )


def regression_fit_and_predict( X_train, y_train, X_val, y_val, X_test, num_prev_frames = CONFIG.NUM_PREV_FRAMES ):
  '''
  fit regression models
  '''
  
  if CONFIG.IMG_TYPE == 'kitti':
    model = xgb.XGBRegressor(max_depth=5, colsample_bytree=0.5, n_estimators=100, reg_alpha=0.4, reg_lambda=0.4)
  elif CONFIG.IMG_TYPE == 'mot':
    model = xgb.XGBRegressor(max_depth=4, colsample_bytree=0.5, subsample=0.3, n_estimators=60, reg_alpha=0.4, reg_lambda=0.4, gamma=10)
    
  model.fit( X_train, y_train )
     
  y_train_pred = np.clip( model.predict(X_train), 0, 1 )
  y_val_pred = np.clip( model.predict(X_val), 0, 1 )
  y_test_R_pred = np.clip( model.predict(X_test), 0, 1 )

  return y_train_pred, y_val_pred, y_test_R_pred, model


def classification_fit_and_predict( X_train, y_train, X_val, y_val, X_test, num_prev_frames = CONFIG.NUM_PREV_FRAMES ):
  '''
  fit classification models
  '''
  
  coefs_model = np.zeros((X_train.shape[1]))
  if CONFIG.IMG_TYPE == 'kitti':
    model = xgb.XGBClassifier(n_estimators=80, max_depth=6, learning_rate=0.1, subsample=0.5, reg_alpha=0.5, reg_lambda=0.5) 
    
  elif CONFIG.IMG_TYPE == 'mot':
    model = xgb.XGBClassifier(n_estimators=80, max_depth=6, learning_rate=0.1, subsample=0.5, reg_alpha=0.5, reg_lambda=0.5) 
    
  model.fit( X_train, y_train )
  coefs_model = np.array(model.feature_importances_)
    
  y_train_pred = model.predict_proba(X_train)
  y_val_pred = model.predict_proba(X_val)
  y_test_R_pred = model.predict_proba(X_test)
  
  return y_train_pred, y_val_pred, y_test_R_pred, coefs_model


def comp_iou_mask(masks1, masks2):
  '''
  compute IoU
  '''
  
  # shape: [height, hidth, num_instances]
  if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
    return np.zeros((masks1.shape[-1], masks2.shape[-1]))

  masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
  masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
  pixel1 = np.sum(masks1, axis=0)
  pixel2 = np.sum(masks2, axis=0)

  intersections = np.dot(masks1.T, masks2)
  union = pixel1[:, None] + pixel2[None, :] - intersections
  overlaps = intersections / union
  return overlaps
  
  
def match_gt_pred(gt_boxes, gt_class_ids, gt_masks, pred_boxes, pred_class_ids, pred_scores, pred_masks, img_num_gt, img_num_pred, iou_threshold=0.5):
  '''
  match ground truth instances with predictions
  '''

  # sort predictions by score from high to low
  indices = np.argsort(pred_scores)[::-1]
  pred_boxes = pred_boxes[indices]
  pred_class_ids = pred_class_ids[indices]
  pred_scores = pred_scores[indices]
  pred_masks = pred_masks[..., indices]
  img_num_pred = img_num_pred[indices]

  # compute IoU overlaps [pred_masks, gt_masks]
  overlaps = comp_iou_mask(pred_masks, gt_masks)

  # each prediction has the index of the matched gt
  pred_match = -1 * np.ones([pred_boxes.shape[0]])
  # each gt has the index of the matched prediction
  gt_match = -1 * np.ones([gt_boxes.shape[0]])
  
  for i in range(len(pred_boxes)):
    gc.collect()
    sorted_ixs = np.argsort(overlaps[i])[::-1]
    for j in sorted_ixs:
      if gt_match[j] > -1:
        continue
      if overlaps[i, j] < iou_threshold:
        break
      if (pred_class_ids[i] == gt_class_ids[j]) and (img_num_pred[i] == img_num_gt[j]):
        gt_match[j] = i
        pred_match[i] = j
        break

  return gt_match, pred_match, overlaps 
  
  
def compute_ap(gt_boxes, gt_class_ids, gt_masks, pred_boxes, pred_class_ids, pred_scores, pred_masks, img_num_gt, img_num_pred, iou_threshold=0.5):
  '''
  compute average precision per class
  '''

  gt_match, pred_match, overlaps = match_gt_pred(gt_boxes, gt_class_ids, gt_masks, pred_boxes, pred_class_ids, pred_scores, pred_masks, img_num_gt, img_num_pred, iou_threshold)
  # compute precision and recall at each prediction box step
  precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
  recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)
  # pad with start and end values to simplify the calculation
  precisions = np.concatenate([[0], precisions, [0]])
  recalls = np.concatenate([[0], recalls, [1]])
  # ensure precision values decrease but don't increase
  for i in range(len(precisions) - 2, -1, -1):
    precisions[i] = np.maximum(precisions[i], precisions[i + 1])
  # compute AP over recall range
  indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
  mAP = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
  return mAP, gt_match, pred_match


def comp_mean_average_precision(name, num_prev_frames=0, score_th=0, instances_new=[], y0a_mc=[]):
  '''
  compute mean average precision 
  '''
  result_path = os.path.join(CONFIG.RESULTS_MAP_DIR, 'results_mAP_' + name + '_npf' + str(num_prev_frames) + '.txt')
  if not os.path.exists( CONFIG.RESULTS_MAP_DIR ):
    os.makedirs( CONFIG.RESULTS_MAP_DIR )
  
  epsilon = CONFIG.EPS_MATCHING
  num_reg = CONFIG.NUM_REG_MATCHING
  
  if instances_new != []:
    array_vid = np.asarray(instances_new['vid_no'])
    array_frame = np.asarray(instances_new['frame_no'])
    if CONFIG.IMG_TYPE == 'kitti':
      array_inst = np.stack(instances_new['inst_array'])
    array_score = np.asarray(instances_new['score_val'])
  
  if len(y0a_mc) > 0:
    max_inst = int( np.load(CONFIG.HELPER_DIR + "max_inst.npy") )
  else:
    max_inst = 0
  
  mAP = np.zeros((len(CONFIG.CLASSES)))    
  tp_fp_fn = np.zeros((3), dtype='int16')  
  
  with open(result_path, 'wt') as fi:
  
    for cl, j in zip(CONFIG.CLASSES, range(len(CONFIG.CLASSES)) ):
      print('class', cl, file=fi)

      instance_masks_gt = []
      gt_class_ids = []
      instance_masks_pred = []
      pred_class_ids = []
      pred_scores = []
      img_num_gt = []
      img_num_pred = []
      
      counter_y0a = 0
      
      list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR )) 
        
      for vid,v in zip(list_videos, range(len(list_videos))):
        
        if instances_new != [] and CONFIG.IMG_TYPE == 'mot':
          array_inst_tmp = []
          gt = ground_truth_load(vid, 0)
          for i in range(array_vid.shape[0]):
            if array_vid[i] == vid:
              array_inst_tmp.append(instances_new['inst_array'][i])
            else:
              array_inst_tmp.append(np.zeros(gt.shape,dtype='int16'))
          array_inst = np.stack(array_inst_tmp)
        
        images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + '/' ))
        
        for k in range(len(images_all)):
          if k >= num_prev_frames:

            gt_image = ground_truth_load(vid, k)
            
            if CONFIG.IMG_TYPE == 'mot' and vid == '0005':
              gt = ground_truth_load('0002', 0)
              gt_image_tmp = np.zeros((gt.shape[0], gt.shape[1]))
              gt_image_tmp[:gt_image.shape[0],:gt_image.shape[1]] = gt_image
              gt_image = gt_image_tmp.copy()
              
            obj_ids = np.unique(gt_image)
            for i in range(len(obj_ids)):
              if obj_ids[i] != 0 and obj_ids[i] != 10000 and (obj_ids[i] // 1000) == cl:
                m_i = np.zeros(np.shape(gt_image))
                m_i[gt_image==obj_ids[i]] = 1
                instance_masks_gt.append(m_i)
                gt_class_ids.append(obj_ids[i] // 1000)
                img_num_gt.append(k+1000*v)
              
            inst_image = time_series_instances_load(vid, k, epsilon, num_reg)
            inst_image[inst_image<0] *= -1
            scores = score_small_load(vid, k)
            if instances_new != []:
              inst_image = np.concatenate((inst_image, array_inst[np.logical_and(array_vid==vid, array_frame==k)]), axis=0)
              scores = np.concatenate((scores, array_score[np.logical_and(array_vid==vid, array_frame==k)]), axis=0)
            
            if CONFIG.IMG_TYPE == 'mot' and vid == '0005':
              gt = ground_truth_load('0002', 0)
              inst_image_tmp = np.zeros((inst_image.shape[0], gt.shape[0], gt.shape[1]))
              inst_image_tmp[:,:inst_image.shape[1],:inst_image.shape[2]] = inst_image
              inst_image = inst_image_tmp.copy()
            
            for i in range(inst_image.shape[0]):
              if (inst_image[i].max() // 10000) == cl and ( (len(y0a_mc) == 0 and scores[i] >= score_th) or (len(y0a_mc) > 0 and y0a_mc[counter_y0a+(inst_image[i].max()%10000)-1]==0) ):

                m_i = np.zeros(( inst_image.shape[1], inst_image.shape[2] ))
                m_i[inst_image[i,:,:]!=0] = 1
                instance_masks_pred.append(m_i)
                pred_class_ids.append(inst_image[i].max() // 10000)   
                pred_scores.append(scores[i]) 
                img_num_pred.append(k+1000*v)
                
            counter_y0a += max_inst
      
      print('pack gt')
      # Pack instance masks into an array
      if len(instance_masks_gt) > 0:
        # [height, width, num_instances]
        gt_masks = np.zeros((instance_masks_gt[0].shape[0], instance_masks_gt[0].shape[1], len(instance_masks_gt)), dtype=bool)
        for i in range(len(instance_masks_gt)):
          gt_masks[:,:,i] = np.array(instance_masks_gt[i], dtype=bool)
        gt_class_ids = np.array(gt_class_ids, dtype=np.int32)
        # [num_instances, (y1, x1, y2, x2)]
        gt_boxes = create_bb(gt_masks)
        img_num_gt = np.array(img_num_gt)
      
      print('pack pred')
      if len(instance_masks_pred) > 0:
        pred_masks = np.zeros((instance_masks_pred[0].shape[0], instance_masks_pred[0].shape[1], len(instance_masks_pred)), dtype=bool)
        for i in range(len(instance_masks_pred)):
          pred_masks[:,:,i] = np.array(instance_masks_pred[i], dtype=bool)
        pred_class_ids = np.array(pred_class_ids, dtype=np.int32)
        pred_scores = np.array(pred_scores)
        pred_boxes = create_bb(pred_masks)
        img_num_pred = np.array(img_num_pred) 

        print('number of gt instances:', gt_class_ids.shape[0], file=fi)
        print('number of predicted instances:', pred_class_ids.shape[0], file=fi)
        print('compute mAP')
        mAP[j], gt_match, pred_match = compute_ap(gt_boxes, gt_class_ids, gt_masks, pred_boxes, pred_class_ids, pred_scores, pred_masks, img_num_gt, img_num_pred)
        print('number of matches', np.count_nonzero(gt_match>-1), np.count_nonzero(pred_match>-1), file=fi)
        print('number of non-matches gt and pred', np.count_nonzero(gt_match==-1), np.count_nonzero(pred_match==-1), file=fi)
        print('class', cl, 'mAP', mAP[j], file=fi)
        
        tp_fp_fn[0] += np.count_nonzero(gt_match>-1)
        tp_fp_fn[1] += np.count_nonzero(pred_match==-1)
        tp_fp_fn[2] += np.count_nonzero(gt_match==-1)
    
    print('AP per class:', mAP, 'mAP:', np.sum(mAP)/len(CONFIG.CLASSES), file=fi)
    print('TP', tp_fp_fn[0], file=fi)
    print('FP', tp_fp_fn[1], file=fi)
    print('FN', tp_fp_fn[2], file=fi)


def compute_detected_fn(iou_threshold, flag_double_pred, max_instances):
  '''
  compute fn separated in non-detected & before first detection - after first detection
  '''
  print('compute fn separated with iou threshold', iou_threshold, 'and flag_double_pred', flag_double_pred)
  
  epsilon = CONFIG.EPS_MATCHING
  num_reg = CONFIG.NUM_REG_MATCHING
  
  max_gt = 0
  max_img_per_vid = 0
  list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR ))
  for vid in list_videos: 
    images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + '/' ))
    max_img_per_vid = max(max_img_per_vid, len(images_all))
    for n in range(len(images_all)):
      gt_image = ground_truth_load(vid, n)
      gt_list = np.unique(gt_image)
      if gt_list[-1] == 10000:
        gt_list = gt_list[:-1]
      max_gt = max(max_gt, gt_list.max())
  print('max gt instances', max_gt)
  print('max images per video', max_img_per_vid)

  detected = np.zeros((len(list_videos), max_img_per_vid, max_gt+1))
  
  for vid,v in zip(list_videos, range(len(list_videos))):
    images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + '/' ))
    for n in range(len(images_all)):
      gt_image = ground_truth_load(vid, n)
      inst_image  = time_series_instances_load( vid, n, epsilon, num_reg )
      inst_image[inst_image<0] *= -1
      scores = score_small_load(vid, n)
      
      pred_match, _ = compute_matches_gt_pred(np.array(gt_image, dtype='int16'), inst_image, scores, iou_threshold, flag_double_pred, 0)
      
      for i in range(len(pred_match)):
        if pred_match[i] > -1:
          detected[v, n:, pred_match[i]] = 1
 
  print('save detected fn array', np.shape(detected))
          
  return detected


def bb_iou(bb_a, bb_b):
  '''
  compute IoU of two boundig boxes
  '''
  
  x1 = max(bb_a[0], bb_b[0])
  y1 = max(bb_a[1], bb_b[1])
  x2 = min(bb_a[2], bb_b[2])
  y2 = min(bb_a[3], bb_b[3])
  
  I = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

  A_a = (bb_a[2] - bb_a[0] + 1) * (bb_a[3] - bb_a[1] + 1)
  A_b = (bb_b[2] - bb_b[0] + 1) * (bb_b[3] - bb_b[1] + 1)

  iou = I / float(A_a + A_b - I)
  return iou


def get_occlusion_level(iou=-1, occlusions=[]):
  '''
  occlusion level
  '''
  if iou == -1:
    occlusions = np.zeros((11))
    return occlusions
  
  if iou == 0:
    occlusions[0] += 1
  elif iou <= 0.1:
    occlusions[1] += 1
  elif iou <= 0.2:
    occlusions[2] += 1
  elif iou <= 0.3:
    occlusions[3] += 1
  elif iou <= 0.4:
    occlusions[4] += 1
  elif iou <= 0.5:
    occlusions[5] += 1
  elif iou <= 0.6:
    occlusions[6] += 1
  elif iou <= 0.7:
    occlusions[7] += 1
  elif iou <= 0.8:
    occlusions[8] += 1
  elif iou <= 0.9:
    occlusions[9] += 1
  elif iou <= 1.0:
    occlusions[10] += 1
    
  return occlusions


def compute_num_fn_i(vid, v, n, iou_threshold, flag_double_pred, flag_double_gt, detected_fn, score_th=0, y0a_mc=[], max_inst=0, counter_y0a=0, instances_new=[], scores_new=[]):
  
  gc.collect()
  
  numbers = { 'num_fn': np.zeros((2)), 'num_fp': np.zeros((1)), 'num_tp_1': np.zeros((1)), 'num_tp_2': np.zeros((1)), 'occlusions': get_occlusion_level(), 'occlusions_1': get_occlusion_level()}

  epsilon = CONFIG.EPS_MATCHING
  num_reg = CONFIG.NUM_REG_MATCHING
        
  gt_image = ground_truth_load(vid, n)
  inst_image  = time_series_instances_load( vid, n, epsilon, num_reg )
  inst_image[inst_image<0] *= -1
  scores = score_small_load(vid, n)
  if instances_new != []:
    inst_image = np.concatenate((inst_image, instances_new), axis=0)
    scores = np.concatenate((scores, scores_new), axis=0)
  if y0a_mc != []:
    inst_sort = []
    scores_sort = [] 
    for i in range(inst_image.shape[0]):
      if y0a_mc[counter_y0a+(inst_image[i].max()%10000)-1]==0:
        inst_sort.append(inst_image[i])
        scores_sort.append(scores[i])
    if len(inst_sort) > 0:
      inst_image = np.stack(inst_sort)
      scores = np.asarray(scores_sort)
    else: 
      inst_image = np.zeros((0,gt_image.shape[0],gt_image.shape[1]), dtype='int16')
      scores = np.zeros((0))
  if score_th > 0:
    inst_sort = []
    scores_sort = [] 
    for i in range(inst_image.shape[0]):
      if scores[i] >= score_th:
        inst_sort.append(inst_image[i])
        scores_sort.append(scores[i])
    if len(inst_sort) > 0:
      inst_image = np.stack(inst_sort)
      scores = np.asarray(scores_sort)
    else: 
      inst_image = np.zeros((0,gt_image.shape[0],gt_image.shape[1]), dtype='int16')
      scores = np.zeros((0))
      
  pred_match, gt_match = compute_matches_gt_pred(np.array(gt_image, dtype='int16'), inst_image, scores, iou_threshold, flag_double_pred, flag_double_gt)
  
  gt_id_list = []
  instance_masks_gt = []
  for i in np.unique(gt_image):
    if (i != 0) and (i != 10000):
      gt_id_list.append(i)
      # bounding boxes
      m_i = np.zeros(np.shape(gt_image))
      m_i[gt_image==i] = 1
      instance_masks_gt.append(m_i)
  if len(instance_masks_gt) > 0:
    gt_masks = np.stack(instance_masks_gt, axis=2).astype(np.bool)
    # (y1, x1, y2, x2)
    gt_boxes = create_bb(gt_masks)
  
  for i in range(len(gt_match)):
    if gt_match[i] == -1:
      if detected_fn[v, n, gt_id_list[i]] == 0:
        numbers['num_fn'][0] += 1
      else:
        numbers['num_fn'][1] += 1
      
      gt_image_bb = np.zeros((gt_image.shape[0],gt_image.shape[1]))
      gt_image_bb[gt_boxes[i,0]:gt_boxes[i,2]+1, gt_boxes[i,1]:gt_boxes[i,3]+1] = 1
      gt_image_bb_tmp = np.zeros((gt_image.shape[0],gt_image.shape[1]))
      for j in range(len(gt_match)):
        if j != i and bb_iou(gt_boxes[j],gt_boxes[i]) > 0:
          gt_image_bb_tmp[gt_boxes[j,0]:gt_boxes[j,2]+1, gt_boxes[j,1]:gt_boxes[j,3]+1] = 1

      iou = 0
      if np.sum(gt_image_bb_tmp>0) > 0:
        intersection = np.sum( np.logical_and(gt_image_bb>0,gt_image_bb_tmp>0) )
        union = np.sum(gt_image_bb>0) + np.sum(gt_image_bb_tmp>0) - intersection
        iou = intersection/union
      
      if detected_fn[v, n, gt_id_list[i]] == 0:
        numbers['occlusions_1'] = get_occlusion_level(iou, numbers['occlusions_1'])
      else:
        numbers['occlusions'] = get_occlusion_level(iou, numbers['occlusions'])      

  numbers['num_fp'][0] += np.count_nonzero(pred_match==-1)
  numbers['num_tp_1'][0] += np.count_nonzero(gt_match!=-1)
  numbers['num_tp_2'][0] += np.count_nonzero(pred_match!=-1)
  
  return numbers 

def compute_num_fn(name, detected_fn, iou_threshold=0.5, flag_double_pred=0, flag_double_gt=0, num_prev_frames=0, score_th=0, instances_new=[], y0a_mc=[]):
  '''
  compute false negatives for all images and sequences
  '''

  if instances_new != []:
    array_vid = np.asarray(instances_new['vid_no'])
    array_frame = np.asarray(instances_new['frame_no'])
    if CONFIG.IMG_TYPE == 'kitti':
      array_inst = np.stack(instances_new['inst_array'])
    array_score = np.asarray(instances_new['score_val'])
  
  if len(y0a_mc) > 0:
    max_inst = int( np.load(CONFIG.HELPER_DIR + "max_inst.npy") )
  else:
    max_inst = int( np.load(CONFIG.HELPER_DIR + "max_inst.npy") ) #0
  
  # non-detected & before first detection, after first detection 
  num_fn = np.zeros((2))
  num_fp = 0
  num_tp_1 = 0
  num_tp_2 = 0
  
  counter_y0a = 0
  
  occlusions = get_occlusion_level()
  occlusions_1 = get_occlusion_level()
  
  list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR ))
  for vid,v in zip(list_videos, range(len(list_videos))):
    images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + '/' ))
    
    if instances_new != [] and CONFIG.IMG_TYPE == 'mot':
      array_inst_tmp = []
      gt = ground_truth_load(vid, 0)
      for i in range(array_vid.shape[0]):
        if array_vid[i] == vid:
          array_inst_tmp.append(instances_new['inst_array'][i])
        else:
          array_inst_tmp.append(np.zeros(gt.shape,dtype='int16'))
      array_inst = np.stack(array_inst_tmp)
    
    p = Pool(CONFIG.NUM_CORES)
    if instances_new != []:
      p_args = [ (vid, v, n, iou_threshold, flag_double_pred, flag_double_gt, detected_fn, score_th, y0a_mc, max_inst, counter_y0a+max_inst*(n-num_prev_frames), array_inst[np.logical_and(array_vid==vid, array_frame==n)], array_score[np.logical_and(array_vid==vid, array_frame==n)]) for n in range(num_prev_frames,len(images_all)) ]
    else:
      p_args = [ (vid, v, n, iou_threshold, flag_double_pred, flag_double_gt, detected_fn, score_th, y0a_mc, max_inst, counter_y0a+max_inst*(n-num_prev_frames)) for n in range(num_prev_frames,len(images_all)) ]
    numbers = p.starmap( compute_num_fn_i, p_args )
    p.close()

    for i in range(len(numbers)):
      num_tp_1 += numbers[i]['num_tp_1'][0]
      num_tp_2 += numbers[i]['num_tp_2'][0]
      num_fp += numbers[i]['num_fp'][0]
      num_fn += numbers[i]['num_fn']
      occlusions += numbers[i]['occlusions']
      occlusions_1 += numbers[i]['occlusions_1']
    
    counter_y0a += max_inst*(len(images_all)-num_prev_frames)

  if not os.path.exists( CONFIG.RESULTS_FN_FP_DIR ):
    os.makedirs( CONFIG.RESULTS_FN_FP_DIR )
  with open(os.path.join(CONFIG.RESULTS_FN_FP_DIR, 'results_fnfp' + name + '_th' + str(iou_threshold) + '_p' + str(flag_double_pred) + '_g' + str(flag_double_gt) + '_npf' + str(num_prev_frames) + '.txt'), 'wt') as fi:
    print('true positives (gt/pred): ', num_tp_1, num_tp_2, file=fi)
    print('false positives: ', num_fp, file=fi)
    print('false negatives: ', np.sum(num_fn), file=fi)
    print('non-detected & before first detection, after first detection: ', num_fn, file=fi)
    print('occlusion level (after first detection) : ', occlusions, file=fi)
    print('occlusion level (non-detected & before first detection) : ', occlusions_1, file=fi)


