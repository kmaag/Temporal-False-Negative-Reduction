import numpy as np
cimport numpy as np
import pickle
from scipy.stats import linregress

from global_defs import CONFIG
from in_out      import time_series_instances_load, ground_truth_load, score_small_load, depth_load, helpers_p_load



def compute_matches_gt_pred( np.ndarray gt_in1, np.ndarray instances_in1, score=[], float iou_threshold=0.5, int flag_double_pred=0, int flag_double_gt=0 ):
  # flag_double_pred=0 & flag_double_gt=0: 1 pred - 1 gt
  # flag_double_pred=1: 1 pred - >=1 gt - reduces fn
  # flag_double_gt=1: 1 gt - >=1 pred - reduces fp
  
  cdef int m, k, max_index, x, y, I, U, idx
  cdef float tmp_max_iou
  
  cdef short int[:,:] gt
  cdef short int[:,:] gt_dp
  cdef short int[:,:,:] instances_id
  
  gt_in = gt_in1.copy()
  gt = gt_in
  gt_class_in = gt_in // 1000
  
  instances_in = instances_in1.copy() 
  instances_in[instances_in<0] *= -1
  instances_id_in = instances_in  % 10000
  instances_id = instances_id_in
  
  gt_id_list = []
  for k in np.unique(gt_in):
    if (k != 0) and (k != 10000):
      gt_id_list.append(k)
  
  # gt_match: for each GT instance (sorted) it has the id (1,2,.) of the matched predicted instance
  # pred_match: for each predicted instance (index-wise), it has the id (1000,.) of the matched ground truth
  gt_match = np.zeros((len(gt_id_list)), dtype='int16' ) -1
  pred_match = np.zeros((instances_in.shape[0]), dtype='int16' ) -1

  ind_field_counter = np.zeros((instances_in.shape[0]), dtype='float32' )
  
  # sort instance predictions by size
  for m in range( 0, instances_in.shape[0] ):
    if score != []:
      ind_field_counter[m] = score[m] 
    else:
      ind_field_counter[m] = np.count_nonzero(instances_id_in[m]>0)
    
  for m in range( 0, instances_in.shape[0] ):
    
    max_index = int(np.argmax(ind_field_counter))
    class_inst = int(instances_in[max_index].max() // 10000)
    
    tmp_max_gt_id = -1
    tmp_max_iou = 0
    
    for k, idx in zip(gt_id_list, range(len(gt_id_list))):
      
      if np.sum(gt_in[gt_in==k]) > 0:
      
        if class_inst == gt_class_in[gt_in==k].max():
          I = 0
          U = 0
          
          for x in range(instances_in.shape[1]):
            for y in range(instances_in.shape[2]):
              
              if instances_id[max_index,x,y] > 0 and gt[x,y] == k:
                I += 1
                U += 1
              
              elif instances_id[max_index,x,y] > 0 or gt[x,y] == k: 
                U += 1
                
          if U > 0:
            if (float(I) / float(U)) > tmp_max_iou:
              tmp_max_iou = (float(I) / float(U))
              tmp_max_gt_id = idx
            
    if tmp_max_gt_id > -1 and tmp_max_iou >= iou_threshold:
      if flag_double_gt == 0:
        gt_in[gt_in==gt_id_list[tmp_max_gt_id]] = 0
      pred_match[max_index] = gt_id_list[tmp_max_gt_id]
      gt_match[tmp_max_gt_id] = instances_id_in[max_index].max()
    ind_field_counter[max_index] = -1
      
  if flag_double_pred == 1:

    gt_in_dp = gt_in1.copy()
    gt_dp = gt_in_dp
    gt_dp_class_in = gt_in_dp // 1000
      
    for k, idx in zip(gt_id_list, range(len(gt_id_list))):
      
      class_inst = int(gt_dp_class_in[gt_in_dp==k].max())
      
      if gt_match[idx] == -1:
        
        tmp_max_pred_id = -1
        tmp_max_iou = 0
        
        for m in range( 0, instances_in.shape[0] ):
          
          if class_inst == int(instances_in[m].max() // 10000):
            I = 0
            U = 0
        
            for x in range(instances_in.shape[1]):
              for y in range(instances_in.shape[2]):

                if gt_dp[x,y] == k and instances_id[m,x,y] > 0:
                  I += 1
                  U += 1
                
                elif gt_dp[x,y] == k or instances_id[m,x,y] > 0: 
                  U += 1
                  
            if U > 0:
              if (float(I) / float(U)) > tmp_max_iou:
                tmp_max_iou = (float(I) / float(U))
                tmp_max_pred_id = m
              
        if tmp_max_pred_id > -1 and tmp_max_iou >= iou_threshold:
          pred_match[tmp_max_pred_id] = k
          gt_match[idx] = instances_id_in[tmp_max_pred_id].max()
  
  return pred_match, gt_match
  
  
def analyze_instances_vid( vid, num_img, list_gt_ids, c, flag_detect=0 ):

  cdef int epsilon, num_reg, imx, imy, x, y, k, i
  
  cdef short int[:,:,:] instances_id
  cdef short int[:,:] gt
  
  epsilon = CONFIG.EPS_MATCHING
  num_reg = CONFIG.NUM_REG_MATCHING 
  
  seg_0 = time_series_instances_load( vid, 0, epsilon, num_reg)
  imx = seg_0.shape[1]
  imy = seg_0.shape[2]
  
  if flag_detect == 1:
    instances_new = helpers_p_load('inst_time_series_tested')
    array_vid = np.asarray(instances_new['vid_no'])
    array_frame = np.asarray(instances_new['frame_no'])
    array_score = np.asarray(instances_new['score_val'])
    if CONFIG.IMG_TYPE == 'kitti':
      array_inst = np.stack(instances_new['inst_array'])
    if CONFIG.IMG_TYPE == 'mot':
      array_inst_tmp = []
      gt_tmp = ground_truth_load(vid, 0)
      for i in range(array_vid.shape[0]):
        if array_vid[i] == vid:
          array_inst_tmp.append(instances_new['inst_array'][i])
        else:
          array_inst_tmp.append(np.zeros(gt_tmp.shape,dtype='int16'))
      array_inst = np.stack(array_inst_tmp)
    
  
  tracking_metrics = { "num_frames": np.zeros((1)), "gt_obj": np.zeros((1)), "fp": np.zeros((1)), "misses": np.zeros((1)), "mot_a": np.zeros((1)), "dist_bb": np.zeros((1)), "dist_geo": np.zeros((1)), "matches": np.zeros((1)), "mot_p_bb": np.zeros((1)), "mot_p_geo": np.zeros((1)), "far": np.zeros((1)), "f_measure": np.zeros((1)), "precision": np.zeros((1)), "recall": np.zeros((1)), "switch_id": np.zeros((1)), "num_gt_ids": np.zeros((1)), "mostly_tracked": np.zeros((1)), "partially_tracked": np.zeros((1)), "mostly_lost": np.zeros((1)), "switch_tracked": np.zeros((1)), 'totally_tracked': np.zeros((1)), 'totally_lost': np.zeros((1)) }

  tracking_unique_gt = { 'last_match_id': np.ones((list_gt_ids.shape[0]))*-1, 'num_tracked': np.zeros((list_gt_ids.shape[0])), 'lifespan_gt_id': np.zeros((list_gt_ids.shape[0])), "last_tracked": np.ones((list_gt_ids.shape[0]))*-1 }
  
  for n in range(num_img):
    print(vid, n)
    
    instances_in = np.array( time_series_instances_load( vid, n, epsilon, num_reg), dtype='int16' )
    if flag_detect == 1:
      instances_new = np.array(array_inst[np.logical_and(array_vid==vid, array_frame==n)], dtype='int16' )
      instances_in = np.concatenate((instances_in, instances_new), axis=0)
    instances_in[instances_in<0] *= -1
    if c == 1:
      instances_in[instances_in>=20000] = 0
    elif c == 2:
      instances_in[instances_in<20000] = 0
    if np.sum(instances_in) == 0:
      size_new_instances = 0
    else:
      size_new_instances = len(np.unique(instances_in))-1
    instances_c_in = np.zeros((size_new_instances, imx, imy), dtype='int16')
    counter = 0
    for i in range(instances_in.shape[0]):
      if np.sum(instances_in[i]) > 0:
        instances_c_in[counter] = instances_in[i]
        counter += 1
    instances_id_in = instances_c_in % 10000
    instances_id = instances_id_in
    
    gt_in = np.array( ground_truth_load(vid, n), dtype='int16')
    if c == 1:
      gt_in[gt_in>=2000] = 0
    elif c == 2:
      gt_in[gt_in<2000] = 0
    gt = gt_in
    
    score = score_small_load( vid, n )
    if flag_detect == 1:
      scores_new = array_score[np.logical_and(array_vid==vid, array_frame==n)]
      score = np.concatenate((score, scores_new), axis=0)
    pred_match, gt_match = compute_matches_gt_pred(gt_in, instances_c_in, score)
    
    tracking_metrics['gt_obj'] += len(gt_match)
    tracking_metrics['fp'] += np.count_nonzero(pred_match==-1)
    tracking_metrics['misses'] += np.count_nonzero(gt_match==-1)
    tracking_metrics['matches'] += np.count_nonzero(gt_match>-1)
    
    gt_id_list = []
    for k in np.unique(gt_in):
      if (k != 0) and (k != 10000):
        gt_id_list.append(k)
              
    for k, idx in zip(gt_id_list, range(len(gt_id_list))):
      
      if gt_match[idx] > -1:
      
        idx_instance = -1
        for i in range(instances_c_in.shape[0]):
          if instances_id_in[i].max() == gt_match[idx]:
            idx_instance = i
            break
        
        mean_x_gt = 0
        mean_y_gt = 0
        counter_gt = 0
        min_x_gt = imx
        max_x_gt = 0
        min_y_gt = imy
        max_y_gt = 0
        
        mean_x_instance = 0
        mean_y_instance = 0
        counter_instance = 0
        min_x_instance = imx
        max_x_instance = 0
        min_y_instance = imy
        max_y_instance = 0
        
        for x in range(imx):
          for y in range(imy):
            
            if gt[x,y] == k:
              mean_x_gt += x
              mean_y_gt += y
              counter_gt += 1
              min_x_gt = min(min_x_gt, x)
              max_x_gt = max(max_x_gt, x)
              min_y_gt = min(min_y_gt, y)
              max_y_gt = max(max_y_gt, y)
              
            if instances_id[idx_instance,x,y] > 0:
              mean_x_instance += x
              mean_y_instance += y
              counter_instance += 1
              min_x_instance = min(min_x_instance, x)
              max_x_instance = max(max_x_instance, x)
              min_y_instance = min(min_y_instance, y)
              max_y_instance = max(max_y_instance, y)
              
        mean_x_gt /= counter_gt
        mean_y_gt /= counter_gt
        mean_x_instance /= counter_instance
        mean_y_instance /= counter_instance
        
        tracking_metrics['dist_bb'] += ( (float(min_x_gt+max_x_gt)/2 - float(min_x_instance+max_x_instance)/2)**2 + (float(min_y_gt+max_y_gt)/2 - float(min_y_instance+max_y_instance)/2)**2 )**0.5
        
        tracking_metrics['dist_geo'] += ( (mean_x_gt - mean_x_instance)**2 + (mean_y_gt - mean_y_instance)**2 )**0.5
  
    for k in range(len(list_gt_ids)):  
      
      if list_gt_ids[k] in gt_id_list:
        tracking_unique_gt['lifespan_gt_id'][k] += 1
        
        if tracking_unique_gt['last_tracked'][k] == -1 and list_gt_ids[k] in pred_match:
          tracking_unique_gt['last_tracked'][k] = 1
          
        elif tracking_unique_gt['last_tracked'][k] == 1 and list_gt_ids[k] not in pred_match:
          tracking_unique_gt['last_tracked'][k] = 0
        
        elif tracking_unique_gt['last_tracked'][k] == 0 and list_gt_ids[k] in pred_match:
          tracking_unique_gt['last_tracked'][k] = 1
          tracking_metrics['switch_tracked'] += 1
           
      if list_gt_ids[k] in pred_match:
        tracking_unique_gt['num_tracked'][k] += 1
        
        if tracking_unique_gt['last_match_id'][k] == -1:
          tracking_unique_gt['last_match_id'][k] = gt_match[gt_id_list.index(list_gt_ids[k])]
        
        elif tracking_unique_gt['last_match_id'][k] != gt_match[gt_id_list.index(list_gt_ids[k])]:
          
          tracking_unique_gt['last_match_id'][k] = gt_match[gt_id_list.index(list_gt_ids[k])]
          tracking_metrics['switch_id'] += 1
   
  tracking_metrics['num_frames'] += num_img
  
  tracking_metrics['mot_a'] = 1 - ((tracking_metrics['misses'] + tracking_metrics['fp'] + tracking_metrics['switch_id'])/tracking_metrics['gt_obj'])
  
  tracking_metrics['mot_p_bb'] = tracking_metrics['dist_bb'] / tracking_metrics['matches']
  
  tracking_metrics['mot_p_geo'] = tracking_metrics['dist_geo'] / tracking_metrics['matches']
  
  tracking_metrics['far'] = tracking_metrics['fp'] / tracking_metrics['num_frames'] * 100
  
  tracking_metrics['f_measure'] = (2 * tracking_metrics['matches']) / (2 * tracking_metrics['matches'] + tracking_metrics['misses'] + tracking_metrics['fp'])
  
  tracking_metrics['precision'] = tracking_metrics['matches'] / ( tracking_metrics['matches'] + tracking_metrics['fp'])
  
  tracking_metrics['recall'] = tracking_metrics['matches'] / ( tracking_metrics['matches'] + tracking_metrics['misses'])
  
  tracking_metrics['num_gt_ids'] += len(list_gt_ids)
  
  for k in range(len(list_gt_ids)):
    
    quotient = tracking_unique_gt['num_tracked'][k] / tracking_unique_gt['lifespan_gt_id'][k]
    
    if quotient == 1.0:
      tracking_metrics['totally_tracked'] += 1
    elif quotient >= 0.8:
      tracking_metrics['mostly_tracked'] += 1
    elif quotient >= 0.2:
      tracking_metrics['partially_tracked'] += 1
    elif quotient > 0:
      tracking_metrics['mostly_lost'] += 1
    else:
      tracking_metrics['totally_lost'] += 1
  
  path_tmp = 'inst_pred_'
  if flag_detect == 1:
    path_tmp = 'inst_pred_detect_'
  pickle.dump( tracking_metrics, open( CONFIG.ANALYZE_INSTANCES_DIR + path_tmp + vid + '_class' + str(c) + '.p', 'wb' ) )  

  
def compute_missed(np.ndarray inst_image_steps_in):
  
  cdef int id_inst, x, y, c, reg_steps, imx, imy, nm, x_shift, y_shift, x_t, y_t
  
  cdef float[:,:] mean_field
  cdef float[:] mean_counter_field
  cdef short int[:,:] shifted_comp
  
  inst_image_steps = inst_image_steps_in
  id_inst = inst_image_steps_in.max()
  
  reg_steps = inst_image_steps_in.shape[0]
  imx = inst_image_steps_in.shape[1]
  imy = inst_image_steps_in.shape[2]
  
  mean_field_in = np.zeros( (reg_steps,2), dtype='float32' )
  mean_field = mean_field_in
  mean_counter_field_in = np.zeros( (reg_steps), dtype='float32' )
  mean_counter_field = mean_counter_field_in
  
  for x in range(imx):
    for y in range(imy):
      for c in range(reg_steps):
        if inst_image_steps[c,x,y] > 0:
          mean_field[c,0] += x
          mean_field[c,1] += y
          mean_counter_field[c] += 1
  
  mean_x_list = []
  mean_y_list = []
  mean_t_list = []
  index_last_inst = 0
  
  for c in range(reg_steps):
    if mean_counter_field[c] > 0:
      mean_field[c,0] /= mean_counter_field[c]
      mean_field[c,1] /= mean_counter_field[c]
      mean_x_list.append(mean_field[c,0])
      mean_y_list.append(mean_field[c,1])
      mean_t_list.append(c)
      index_last_inst = c

  # linear regression of geometric centers
  b_x, a_x, _, _, _ = linregress(mean_t_list, mean_x_list)
  b_y, a_y, _, _, _ = linregress(mean_t_list, mean_y_list)
  
  shifted_inst = list([])
  num_frame = list([])
  
  for nm in range(index_last_inst+1,reg_steps):
    
    if inst_image_steps_in[nm].max() > 0:
      break
    
    pred_x = a_x + b_x * nm
    pred_y = a_y + b_y * nm

    x_shift = int(pred_x - mean_x_list[-1])
    y_shift = int(pred_y - mean_y_list[-1])
    
    shifted_comp_in = np.zeros((imx, imy), dtype='int16')
    shifted_comp = shifted_comp_in
    
    # compute the shifted instance
    for x in range(imx):
      for y in range(imy):
        if inst_image_steps[index_last_inst,x,y] > 0:
          x_t = x + x_shift
          y_t = y + y_shift
          if x_t>=0 and x_t<imx and y_t>=0 and y_t<imy:
            shifted_comp[x_t,y_t] = id_inst
    
    if shifted_comp_in.max() > 0:
      shifted_inst.append(shifted_comp_in)
      num_frame.append(-reg_steps+nm)
    
  return shifted_inst, num_frame
    
    
def compute_consequetive_instances(vid, int num_images, int inst_id, np.ndarray frame_ind_in, np.ndarray values_in):
  
  cdef int epsilon, num_reg, imx, imy, i, index, inst_num, id_inst
  
  cdef short int[:] frame_ind
  cdef short int[:,:] shifted_comp
  cdef short int[:,:,:] instances
  
  epsilon = CONFIG.EPS_MATCHING
  num_reg = CONFIG.NUM_REG_MATCHING 
  
  instances_in = np.array( time_series_instances_load( vid, 0, epsilon, num_reg), dtype='int16' )
  imx = instances_in.shape[1]
  imy = instances_in.shape[2]
  
  frame_ind_in = np.array(frame_ind_in, dtype='int16')
  frame_ind = frame_ind_in
  values = values_in
  
  shifted_inst = list([])
  num_frame = list([])
  score_val = list([])
  
  mean_t_list = []
  mean_x_list = []
  mean_y_list = []
  mean_score_list = []
  
  mean_t_list.append(frame_ind[0])
  mean_x_list.append(values[0,1])
  mean_y_list.append(values[0,2])
  mean_score_list.append(values[0,3])
  index = 1
  
  for i in range(frame_ind[1],num_images):
    if i in frame_ind_in:
      
      mean_t_list.append(frame_ind[index])
      mean_x_list.append(values[index,1])
      mean_y_list.append(values[index,2])
      mean_score_list.append(values[index,3])
      index += 1
      
    elif i-mean_t_list[-1] <= 10:
      
      # linear regression of geometric centers
      b_x, a_x, _, _, _ = linregress(mean_t_list, mean_x_list)
      b_y, a_y, _, _, _ = linregress(mean_t_list, mean_y_list)
      
      pred_x = a_x + b_x * i
      pred_y = a_y + b_y * i
      
      instances_in = np.array( time_series_instances_load( vid, mean_t_list[-1], epsilon, num_reg), dtype='int16' )
      instances_in[instances_in<0] *= -1
      instances = instances_in
      for inst_num in range(instances_in.shape[0]):
        if inst_id == (instances_in[inst_num].max() % 10000):
          break
      id_inst = instances_in[inst_num].max()
        
      
      x_shift = int(pred_x - mean_x_list[-1])
      y_shift = int(pred_y - mean_y_list[-1])
      
      shifted_comp_in = np.zeros((imx, imy), dtype='int16')
      shifted_comp = shifted_comp_in
      
      # compute the shifted instance
      for x in range(imx):
        for y in range(imy):
          if instances[inst_num,x,y] > 0:
            x_t = x + x_shift
            y_t = y + y_shift
            if x_t>=0 and x_t<imx and y_t>=0 and y_t<imy:
              shifted_comp[x_t,y_t] = id_inst
      
      # instance may have run over the boundary and is no longer visible
      if shifted_comp_in.max() > 0:
          
        shifted_inst.append(shifted_comp_in)
        num_frame.append(i)
        score_val.append(np.mean(mean_score_list))
      
  return shifted_inst, num_frame, score_val
  

def shifted_iou( inst1_in, inst2_in):
  
  cdef int imx, imy, x, y, counter, x_shift, y_shift, x_t, y_t 
  cdef float mean_x_1, mean_y_1, mean_x_2, mean_y_2, intersection, union, iou=0
  
  cdef char[:,:] inst1
  cdef char[:,:] inst2
  cdef char[:,:] shifted_comp
  
  inst1_in = np.asarray( inst1_in, dtype='uint8' )
  inst1 = inst1_in
  inst2_in = np.asarray( inst2_in, dtype='uint8' )
  inst2 = inst2_in
  
  imx = inst1_in.shape[0]
  imy = inst1_in.shape[1]
  
  mean_x_1 = 0
  mean_y_1 = 0
  counter = 0
  for x in range(imx):
    for y in range(imy):
      if inst1[x,y] == 1:
        mean_x_1 += x
        mean_y_1 += y
        counter = counter + 1   
  mean_x_1 /= counter
  mean_y_1 /= counter
  
  mean_x_2 = 0
  mean_y_2 = 0
  counter = 0
  for x in range(imx):
    for y in range(imy):
      if inst2[x,y] == 1:
        mean_x_2 += x
        mean_y_2 += y
        counter = counter + 1
  mean_x_2 /= counter
  mean_y_2 /= counter
  
  x_shift = int(mean_x_1 - mean_x_2)
  y_shift = int(mean_y_1 - mean_y_2)
  
  shifted_comp_in = np.zeros((imx, imy), dtype='uint8')
  shifted_comp = shifted_comp_in
  
  # compute the shifted instance
  for x in range(imx):
    for y in range(imy):
      if inst2[x,y] == 1:
        x_t = x + x_shift
        y_t = y + y_shift
        if x_t>=0 and x_t<imx and y_t>=0 and y_t<imy:
          shifted_comp[x_t,y_t] = 1
  
  intersection = 0
  union = 0
  for x in range(imx):
    for y in range(imy):
      if shifted_comp[x,y] == 1 and inst1[x,y] > 0:
        intersection = intersection + 1
      if shifted_comp[x,y] == 1 or inst1[x,y] > 0:
        union = union + 1
        
  if union > 0:     
    iou = intersection / union 
  return iou
  

def test_occlusion( inst1_in, index, inst2_in):
  
  cdef int imx, imy, x, y, counter, x_shift, y_shift, x_t, y_t, j, flag_occluded 
  cdef float mean_x_1, mean_y_1, mean_x_2, mean_y_2, pixel_inst, overlap, occlusion=0
  
  cdef short int[:,:,:] inst1
  cdef char[:,:] inst2
  cdef char[:,:] shifted_comp
  
  inst1_in = np.asarray( inst1_in, dtype='int16' )
  inst1 = inst1_in
  inst2_in = np.asarray( inst2_in, dtype='uint8' )
  inst2 = inst2_in
  
  imx = inst1_in.shape[1]
  imy = inst1_in.shape[2]
  
  mean_x_1 = 0
  mean_y_1 = 0
  counter = 0
  for x in range(imx):
    for y in range(imy):
      if inst1[index,x,y] > 0:
        mean_x_1 += x
        mean_y_1 += y
        counter = counter + 1   
  mean_x_1 /= counter
  mean_y_1 /= counter
  
  mean_x_2 = 0
  mean_y_2 = 0
  counter = 0
  for x in range(imx):
    for y in range(imy):
      if inst2[x,y] == 1:
        mean_x_2 += x
        mean_y_2 += y
        counter = counter + 1
  mean_x_2 /= counter
  mean_y_2 /= counter
  
  x_shift = int(mean_x_1 - mean_x_2)
  y_shift = int(mean_y_1 - mean_y_2)
  
  shifted_comp_in = np.zeros((imx, imy), dtype='uint8')
  shifted_comp = shifted_comp_in
  
  # compute the shifted instance
  for x in range(imx):
    for y in range(imy):
      if inst2[x,y] == 1:
        x_t = x + x_shift
        y_t = y + y_shift
        if x_t>=0 and x_t<imx and y_t>=0 and y_t<imy:
          shifted_comp[x_t,y_t] = 1
  
  pixel_inst = 0
  overlap = 0
  for x in range(imx):
    for y in range(imy):
      if shifted_comp[x,y] == 1:
        pixel_inst = pixel_inst + 1
      
        flag_occluded = 0
        for j in range(inst1_in.shape[0]):
          if j != index and inst1[j,x,y] > 0: 
            flag_occluded = 1
        if flag_occluded == 1:
          overlap = overlap + 1
        
  if pixel_inst > 0:     
    occlusion = overlap / pixel_inst 
  return occlusion
  
  
  
  
  
  
  

