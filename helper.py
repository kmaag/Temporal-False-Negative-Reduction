#!/usr/bin/env python3
'''
script including
functions for easy usage in main scripts
'''

import os
import subprocess
import numpy as np

from global_defs import CONFIG
from in_out      import time_series_instances_load


def name_to_latex( name ):
  '''
  metric names in latex
  '''

  mapping = {'D': '$\\bar D$',
             'D_bd': '${\\bar D}_{bd}$',
             'D_in': '${\\bar D}_{in}$',
             'D_rel_in': '$\\tilde{\\bar D}_{in}$',
             'D_rel': '$\\tilde{\\bar D}$',
             'S': '$S$',
             'S_bd': '${S}_{bd}$',
             'S_in': '${S}_{in}$',
             'S_rel_in': '$\\tilde{S}_{in}$',
             'S_rel': '$\\tilde{S}$',
             'mean_x' : '${\\bar i}_{v}$',
             'mean_y' : '${\\bar i}_{h}$', 
             'class' : '$c$',
             'score' : '$s$',
             'survival' : '$v$',
             'ratio' : '$r$', 
             'deformation' : '$f$',
             'occlusion' : '$o$',
             'diff_mean' : '$d_{c}$',
             'diff_size' : '$d_{s}$',
             'diff_depth' : '$d_{d}$',
             'iou_o' : '$IoU$',
             'iou_s' : '$IoU$'}            
  if str(name) in mapping:
    return mapping[str(name)]
  else:
    return str(name) 
  
  
def name_to_latex_scatter_plot( name ):
  '''
  metric names in latex for scatter plots
  '''
  
  mapping = {'D': '$\\bar D$',
             'D_bd': '${\\bar D}_{bd}$',
             'D_in': '${\\bar D}_{in}$',
             'D_rel_in': '$\\tilde{\\bar D}_{in}/\\tilde{\\bar D}_{in,max}$',
             'D_rel': '$\\tilde{\\bar D}/\\tilde{\\bar D}_{max}$',
             'S': '$S/S_{max}$',
             'S_bd': '$S_{bd}/S_{bd,max}$',
             'S_in': '$S_{in}/S_{in,max}$',
             'S_rel_in': '$\\tilde{S}_{in}/\\tilde{S}_{in,max}$',
             'S_rel': '$\\tilde{S}/\\tilde{S}_{max}$',
             'mean_x' : '${\\bar i}_{v}$',
             'mean_y' : '${\\bar i}_{h}$', 
             'class' : '$c$',
             'score' : '$s$',
             'survival' : '$v$',
             'ratio' : '$r$', 
             'deformation' : '$f$',
             'occlusion' : '$o$',
             'diff_mean' : '$d_{c}$',
             'diff_size' : '$d_{s}$',
             'diff_depth' : '$d_{d}$',
             'iou_o' : '$IoU$',
             'iou_s' : '$IoU$'}        
  if str(name) in mapping:
    return mapping[str(name)]
  else:
    return str(name) 


def compute_max_inst():
  '''
  compute the maximal ID in all frames
  '''
  
  if os.path.isfile( CONFIG.HELPER_DIR + 'max_inst.npy' ):
    max_inst = np.load(CONFIG.HELPER_DIR + 'max_inst.npy')
      
  else:
    epsilon = CONFIG.EPS_MATCHING
    num_reg = CONFIG.NUM_REG_MATCHING
    
    max_inst = 0
    
    list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR ))
    for vid in list_videos: 
      
      images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + '/' ))
      for i in range(len(images_all)):
        
        ts_inst = time_series_instances_load(vid, i, epsilon, num_reg)
        ts_inst[ts_inst<0] *= -1
        ts_inst = ts_inst % 10000
        
        if ts_inst.shape[0] > 0:
          max_inst = max(max_inst, ts_inst.max())
          
      print('max:', max_inst)
    
    if not os.path.exists( CONFIG.HELPER_DIR ):
      os.makedirs( CONFIG.HELPER_DIR )
    np.save(os.path.join(CONFIG.HELPER_DIR, 'max_inst'), max_inst)
    
  print('maximal number of instances:', max_inst)
  return max_inst


def comp_inst_in_bd(instances):
  '''
  divide the values of an instance in inner and boundary
  '''
  
  inst_j = instances.copy()
  inst_j[inst_j>0] = 1
  # addition of 8 neigbours for every pixel (not included image boundary)
  inst_j_small = inst_j[1:instances.shape[0]-1,1:instances.shape[1]-1] \
                + inst_j[0:instances.shape[0]-2,1:instances.shape[1]-1] \
                + inst_j[2:instances.shape[0],1:instances.shape[1]-1] \
                + inst_j[1:instances.shape[0]-1,2:instances.shape[1]] \
                + inst_j[1:instances.shape[0]-1,0:instances.shape[1]-2] \
                + inst_j[0:instances.shape[0]-2,0:instances.shape[1]-2] \
                + inst_j[2:instances.shape[0],0:instances.shape[1]-2] \
                + inst_j[0:instances.shape[0]-2,2:instances.shape[1]] \
                + inst_j[2:instances.shape[0],2:instances.shape[1]]
  # 1 for interior, 0 for boundary and background
  inst_j_small = (inst_j_small == 9 ) 
  # 0 background, 1 boundary, 2 interior
  inst_j[1:instances.shape[0]-1,1:instances.shape[1]-1] += inst_j_small 
  instances[inst_j==1] *= -1
  
  return instances
        
        
def time_series_metrics_to_nparray( metrics, names, normalize=False, all_metrics=[] ):
  '''
  metrics to np array, normalized and 0s stay in (no of instances * no of images, no of metrics)
  '''
  
  I = range(len(metrics['S']))
  M_with_zeros = np.zeros((len(I), len(names)))
  I = np.asarray(metrics['S']) > 0
  
  M = np.asarray( [ np.asarray(metrics[ m ])[I] for m in names ] )
  MM = M.copy()
  print(np.shape(M))
  # normalize: E = 0 and sigma = 1
  if normalize == True:
    for i in range(M.shape[0]):
      if names[i] != 'class':
        M[i] = ( np.asarray(M[i]) - np.mean(MM[i], axis=-1 ) ) / ( np.std(MM[i], axis=-1 ) + 1e-10 )
  M = np.squeeze(M.T)
  
  counter = 0
  for i in range(M_with_zeros.shape[0]):
    if I[i] == True and M_with_zeros.shape[1]>1:
      M_with_zeros[i,:] = M[counter,:]
      counter += 1
    if I[i] == True and M_with_zeros.shape[1]==1:
      M_with_zeros[i] = M[counter]
      counter += 1
  
  return M_with_zeros        


def split_tvs_and_concatenate( Xa, ya, y0a, train_val_test_string, run=0 ):
  '''
  0s will be sorted out, the metrics of the previous frames (NUM_PREV_FRAMES) will be included 
  ''' 
  
  np.random.seed( run )
  num_images = CONFIG.NUM_IMAGES
  num_prev_frames = CONFIG.NUM_PREV_FRAMES
  max_inst = int( np.load(CONFIG.HELPER_DIR + 'max_inst.npy') )
  
  print('Concatenate timeseries dataset and create train/val/test splitting')
    
  list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR ))
  
  ya = np.squeeze(ya)
  y0a = np.squeeze(y0a)
  
  Xa_train = np.zeros(( (num_images-(len(list_videos)*num_prev_frames)) * max_inst, Xa.shape[1] * (num_prev_frames+1)))
  ya_train = np.zeros(( (num_images-(len(list_videos)*num_prev_frames)) * max_inst ))
  y0a_train = np.zeros(( (num_images-(len(list_videos)*num_prev_frames)) * max_inst ))
  
  Xa_val = np.zeros(( (num_images-(len(list_videos)*num_prev_frames)) * max_inst, Xa.shape[1] * (num_prev_frames+1)))
  ya_val = np.zeros(( (num_images-(len(list_videos)*num_prev_frames)) * max_inst ))
  y0a_val = np.zeros(( (num_images-(len(list_videos)*num_prev_frames)) * max_inst ))
  
  Xa_test = np.zeros(( (num_images-(len(list_videos)*num_prev_frames)) * max_inst, Xa.shape[1] * (num_prev_frames+1)))
  ya_test = np.zeros(( (num_images-(len(list_videos)*num_prev_frames)) * max_inst ))
  y0a_test = np.zeros(( (num_images-(len(list_videos)*num_prev_frames)) * max_inst ))
  
  counter = 0
  counter_train = 0
  counter_val = 0
  counter_test = 0
  
  for vid,v in zip(list_videos, range(len(list_videos))):
    images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + '/' ))
    
    if CONFIG.IMG_TYPE == 'mot' and train_val_test_string[v] == 'v':
      split_point = int( len(images_all)/3 - num_prev_frames/2 )
      
    for i in range(len(images_all)):
      
      if i >= num_prev_frames:
          
        tmp = np.zeros(( max_inst, Xa.shape[1] * (num_prev_frames+1) ))
        for j in range(0,num_prev_frames+1):
          tmp[:,Xa.shape[1]*j:Xa.shape[1]*(j+1)] = Xa[max_inst*(counter-j):max_inst*(counter-j+1)] 
        
        if train_val_test_string[v] == 't':
        
          Xa_train[max_inst*counter_train:max_inst*(counter_train+1),:] = tmp
          ya_train[max_inst*counter_train:max_inst*(counter_train+1)] = ya[max_inst*counter:max_inst*(counter+1)]
          y0a_train[max_inst*counter_train:max_inst*(counter_train+1)] = y0a[max_inst*counter:max_inst*(counter+1)]
          counter_train +=1
        
        elif CONFIG.IMG_TYPE == 'kitti':
          
          if train_val_test_string[v] == 'v':
            
            Xa_val[max_inst*counter_val:max_inst*(counter_val+1),:] = tmp
            ya_val[max_inst*counter_val:max_inst*(counter_val+1)] = ya[max_inst*counter:max_inst*(counter+1)]
            y0a_val[max_inst*counter_val:max_inst*(counter_val+1)] = y0a[max_inst*counter:max_inst*(counter+1)]
            counter_val +=1
            
          elif train_val_test_string[v] == 's':
            
            Xa_test[max_inst*counter_test:max_inst*(counter_test+1),:] = tmp
            ya_test[max_inst*counter_test:max_inst*(counter_test+1)] = ya[max_inst*counter:max_inst*(counter+1)]
            y0a_test[max_inst*counter_test:max_inst*(counter_test+1)] = y0a[max_inst*counter:max_inst*(counter+1)]
            counter_test +=1
        
        elif CONFIG.IMG_TYPE == 'mot':
          
          if train_val_test_string[v] == 'v':
            
            if i <= split_point:
            
              Xa_val[max_inst*counter_val:max_inst*(counter_val+1),:] = tmp
              ya_val[max_inst*counter_val:max_inst*(counter_val+1)] = ya[max_inst*counter:max_inst*(counter+1)]
              y0a_val[max_inst*counter_val:max_inst*(counter_val+1)] = y0a[max_inst*counter:max_inst*(counter+1)]
              counter_val +=1
              
            elif i > split_point + num_prev_frames:
              
              Xa_test[max_inst*counter_test:max_inst*(counter_test+1),:] = tmp
              ya_test[max_inst*counter_test:max_inst*(counter_test+1)] = ya[max_inst*counter:max_inst*(counter+1)]
              y0a_test[max_inst*counter_test:max_inst*(counter_test+1)] = y0a[max_inst*counter:max_inst*(counter+1)]
              counter_test +=1     
      
      counter += 1
    
  # delete rows with only zeros in frame t
  not_del_rows_train = ~(Xa_train[:,0:Xa.shape[1]]==0).all(axis=1)
  Xa_train = Xa_train[not_del_rows_train]
  ya_train = ya_train[not_del_rows_train]
  y0a_train = y0a_train[not_del_rows_train]  
  
  not_del_rows_val = ~(Xa_val[:,0:Xa.shape[1]]==0).all(axis=1)
  Xa_val = Xa_val[not_del_rows_val]
  ya_val = ya_val[not_del_rows_val]
  y0a_val = y0a_val[not_del_rows_val]  
  
  not_del_rows_test = ~(Xa_test[:,0:Xa.shape[1]]==0).all(axis=1)
  Xa_test = Xa_test[not_del_rows_test]
  ya_test = ya_test[not_del_rows_test]
  y0a_test = y0a_test[not_del_rows_test]  
  
  ya_train = np.squeeze(ya_train)
  y0a_train = np.squeeze(y0a_train)
  ya_val = np.squeeze(ya_val)
  y0a_val = np.squeeze(y0a_val)
  ya_test = np.squeeze(ya_test)
  y0a_test = np.squeeze(y0a_test)
  
  print('shapes train', np.shape(Xa_train), 'val', np.shape(Xa_val), 'test', np.shape(Xa_test))
  return Xa_train, Xa_val, Xa_test, ya_train, ya_val, ya_test, y0a_train, y0a_val, y0a_test
  

def concatenate_val_for_visualization( Xa, ya ):
  '''
  concatenate validation set for visualization
  ''' 
  
  num_imgs = CONFIG.NUM_IMAGES
  num_prev_frames = CONFIG.NUM_PREV_FRAMES
  max_inst = int( np.load(CONFIG.HELPER_DIR + 'max_inst.npy') )
  
  ya = np.squeeze(ya)
    
  list_videos = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR ))
  
  # validation data 
  # prediction with components in num_prev_frames previous frames
  Xa_zero_val = np.zeros(( (num_imgs-(len(list_videos)*num_prev_frames)) * max_inst, Xa.shape[1] * (num_prev_frames+1)))
  ya_zero_val = np.zeros(( (num_imgs-(len(list_videos)*num_prev_frames)) * max_inst ))
  
  plot_image_list = []
  counter = 0
  counter_new = 0
  for vid in list_videos:
    images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + '/' ))
    for i in range(len(images_all)):
      
      if i >= num_prev_frames:
        
        plot_image_list.append( ( vid, i, counter_new ) )
          
        tmp = np.zeros(( max_inst, Xa.shape[1] * (num_prev_frames+1) ))
        for j in range(0,num_prev_frames+1):
          tmp[:,Xa.shape[1]*j:Xa.shape[1]*(j+1)] = Xa[max_inst*(counter-j):max_inst*(counter-j+1)] 
            
        Xa_zero_val[max_inst*counter_new:max_inst*(counter_new+1),:] = tmp
        ya_zero_val[max_inst*counter_new:max_inst*(counter_new+1)] = ya[max_inst*counter:max_inst*(counter+1)]
        counter_new +=1
      
      counter += 1
  
  # delete rows with only zeros in frame t
  not_del_rows_val = ~(Xa_zero_val[:,0:Xa.shape[1]]==0).all(axis=1)
  Xa_val = Xa_zero_val[not_del_rows_val]
  ya_val = ya_zero_val[not_del_rows_val]
  
  ya_val = np.squeeze(ya_val)
  ya_zero_val = np.squeeze(ya_zero_val)
    
  return Xa_val, ya_val, ya_zero_val, not_del_rows_val, plot_image_list   


