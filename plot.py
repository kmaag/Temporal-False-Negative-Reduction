#!/usr/bin/env python3
'''
script including
functions for visualizations
'''

import os
import time
import numpy as np
import pandas as pd
from skimage import feature
from PIL import Image, ImageDraw
from scipy.stats import pearsonr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rc('font', size=10, family='serif')
plt.rc('text', usetex=True)

from global_defs import CONFIG
from calculate   import create_bb
from helper      import name_to_latex_scatter_plot, name_to_latex
from in_out      import depth_load, time_series_instances_load, get_save_path_image_i, ground_truth_load, score_small_load, helpers_p_load, metrics_load
from metrics     import compute_matches_gt_pred
import labels as labels

trainId2label = { label.trainId : label for label in reversed(labels.kitti_labels) }


def hex_to_rgb(input1):
  value1 = input1.lstrip('#')
  return tuple(int(value1[i:i+2], 16) for i in (0, 2 ,4))

  
def compute_iou_color( iou ):
  
  iou_color = np.zeros((3))
  iou_color[0] = np.asarray( iou )
  iou_color[0] = 1-0.5*iou_color[0]
  iou_color[1] = np.asarray( iou )
  iou_color[2] = 0.3+0.35*np.asarray( iou )
  return iou_color*255 


def plot_fn_inst( vid, n, save_path, colors_list, instances_new=[], scores_new=[] ):

  t = time.time()
  
  epsilon = CONFIG.EPS_MATCHING
  num_reg = CONFIG.NUM_REG_MATCHING
  
  input_image = np.asarray( Image.open(get_save_path_image_i(vid, n)) )
  gt_image = ground_truth_load(vid, n)
  inst_image  = time_series_instances_load( vid, n, epsilon, num_reg )
  scores = score_small_load(vid, n)
  
  _, gt_match_orig = compute_matches_gt_pred(np.array(gt_image, dtype='int16'), inst_image, scores, CONFIG.IOU_THRESHOLD, CONFIG.FLAG_DOUBLE_PRED, CONFIG.FLAG_DOUBLE_GT)
  
  if instances_new != []:
    num_orig = inst_image.shape[0]
    inst_image = np.concatenate((inst_image, instances_new), axis=0)
    scores = np.concatenate((scores, scores_new), axis=0)
    
  pred_match, gt_match = compute_matches_gt_pred(np.array(gt_image, dtype='int16'), inst_image, scores, CONFIG.IOU_THRESHOLD, CONFIG.FLAG_DOUBLE_PRED, CONFIG.FLAG_DOUBLE_GT)
  
  if np.sum(gt_match_orig==-1) != np.sum(gt_match==-1):
    gt_match[gt_match_orig!=gt_match] = -2

  instance_masks = []
  gt_class_ids = []
  obj_ids = np.unique(gt_image)
  for i in range(len(obj_ids)):
    if obj_ids[i] != 0 and obj_ids[i] != 10000:
        m_i = np.zeros(np.shape(gt_image))
        m_i[gt_image==obj_ids[i]] = 1
        instance_masks.append(m_i)
        gt_class_ids.append(obj_ids[i] // 1000)
  # Pack instance masks into an array
  if len(instance_masks) > 0:
    mask_tmp = np.stack(instance_masks, axis=2).astype(np.bool)
    # [num_instances, (y1, x1, y2, x2)]
    gt_bbox = create_bb(mask_tmp)
  
  instance_masks = []
  seg_class_ids = []
  for i in range(inst_image.shape[0]):
    m_i = np.zeros(( inst_image.shape[1], inst_image.shape[2] ))
    m_i[inst_image[i,:,:]!=0] = 1
    instance_masks.append(m_i)
    seg_class_ids.append(inst_image[i].max() // 10000)        
  if len(instance_masks) > 0:
    mask_tmp = np.stack(instance_masks, axis=2).astype(np.bool)
    seg_bbox = create_bb(mask_tmp)
  
  I1 = input_image.copy()
  tmp = np.zeros((3))
  for i in range(input_image.shape[0]):
    for j in range(input_image.shape[1]):
      if gt_image[i,j] > 0 and gt_image[i,j] < 10000:   
        tmp = np.asarray( hex_to_rgb( colors_list[ (gt_image[i,j]-1) % len(colors_list) ] ) )
        I1[i,j,:] = tmp * 0.6 + input_image[i,j,:] * 0.4
      elif gt_image[i,j] == 10000:
        tmp = np.asarray((255,255,255))
        I1[i,j,:] = tmp * 0.6 + input_image[i,j,:] * 0.4
  
  I2 = input_image.copy()
  for i in range(input_image.shape[0]):
    for j in range(input_image.shape[1]):
      tmp = np.zeros((3))
      counter = 0
      for k in range(inst_image.shape[0]):
        if inst_image[k,i,j] > 0: 
          tmp += np.asarray( hex_to_rgb( colors_list[ int(inst_image[k,i,j]-1) % len(colors_list) ] ) )
          counter += 1
        elif inst_image[k,i,j] < 0: 
          counter += 1   
      if counter > 0:
        tmp /= counter
        I2[i,j,:] = tmp * 0.6 + input_image[i,j,:] * 0.4
  
  img12 = np.concatenate( (I1,I2), axis=0 )
  
  depth = depth_load(vid, n)
  depth[depth==0] = depth[depth>0].min()
  plt.imsave(save_path + '/tmp' + str(n).zfill(6) + '.png', np.log10(depth), cmap='inferno')
  I3 = np.asarray( Image.open(save_path + '/tmp' + str(n).zfill(6) + '.png').convert('RGB') )
  os.remove(save_path + '/tmp' + str(n).zfill(6) + '.png')

  I4 = input_image.copy()
  for k in range(len(pred_match)):
    iou = 0
    I = np.sum( np.logical_and(inst_image[k]!=0,gt_image==pred_match[k]) )
    U = np.sum(inst_image[k]!=0) + np.sum(gt_image==pred_match[k]) - I
    if U > 0:
      iou = float(I) / float(U)
    iou_color = compute_iou_color(iou)
    for i in range(input_image.shape[0]):
      for j in range(input_image.shape[1]):
        if inst_image[k,i,j] < 0:
          I4[i,j,:] = np.zeros((3)) * 0.6 + input_image[i,j,:] * 0.4
        elif inst_image[k,i,j] > 0:
          I4[i,j,:] = iou_color * 0.6 + input_image[i,j,:] * 0.4

  img34 = np.concatenate( (I3,I4), axis=0 )
  
  img = np.concatenate( (img12,img34), axis=1 )
  image = Image.fromarray(img.astype('uint8'), 'RGB')

  draw = ImageDraw.Draw(image)
  if len(seg_class_ids) > 0:
    for i in range(seg_bbox.shape[0]):
      color_class = trainId2label[ seg_class_ids[i] ].color
      draw.rectangle([seg_bbox[i,1], seg_bbox[i,0]+input_image.shape[0], seg_bbox[i,3], seg_bbox[i,2]+input_image.shape[0]], fill=None, outline=color_class)
      
      if instances_new != [] and i >= num_orig: 
        draw.rectangle([seg_bbox[i,1]-1, seg_bbox[i,0]-1+input_image.shape[0], seg_bbox[i,3]+1, seg_bbox[i,2]+1+input_image.shape[0]], fill=None, outline='deeppink')
        draw.rectangle([seg_bbox[i,1]-1+input_image.shape[1], seg_bbox[i,0]-1, seg_bbox[i,3]+1+input_image.shape[1], seg_bbox[i,2]+1], fill=None, outline='deeppink')
  if len(gt_class_ids) > 0:
    for i in range(gt_bbox.shape[0]):
      color_class = trainId2label[ gt_class_ids[i] ].color
      draw.rectangle([gt_bbox[i,1], gt_bbox[i,0], gt_bbox[i,3], gt_bbox[i,2]], fill=None, outline=color_class)
      if gt_match[i] == -2:
        draw.rectangle([gt_bbox[i,1]-1, gt_bbox[i,0]-1, gt_bbox[i,3]+1, gt_bbox[i,2]+1], fill=None, outline='lime')
        draw.rectangle([gt_bbox[i,1]-1+input_image.shape[1], gt_bbox[i,0]-1, gt_bbox[i,3]+1+input_image.shape[1], gt_bbox[i,2]+1], fill=None, outline='lime')
      elif gt_match[i] == -1:
        draw.rectangle([gt_bbox[i,1]-1, gt_bbox[i,0]-1, gt_bbox[i,3]+1, gt_bbox[i,2]+1], fill=None, outline='cyan')
        draw.rectangle([gt_bbox[i,1]-1+input_image.shape[1], gt_bbox[i,0]-1, gt_bbox[i,3]+1+input_image.shape[1], gt_bbox[i,2]+1], fill=None, outline='cyan')
  
  if -2 in gt_match:
    image.save(save_path + '/img' + str(n).zfill(6) + '_fn_detect.png')
  elif -1 in gt_match:
    image.save(save_path + '/img' + str(n).zfill(6) + '_fn.png')
  else:
    image.save(save_path + '/img' + str(n).zfill(6) + '.png')
  
  print('plot image', n, ': time needed ', time.time()-t)  


def plot_time_series_per_gt( vid ):
  
  epsilon = CONFIG.EPS_MATCHING
  num_reg = CONFIG.NUM_REG_MATCHING
  
  save_path = CONFIG.IMG_FN_INSTANCES_DIR + vid + '_' + str(CONFIG.IOU_THRESHOLD) + '_' + str(CONFIG.FLAG_DOUBLE_PRED) + str(CONFIG.FLAG_DOUBLE_GT) + '_gt/'
  if not os.path.exists( save_path ):
    os.makedirs( save_path )
  
  instances_new = helpers_p_load('inst_time_series_tested')
  array_vid = np.asarray(instances_new['vid_no'])
  array_frame = np.asarray(instances_new['frame_no'])
  array_score = np.asarray(instances_new['score_val'])
  if CONFIG.IMG_TYPE == 'kitti':
    array_inst = np.stack(instances_new['inst_array'])
  if CONFIG.IMG_TYPE == 'mot':
    array_inst_tmp = []
    gt = ground_truth_load(vid, 0)
    for i in range(array_vid.shape[0]):
      if array_vid[i] == vid:
        array_inst_tmp.append(instances_new['inst_array'][i])
      else:
        array_inst_tmp.append(np.zeros(gt.shape,dtype='int16'))
    array_inst = np.stack(array_inst_tmp)
  
  images_all = sorted(os.listdir( CONFIG.TIME_SERIES_INST_DIR + vid + '/' ))
  
  list_gt_ids = []
  
  for n in range(len(images_all)):
    gt_image = ground_truth_load(vid, n) 
    list_gt_ids.extend(np.unique(gt_image))
  list_gt_ids = np.unique(list_gt_ids)
  list_gt_ids = list_gt_ids[1:-1]
  print(vid, 'gt Ids', len(list_gt_ids), list_gt_ids)
  
  # size, iou, iou detect
  size_iou = np.zeros((len(list_gt_ids), len(images_all), 3))
  
  for n in range(len(images_all)):
      
    gt_image = ground_truth_load(vid, n) 
    
    for k in np.unique(gt_image):
      if k != 0 and k != 10000:
        for j in range(len(list_gt_ids)):
          if list_gt_ids[j] == k:
            break
        size_iou[j,n,0] = np.sum(gt_image==k)
      
    inst_image  = time_series_instances_load( vid, n, epsilon, num_reg )
    scores = score_small_load(vid, n)
  
    pred_match, gt_match_orig = compute_matches_gt_pred(np.array(gt_image, dtype='int16'), inst_image, scores, CONFIG.IOU_THRESHOLD, CONFIG.FLAG_DOUBLE_PRED, CONFIG.FLAG_DOUBLE_GT)
    
    for k in range(len(pred_match)):
      if pred_match[k] != -1:
        
        for j in range(len(list_gt_ids)):
          if list_gt_ids[j] == pred_match[k]:
            break
        
        I = np.sum( np.logical_and(inst_image[k]!=0,gt_image==pred_match[k]) )
        U = np.sum(inst_image[k]!=0) + np.sum(gt_image==pred_match[k]) - I
        if U > 0:
          size_iou[j,n,1] = float(I) / float(U)
       
    inst_image = np.concatenate((inst_image, array_inst[np.logical_and(array_vid==vid, array_frame==n)]), axis=0)
    scores = np.concatenate((scores, array_score[np.logical_and(array_vid==vid, array_frame==n)]), axis=0)
    
    pred_match, gt_match = compute_matches_gt_pred(np.array(gt_image, dtype='int16'), inst_image, scores, CONFIG.IOU_THRESHOLD, CONFIG.FLAG_DOUBLE_PRED, CONFIG.FLAG_DOUBLE_GT)

    for k in range(len(pred_match)):
      if pred_match[k] != -1:
        
        for j in range(len(list_gt_ids)):
          if list_gt_ids[j] == pred_match[k]:
            break
        
        I = np.sum( np.logical_and(inst_image[k]!=0,gt_image==pred_match[k]) )
        U = np.sum(inst_image[k]!=0) + np.sum(gt_image==pred_match[k]) - I
        if U > 0:
          size_iou[j,n,2] = float(I) / float(U)
  
  for i in range(len(list_gt_ids)):
    
    if list_gt_ids[i] == 1021:
  
      print('plot gt', i)
      
      index = np.where(size_iou[i,:,0] > 0)[0]
      list_img_num = np.arange(index[0], index[-1]+1)
      
      font_size = 14.9
      dpi_val = 200
      f, (ax, ax1, ax2) = plt.subplots(3, 1, sharex=True)
      f.set_size_inches((gt_image.shape[1]-468)/dpi_val,(gt_image.shape[0]+140)/dpi_val*2)
      ax.plot(list_img_num, size_iou[i,index[0]:index[-1]+1,0], color='dodgerblue') 
      ax.set_ylabel('S', fontsize=font_size, labelpad=-19)
      ax1.plot(list_img_num, size_iou[i,index[0]:index[-1]+1,1], color='salmon') 
      ax1.set_ylabel('IoU', fontsize=font_size, labelpad=-10)
      ax2.plot(list_img_num, size_iou[i,index[0]:index[-1]+1,2], color='purple') 
      ax2.set_ylabel('IoU$_d$', fontsize=font_size, labelpad=-10)
      plt.xlabel('frame', fontsize=font_size, labelpad=-3)
      ax.tick_params(axis ='y', labelsize=font_size)
      ax1.tick_params(axis ='y', labelsize=font_size)
      ax2.tick_params(axis ='x', labelsize=font_size)
      ax2.tick_params(axis ='y', labelsize=font_size)
      ax.set_yticks([ (np.nanmin(np.array(size_iou[i,index[0]:index[-1]+1,0]))).round(0), (np.nanmax(np.array(size_iou[i,index[0]:index[-1]+1,0]))).round(0) ])
      ax1.set_yticks([ (np.nanmin(np.array(size_iou[i,index[0]:index[-1]+1,1]))).round(2), (np.nanmax(np.array(size_iou[i,index[0]:index[-1]+1,1]))).round(2) ])
      ax2.set_yticks([ (np.nanmin(np.array(size_iou[i,index[0]:index[-1]+1,2]))).round(2), (np.nanmax(np.array(size_iou[i,index[0]:index[-1]+1,2]))).round(2) ])
      f.savefig(save_path + 'time_series' + str(list_gt_ids[i]), dpi=dpi_val, bbox_inches='tight')
      plt.close()
    
  
def add_scatterplot_vs_iou(ious, sizes, dataset, name, setylim=True):
  
  rho = pearsonr(ious,dataset)
  plt.title(r'$\rho = {:.05f}$'.format(rho[0]), fontsize=35)
  plt.scatter( ious, dataset, s = sizes/50, marker='.', c='cornflowerblue', alpha=0.1 )  
  plt.xlabel('$\mathit{IoU}$', fontsize=35, labelpad=-15)
  plt.ylabel(name, fontsize=35, labelpad=-45)
  plt.xticks((0,1),fontsize=35)
  plt.yticks((np.amin(dataset),np.amax(dataset)),fontsize=35)
  plt.subplots_adjust(left=0.15)
  

def plot_scatter_metric_iou(list_metrics):
  
  print('plot scatter metric vs iou')
  
  if len(list_metrics) == 20:
    num_x = 5
    num_y = 4
  
  size_x = 11.0 
  size_y = 9.0 
  
  if not os.path.exists( CONFIG.IMG_METRICS_DIR + 'scatter/' ):
    os.makedirs( CONFIG.IMG_METRICS_DIR + 'scatter/' )
  
  epsilon = CONFIG.EPS_MATCHING
  num_reg = CONFIG.NUM_REG_MATCHING
  runs = CONFIG.NUM_RESAMPLING
  
  if CONFIG.FLAG_OBJ_SEG == 0:
    name_iou = 'iou_o'
  else:
    name_iou = 'iou_s'
  
  tvs = np.load(CONFIG.HELPER_DIR + 'tvs_runs' + str(runs ) + '.npy')  
  metrics = metrics_load( tvs[0], 0, epsilon, num_reg, 1 ) 
  df_full = pd.DataFrame( data=metrics ) 
  df_full = df_full.copy().loc[df_full['S'].to_numpy().nonzero()[0]]
  
  plt.rc('axes', titlesize=30)
  plt.rc('figure', titlesize=25)
  
  fig = plt.figure(frameon=False)
  fig.set_size_inches(size_x*num_x,size_y*num_y)
  
  result_path = CONFIG.IMG_METRICS_DIR + 'scatter/scatter' + str(len(list_metrics)) + '.txt'
  with open(result_path, 'wt') as fi:
    
    for i in range(len(list_metrics)):
      fig.add_subplot(num_y,num_x,i+1)
      if 'S' in list_metrics[i]:
        add_scatterplot_vs_iou(df_full[name_iou], 5000, df_full[list_metrics[i]]/df_full[list_metrics[i]].max(), name_to_latex_scatter_plot(list_metrics[i]))
      elif 'rel' in list_metrics[i]:
        add_scatterplot_vs_iou(df_full[name_iou], df_full['S'], df_full[list_metrics[i]]/df_full[list_metrics[i]].max(), name_to_latex_scatter_plot(list_metrics[i]))
      else:
        add_scatterplot_vs_iou(df_full[name_iou], df_full['S'], df_full[list_metrics[i]], name_to_latex_scatter_plot(list_metrics[i]))
        
      print(list_metrics[i], '{:.5f}'.format(pearsonr(df_full[name_iou], df_full[list_metrics[i]])[0]), file=fi)

  plt.savefig(CONFIG.IMG_METRICS_DIR + 'scatter/scatter' + str(len(list_metrics)) + '.png', bbox_inches='tight')
  plt.close()


def plot_regression_scatter( Xa_test, ya_test, ya_test_pred, X_names, num_frames ):
  
  plt.rc('axes', titlesize=10)
  plt.rc('figure', titlesize=10)
  cmap=plt.get_cmap('tab20')

  S_ind = 0
  for S_ind in range(len(X_names)):
    if X_names[S_ind] == 'S':
      break
  
  figsize=(3.0,13.0/5.0)
  plt.figure(figsize=figsize, dpi=300)
  plt.clf()
  
  sizes = np.squeeze(Xa_test[:,S_ind]*np.std(Xa_test[:,S_ind]))
  sizes = sizes - np.min(sizes)
  sizes = sizes / np.max(sizes) * 50 #+ 1.5      
  x = np.arange(0., 1, .01)
  plt.plot( x, x, color='black' , alpha=0.5, linestyle='dashed')
  plt.scatter( ya_test, np.clip(ya_test_pred,0,1), s=sizes, linewidth=.5, c=cmap(0), edgecolors=cmap(1), alpha=0.25 )
  plt.xlabel('$\mathit{IoU}$')
  plt.ylabel('predicted $\mathit{IoU}$')
  plt.savefig(CONFIG.ANALYZE_DIR + 'scatter/scatter_test_npf' + str(num_frames) + '.png', bbox_inches='tight')
  plt.close()


def plot_coef_timeline( mean_stats, X_names ):
  
  num_prev_frames = CONFIG.NUM_PREV_FRAMES

  switcher = {
          'S'           : 'palegreen',          
          'S_bd'        : 'limegreen', 
          'S_in'        : 'olivedrab',
          'S_rel'       : 'mediumseagreen',
          'S_rel_in'    : 'darkgreen',
          'mean_x'      : 'gold',
          'mean_y'      : 'orange',
          'class'       : 'darkturquoise',
          'ratio'       : 'peru',
          'deformation' : 'mediumpurple',
          'survival'    : 'aquamarine',
          'diff_mean'   : 'lightskyblue',
          'diff_size'   : 'deepskyblue',
          'diff_depth'  : 'cornflowerblue',
          'occlusion'   : 'blueviolet',
          'score'       : 'lightcoral',
          'D'           : 'lightpink',          
          'D_bd'        : 'palevioletred', 
          'D_in'        : 'deeppink',
          'D_rel'       : 'mediumvioletred',
          'D_rel_in'    : 'purple'
  }

  size_font = 20
  
  for i in range(num_prev_frames+1):
    
    coefs = np.asarray(mean_stats['coef'][i])
    num_timeseries = np.arange(0, i+1)
    x_ticks = []
    
    f1 = plt.figure(figsize=(10,6.2))
    plt.clf()
    
    # 1: smaller, shift to the left, 2: smaller, shift to the bottom
    ax = f1.add_axes([0.11, 0.12, 0.6, 0.75])
    
    for c in range(len(X_names)):
      if c == 0:
        x_ticks.append( '$t$' )
      else:
        x_ticks.append( '$t-$' + str(c) )

      index = 0
      for index in range(len(X_names)):
        if X_names[index] == X_names[c]:
          break
        
      coef_c = np.zeros((i+1))
      for k in range(i+1):
        coef_c[k] = coefs[k*len(X_names)+index]
        
      plt.plot(num_timeseries, coef_c, color=switcher.get(X_names[c], 'red'), marker='o', label=name_to_latex(X_names[c]), alpha=0.8)  
    
    plt.xticks(fontsize = size_font)
    plt.yticks(fontsize = size_font)
    plt.xlabel('frame', fontsize=size_font)
    plt.xticks(num_timeseries, (x_ticks))
    plt.ylabel('feature importance', fontsize=size_font)
    plt.legend(bbox_to_anchor=(1., 0.5), loc=6, ncol=2, borderaxespad=0.6, prop={'size': 16})
    save_path = CONFIG.ANALYZE_DIR +'feature_importance/coef' + str(i)
    f1.savefig(save_path, dpi=300)
    plt.close()


def plot_train_val_test_timeline( num_timeseries, train, train_std, val, val_std, test_R, test_R_std, analyze_type ):
  
  size_font = 26
  
  f1 = plt.figure(1,frameon=False) 
  plt.clf()
  plt.plot(num_timeseries, train, color='violet', marker='o', label='train')  
  plt.fill_between(num_timeseries, train-train_std, train+train_std, color='violet', alpha=0.05 )
  plt.plot(num_timeseries, val, color='midnightblue', marker='o', label='val')
  plt.fill_between(num_timeseries, val-val_std, val+val_std, color='midnightblue', alpha=0.05 )
  plt.plot(num_timeseries, test_R, color='deepskyblue', marker='o', label='test')  
  plt.fill_between(num_timeseries, test_R-test_R_std, test_R+test_R_std, color='deepskyblue', alpha=0.05 )
  plt.xlabel('Frames', fontsize=size_font)
  if analyze_type == 'r2':
    plt.ylabel('$R^2$', fontsize=size_font)
    name = 'r2_timeline.png'
  elif analyze_type == 'auc':
    plt.ylabel('$AUROC$', fontsize=size_font)
    name = 'auc_timeline.png'
  elif analyze_type == 'acc':
    plt.ylabel('$ACC$', fontsize=size_font)
    name = 'acc_timeline.png'
  plt.xticks(fontsize = size_font)
  plt.yticks(fontsize = size_font)
  plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode='expand', prop={'size': 20}, borderaxespad=0.)
  save_path = CONFIG.ANALYZE_DIR +'train_val_test_timeline/' + name
  f1.savefig(save_path, bbox_inches='tight', dpi=400)
  plt.close()
  
  f1 = plt.figure(1,frameon=False) 
  plt.clf()
  plt.plot(num_timeseries, test_R, color='deepskyblue', marker='o', label=str(test_R.max())+', '+str(np.argmax(test_R)+1))  
  plt.fill_between(num_timeseries, test_R-test_R_std, test_R+test_R_std, color='deepskyblue', alpha=0.05 )
  plt.xlabel('Frames', fontsize=size_font)
  if analyze_type == 'r2':
    plt.ylabel('$R^2$', fontsize=size_font)
    name = 'r2_timeline_test.png'
  elif analyze_type == 'auc':
    plt.ylabel('$AUROC$', fontsize=size_font)
    name = 'auc_timeline_test.png'
  elif analyze_type == 'acc':
    plt.ylabel('$ACC$', fontsize=size_font)
    name = 'acc_timeline_test.png'
  plt.xticks(fontsize = size_font)
  plt.yticks(fontsize = size_font)
  plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode='expand', prop={'size': 20}, borderaxespad=0.)
  save_path = CONFIG.ANALYZE_DIR +'train_val_test_timeline/' + name
  f1.savefig(save_path, bbox_inches='tight', dpi=400)
  plt.close()


def plot_fn_vs_fp(save_path, fp_fn_classic, fp_fn_meta):
  
  size_text = 22
  
  sort_classic = np.argsort(fp_fn_classic[:,2])
  sort_meta = np.argsort(fp_fn_meta[:,2])
  
  f1 = plt.figure(1,frameon=False) 
  plt.clf()
  if np.ndim(fp_fn_classic) == 2:
    plt.plot(fp_fn_classic[sort_classic,0]/1000, fp_fn_classic[sort_classic,1]/1000, color='olivedrab', marker='o', linewidth=2, markersize=10, label='score', alpha=0.7)  
    plt.plot(fp_fn_meta[sort_meta,0]/1000, fp_fn_meta[sort_meta,1]/1000, color='tab:pink', marker='o', linewidth=2, markersize=10, label='ours', alpha=0.7)
  elif np.ndim(fp_fn_classic) == 3:
    for i in range(fp_fn_classic.shape[1]):
      plt.plot(fp_fn_classic[sort_classic,i,0]/1000, fp_fn_classic[sort_classic,i,1]/1000, color='olivedrab', marker='o', linewidth=2, markersize=10, alpha=0.7)  
      plt.plot(fp_fn_meta[sort_meta,i,0]/1000, fp_fn_meta[sort_meta,i,1]/1000, color='tab:pink', marker='o', linewidth=2, markersize=10, alpha=0.7)
  matplotlib.rcParams['legend.numpoints'] = 1
  matplotlib.rcParams['legend.handlelength'] = 0
  plt.xlabel('$\#$ false positives ($\\times 10^3$)', fontsize=size_text) 
  plt.ylabel('$\#$ false negatives ($\\times 10^3$)', fontsize=size_text)
  plt.xticks(fontsize=size_text)
  plt.yticks(fontsize=size_text)
  plt.legend(fontsize=size_text)
  f1.savefig(save_path + 'png', dpi=400, bbox_inches='tight')
  plt.close()
  
  if np.ndim(fp_fn_classic) == 2:
    sort_classic = np.argsort(fp_fn_classic[:,2]*-1)
    save_path1 = save_path.replace('fn_fp_iou','rp_iou')
    fp_fn_classic[fp_fn_classic[:,2]==1,3] = 1
    fp_fn_meta[fp_fn_meta[:,2]==0,3] = 1
    prec_classic = fp_fn_classic[sort_classic,3] / (fp_fn_classic[sort_classic,3]+fp_fn_classic[sort_classic,0])
    prec_meta = fp_fn_meta[sort_meta,3] / (fp_fn_meta[sort_meta,3]+fp_fn_meta[sort_meta,0])
    rec_classic = fp_fn_classic[sort_classic,3] / (fp_fn_classic[sort_classic,3]+fp_fn_classic[sort_classic,1])
    rec_meta = fp_fn_meta[sort_meta,3] / (fp_fn_meta[sort_meta,3]+fp_fn_meta[sort_meta,1])
    
    ap_classic = rec_classic[0] * prec_classic[0]
    ap_meta = rec_meta[0] * prec_meta[0]
    for i in range(1,len(rec_classic)):
      ap_classic += (rec_classic[i]-rec_classic[i-1])*prec_classic[i]
      ap_meta += (rec_meta[i]-rec_meta[i-1])*prec_meta[i]
    print('ap', ap_classic, ap_meta)
    
    rec_classic = rec_classic[1:]
    prec_classic = prec_classic[1:]
    rec_meta = rec_meta[1:]
    prec_meta = prec_meta[1:]
    
    f1 = plt.figure(1,frameon=False) 
    plt.clf()
    plt.plot(rec_classic, prec_classic, color='lightseagreen', marker='o', linewidth=2, markersize=10, label='score {:.02f}$\%$'.format(ap_classic*100), alpha=0.7)  
    plt.plot(rec_meta, prec_meta, color='mediumvioletred', marker='o', linewidth=2, markersize=10, label='ours {:.02f}$\%$'.format(ap_meta*100), alpha=0.7)
    matplotlib.rcParams['legend.numpoints'] = 1
    matplotlib.rcParams['legend.handlelength'] = 0
    plt.xlabel('recall', fontsize=size_text) 
    plt.ylabel('precision', fontsize=size_text)
    plt.xticks(fontsize=size_text)
    plt.yticks(fontsize=size_text)
    plt.legend(fontsize=size_text)
    f1.savefig(save_path1 + 'png', dpi=400, bbox_inches='tight')
    plt.close()
    




