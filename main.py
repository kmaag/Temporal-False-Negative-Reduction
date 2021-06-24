#!/usr/bin/env python3
"""
main script executing tasks defined in global settings file
"""

from global_defs    import CONFIG
from main_functions import detect_fn, plot_fn_instances, analyze_instance_prediction, compute_time_series_metrics, visualize_time_series_metrics, analyze_metrics, compute_mean_ap, compute_fn_fp, plot_fn_fp

 
 
def main():
  
  """ COMMENT:
  Detection of false negative instances, results are stored in HELPER_DIR. 
  """  
  if CONFIG.DETECT_FN:
    run = detect_fn()   
    run.detect_fn_instances() 
    
  
  """ COMMENT:
  For visualizing the false negative instances, the time series instances are used and in IMG_FN_INSTANCES_DIR the resulting visualization images (*.png) are stored. 
  """  
  if CONFIG.PLOT_FN_INSTANCES:
    run = plot_fn_instances()   
    run.plot_fn_instances_per_image() 
    run.visualize_time_series_per_gt()
  
  
  """ COMMENT:
  Analyze the prediction of instances and save results in ANALYZE_INSTANCES_DIR.
  """
  if CONFIG.ANALYZE_TRACKING:
    run = analyze_instance_prediction()  
    run.analyze_instances()
    
  
  """ COMMENT:
  Compute time series metrics and save in METRICS_DIR as pickle (*.p) files.
  """    
  if CONFIG.COMPUTE_METRICS:
    run = compute_time_series_metrics()
    run.compute_time_series_metrics_per_image()
    
    
  """ COMMENT:
  For visualizing the metrics over time, the time series metrics are used and saved in IMG_METRICS_DIR the resulting visualization images (*.png) are stored. 
  """ 
  if CONFIG.VISUALIZE_METRICS:
    run = visualize_time_series_metrics()
    run.visualize_metrics_vs_iou()                          


  """ COMMENT:
  For analyzing meta tasks performance based on the derived metrics, the underlying metrics for the meta model need to be computed and saved in METRICS_DIR defined in "global_defs.py". Results for viewing are saved in ANALYZE_DIR. The calculation results file is saved in ANALYZE_DIR/stats. 
  """
  if CONFIG.ANALYZE_METRICS:
    run = analyze_metrics() 
    run.analyze_time_series_metrics()
  
  
  """ COMMENT:
  Calculate the mean average precision to evaluate the prediction of the network.
  """
  if CONFIG.COMPUTE_MAP:
    run = compute_mean_ap()  
    run.compute_map()
  
  
  """ COMMENT:
  Calculate false negatives/positives to evaluate the prediction of the network after application of meta classification.
  """
  if CONFIG.COMPUTE_FN_FP:
    run = compute_fn_fp()  
    run.compute_fn_fp_vid()
  
  
  """ COMMENT:
  Visualization of the fn vs. fp detections.
  """
  if CONFIG.PLOT_FN_FP:
    run = plot_fn_fp()  
    run.plot_fnfp()
    
    
    
if __name__ == '__main__':
  
  print( "===== START =====" )
  main()
  print( "===== DONE! =====" )
  
  
