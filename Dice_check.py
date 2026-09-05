import numpy as np
import nibabel as nib
from pathlib import Path
import statistics
import csv
from datetime import date
#import lesion_post_processing
from skimage import measure
from skimage.measure import regionprops
from skimage.measure import label
from scipy.ndimage import distance_transform_edt

import argparse

def calculating_dice(gt, predictions):
    predictions = predictions.astype(np.uint8)
    gt = gt.astype(np.uint8)

    # Make sure shape agrees with case
    if not predictions.shape == gt.shape:
        raise ValueError(
            ("Predictions for case {} have shape {} "
            "which do not match ground truth shape of {}").format( predictions.shape, gt.shape
            )
        )

    try:
        # Compute tumor+liver Dice
        tk_pd = np.greater(predictions, 0)
        tk_gt = np.greater(gt, 0)
        tk_dice = 2*np.logical_and(tk_pd, tk_gt).sum()/(
            tk_pd.sum() + tk_gt.sum()
        )
    except ZeroDivisionError:
        return 0.0, 0.0

    try:
        # Compute tumor Dice
        tu_pd = np.greater(predictions, 1)
        tu_gt = np.greater(gt, 1)
        if (tu_pd.sum() + tu_gt.sum()) == 0:
            return tk_dice, 0.0
        else:
            tu_dice = 2*np.logical_and(tu_pd, tu_gt).sum()/(
                tu_pd.sum() + tu_gt.sum()
            )
    except ZeroDivisionError:
        return tk_dice, 0.0

    return tk_dice, tu_dice

def calculating_hausdorff(gt, predictions, percentile=95):
    """
    Compute Hausdorff distance for liver+tumor and tumor.
    Returns HD and HD95 for both.
    gt: numpy array with labels (0=background, 1=liver, 2=tumor)
    predictions: numpy array with labels
    percentile: percentile for robust HD (default 95)
    """
    predictions = predictions.astype(np.uint8)
    gt = gt.astype(np.uint8)

    # Compute liver+tumor Hausdorff
    tk_pd = np.greater(predictions, 0)
    tk_gt = np.greater(gt, 0)
    tk_hd, tk_hd95 = _surface_distances_hd(tk_gt, tk_pd, percentile)

    # Compute tumor Hausdorff
    tu_pd = np.greater(predictions, 1)
    tu_gt = np.greater(gt, 1)
    tu_hd, tu_hd95 = _surface_distances_hd(tu_gt, tu_pd, percentile)

    return tk_hd, tk_hd95, tu_hd, tu_hd95

def _surface_distances_hd(gt_mask, pred_mask, percentile=95):
    """
    Compute HD and HD95 between two binary masks using surface distances.
    Returns (hd, hd_percentile). Returns (np.inf, np.inf) if either mask is empty.
    """
    if gt_mask.sum() == 0 or pred_mask.sum() == 0:
        return np.inf, np.inf

    # distance transform of the complement gives distance to nearest foreground voxel
    dist_from_gt = distance_transform_edt(~gt_mask)
    dist_from_pred = distance_transform_edt(~pred_mask)

    # surface voxels (border voxels of each mask)
    gt_surface = gt_mask & (distance_transform_edt(gt_mask) <= 1)
    pred_surface = pred_mask & (distance_transform_edt(pred_mask) <= 1)

    # distances from prediction surface to gt, and from gt surface to prediction
    dist_pred_to_gt = dist_from_gt[pred_surface]
    dist_gt_to_pred = dist_from_pred[gt_surface]

    all_distances = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])

    hd = float(np.max(all_distances))
    hd_perc = float(np.percentile(all_distances, percentile))

    return hd, hd_perc

def calculating_metrics(gt, predictions, class_n):
    predictions = predictions.astype(np.uint8)
    gt = gt.astype(np.uint8)
    try:
        # Compute accuracy, precision, recall, sensetivity, voe, vd,
        gt_p = np.greater(gt, class_n-1)
        gt_n = np.less(gt,class_n)

        pos_pred = np.greater(prediction, class_n-1)
        neg_pred = np.less(prediction, class_n)

        fp = np.logical_and(pos_pred, gt_n).sum()
        fn = np.logical_and(neg_pred, gt_p).sum()

        tn = np.logical_and(neg_pred, gt_n).sum()
        tp = np.logical_and(pos_pred, gt_p).sum()
        #voe
        voe=100*(pos_pred.sum()-gt_p.sum())/gt_p.sum()
        #sensetivity
        sn = float(tp/(tp+fn+1))
        #specificity
        sp = float(tn/(tn+fp+1))
        #accuracy
        ac = float((tn+tp)/(tp+tn+fp+fn+1))
        #precision
        pr =  float(tp/(tp+fp+1))
        #recal
        rc = float(tp/(tp+fn+1))

    except ZeroDivisionError:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    return sn, sp, ac, pr, rc, voe

def lesion_calc ( gt, prediction, case_name, file_name1 = Path('tumors.csv'), file_name2 = Path('sizes_tumors.csv'), write=True, count=0, print_res=False):
    """
    gt: numpy file,
    prediction: numpy file,
    Lesion should be class 2!!
    file_name: Pathlib file with '.csv' extension
    write: make False if you don't wanna write output to the file
    print_res: make True if you wanna print results
    """

    segmentation = (prediction == 2)
    #print(segmentation.sum())
    labels = label(segmentation)
    props1 = regionprops(labels) #lesions in prediction

    gt_ = (gt == 2)
    #print(gt_.sum())
    gt_labels = label(gt_)
    props2 = regionprops(gt_labels) #lesion in gt
    #print(props2)
    pro2 = measure.regionprops_table(gt_labels, gt_labels,
                        properties=['label', 'bbox', 'area'])
    if write==True:
        if file_name2.exists()==False:
            with open(file_name2, 'w', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter='|')
                spamwriter.writerow(['case', 'tumors_area', 'found', 'dice'])

    print("checking len", len(props2))
    if len(props2)>0:
        for i in range(0, len(props2)):
            #trying to make Dice 50 percent from tumor

            vol_pred = segmentation[pro2['bbox-0'][i]:pro2['bbox-3'][i], pro2['bbox-1'][i]:pro2['bbox-4'][i], pro2['bbox-2'][i]:pro2['bbox-5'][i]].astype(int)
            vol_gt = gt[pro2['bbox-0'][i]:pro2['bbox-3'][i], pro2['bbox-1'][i]:pro2['bbox-4'][i], pro2['bbox-2'][i]:pro2['bbox-5'][i]]
            vol_gt[vol_gt < 2] = 0
            vol_gt[vol_gt == 2] = 1
            perc = vol_pred.sum()/vol_gt.sum()
            print('checking percentage per tumor', perc)
            #i think was an error here
            if perc >= 0.1:
                count = count + 1
                temp_count = 1
            else:
                count = count
                temp_count = 0
            tumor_dice, _ = calculating_dice(vol_gt, vol_pred)
            print('Dice per tumor: ', tumor_dice)
            with open(file_name2, 'a+', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter='|')
                spamwriter.writerow([case_name, pro2['area'][i], temp_count, tumor_dice])

    #if print_res==True:
    print('in prediction found ', len(props1), 'lesions')
    print('in gt there are', len(props2), 'lesions')
    print('correctly found lesions:', count)

    if write==True:
        if file_name1.exists()==False:
            with open(file_name1, 'w', newline='') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter='|')
                spamwriter.writerow(['case', 'tumors_gt', 'tumors_prediction', 'correct'])
        with open(file_name1, 'a+', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter='|')
            spamwriter.writerow([case_name, len(props2), len(props1), count])

    return len(props2), len(props1), count

def case_metrics(gt, prediction, file_metr, case_name, export_cases=False,voe_liver=[], voe_tumor=[], tk_dice=[], tu_dice=[], sn_liver=[], sp_liver=[], ac_liver=[], pr_liver=[], rc_liver=[], sn_tumor=[], sp_tumor=[], ac_tumor=[], pr_tumor=[], rc_tumor = [], hd_liver=[], hd95_liver=[], hd_tumor=[], hd95_tumor=[]):
    if file_metr.exists()==False:
        with open(file_metr, 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter='|')
            spamwriter.writerow(['case', 'object', 'sn', 'sp', 'ac', 'pr', 'rc', 'dice', 'voe', 'hd', 'hd95'])

    tk, tu = calculating_dice(gt, prediction)
    tk_h, tk_h95, tu_h, tu_h95 = calculating_hausdorff(gt, prediction)

    tk_dice.append(tk)
    tu_dice.append(tu)
    hd_liver.append(tk_h)
    hd95_liver.append(tk_h95)
    hd_tumor.append(tu_h)
    hd95_tumor.append(tu_h95)
    print(Path(case).name, 'dice for liver+tumor and tumor are', tk, tu)
    print(Path(case).name, 'HD95 for liver+tumor and tumor are', tk_h95, tu_h95)

    sn, sp, ac, pr, rc, voe = calculating_metrics(gt, prediction, 1)

    sn_liver.append(sn)
    sp_liver.append(sp)
    ac_liver.append(ac)
    pr_liver.append(pr)
    rc_liver.append(rc)
    voe_liver.append(voe)

    if export_cases==True:
        with open(file_metr, 'a+', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter='|')
            spamwriter.writerow([case_name, 'liver', round(sn, 3), round(sp, 3), round(ac, 3), round(pr, 3), round(rc, 3),round(tk, 3),  round(voe,3), round(tk_h, 3), round(tk_h95, 3)] )

    sn, sp, ac, pr, rc, voe = calculating_metrics(gt, prediction, 2)

    sn_tumor.append(sn)
    sp_tumor.append(sp)
    ac_tumor.append(ac)
    pr_tumor.append(pr)
    rc_tumor.append(rc)
    voe_tumor.append(voe)

    if export_cases==True:
        with open(file_metr, 'a+', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter='|')
            spamwriter.writerow([case_name, 'tumor', round(sn, 3), round(sp, 3), round(ac, 3), round(pr, 3), round(rc, 3),round(tu, 3), round(voe,3), round(tu_h, 3), round(tu_h95, 3)])

    return tk_dice, tu_dice, sn_liver, sp_liver, ac_liver , pr_liver, rc_liver, sn_tumor, sp_tumor, ac_tumor, pr_tumor, rc_tumor, hd_liver, hd95_liver, hd_tumor, hd95_tumor

parser = argparse.ArgumentParser()
parser.add_argument('-pred_dir', '--folder_prediction', dest='folder_prediction', help='Input dir with inference ', default=None)
parser.add_argument('-gt_dir', '--folder_gt', dest='folder_gt', help='Input dir with gt to calculate metrics', default=None)
parser.add_argument('-csv', '--csv', dest='file_main_results', help='CSV file to save all results in one table', default = '27_07_local_results_DL.csv')

args = parser.parse_args()

#folder_gt = Path('D:\\Yuliia_data\\Yuliia_data\\COMET_2023\\MR_img_and_seg_initial_batch\\seg_revised_2023\\all')
#folder_gt = Path('D:\\Yuliia_data\\Yuliia_data\\COMET_2023\\MR_img_and_seg_initial_batch\\seg_revised_2023\\all')
folder_gt = Path('D:\\server_folder\\data_folder\\test_2nd_batch\\seg')
#Path('D:\\Yuliia_data\\Yuliia_data\\COMET_2023\\MR_img_and_seg_initial_batch\\seg_2022\\all') # Update this path to the folder containing ground truth files
folder_prediction = Path('D:\\server_folder\\RESULTS_3paper\\Experiment_14\\inference_2' ) # Update this path to the folder containing prediction files
#folder_gt = Path(args.folder_gt )
#folder_prediction = Path(args.folder_prediction)
#folder_gt  = Path('D:\\Yuliia_data\\Yuliia_data\\lits\\seg\\test_from_gabi_gz')
file_main_results  = Path('metrics_initial_vs_correction.csv')
#file_main_results = Path(args.file_main_results)

file_metr = Path('meticks_by_case_exp_14_check.csv')
file_out = Path('lesion_cal_exp_14_check.csv')
file_area= Path('area_cal_config_exp_14_check.csv')

count_gt, count_prediction, count_correc = 0, 0, 0
index = 0
#file_main_results = Path('results_evrything.csv')

#fold_scv.mkdir(parents=True, exist_ok=True)
fold_scv = folder_prediction.parent

for case in folder_prediction.iterdir():
    case1 = case.name
    prediction = nib.load(folder_prediction/case1).get_fdata()
    print(folder_gt/case1)
    print(prediction.shape)

    gt = nib.load(folder_gt/case1).get_fdata()
    print(gt.shape)
    gt = np.round(gt)

    tk_dice, tu_dice, sn_liver, sp_liver, ac_liver , pr_liver, rc_liver, sn_tumor, sp_tumor, ac_tumor, pr_tumor, rc_tumor, hd_liver, hd95_liver, hd_tumor, hd95_tumor = case_metrics(gt, prediction, fold_scv/file_metr, case1, export_cases=True)
    n_gt, n_prediction, n_correct = lesion_calc(gt, prediction, case1, fold_scv/file_out, fold_scv/file_area)
    count_gt = count_gt + n_gt
    count_prediction = count_prediction + n_prediction
    count_correc = count_correc + n_correct
    index = index + 1


if file_main_results.exists()==False:
    with open(file_main_results, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter='|')
        spamwriter.writerow(['version', 'date', 'object', 'sn', 'sp', 'ac', 'pr', 'rc', 'dice', 'hd', 'hd95', 'lesion_gt','lesion_found','correct_lesion','location'])

version = folder_prediction.parent.name
print(version)

with open(file_main_results, 'a+', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter='|')
    spamwriter.writerow([version,date.today(), 'liver', round(statistics.mean(sn_liver), 3), round(statistics.mean(sp_liver), 3), round(statistics.mean(ac_liver), 3), round(statistics.mean(pr_liver), 3), round(statistics.mean(rc_liver), 3),round(statistics.mean(tk_dice), 3), round(statistics.mean(hd_liver), 3), round(statistics.mean(hd95_liver), 3), count_gt, count_prediction, count_correc, folder_prediction])
    spamwriter.writerow([version,date.today(), 'tumor', round(statistics.mean(sn_tumor), 3), round(statistics.mean(sp_tumor), 3), round(statistics.mean(ac_tumor), 3), round(statistics.mean(pr_tumor), 3), round(statistics.mean(rc_tumor), 3),round(statistics.mean(tu_dice), 3), round(statistics.mean(hd_tumor), 3), round(statistics.mean(hd95_tumor), 3), count_gt, count_prediction, count_correc, folder_prediction])

print ('mean tumor+liver Dice', statistics.mean(tk_dice),'mean tumor Dice', statistics.mean(tu_dice))
# print ('mean tumor+liver Sensetivity', statistics.mean(sn_liver),'mean tumor Dice', statistics.mean(sn_tumor))
# print ('mean tumor+liver Specificity', statistics.mean(sp_liver),'mean tumor Dice', statistics.mean(sp_tumor))
# print ('mean tumor+liver Accuracy', statistics.mean(ac_liver),'mean tumor Dice', statistics.mean(ac_tumor))
# print ('mean tumor+liver Precision', statistics.mean(pr_liver),'mean tumor Dice', statistics.mean(pr_tumor))
# print ('mean tumor+liver Recall', statistics.mean(rc_liver),'mean tumor Dice', statistics.mean(rc_tumor))