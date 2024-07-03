# dataset_path 
# 702
# -i data/nnUNet_raw/Dataset702_AbdomenMR/imagesTs
# -g data/nnUNet_raw/Dataset702_AbdomenMR/labelsTs

# 703
# -i data/nnUNet_raw/Dataset703_NeurIPSCell/imagesVal
# -g data/nnUNet_raw/Dataset703_NeurIPSCell/labelsVal-instance-mask

# 704
# -i data/nnUNet_raw/Dataset704_Endovis17/imagesTs
# -g data/nnUNet_raw/Dataset704_Endovis17/labelsTs 


dataset_id="702"
exp_name="pretrained_weight_Bot_3d"
output_path="infer_result/${dataset_id}/${exp_name}"
mkdir -p $output_path
save_path="evaluate_result/${dataset_id}/pretrained_weight"

CUDA_VISIBLE_DEVICES=2 nnUNetv2_predict -i data/nnUNet_raw/Dataset702_AbdomenMR/imagesTs \
-o $output_path -d ${dataset_id} \
-f ${dataset_id} -tr nnUNetTrainerUxLSTMBot \
--disable_tta -c 3d_fullres

python evaluation/abdomen_DSC_Eval.py \
--gt_path /home/ubuntu/public_c/crl/cchenzong/U_xlstm/data/nnUNet_raw/Dataset702_AbdomenMR/labelsTs \
--seg_path $output_path \
--save_path $save_path

python evaluation/abdomen_NSD_Eval.py \
--gt_path /home/ubuntu/public_c/crl/cchenzong/U_xlstm/data/nnUNet_raw/Dataset702_AbdomenMR/labelsTs \
--seg_path $output_path \
--save_path $save_path



# dataset_id="703"
# exp_name="pretrained_weight_Bot"
# output_path="infer_result/${dataset_id}/${exp_name}"
# mkdir -p $output_path
# save_path="evaluate_result/703/pretrained_weight"

# nnUNetv2_predict -i data/nnUNet_raw/Dataset703_NeurIPSCell/imagesVal/ \
# -o $output_path -d ${dataset_id} \
# -f ${dataset_id} -tr nnUNetTrainerUxLSTMBot \
# --disable_tta -c 2d

#  python evaluation/compute_cell_metric.py \
# -g data/nnUNet_raw/Dataset703_NeurIPSCell/labelsVal-instance-mask \
# -s $output_path \
# -o $save_path



# dataset_id="704"
# exp_name="pretrained_weight_Bot"
# output_path="infer_result/${dataset_id}/${exp_name}"
# mkdir -p $output_path
# save_path="evaluate_result/${dataset_id}/pretrained_weight"

# nnUNetv2_predict -i data/nnUNet_raw/Dataset704_Endovis17/imagesTs \
# -o $output_path -d ${dataset_id} \
# -f ${dataset_id} -tr nnUNetTrainerUxLSTMBot \
# --disable_tta -c 2d

# python evaluation/endoscopy_DSC_Eval.py \
# --gt_path data/nnUNet_raw/Dataset704_Endovis17/labelsTs \
# --seg_path $output_path \
# --save_path $save_path

# python evaluation/endoscopy_NSD_Eval.py \
# --gt_path data/nnUNet_raw/Dataset704_Endovis17/labelsTs \
# --seg_path $output_path \
# --save_path $save_path


