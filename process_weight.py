import os
import shutil
dataset_name = {
    '702': 'Dataset702_AbdomenMR',
    '703': 'Dataset703_NeurIPSCell',
    '704': 'Dataset704_Endovis17'
}

weights_path = 'pretrained_model'
target_path = 'data/nnUNet_results'

for weight_name in os.listdir(weights_path):
    if '3d' not in weight_name:
        dataset_id, model_type =  os.path.splitext(weight_name)[0].split('_')

        weight_target_folder = os.path.join(target_path, dataset_name[dataset_id], f'nnUNetTrainerUxLSTM{model_type}__nnUNetPlans__2d', f'fold_{dataset_id}')
        os.makedirs(weight_target_folder, exist_ok=True)

        weight_target_path = os.path.join(weight_target_folder, 'checkpoint_final.pth')
        weight_source_path = os.path.join(weights_path, weight_name)
        shutil.copy(weight_source_path, weight_target_path)
        print(weight_source_path, weight_target_path)
    else:
        dataset_id, model_type, dataset_type =  os.path.splitext(weight_name)[0].split('_')
        weight_target_folder = os.path.join(target_path, dataset_name[dataset_id], f'nnUNetTrainerUxLSTM{model_type}__nnUNetPlans__3d_fullres', f'fold_{dataset_id}')
        os.makedirs(weight_target_folder, exist_ok=True)

        weight_target_path = os.path.join(weight_target_folder, 'checkpoint_final.pth')
        weight_source_path = os.path.join(weights_path, weight_name)
        shutil.copy(weight_source_path, weight_target_path)
        print(weight_source_path, weight_target_path)

    weight_config_path = f'data/nnUNet_preprocessed/{dataset_name[dataset_id]}'
    if not os.path.exists( weight_config_path):
        raise Exception(f"Dataset {dataset_id} not processed")
    for filename in os.listdir(weight_config_path):
        if filename.endswith('.json'):
            source_file = os.path.join(weight_config_path, filename)
            target_file = os.path.join(weight_target_folder, filename)
            shutil.copy(source_file, target_file)