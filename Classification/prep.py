import pandas as pd
import numpy as np
import os
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.model_selection import train_test_split 

def prepare_final_csv(csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df.fillna(0)
    if 'AVSD' in df.columns:
        df = df.drop(columns=['AVSD'])
        
    test_ids = [1001, 1016, 1022, 1023, 1024, 1043, 1044, 1046, 1052, 1060, 
                1064, 1077, 1081, 1091, 1102, 1105, 1106, 1112, 1122, 1124, 1125, 1161]
   
    df_test = df[df['index'].isin(test_ids)]
    test_path = os.path.join(output_dir, "test_data.csv")
    df_test.to_csv(test_path, index=False)

    df_rest = df[~df['index'].isin(test_ids)].copy()
    
    rare_diseases = ['PAS', 'AAH', 'IAA', 'CAT', 'DAA', 'TGA', 'APVC']
    all_diseases = [c for c in df.columns if c != 'index']
    
    df_rest['is_rare'] = df_rest[rare_diseases].sum(axis=1)
    df_train_rare = df_rest[df_rest['is_rare'] > 0].drop(columns=['is_rare'])

    df_sisa = df_rest[df_rest['is_rare'] == 0].drop(columns=['is_rare'])
    
    df_sisa['total_disease'] = df_sisa[all_diseases].sum(axis=1)
    df_common = df_sisa[df_sisa['total_disease'] > 0].drop(columns=['total_disease'])
    df_pure_normal = df_sisa[df_sisa['total_disease'] == 0].drop(columns=['total_disease'])
    
    X_common = df_common[['index']].values
    y_common = df_common.drop(columns=['index']).values
   
    test_ratio_common = 18 / len(df_common)
    X_train_c, y_train_c, X_val_c, y_val_c = iterative_train_test_split(X_common, y_common, test_size=test_ratio_common)
    
    df_train_common = pd.DataFrame(np.hstack((X_train_c, y_train_c)), columns=df.columns)
    df_val_common = pd.DataFrame(np.hstack((X_val_c, y_val_c)), columns=df.columns)
    
    df_train_normal, df_val_normal = train_test_split(df_pure_normal, test_size=4, random_state=42)
    
    df_train_final = pd.concat([df_train_rare, df_train_common, df_train_normal], ignore_index=True)
    df_val_final = pd.concat([df_val_common, df_val_normal], ignore_index=True)

    df_train_final = df_train_final.sample(frac=1, random_state=42).reset_index(drop=True)
    df_val_final = df_val_final.sample(frac=1, random_state=42).reset_index(drop=True)

    train_path = os.path.join(output_dir, "train_data.csv")
    val_path = os.path.join(output_dir, "val_data.csv")
    
    df_train_final.to_csv(train_path, index=False)
    df_val_final.to_csv(val_path, index=False)

if __name__ == "__main__":
    INPUT_CSV = "/home/s2516118/skripsi/ImageCHD_info/imageCHD_dataset_info.csv"
    OUTPUT_DIR = "/home/s2516118/skripsi/ImageCHD_info"
    prepare_final_csv(INPUT_CSV, OUTPUT_DIR)