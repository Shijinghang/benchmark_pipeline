import os
import pandas as pd
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# 获取result文件夹下的所有.tsv文件
result_folder = './result'
tsv_files = [f for f in os.listdir(result_folder) if f.endswith('.tsv')]

# 创建一个空的DataFrame来存储合并的结果
combined_df = pd.DataFrame()

# 处理每个文件
for file in tsv_files:
    file_path = os.path.join(result_folder, file)
    df = pd.read_csv(file_path, sep='\t')
    
    # 获取文件名（不含扩展名）作为前缀
    file_prefix = os.path.splitext(file)[0]
    
    # 为score和prediction字段添加前缀
    df.rename(columns={'score': f'{file_prefix}_score', 'prediction': f'{file_prefix}_prediction'}, inplace=True)
    
    # 如果combined_df为空，直接赋值，否则按index列合并
    if combined_df.empty:
        combined_df = df
    else:
        combined_df = pd.merge(combined_df, df, on=['index', 'image_path','question', 'A', 'B', 'C', 'D', 'answer', 'explian'], how='outer')

# 添加sum_score列，计算所有带有_score后缀列的总和
score_columns = [col for col in combined_df.columns if col.endswith('_score')]
combined_df['sum_score'] = combined_df[score_columns].sum(axis=1)
# 保存合并后的结果到一个新的.tsv文件
output_file_path = './llms_combined_results.tsv'
combined_df.to_csv(output_file_path, sep='\t', index=False)
# 筛选总分数大于3的所有行
filtered_df = combined_df[combined_df['sum_score'] >= 3]

# 仅保留指定字段
filtered_df = filtered_df[['index', 'image_path',  'question', 'A', 'B', 'C', 'D', 'answer', 'explian', 'sum_score']]

# 重新编号index字段从1开始
filtered_df['index'] = range(1, len(filtered_df) + 1)

# 保存筛选后的结果到2024_p2_mcq_plus.tsv
filtered_output_file_path = './2024_p2_mcq_plus.tsv'
filtered_df.to_csv(filtered_output_file_path, sep='\t', index=False)