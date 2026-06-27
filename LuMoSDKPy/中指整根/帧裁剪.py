import pandas as pd

input_csv = "clean_glove_one_row_per_frame.csv"
output_csv = "clean_glove_one_row_per_frame_cut.csv"

start_frame = 6500
end_frame = 15250

df = pd.read_csv(input_csv)

df_cut = df[(df["frame_idx"] >= start_frame) & (df["frame_idx"] <= end_frame)].copy()
df_cut.reset_index(drop=True, inplace=True)

# 重新编号
df_cut["frame_idx"] = range(len(df_cut))

df_cut.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"截取后帧数: {len(df_cut)}")
print(f"已保存到: {output_csv}")
print(f"新的 frame_idx 范围: {df_cut['frame_idx'].min()} ~ {df_cut['frame_idx'].max()}")