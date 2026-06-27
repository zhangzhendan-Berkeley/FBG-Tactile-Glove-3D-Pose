import pandas as pd
import os

# 输入文件
INPUT_FILE = "glove_pose_for_senior_data_py.txt"

# 输出目录
OUTPUT_DIR = "data"

# 切分比例
TRAIN_RATIO = 1
VAL_RATIO = 0.
TEST_RATIO = 0.


def main():

    df = pd.read_csv(INPUT_FILE)

    n = len(df)

    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    val_path = os.path.join(OUTPUT_DIR, "val.csv")
    test_path = os.path.join(OUTPUT_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("数据切分完成")
    print(f"总帧数: {n}")
    print(f"train: {len(train_df)}")
    print(f"val:   {len(val_df)}")
    print(f"test:  {len(test_df)}")
    print()
    print(f"保存路径:")
    print(train_path)
    print(val_path)
    print(test_path)


if __name__ == "__main__":
    main()