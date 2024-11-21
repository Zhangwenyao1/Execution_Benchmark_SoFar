# 定义 open6dor_asset 目录路径
open6dor_asset_dir="/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/objaverse_rescale"

# 遍历每个子文件夹
for folder in "$open6dor_asset_dir"/*; do
    if [ -d "$folder" ]; then  # 如果是文件夹
        echo "Processing folder: $folder"
        obj2mjcf --obj-dir "$folder" --save-mjcf  --overwrite
    fi
done

open6dor_asset_dir="/data/workspace/LIBERO/Open6Dor/Open6dorAsset/objects/ycb_16k_backup"

# 遍历每个子文件夹
for folder in "$open6dor_asset_dir"/*; do
    if [ -d "$folder" ]; then  # 如果是文件夹
        echo "Processing folder: $folder"
        obj2mjcf --obj-dir "$folder" --save-mjcf  --overwrite
    fi
done