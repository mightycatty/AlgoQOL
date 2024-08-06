img2dataset --url_list /mnt/juicefs/cv_tikv/heshuai/datasets/laion400m/the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta --input_format "parquet"\
         --url_col "URL" \
           --output_folder /mnt/juicefs/cv_tikv/heshuai/datasets/laion400m/data --processes_count 1 --thread_count 32 --image_size 512 \
            --enable_wandb False