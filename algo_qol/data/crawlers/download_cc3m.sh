img2dataset --url_list /mnt/juicefs/cv_tikv/heshuai/datasets/cc3m/Validation_GCC-1.1.0-Validation.tsv --input_format "tsv"\
         --url_col "url"  \
           --output_folder /mnt/juicefs/cv_tikv/heshuai/datasets/cc3m/val --processes_count 16 --thread_count 64 --image_size 512 \
             --enable_wandb False