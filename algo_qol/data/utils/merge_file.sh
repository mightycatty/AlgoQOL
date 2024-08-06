# Iterate over 20 folders ending with "20210409_i"
for i in {0..43}; do
  # Construct the source and destination paths
  source_path="/logs/heshuai03/datasets/object356val/images/patch$i"
  dest_path="/logs/heshuai03/datasets/object356val/images/merge"

  # Execute the juicefs sync command with 16 threads
  sudo juicefs sync "$source_path" "$dest_path" --threads 16
done