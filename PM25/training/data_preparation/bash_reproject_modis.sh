#!/bin/bash

# Directory containing NetCDF files
input_dir="./modis_pm25_data"
output_dir="./modis_pm25_data_crop"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Iterate over all .nc files in the input directory
for file in "$input_dir"/*.nc; do
  # Get the filename without the path and extension
  filename=$(basename "$file" .nc)

  # Define the output filename
  output_file="$output_dir/${filename}_cropped_projected.nc"

  gdalwarp -t_srs EPSG:4326 -te -0.5861507 51.2401733 0.3231240 51.7291263 \
           -r bilinear "$file" "$output_file"
  echo "Processed $file -> $output_file"
done
