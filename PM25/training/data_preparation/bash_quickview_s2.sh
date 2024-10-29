#!/bin/bash

# Define the input and output directories
input_dir="./S2_London_2018"
output_dir="./S2_London_2018_thumbview"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Loop through each .tif file in the input directory
for file in "$input_dir"/*.tif; do
    # Extract the base filename without extension
    base_name=$(basename "$file" .tif)

    # Define the output filename in the output directory
    output_file="$output_dir/${base_name}_RGB_Reduced.tif"

    # Run gdal_translate with specified bands and reduced resolution
    gdal_translate -b 4 -b 3 -b 2 -of GTiff -outsize 10% 10% "$file" "$output_file"

    echo "Processed $file -> $output_file"
done
