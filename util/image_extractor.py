import tarfile
import os
import sys

if (len(sys.argv) < 4):
    print("Usage: image_extractor.py archive_path output_dir image_count")

archive_path = sys.argv[1]
output_dir = sys.argv[2]
max_images = sys.argv[3]

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"Starting extraction {archive_path} to {output_dir} with {max_images} image count.")

with tarfile.open(archive_path, "r:bz2") as tar:
    count = 0
    for member in tar:
        if member.isfile() and member.name.lower().endswith(('.jpg', '.jpeg', '.png')):
            member.name = os.path.basename(member.name) 
            tar.extract(member, output_dir)
            count += 1
            
            if count % 500 == 0:
                print(f"{count} images extracted.")
            
            if count >= max_images:
                break

print(f"{count} images extracted to {output_dir}.")