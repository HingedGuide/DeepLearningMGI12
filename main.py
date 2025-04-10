import zipfile
import os
import shutil

# Extract zip
with zipfile.ZipFile('UCMerced_LandUse.zip', 'r') as zip_ref:
    zip_ref.extractall('UCMImages')

# Move 'Images' directory to current working dir
shutil.move('UCMImages/UCMerced_LandUse/Images', './Images')

# Clean up
shutil.rmtree('UCMImages')
if os.path.exists('README.md'):
    os.remove('README.md')
os.remove('UCMerced_LandUse.zip')

# List files in current directory (optional)
print("Files in current directory:")
print(os.listdir('.'))

# Paths for further use
UCM_images_path = "Images/"
Multilabels_path = "LandUse_Multilabeled.txt"