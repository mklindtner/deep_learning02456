import os
import random
import shutil

# Path to your Speech folder
speech_folder = 'Speech'

# Path to the Speech_test folder (to be created)
test_folder = 'Speech_test'

# Create Speech_test folder if it doesn't exist
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

# List all files in the Speech folder
all_files = [f for f in os.listdir(speech_folder) if os.path.isfile(os.path.join(speech_folder, f))]

# Calculate 30% of the files
num_files_to_select = int(len(all_files) * 0.3)

# Randomly select files
selected_files = random.sample(all_files, num_files_to_select)

# Move the selected files to Speech_test folder
for file in selected_files:
    shutil.move(os.path.join(speech_folder, file), test_folder)

print(f'Moved {num_files_to_select} files to {test_folder}')
