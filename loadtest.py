from pydub import AudioSegment
import os
import shutil

# Source directory with WAV files
source_dir = '/work3/s164396/data/DNS-Challenge-4/datasets_fullband/clean_fullband/vctk_wav48_silence_trimmed/p225/'

# Destination directory for short WAV files
destination_dir = 'Audio/Speech/'

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)
duration_to_extract = 6000
# Define the threshold duration (in milliseconds)
threshold_duration = 6000  # 5 seconds
i=0
# Iterate through the files in the source directory
for filename in os.listdir(source_dir):
    if filename.endswith('.wav'):
        # Load the WAV file using pydub
        audio = AudioSegment.from_file(os.path.join(source_dir, filename))

        # Check if the duration is less than the threshold
        if len(audio) > threshold_duration:
            # Move the file to the destination directory
            extracted_audio = audio[:duration_to_extract]
            extracted_audio.export(os.path.join(destination_dir, filename), format="wav")
            i+=1
        else: print("TOO small")
print("Task completed.", i)

# Source directory with WAV files
source_dir = '/work3/s164396/data/DNS-Challenge-4/datasets_fullband/noise_fullband/'

# Destination directory for short WAV files
destination_dir = 'Audio/Noise/'

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)
# Define the duration to extract (in milliseconds)
duration_to_extract = 6000 # 5 seconds

# Limit the number of files to process
files_to_process = 28
processed_files = 0

# Iterate through the files in the source directory
for filename in os.listdir(source_dir):
    if filename.endswith('.wav'):
        # Load the WAV file using pydub
        audio = AudioSegment.from_file(os.path.join(source_dir, filename))
        if len(audio) > threshold_duration:
            # Extract the first 5 seconds of audio
            extracted_audio = audio[:duration_to_extract]

            # Save the extracted segment to the destination directory
            extracted_audio.export(os.path.join(destination_dir, filename), format="wav")

            print(f"Extracted {filename} to the destination directory.")
            processed_files += 1

            if processed_files >= files_to_process:
                break  # Exit the loop after processing the desired number of files
       

print(f"Task completed. Processed {processed_files} files.")
