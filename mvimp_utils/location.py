"""Location for all available function."""
import os

LOC = os.getcwd()
if LOC.split("/")[-1] != "MVIMP":
    raise ValueError("Please change directory to the root of MVIMP.")

ANIMEGAN_PREFIX = os.path.join(LOC, "third_party/AnimeGAN")
DAIN_PREFIX = os.path.join(LOC, "third_party/DAIN")
Photo_3D = os.path.join(LOC, "third_party/Photo3D")

input_data_dir = os.path.join(LOC, "Data/Input")
output_data_dir = os.path.join(LOC, "Data/Output")
input_videos = os.path.join(LOC, "Data/Input_Videos")
output_videos = os.path.join(LOC, "Data/OutputVideos")
if not os.path.exists(input_data_dir):
    os.makedirs(input_data_dir)
if not os.path.exists(output_data_dir):
    os.makedirs(output_data_dir)
if not os.path.exists(input_videos):
    os.makedirs(input_videos)
if not os.path.exists(output_videos):
    os.makedirs(output_videos)
