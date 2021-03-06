"""
Input: A single video file;
Output: A single inserted video file.
"""

from mvimp_utils.location import *
import torch
from mvimp_utils.ffmpeg_helper import video_fusion, video_extract, frames_info, fps_info
from mvimp_utils.file_op_helper import file_order, clean_folder
import argparse
import numpy as np
import cv2
import shutil

torch.backends.cudnn.benchmark = True


def config():
    parser = argparse.ArgumentParser(description="Inference DAIN.")
    parser.add_argument(
        "--input_video", "-input", type=str, help="indicate the input files",
    )
    parser.add_argument(
        "--time_step", "-ts", type=float, default=0.5, help="choose the time steps"
    )
    parser.add_argument(
        "--high_resolution",
        "-hr",
        default=False,
        type=bool,
        help="split the frames when handling 720+ videos",
    )
    parser.add_argument(
        "--input_video_fps", "-ifps", type=float, help="Input Video FPS for Missing Frames only Anime MODE"
    )
    parser.add_argument(
        "--anime_fill_frames",
        "-aff",
        default=False,
        type=bool,
        help="Fill in only the missing frames",
    )
    return parser.parse_args()


if __name__ == "__main__":
    os.chdir(DAIN_PREFIX)
    print(f"Current PyTorch version is {torch.__version__}")
    args = config()
    
    if not args.anime_fill_frames:
        assert args.time_step in [
            0.5,
            0.25,
            0.125,
        ], "Time step must be one of 0.5/0.25/0.125."

        # STAGE 1: video pre-processing
        # if len(os.listdir(input_data_dir)) > 1:
        #     raise FileExistsError("You can only process one video at a time..")
        video_file_link = os.path.join(input_data_dir, args.input_video)
        frame_num = frames_info(video_file_link)
        if frame_num < 2:
            raise FileNotFoundError(
                "Video File not found, or incompatible."
            )
        fps = fps_info(video_file_link)
        target_fps = float(fps) / args.time_step
        video_extract(src=video_file_link, dst=input_data_dir, thread=4)
        os.remove(video_file_link)
        print(
            f"\n--------------------SUMMARY--------------------\n"
            f"Current input video file is {args.input_video},\n"
            f"{args.input_video}'s fps is {fps},\n"
            f"{args.input_video} has {frame_num} frames.\n"
            f"Now we will process this video to {target_fps} fps.\n"
            f"--------------------NOW END--------------------\n\n"
        )
    else:
        print(
            f"\n--------------------SUMMARY--------------------\n"
            f"Interpolating Anime missing frames only!,\n"
            f"--------------------NOW END--------------------\n\n"
        )

    # STAGE 2: Inference
    os.system(
        f"python3 -W ignore vfi_helper.py "
        f"--src {input_data_dir} "
        f"--dst {output_data_dir} "
        f"--high_resolution {args.high_resolution} "
        f"--anime_fill_frames {args.anime_fill_frames} "
        f"--time_step {args.time_step} "
    )

    # STAGE 3: video post-processing
    os.chdir(DAIN_PREFIX)
    clean_folder(input_data_dir)
    file_order(src=output_data_dir, dst=input_data_dir)
    
    if args.anime_fill_frames:
        output_video_file = "output_anime.mp4"
        video_fusion(
            src=input_data_dir + "/%10d.png",
            dst=os.path.join(output_data_dir, output_video_file),
            fps=args.input_video_fps,
            thread=4,
        )
    else:
        output_video_file = f"{args.input_video.split('.')[0]}-{target_fps}.{args.input_video.split('.')[1]}"
        video_fusion(
            src=input_data_dir + "/%10d.png",
            dst=os.path.join(output_data_dir, output_video_file),
            fps=target_fps,
            thread=4,
        )
    
    shutil.move(os.path.join(output_data_dir, output_video_file), "/content/gdrive/My Drive/DAIN/Output/" + output_video_file)
    clean_folder(input_data_dir)
