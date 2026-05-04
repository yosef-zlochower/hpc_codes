#!/bin/bash

#this scripts takes the png files produced by gnuplot and turns them
# into an mp4 file

ffmpeg  -f image2  -i U_%07d.png -vcodec libx264 -crf 22  -pix_fmt yuv420p test.mp4
