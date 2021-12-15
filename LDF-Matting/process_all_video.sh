#!/usr/bin/bash
for file in $(ls ../input_video/)
do
    echo $file
    python LDFbased-merger.py ../input_video/$file background.mp4 $file-output.mp4
done
