#!/usr/bin/env bash
set -euxo pipefail

# this script only downloads, extracts, renames and preprocesses zuco 1
# if you wish to also download, extract, rename and preprocess zuco 2 uncomment everything below
mkdir -p data/zuco
#
# mkdir -p data/zuco2  # optional to also get zuco 2

pip install osfclient

for zuco_data in q3zws # 2urht  # optional to also get zuco2
# do
    osf -p $zuco_data clone
    # if [ "$zuco_data" == "2urht" ];   # optional to also get zuco2then
        for task in task1-\ SR task2\ -\ NR task3\ -\ TSR
        do
            mv $zuco_data"/osfstorage/$task" "data/zuco/"${task:0:5}
            mv "data/zuco/${task:0:5}/Matlab files" "data/zuco/${task:0:5}/Matlab_files"
        done
    # else  # optional to also get zuco2
        # for task in task1\ -\ NR task2\ -\ TSR  # optional to also get zuco2
        # do  # optional to also get zuco2
        #     mv $zuco_data"/osfstorage/$task" "data/zuco2/"${task:0:5}  # optional to also get zuco2
        #     mv "data/zuco2/${task:0:5}/Matlab files" "data/zuco2/${task:0:5}/Matlab_files"  # optional to also get zuco2
        # done  # optional to also get zuco2
    # fi  # optional to also get zuco2
done

python3 -m scripts.zuco_create_wordinfor_scanpath_files --zuco-task zuco11
python3 -m scripts.zuco_create_wordinfor_scanpath_files --zuco-task zuco12
# python3 -m scripts.zuco_create_wordinfor_scanpath_files --zuco-task zuco21  # optional to also get zuco2
