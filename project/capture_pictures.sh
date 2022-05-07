#!/bin/bash
set -e
n=1
while [ $n -le 100 ];
do
    nvgstcapture --image-res=6 --automate --file-name jforecast --capture-auto &
    wait $!;
    (( n++ ));
    sleep 10;
done
