#!/bin/bash
set -e
nvgstcapture --image-res=6 --automate --file-name jforecast --capture-auto &
wait $!
