#!/bin/bash
set -e
cd `dirname $0`

mkdir frames

for date in {17,18,19,24,25}; do
    # echo $date
    for room in {livingroom,openoffice}; do
        dateroom=$date-$room
        command="squashfs frames_squashed/2023-01-$dateroom frames/2023-01-$dateroom"
        echo "Running '$command'"
        $command
    done
done
