#!/bin/bash

dir1=$(mktemp -d)
dir2=$(mktemp -d)

./harris_hessian_freak -x -b "$dir1"
./harris_hessian_freak -r -x -b "$dir2"

for i in $(ls "$dir1")
do
    diff "$dir1/$i" "$dir2/$i"
done

### rm -rv "$dir1"
### rm -rv "$dir2"
