#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Program requires one param: folder"
    exit
fi

names_file=$(mktemp)

echo "//This file is generated and should not be manually edited"

for i in $(find "$1" -type f); do
    cname=$(basename $i)
    cname=${cname//-/_}
    cname=${cname//./_}

    echo "\"$i\",$cname" >> "$names_file"

    echo "static unsigned char $cname[] = {"
    hexdump -v -e '" " 16/1 "  0x%02x, " "\n"' $i | \
       sed -e '$s/0x  ,//g'
    echo "};"
done

echo ""
echo "const char* kernel_files[] = {"

cat "$names_file" 

echo "NULL, NULL"
echo "};"

rm "$names_file"