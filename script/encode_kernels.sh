#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Program requires one param: folder"
    exit
fi

names_file=$(mktemp)
sizes_file=$(mktemp)

echo "//This file is generated and should not be manually edited"

for i in $(find "$1" -name "*.cl" -type f); do
    filename=$(basename $i)
    cname=${filename//-/_}
    cname=${cname//./_}

    echo "\"$filename\",$cname," >> "$names_file"
    echo "sizeof( $cname )," >> "$sizes_file"

    echo "static unsigned char $cname[] = {"
    hexdump -v -e '" " 16/1 "  0x%02x, " "\n"' $i | \
       sed -e '$s/0x  ,//g'
    echo "};"
done

echo ""
echo "const void* kernel_files[] = {"

cat "$names_file" 

echo "NULL, NULL"
echo "};"

echo ""

echo "const size_t kernel_sizes[] = {"

cat "$sizes_file"

echo "};"


rm "$names_file"
