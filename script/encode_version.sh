#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Program takes base folder as param"
    exit
fi

cd "$1" 

echo "static char git_commit_info[] = {"
git show -s "--format=commit %H %ci" | \
    hexdump -v -e '" " 16/1 "  0x%02x, " "\n"' | \
    sed -e '$s/0x  ,//g'
echo "   '\0'"
echo "};"
