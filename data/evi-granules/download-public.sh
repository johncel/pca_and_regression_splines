#!/bin/bash

FILES=$1

if [ -z "$FILES" ]
then
    echo "usage $0 <file list>"
    exit 1
fi

while read file
do 
    FILENAME=`echo "$file" |  awk -F/ '{print $NF}'`
    XML=$file.xml
    XML_FILENAME=$FILENAME.xml

    curl $file > $FILENAME
    curl $XML > $XML_FILENAME

done < $FILES
