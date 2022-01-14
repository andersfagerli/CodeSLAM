#!/bin/bash

TRAIN_FOLDER="$1"
FOLDER_NUMBER=$(echo $TRAIN_FOLDER | tr "_" "\n"|sed -n '2p')

cd $TRAIN_FOLDER/train

mkdir depth/ instance/ photo/

WD=$(pwd)
COUNTER=0

for d in $FOLDER_NUMBER/*
do
  cd $WD/$d/depth

  TEMP=$COUNTER
  for file in *.png
  do
    mv "$file" "temp$COUNTER.png"
    mv "temp$COUNTER.png" ../../../depth 
    COUNTER=$[$COUNTER+1]
  done
  cd ../../../
  
  cd $WD/$d/photo
  COUNTER=$TEMP
  for file in *.jpg
  do
    mv "$file" "$COUNTER.png"
    mv $COUNTER.png ../../../photo
    COUNTER=$[$COUNTER+1]
  done
  cd ../../../

  echo "Images moved: $COUNTER"
done

rm -rf $FOLDER_NUMBER/

echo "Renaming images ..."

cd depth

for file in *.png; do mv "$file" "${file/temp/}"; done

cd ../../../../