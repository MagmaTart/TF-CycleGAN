mkdir dataset
wget -N https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/apple2orange.zip -O apple2orange.zip
mv apple2orange.zip dataset
cd dataset
unzip apple2orange.zip
rm apple2orange.zip
cd ..

wget -N https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip -O horse2zebra.zip
mv horse2zebra.zip dataset
cd dataset
unzip horse2zebra.zip
rm horse2zebra.zip
cd ..
