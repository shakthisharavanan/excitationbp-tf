mkdir -p checkpoints
cd checkpoints

FILE=vgg_16.ckpt
TARBALL=http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz

if [ ! -f $FILE ]; then
	echo "Downloading $FILE tarball."
	wget $TARBALL
	tar xvzf vgg_16_2016_08_28.tar.gz
	rm vgg_16_2016_08_28.tar.gz
	echo "$FILE downloaded."
else
	echo "$FILE already exists"
fi

cd ..