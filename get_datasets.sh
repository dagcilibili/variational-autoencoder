# Get CIFAR10
if [ ! `uname` = "Darwin" ]; then
  export WGET='wget --no-check-certificate'
else
  export WGET='curl -C - --retry 2 -O'
fi
$WGET http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzvf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz 
