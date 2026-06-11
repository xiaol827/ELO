Here are instructions for quick remote setup


```
apt-get update
apt install tmux vim rsync htop -y
tmux

mkdir l2o_install
cd l2o_install

wget https://repo.anaconda.com/miniconda/Miniconda3-py311_24.7.1-0-Linux-x86_64.sh
bash Miniconda3-py311_24.7.1-0-Linux-x86_64.sh -b -p $PWD/miniconda3
source $PWD/miniconda3/bin/activate
```

