ARG UBUNTU_VERSION=20.04

#FROM tensorflow/tensorflow as base
FROM ubuntu:${UBUNTU_VERSION} as base
ENV DEBIAN_FRONTEND noninteractive
ENV LANG C.UTF-8
RUN apt update && apt upgrade -y
RUN apt install software-properties-common -y
#RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install python3-pip -y
RUN apt-get install -qqy x11-apps
ENV DISPLAY :0
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 500
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 500
RUN update-alternatives --set python /usr/bin/python3.8
RUN update-alternatives --set python3 /usr/bin/python3.8

#FROM base as tf
#RUN ln -s $(which python3) /usr/local/bin/python
# Options:
#   tensorflow
#   tensorflow-gpu
#   tf-nightly
#   tf-nightly-gpu
# Set --build-arg TF_PACKAGE_VERSION=1.11.0rc0 to install a specific version.
# Installs the latest version by default.
#ARG TF_PACKAGE=tensorflow
#ARG TF_PACKAGE_VERSION=
#RUN python3 -m pip install --no-cache-dir tensorflow #${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}

FROM base AS r
RUN apt update && apt upgrade -y
RUN apt install -y r-base r-base-core r-recommended r-base-dev r-cran-tidyverse
RUN Rscript -e 'install.packages("baseline", repos="https://cloud.r-project.org")'

FROM r AS latex
RUN apt update && apt upgrade -y
RUN apt install fontconfig -y
RUN mkdir -p /usr/share/fonts/truetype/myfonts/
COPY Helvetica.ttf /usr/share/fonts/truetype/myfonts/
RUN fc-cache -f -v /usr/share/fonts/truetype/myfonts/
RUN apt install texlive-latex-recommended -y

FROM latex AS dev
RUN apt update && apt upgrade -y
RUN apt install git vim zsh curl fzf fonts-hack-ttf -y
RUN chsh -s $(which zsh)
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
RUN git clone --depth=1 https://github.com/romkatv/powerlevel10k.git ~/powerlevel10k
RUN echo 'source ~/powerlevel10k/powerlevel10k.zsh-theme' >>~/.zshrc
COPY p10k.zsh /root/.p10k.zsh
COPY zshrc /root/.zshrc
# RUN update-alternatives --set editor /usr/bin/vim.basic 

FROM dev AS vim
#RUN add-apt-repository ppa:neovim-ppa/unstable -y
RUN apt update -y
RUN apt install -y neovim git ripgrep fd-find nodejs npm
RUN git clone https://github.com/NvChad/NvChad ~/.config/nvim
#RUN nvim -c "autocmd User PackerComplete" -c "PackerSync"
#RUN git clone --depth 1 https://github.com/NTBBloodbath/doom-nvim.git ${XDG_CONFIG_HOME:-$HOME/.config}/nvim

FROM vim AS figarodev
RUN mkdir dsProject/
RUN cd dsProject/
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install rpy2==3.4.3
RUN apt install build-essential swig python3-dev -y
WORKDIR /dsProject/
RUN mkdir -p 01_Preprocessing/notebooks/ 01_Preprocessing/scripts/python/ 01_Preprocessing/scripts/R/\
    02_Modelling/notebooks/ 02_Modelling/scripts/python/ 02_Modelling/scripts/R/\
    03_Models/ 04_Reports/documents/ 04_Reports/figures/ data/raw/ data/interim/ data/processed/ tools/
COPY tools/ tools/
RUN chmod +x -R *
RUN pip install -e tools/

#FROM figarodev AS torch
#RUN pip install --no-cache-dir torch==1.10.1+cpu torchvision==0.11.2+cpu torchaudio==0.10.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
#RUN pip install --no-cache-dir pytorch-lightning 
EXPOSE 8888
ENTRYPOINT ["/bin/zsh"]
