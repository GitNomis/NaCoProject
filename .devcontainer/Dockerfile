FROM cupy/cupy

WORKDIR /workspaces/NaCoProject
COPY . ./

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Amsterdam

# install python packages & git
RUN apt-get update && apt-get install -y pip git
RUN pip install --no-cache -r requirements.txt
# install package to display stuff
RUN apt-get install -y python3-tk