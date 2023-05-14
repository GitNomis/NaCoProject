FROM cupy/cupy

WORKDIR /workspaces/NaCoProject
COPY . ./

# install python packages & git
RUN apt-get update && apt-get install -y pip git
RUN pip install --no-cache -r requirements.txt
