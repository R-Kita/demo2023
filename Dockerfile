FROM python:3.9-buster

WORKDIR /src

# COPY requirements.txt ./
# RUN pip install -r requirements.txt

RUN pip install fastapi uvicorn[standard]
RUN pip install python-multipart
RUN pip install numpy Pillow torch torchvision matplotlib 

ENTRYPOINT ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--reload"]
