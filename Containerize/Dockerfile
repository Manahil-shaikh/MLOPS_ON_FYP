FROM python:3.10.11

WORKDIR /app

COPY Segmentation /app/Segmentation
COPY static /app/static
COPY templates /app/templates
COPY No_Feature.h5 /app/No_Feature.h5
COPY Increased_depth_Feature1.h5 /app/Increased_depth_Feature1.h5
COPY newapp.py /app/newapp.py


ADD newapp.py .

RUN pip install opencv-contrib-python numpy pandas tensorflow matplotlib Flask
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*


EXPOSE 5000

CMD [ "python" , "./newapp.py" ]

