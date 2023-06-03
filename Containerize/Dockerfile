FROM python:3.8

WORKDIR /app

COPY Segmentation /app/Segmentation
COPY static /app/static
COPY templates /app/templates
COPY No_Feature.h5 /app/No_Feature.h5
COPY Increased_depth_Feature1.h5 /app/Increased_depth_Feature1.h5
COPY newapp.py /app/newapp.py


ADD newapp.py .

RUN pip install opencv-python numpy pandas 

EXPOSE 5000

CMD [ "python" , "./newapp.py" ]
