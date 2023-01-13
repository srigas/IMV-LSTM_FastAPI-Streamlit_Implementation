# Retrieve python 3.10 base image
FROM python:3.10

# Create a new directory
RUN mkdir /frontend

# Copy everything into it
COPY . /frontend

# And make it the base directory
WORKDIR /frontend

# Install all required libraries
RUN pip install -r requirements.txt

# Expose the port where the client will run
EXPOSE 8501

# Run streamlit
CMD ["streamlit", "run", "client.py"]