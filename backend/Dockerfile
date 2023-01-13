# Retrieve the base image
FROM tiangolo/uvicorn-gunicorn:python3.10

# Create a new directory
RUN mkdir /backend

# Copy everything into it
COPY . /backend

# And make it the base directory
WORKDIR /backend

# Install all required libraries
RUN pip install -r requirements.txt

# Expose the port where the client will run
EXPOSE 8000

# Run the server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
