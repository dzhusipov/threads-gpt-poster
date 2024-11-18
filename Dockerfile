# Use the official Python image based on Alpine Linux
FROM python:3.9-alpine

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install build dependencies
RUN apk update && \
    apk add --no-cache gcc musl-dev libffi-dev make jpeg-dev zlib-dev

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port on which the app will run
EXPOSE 19998

# Set environment variables (optional, can also be set during runtime)
ENV OPENAI_API_KEY=
ENV THREADS_USER_ID=
ENV ACCESS_TOKEN=
ENV PUBLIC_IMAGE_HOST=

EXPOSE 19998

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "19998"]