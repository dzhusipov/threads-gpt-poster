import os
import requests
from openai import OpenAI
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
from dotenv import load_dotenv

import time as time_module  # Rename to avoid conflict

import datetime as dt  # Import the datetime module as dt
import logging
from PIL import Image
from io import BytesIO
import asyncio
import random

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
    handlers=[
        logging.StreamHandler()  # Output logs to the console
    ]
)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
THREADS_USER_ID = os.getenv('THREADS_USER_ID')
ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')

# Create a FastAPI app
app = FastAPI()

# Define the directory where images will be saved
IMAGE_FOLDER = 'downloaded_images'
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

def generate_random_prompt():
    """
    Generates a random image prompt by combining random elements.
    Then sends it to ChatGPT to get a more enriched prompt.
    """
    adjectives = ['beautiful', 'serene', 'mystical', 'vibrant', 'tranquil', 'majestic', 'ethereal', 'enigmatic']
    subjects = ['forest', 'mountain', 'ocean', 'desert', 'waterfall', 'sky', 'galaxy', 'island']
    styles = ['digital painting', 'photorealistic', 'abstract', 'impressionist', 'surreal', 'minimalist', 'fantasy']
    times_of_day = ['at sunrise', 'at sunset', 'under the stars', 'during a storm', 'on a foggy morning', 'in autumn']
    
    adjective = random.choice(adjectives)
    subject = random.choice(subjects)
    style = random.choice(styles)
    time_of_day = random.choice(times_of_day)
    
    initial_prompt = f"A {adjective} {subject} {time_of_day}, {style}"
    logging.info('Generated initial prompt: %s', initial_prompt)
    
    # Send the initial prompt to ChatGPT to get an enriched prompt
    enriched_prompt = get_enriched_prompt(initial_prompt)
    if enriched_prompt:
        logging.info('Enriched prompt: %s', enriched_prompt)
        return enriched_prompt
    else:
        # If enrichment fails, use the initial prompt
        logging.warning('Using initial prompt as enrichment failed.')
        return initial_prompt

def get_enriched_prompt(prompt):
    """
    Sends the prompt to ChatGPT to get a more enriched and detailed prompt.
    """
    
    client = OpenAI()
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a creative assistant that enriches image prompts for art generation. Make the following prompt more detailed and vivid, suitable for generating images with DALL·E."},
                {"role": "user", "content": f"Please enrich this prompt for an image: '{prompt}'"},
            ],
            temperature=0.7,
            max_tokens=60,
        )
        
        enriched_prompt = response['choices'][0]['message']['content'].strip()
        return enriched_prompt
    except Exception as e:
        logging.error('An error occurred while enriching the prompt: %s', e)
        return None


def generate_image(prompt):
    logging.info('Starting image generation with prompt: %s', prompt)
    
    client = OpenAI()

    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1792",
            quality="standard",
            n=1,
        )

        image_url = response.data[0].url
        logging.info('Image generated successfully. URL: %s', image_url)

        # Process the image
        return process_image_from_url(image_url)

    except Exception as e:
        logging.error('An error occurred while generating the image: %s', e)
        return None

def get_image_from_url(image_url):
    logging.info('Downloading image from URL: %s', image_url)
    try:
        # Process the image
        return process_image_from_url(image_url)
    except Exception as e:
        logging.error('An error occurred while getting the image: %s', e)
        return None

def process_image_from_url(image_url):
    try:
        # Download the image
        response = requests.get(image_url)
        if response.status_code == 200:
            image_data = response.content
            logging.info('Image downloaded successfully.')
        else:
            logging.error('Failed to download image. Status code: %d', response.status_code)
            return None

        # Open the image
        image = Image.open(BytesIO(image_data))
        image_format = image.format

        # Convert the image format to PNG or JPG if necessary
        if image_format not in ['PNG', 'JPEG', 'JPG']:
            logging.info('Converting image to PNG format.')
            image = image.convert('RGB')
            image_format = 'PNG'

        # Save the image locally in the IMAGE_FOLDER
        filename = f"image_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.{image_format.lower()}"
        filepath = os.path.join(IMAGE_FOLDER, filename)
        image.save(filepath, format=image_format)
        logging.info('Image saved locally as: %s', filepath)

        # Construct the public URL for the image
        public_image_url = f"https://threads.dasm.asia/images/{filename}"
        return public_image_url

    except Exception as e:
        logging.error('An error occurred while processing the image: %s', e)
        return None

def post_to_threads(image_url, text):
    logging.info('Starting post to Threads with image URL: %s', image_url)
    
    # Step 1: Create the media container
    threads_api_url = f"https://graph.threads.net/v1.0/{THREADS_USER_ID}/threads"
    params = {
        'media_type': 'IMAGE',
        'image_url': image_url,
        # 'text': text,
        'access_token': ACCESS_TOKEN
    }
    try:
        response = requests.post(threads_api_url, params=params)
        if response.status_code == 200:
            response_data = response.json()
            logging.info('Media container created successfully. Response: %s', response_data)
            media_container_id = response_data.get('id')
            if media_container_id:
                # Step 2: Publish the thread using the media container ID
                publish_url = f"https://graph.threads.net/v1.0/{THREADS_USER_ID}/threads_publish"
                publish_params = {
                    'creation_id': media_container_id,
                    'access_token': ACCESS_TOKEN
                }
                publish_response = requests.post(publish_url, params=publish_params)
                if publish_response.status_code == 200:
                    logging.info('Post published successfully. Response: %s', publish_response.text)
                else:
                    logging.error('Failed to publish post. Status code: %d, Response: %s', publish_response.status_code, publish_response.text)
            else:
                logging.error('Media container ID not found in response')
        else:
            logging.error('Failed to create media container. Status code: %d, Response: %s', response.status_code, response.text)
    except Exception as e:
        logging.error('Exception during posting: %s', e)

@app.get("/images/{filename}")
async def get_image(filename: str):
    filepath = os.path.join(IMAGE_FOLDER, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath)
    else:
        return Response(content="Image not found", status_code=404)

# Modify the job function if necessary
def job(use_image_url=None):
    logging.info('Job started')
    if use_image_url:
        # Use image from the provided URL
        image_url = get_image_from_url(use_image_url)
        post_text = f"Image from URL: {use_image_url}"
    else:
        # Generate a new prompt and image
        prompt = generate_random_prompt()
        image_url = generate_image(prompt)
        post_text = prompt

    if image_url:
        post_to_threads(image_url, post_text)
    else:
        logging.error('Image URL is None, skipping post')
    logging.info('Job finished')

@app.get("/images/{filename}")
async def get_image(filename: str):
    filepath = os.path.join(IMAGE_FOLDER, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath)
    else:
        return Response(content="Image not found", status_code=404)

@app.get("/job")
async def run_job_endpoint():
    # Run the job in an executor to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, job)
    
    return {"message": "Job executed successfully"}

# Use the startup event to run code after the server starts
@app.on_event("startup")
async def startup_event():
    # Run the scheduling loop in the background
    asyncio.create_task(run_job_scheduler())

async def run_job_scheduler():
    # Define the times when the job should run (24-hour format)
    run_times = [dt.time(8, 0), dt.time(14, 0), dt.time(20, 0)]  # 8:00 AM, 2:00 PM, 8:00 PM

    while True:
        now = dt.datetime.now()
        # Find the next scheduled run time
        next_run_time = None
        for run_time in run_times:
            scheduled_time = dt.datetime.combine(now.date(), run_time)
            if scheduled_time > now:
                next_run_time = scheduled_time
                break
        if not next_run_time:
            # All run times have passed today, schedule for the first time tomorrow
            next_run_time = dt.datetime.combine(now.date() + dt.timedelta(days=1), run_times[0])

        # Calculate the time to sleep until the next run time
        sleep_seconds = (next_run_time - now).total_seconds()
        logging.info(f"Next job scheduled at {next_run_time.strftime('%Y-%m-%d %H:%M:%S')}, sleeping for {sleep_seconds} seconds.")

        # Sleep until the next scheduled time
        await asyncio.sleep(sleep_seconds)

        # Run the job
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, job)

if __name__ == "__main__":
    # Start the FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=19998)