import os
import re
import time
import csv
from selenium import webdriver
from bs4 import BeautifulSoup

# URLs for different genres
GENRE_URLS = [
    "https://zaycev.net/genres/electronic/index.html",
    "https://zaycev.net/genres/rap/index.html",
    "https://zaycev.net/genres/pop/index.html",
    "https://zaycev.net/genres/easy/index.html",
    "https://zaycev.net/genres/classical/index.html",
    "https://zaycev.net/genres/shanson/index.html",
    "https://zaycev.net/genres/metal/index.html",
    "https://zaycev.net/genres/indie/index.html",
    "https://zaycev.net/genres/blues/index.html",
    "https://zaycev.net/genres/reggae/index.html",
    "https://zaycev.net/genres/dance/index.html"
]

# Paths for saving dataset
DATASET_PATH = "music_dataset"
os.makedirs(DATASET_PATH, exist_ok=True)

# Set up CSV metadata file
metadata_file = os.path.join(DATASET_PATH, "metadata.csv")
with open(metadata_file, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["genre", "title", "artist", "duration", "url"])

# Function to clean text
def clean_text(text):
    return text.strip() if text else "Unknown"

# Configure the WebDriver
def setup_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)
    return driver

# Function to scrape tracks in a genre
def scrape_genre_tracks(driver, genre_url, genre_name):
    driver.get(genre_url)
    time.sleep(3)  # Allow time for page load
    
    while True:
        # Parse page source with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, "html.parser")
        tracks = soup.select("div[data-qa='track']")
        
        for track in tracks:
            title = clean_text(track.get("title"))
            artist = clean_text(track.select_one("a[data-qa='artist-link']").text if track.select_one("a[data-qa='artist-link']") else "Unknown Artist")
            duration = clean_text(track.select_one("time[data-qa='track-duration']").text if track.select_one("time[data-qa='track-duration']") else "Unknown Duration")
            track_url = "https://zaycev.net" + track.select_one("a[data-qa='track-link']")["href"]

            # Write track info to CSV
            with open(metadata_file, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([genre_name, title, artist, duration, track_url])

        # Check and click "Показать ещё" button if present
        try:
            load_more_button = driver.find_element_by_css_selector("button[data-qa='load-more-tracks-button']")
            driver.execute_script("arguments[0].click();", load_more_button)
            time.sleep(3)  # Allow time for additional tracks to load
        except:
            print(f"All tracks loaded for genre: {genre_name}")
            break

# Main script execution
driver = setup_driver()
for genre_url in GENRE_URLS:
    genre_name = genre_url.split("/")[-2]
    print(f"Scraping genre: {genre_name}")
    scrape_genre_tracks(driver, genre_url, genre_name)

driver.quit()
print("All genres scraped. Data saved to metadata.csv.")