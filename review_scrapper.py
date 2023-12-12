from google_play_scraper import app, Sort, reviews
import csv

# Replace 'com.example.app' with the actual package name of the app
app_package = 'com.facebook.katana'

# Scraping app information
app_info = app(app_package)

# Specify the filename for the CSV file
csv_filename = 'fb.csv'
# Specify the number of reviews you want to scrape
num_reviews = 155000

# Scraping reviews
result, _ = reviews(app_package, lang='en', sort=Sort.MOST_RELEVANT, count=num_reviews)

# Extracting review details
reviews_list = []
for review in result:
    review_text = review['content']
    review_rating = review['score']
    reviews_list.append((review_text, review_rating))

# Saving reviews to a CSV file
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Review Text', 'Rating']
    writer = csv.writer(csvfile)
    writer.writerow(fieldnames)
    writer.writerows(reviews_list)

print(f'Reviews scraped and saved to {csv_filename}')
