from twitter_scraper import get_tweets
import csv


with open('pages.txt', 'r') as pages:
    pages = pages.readlines()

for page in pages:
    tweets = get_tweets(page)
    filename = f'pages/{page}.csv'
    with open(filename, 'w') as pagefile:
        writer = csv.writer(pagefile)
        print('a')
        try:
            head = ['Date', 'Text', 'Classification']
            writer.writerow(head)
            print('b')
            for tweet in tweets:
                row = [tweet.get('time'), tweet.get('text'), '']
                writer.writerow(row)
                print('c')
        except Exception as e:
            print(e)
