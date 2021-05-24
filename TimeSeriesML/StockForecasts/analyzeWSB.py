'''*****************************************************************************
Purpose: To analyze the sentiments of the reddit
This program uses Vader SentimentIntensityAnalyzer to calculate the ticker compound value. 
You can change multiple parameters to suit your needs. See below under "set program parameters."
Implementation:
I am using sets for 'x in s' comparison, sets time complexity for "x in s" is O(1) compare to list: O(n).
Limitations:
It depends mainly on the defined parameters for current implementation:
It completely ignores the heavily downvoted comments, and there can be a time when
the most mentioned ticker is heavily downvoted, but you can change that in upvotes variable.
Author: github:asad70
-------------------------------------------------------------------
****************************************************************************'''

import praw
from data import *
import time
import pandas as pd
import matplotlib.pyplot as plt
import squarify

start_time = time.time()
reddit = praw.Reddit(user_agent="Comment Extraction",
    client_id="3_xzISBumaCGgg",
    client_secret="MTMEoA3thvueOMCuPJdD2FKtPLi_mw",
    username="USERNAME",
    password="PASSWORD")


'''############################################################################'''
# set the program parameters
subs = ['wallstreetbets' ]     # sub-reddit to search
post_flairs = {'Daily Discussion', 'Weekend Discussion', 'Discussion'}    # posts flairs to search || None flair is automatically considered
goodAuth = {'AutoModerator'}   # authors whom comments are allowed more than once
uniqueCmt = True                # allow one comment per author per symbol
ignoreAuthP = {'example'}       # authors to ignore for posts 
ignoreAuthC = {'example'}       # authors to ignore for comment 
upvoteRatio = 0.33         # upvote ratio for post to be considered, 0.70 = 70%
ups = 5       # define # of upvotes, post is considered if upvotes exceed this #
limit = 5000     # define the limit, comments 'replace more' limit
upvotes = 5     # define # of upvotes, comment is considered if upvotes exceed this #
picks = 15     # define # of picks here, prints as "Top ## picks are:"
picks_ayz = 10   # define # of picks for sentiment analysis
'''############################################################################'''


posts, count, c_analyzed, tickers, titles, a_comments = 0, 0, 0, {}, [], {}
cmt_auth = {}


for sub in subs:
    subreddit = reddit.subreddit(sub)
    hot_python = subreddit.hot()    # sorting posts by hot
    # Extracting comments, symbols from subreddit
    for submission in hot_python:
        flair = submission.link_flair_text 
        author = submission.author.name         
        
        # checking: post upvote ratio # of upvotes, post flair, and author 
        if submission.upvote_ratio >= upvoteRatio and submission.ups > ups and (flair in post_flairs or flair is None) and author not in ignoreAuthP:   
            submission.comment_sort = 'new'     
            comments = submission.comments
            titles.append(submission.title)
            posts += 1
            try: 
                submission.comments.replace_more(limit=limit)   
                for comment in comments:
                    # try except for deleted account?
                    try: auth = comment.author.name
                    except: pass
                    c_analyzed += 1
                    
                    # checking: comment upvotes and author
                    if comment.score > upvotes and auth not in ignoreAuthC:      
                        split = comment.body.split(" ")
                        for word in split:
                            word = word.replace("$", "")        
                            # upper = ticker, length of ticker <= 5, excluded words,                     
                            if word.isupper() and len(word) <= 5 and word not in blacklist and word in us:
                                
                                # unique comments, try/except for key errors
                                if uniqueCmt and auth not in goodAuth:
                                    try: 
                                        if auth in cmt_auth[word]: break
                                    except: pass
                                    
                                # counting tickers
                                if word in tickers:
                                    tickers[word] += 1
                                    a_comments[word].append(comment.body)
                                    cmt_auth[word].append(auth)
                                    count += 1
                                else:                               
                                    tickers[word] = 1
                                    cmt_auth[word] = [auth]
                                    a_comments[word] = [comment.body]
                                    count += 1   
            except Exception as e: print(e)
            
                       

# sorts the dictionary
symbols = dict(sorted(tickers.items(), key=lambda item: item[1], reverse = True))
top_picks = list(symbols.keys())[0:picks]
time = (time.time() - start_time)

# print top picks
#print("It took {t:.2f} seconds to analyze {c} comments in {p} posts in {s} subreddits.\n".format(t=time, c=c_analyzed, p=posts, s=len(subs)))
#print("Posts analyzed saved in titles")
#for i in titles: print(i)  # prints the title of the posts analyzed


#print(f"\n{picks} most mentioned picks: ")
times = []
top = []
for i in top_picks:
    print(f"{i}: {symbols[i]}")
    times.append(symbols[i])
    top.append(f"{i}: {symbols[i]}")
   
    
# Applying Sentiment Analysis
scores, s = {}, {}
 
# Date Visualization
# most mentioned picks    
squarify.plot(sizes=times, label=top, alpha=.7 )
plt.axis('off')
plt.title(f"{picks} most mentioned reddit wallstreetbets stocks")
plt.savefig('e:\Temp\WSBCloud.png')
#plt.show()
