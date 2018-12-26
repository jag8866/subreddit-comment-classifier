'''
Reddit Comment Harvester
Accepts any subreddit, creates a file with all comments in pickled plaintext.
'''
import praw
import pickle



def get_comments(sub, posts=100, pType="all"):
	#Create reddit object
	r = praw.Reddit(client_id='',
					client_secret="", password='',
					user_agent='', username='')

	already_done = set()
	all_comments = []

	#Pull from sub, either top posts of the month, hot, or top of all time
	if pType == "top":
		postGenerator = r.subreddit(sub).top(time_filter='month', limit=posts)
	elif pType == "hot":
		postGenerator = r.subreddit(sub).hot(limit=posts)
	else:
		postGenerator = r.subreddit(sub).top(time_filter='all', limit=posts)

	#Iterate through posts and pull the comments from them
	for i in range(posts):
		post = postGenerator.next()

		post.comments.replace_more(limit=3)
		for comment in post.comments:

			if not hasattr(comment, 'body'):
				continue

			all_comments.append(comment.body)

	#Write pickled comments to file
	f = open(sub + ".pk2", "w+")
	f.write(pickle.dumps(all_comments))
	f.close()

	return all_comments
