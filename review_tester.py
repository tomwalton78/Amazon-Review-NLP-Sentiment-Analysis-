import nlp_model_v3 as rev_sent


def run_one(input_text):
	sent, conf = rev_sent.sentiment(input_text)
	if sent == 'pos':
		result = 'positive'
	else:
		result = 'negative'
	print(
		"The review was %s, with a confidence of: %s %%" % (result, str(conf * 100)))

while True:
	user_input = input("Enter review: ")
	if user_input != '':
		run_one(user_input)
	else:
		break
