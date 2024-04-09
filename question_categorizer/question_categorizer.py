from transformers import pipeline


#question_txt is the question in string format
#question_types is a list of strings of the types/categories of questions
#MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli is the zero-shot-classification model that we are using
def questionCategorizer(question_txt, question_types):
	classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
	output = classifier(question_txt, question_types, multi_label=False)
	print(output)

	max_score_index = output["scores"].index(max(output["scores"]))
	max_score_category = question_types[max_score_index]
	print("Category with the maximum score:", max_score_category)

	return max_score_category


def parseCSV():
	