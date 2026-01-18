def ask_for_name():
	return input("First, what is your name? \n")

def ask_multiple_choice_question(ques, num_opts):
	res = int(input(ques))
	if res <= 0 or res > num_opts:
		print(f"incorrect answer, please provide an answer 1-{num_opts}\n")
		return ask_multiple_choice_question(ques, num_opts)
	return res

def ask_pizza_legality():
	res = input("So you like pizza, how much pineapple can be put on pizza (a lot, a little, none)? \n")
	if res.lower() == 'none':
		print("CONGRATULATIONS, you are a good pizza eater")
	elif 'lot' in res.lower():
		print("That is disgusting, the pizza police will be coming for you")
	elif 'little' in res.lower():
		print("That is just wrong")
	else:
		print("not sure what you mean")
	return res

def report_result(results, name):
	score = 0
	for result in results:
		if isinstance(result, int) or isinstance(result, float):
			score += result
		elif isinstance(result, str):
			score += len(result)
		else:
			score += 1
	print(f"Congratulations, {name} you have scored {score}!")

def take_quiz():
	pizza_legality = ''
	name = ask_for_name()
	print(f"Welcome {name}, let's find out what you like for dinner")
	choice_1 = ask_multiple_choice_question("Which drink? 1. Water, 2. Juice, 3. Beer ", 3)
	choice_2 = ask_multiple_choice_question("Which food? 1. Pizza, 2. Hotdog ", 2)
	if choice_2 == 1:
		pizza_legality = ask_pizza_legality()
	report_result([choice_1, choice_2, pizza_legality], name)

if __name__ == '__main__':
	take_quiz()

