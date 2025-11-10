# Initialize an empty list to store the user's input
user_inputs = []

# Prompt the user to enter text repeatedly until an empty input is provided
while True:
    text_input = input("Enter text (or press Enter to finish): ")

    # Check if the input is empty, indicating the user wants to stop
    if text_input == "":
        break  # Exit the loop

    # Add the non-empty input to the list
    user_inputs.append(text_input)

# Print the collected inputs
print("\nYour collected inputs are:")
for item in user_inputs:
    print(item)