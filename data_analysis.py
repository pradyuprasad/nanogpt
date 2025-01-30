# Read the file
with open('input.txt', 'r') as file:
    text = file.read()

# Get total length
total_chars = len(text)

# Create a dictionary to store character counts
char_counts = {}
for char in text:
    char_counts[char] = char_counts.get(char, 0) + 1

# Calculate and display percentages, sorted by percentage
print("Character distribution (highest to lowest):")
sorted_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
for char, count in sorted_chars:
    percentage = (count / total_chars) * 100
    print(f"'{char}': {percentage:.2f}%")
