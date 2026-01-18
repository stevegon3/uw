import pandas as pd
import numpy as np

def word_bin(series, start_letter, end_letter):
    """
    Filter a pandas Series of strings to include only words that start with letters
    between start_letter and end_letter (inclusive).
    
    Args:
        series (pd.Series): Series of strings to filter
        start_letter (str): Starting letter (inclusive)
        end_letter (str): Ending letter (inclusive)
        
    Returns:
        pd.Series: Filtered series containing only words that start with letters
                   between start_letter and end_letter (inclusive)
    """
    # Convert to lowercase for case-insensitive comparison
    first_letters = series.str[0].str.lower()
    start = start_letter.lower()
    end = end_letter.lower()
    
    # Create boolean mask for words starting between start and end letters (inclusive)
    mask = (first_letters >= start) & (first_letters <= end)
    return series[mask]

# Test the word_bin function
print("\nTesting word_bin function:")
test_series = pd.Series(['apple', 'banana', 'cherry', 'date', 'elderberry', 'fig', 'grape'])
result = word_bin(test_series, 'b', 'e')
print("Words between 'b' and 'e':")

# Create a random number generator with a fixed seed for reproducibility
random_generator = np.random.RandomState(42)  # Using 42 as the seed

# Generate 50 random salaries between 40000 and 60000
salaries_2020 = pd.Series(
    random_generator.integers(40000, 60001, size=50),
    name='2020 Salaries'
)

# Print the first five values
print("\nFirst five 2020 salaries:")
print(salaries_2020.head())

# Create employee names (Employee 1 through Employee 50)
employee_names = pd.Series([f'Employee {i+1}' for i in range(50)], name='employee')

# Create the salaries DataFrame
salaries_df = pd.DataFrame({
    'employee': employee_names,
    'salary_2020': salaries_2020
})

# Display the first few rows of the DataFrame
print("\nSalaries DataFrame:")
print(salaries_df.head())
print(result)

# Sample words
words = pd.Series(['apple', 'banana', 'cherry', 'date', 'elderberry'])

# Create a new Series with capitalized words
capital_words = words.str.capitalize()
print("\nCapitalized Words:")
print(capital_words)

# Filter words starting with 'a' through 'f'
words_a_f = words[words.str[0].str.lower().between('a', 'f')]
print("\nWords starting with 'a' through 'f':")
print(words_a_f)

# Sample single-digit integers
phone_numbers = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 5, 3, 2, 5])

# Print original numbers
print("Original Numbers:")
print(phone_numbers)

# Replace values at indices divisible by 3 with 511
mask = phone_numbers.index % 3 == 0
phone_numbers[mask] = 511

# Print the updated numbers
print("\nUpdated Numbers (indices divisible by 3 replaced with 511):")
print(phone_numbers)
