def parse_search_query(query):
    operators = {'AND', 'OR', 'NOT'}
    words = query.split()
    tokens = []
    current_phrase = []
    for word in words:
        if word.upper() in operators:
            # Add the current phrase to tokens if it's not empty
            if current_phrase:
                tokens.append(' '.join(current_phrase).lower())
                current_phrase = []
            # Add the operator to tokens
            tokens.append(word.upper())
        else:
            current_phrase.append(word)

    # Add the last phrase if it exists
    if current_phrase:
        tokens.append(' '.join(current_phrase).lower())

    # Convert tokens to list of tuples
    result = []
    i = 0
    while i < len(tokens):
        if i == 0:
            # First term has no operator
            result.append((None, tokens[i]))
            i += 1
        else:
            # Subsequent terms have an operator
            if i + 1 < len(tokens):
                result.append((tokens[i], tokens[i + 1]))
                i += 2
            else:
                # This handles the case where there's a trailing operator without a term
                result.append((tokens[i], ''))
                i += 1
    return result


def has_title_in_list(record, record_list):
    """Check if a record with the same title exists in the record_list."""
    return any(existing[3].lower() == record[3].lower() for existing in record_list)


def get_search_results(search_query, records):
    search_terms = parse_search_query(search_query)
    results_so_far = []
    final_results = []

    for operator, term in search_terms:
        if operator is None:
            # First term - search through all records
            for record in records:
                if term in record[3].lower() and not has_title_in_list(record, final_results + results_so_far):
                    results_so_far.append(record)

        elif operator == 'AND':
            # Filter results_so_far to only include records that also match this term
            results_so_far = [record for record in results_so_far
                              if term in record[3].lower()]

        elif operator == 'NOT':
            # Remove records that match this term from results_so_far
            results_so_far = [record for record in results_so_far
                              if term not in record[3].lower()]

        elif operator == 'OR':
            # Add current results to final results (without duplicates)
            for record in results_so_far:
                if not has_title_in_list(record, final_results):
                    final_results.append(record)

            # Reset and start a new search for the OR condition
            results_so_far = []
            for record in records:
                if (term in record[3].lower() and
                        not has_title_in_list(record, final_results) and
                        not has_title_in_list(record, results_so_far)):
                    results_so_far.append(record)

    # Add any remaining results to final results
    for record in results_so_far:
        if not has_title_in_list(record, final_results):
            final_results.append(record)

    return final_results


if __name__ == "__main__":
    check_list = [
        (2022, 1, 1, "lions", "[2022]"),
        (2022, 1, 1, "tigers", "[2022]"),
        (2022, 1, 1, "bears", "[2022]"),
        (2022, 1, 1, "lions tigers", "[2022]"),
        (2022, 1, 1, "tigers bears", "[2022]"),
        (2022, 1, 1, "lions bears", "[2022]"),
        (2022, 1, 1, "lions tigers bears", "[2022]"),
    ]

    print(len(get_search_results("lions AND tigers", check_list)) == 2)
    print(len(get_search_results("lions OR tigers", check_list)) == 6)
    print(len(get_search_results("lions NOT tigers", check_list)) == 2)

    print(len(get_search_results("lions AND tigers NOT bears", check_list)) == 1)
    print(len(get_search_results("lions AND tigers OR bears", check_list)) == 5)
    print(len(get_search_results("lions AND tigers AND bears", check_list)) == 1)

    print(len(get_search_results("lions NOT tigers AND bears", check_list)) == 1)
    print(len(get_search_results("lions NOT tigers OR bears", check_list)) == 5)
    print(len(get_search_results("lions NOT tigers NOT bears", check_list)) == 1)

    print(len(get_search_results("lions OR tigers AND bears", check_list)) == 5)
    print(len(get_search_results("lions OR tigers NOT bears", check_list)) == 5)
    print(len(get_search_results("lions OR tigers OR bears", check_list)) == 7)